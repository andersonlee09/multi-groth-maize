import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_points3d.modules.KPConv.kernels import KPConvLayer
from torch_scatter import scatter_softmax
from timm.models.layers import DropPath, trunc_normal_
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_geometric.nn import voxel_grid
from lib.pointops2.functions import pointops


# ============================================================
# 1. Original utilities
# ============================================================

def grid_sample(pos, batch, size, start=None, return_p2v=True):

    cluster = voxel_grid(
        pos,
        batch,
        size,
        start=start
    )

    if not return_p2v:
        _, cluster = torch.unique(
            cluster,
            sorted=True,
            return_inverse=True
        )
        return cluster

    unique, cluster, counts = torch.unique(
        cluster,
        sorted=True,
        return_inverse=True,
        return_counts=True
    )

    n = unique.shape[0]
    k = counts.max().item()

    p2v_map = cluster.new_zeros(n, k)

    mask = (
        torch.arange(k, device=pos.device).unsqueeze(0)
        < counts.unsqueeze(-1)
    )

    p2v_map[mask] = torch.argsort(cluster)

    return cluster, p2v_map, counts


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


# ============================================================
# 2. MNAD
# Multi-scale Neighborhood Aggregation Downsampling
# ============================================================

class MNAD(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        ratio,
        k,
        radii=None,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.ratio = ratio

        # Three neighborhood scales
        self.nsamples = [
            max(1, k),
            max(1, k * 2),
            max(1, k * 3)
        ]

        # Radii must match the coordinate units.
        # These are initial engineering settings.
        self.radii = (
            tuple(radii)
            if radii is not None
            else (0.05, 0.10, 0.20)
        )

        if len(self.radii) != 3:
            raise ValueError("MNAD requires three radii.")

        if not (
            self.radii[0]
            < self.radii[1]
            < self.radii[2]
        ):
            raise ValueError(
                "MNAD radii must be strictly increasing."
            )

        # Neighbor feature:
        # neighbor feature C
        # relative feature C
        # relative xyz 3
        # distance 1
        input_dim = in_channels * 2 + 4

        self.scale_mlps = nn.ModuleList()

        for _ in range(3):

            self.scale_mlps.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_channels),
                    norm_layer(out_channels),
                    nn.GELU(),

                    nn.Linear(out_channels, out_channels),
                    norm_layer(out_channels),
                    nn.GELU()
                )
            )

        self.fusion = nn.Linear(
            out_channels * 3,
            out_channels,
            bias=False
        )

        self.shortcut = nn.Linear(
            in_channels,
            out_channels,
            bias=False
        )

        self.norm = norm_layer(out_channels)
        self.act = nn.GELU()

    @staticmethod
    def build_offsets(offset, ratio, device):

        counts = torch.diff(
            torch.cat([
                offset.new_zeros(1),
                offset
            ])
        )

        new_counts = torch.clamp(
            torch.ceil(
                counts.float() * ratio
            ).long(),
            min=1
        )

        new_counts = torch.minimum(
            new_counts,
            counts.long()
        )

        new_offset = torch.cumsum(
            new_counts,
            dim=0
        )

        return new_offset.to(
            device=device,
            dtype=torch.int32
        )

    @staticmethod
    def radius_group(
        xyz,
        center_xyz,
        feats,
        offset,
        center_offset,
        radius,
        nsample
    ):
        """
        Radius-constrained grouping.

        Uses pointops.queryandgroup as a candidate search,
        then applies an explicit radius mask.

        Important:
        This is an engineering implementation using the
        existing pointops interface, not a native ball-query
        CUDA implementation.

        Returns:
            grouped_features: [M, K, C]
            relative_xyz: [M, K, 3]
            valid_mask: [M, K]
        """

        # Candidate search.
        # use_xyz=True gives relative xyz plus features.
        grouped = pointops.queryandgroup(
            nsample,
            xyz,
            center_xyz,
            feats,
            None,
            offset,
            center_offset,
            use_xyz=True
        )

        relative_xyz = grouped[:, :, :3]
        grouped_feats = grouped[:, :, 3:]

        distances = torch.linalg.norm(
            relative_xyz,
            dim=-1
        )

        valid_mask = distances <= radius

        return (
            grouped_feats,
            relative_xyz,
            valid_mask
        )

    def forward(self, feats, xyz, offset):

        # ----------------------------------------------------
        # Step 1: FPS
        # ----------------------------------------------------

        n_offset = self.build_offsets(
            offset,
            self.ratio,
            xyz.device
        )

        idx = pointops.furthestsampling(
            xyz.contiguous(),
            offset.int().contiguous(),
            n_offset.int().contiguous()
        )

        idx = idx.long()

        n_xyz = xyz[idx]
        center_feats = feats[idx]

        # ----------------------------------------------------
        # Step 2: Three-scale neighborhood aggregation
        # ----------------------------------------------------

        multi_scale_feats = []

        for scale_idx in range(3):

            radius = self.radii[scale_idx]
            nsample = self.nsamples[scale_idx]

            (
                neighbor_feats,
                relative_xyz,
                valid_mask
            ) = self.radius_group(
                xyz,
                n_xyz,
                feats,
                offset,
                n_offset,
                radius,
                nsample
            )

            m, k, c = neighbor_feats.shape

            center_expand = (
                center_feats.unsqueeze(1)
                .expand(-1, k, -1)
            )

            relative_feats = (
                neighbor_feats - center_expand
            )

            distances = torch.linalg.norm(
                relative_xyz,
                dim=-1,
                keepdim=True
            )

            local_feats = torch.cat(
                [
                    neighbor_feats,
                    relative_feats,
                    relative_xyz,
                    distances
                ],
                dim=-1
            )

            # Shared MLP
            local_feats = self.scale_mlps[
                scale_idx
            ](local_feats)

            # Radius mask before max pooling
            local_feats = local_feats.masked_fill(
                ~valid_mask.unsqueeze(-1),
                torch.finfo(local_feats.dtype).min
            )

            # Max pooling
            pooled = local_feats.max(dim=1).values

            # Fallback when no candidate is valid
            empty = ~valid_mask.any(dim=1)

            if empty.any():
                pooled[empty] = 0

            multi_scale_feats.append(pooled)

        # ----------------------------------------------------
        # Step 3: Multi-scale fusion
        # ----------------------------------------------------

        fused_feats = torch.cat(
            multi_scale_feats,
            dim=-1
        )

        fused_feats = self.fusion(fused_feats)

        # ----------------------------------------------------
        # Step 4: Residual feature fusion
        # ----------------------------------------------------

        shortcut_feats = self.shortcut(
            center_feats
        )

        output_feats = self.act(
            self.norm(
                fused_feats + shortcut_feats
            )
        )

        return output_feats, n_xyz, n_offset


# ============================================================
# 3. Geometry Estimator
# Normal vectors and curvature
# ============================================================

class GeometryEstimator(nn.Module):

    def __init__(self, k=16):
        super().__init__()
        self.k = k

    @torch.no_grad()
    def forward(self, xyz, offset):

        # 1. 查询局部邻域
        grouped = pointops.queryandgroup(
            self.k,
            xyz,
            xyz,
            xyz,
            None,
            offset,
            offset,
            use_xyz=False
        )

        # 2. 关闭局部 AMP
        # 不影响网络其他部分的混合精度训练
        with torch.cuda.amp.autocast(enabled=False):

            # 强制使用 float32
            local_xyz = grouped.to(torch.float32)

            # 3. 计算邻域中心
            mean_xyz = local_xyz.mean(
                dim=1,
                keepdim=True
            )

            # 4. 坐标中心化
            centered = local_xyz - mean_xyz

            # 5. 计算协方差矩阵
            covariance = torch.matmul(
                centered.transpose(1, 2),
                centered
            )

            covariance = covariance / max(
                self.k - 1,
                1
            )

            # 6. 确保协方差矩阵为 float32
            covariance = covariance.to(torch.float32)

            # 7. 特征值分解
            eigenvalues, eigenvectors = torch.linalg.eigh(
                covariance
            )

            # 8. 提取法向量
            normals = eigenvectors[:, :, 0]

            normals = F.normalize(
                normals,
                p=2,
                dim=-1,
                eps=1e-8
            )

            # 9. 计算曲率
            eigenvalues = eigenvalues.clamp_min(0)

            curvature = (
                eigenvalues[:, 0:1]
                /
                (
                    eigenvalues.sum(
                        dim=-1,
                        keepdim=True
                    ) + 1e-8
                )
            )

        return normals, curvature
# ============================================================
# 4. PGGA
# Position Geometric Attention
# ============================================================

class PGGA(nn.Module):

    def __init__(
        self,
        num_heads,
        hidden_dim=32
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Relative position embedding
        self.position_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Normal difference embedding
        self.normal_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Curvature difference embedding
        self.curvature_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Position-normal cross attention
        self.q_normal = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.k_normal = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.v_normal = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        # Position-curvature cross attention
        self.q_curvature = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.k_curvature = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.v_curvature = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        # Fusion
        self.fusion = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Geometry-dependent attention bias
        self.bias_projection = nn.Linear(
            hidden_dim,
            num_heads
        )

        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        xyz,
        normals,
        curvature,
        index_0,
        index_1
    ):

        # ----------------------------------------------------
        # Step 1: Relative geometric descriptors
        # ----------------------------------------------------

        relative_xyz = (
            xyz[index_1] - xyz[index_0]
        )

        # PCA normals have sign ambiguity.
        # Align neighboring normals before subtraction.
        normal_i = normals[index_0]
        normal_j = normals[index_1]

        sign = torch.where(
            (normal_i * normal_j).sum(
                dim=-1,
                keepdim=True
            ) >= 0,
            1.0,
            -1.0
        )

        relative_normal = (
            normal_j * sign - normal_i
        )

        relative_curvature = (
            curvature[index_1]
            - curvature[index_0]
        )

        # ----------------------------------------------------
        # Step 2: Geometric embeddings
        # ----------------------------------------------------

        position_feat = self.position_mlp(
            relative_xyz
        )

        normal_feat = self.normal_mlp(
            relative_normal
        )

        curvature_feat = self.curvature_mlp(
            relative_curvature
        )

        # ----------------------------------------------------
        # Step 3: Position-normal interaction
        # ----------------------------------------------------

        q_n = self.q_normal(position_feat)
        k_n = self.k_normal(normal_feat)
        v_n = self.v_normal(normal_feat)

        score_n = (
            (q_n * k_n).sum(dim=-1, keepdim=True)
            * self.scale
        )

        # ----------------------------------------------------
        # Step 4: Position-curvature interaction
        # ----------------------------------------------------

        q_c = self.q_curvature(position_feat)
        k_c = self.k_curvature(curvature_feat)
        v_c = self.v_curvature(curvature_feat)

        score_c = (
            (q_c * k_c).sum(dim=-1, keepdim=True)
            * self.scale
        )

        # ----------------------------------------------------
        # Step 5: Dual-branch attention
        # ----------------------------------------------------

        # Normalize over the neighbors of each query point.
        weight_n = scatter_softmax(
            score_n,
            index_0,
            dim=0
        )

        weight_c = scatter_softmax(
            score_c,
            index_0,
            dim=0
        )

        output_n = weight_n * v_n
        output_c = weight_c * v_c

        # ----------------------------------------------------
        # Step 6: Geometric fusion
        # ----------------------------------------------------

        fused = torch.cat(
            [output_n, output_c],
            dim=-1
        )

        fused = self.fusion(fused)

        fused = self.norm(
            fused + position_feat
        )

        # ----------------------------------------------------
        # Step 7: Geometry attention bias
        # ----------------------------------------------------

        geometry_bias = self.bias_projection(
            fused
        )

        return geometry_bias


# ============================================================
# 5. WindowAttention + PGGA
# ============================================================

class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        quant_size,
        rel_query=True,
        rel_key=False,
        rel_value=False,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads."
            )

        head_dim = dim // num_heads

        self.scale = (
            qk_scale or head_dim ** -0.5
        )

        self.quant_size = quant_size

        self.rel_query = rel_query
        self.rel_key = rel_key
        self.rel_value = rel_value

        quant_grid_length = int(
            window_size / quant_size
        )

        self.quant_grid_length = quant_grid_length

        if rel_query:
            self.relative_pos_query_table = nn.Parameter(
                torch.zeros(
                    2 * quant_grid_length - 1,
                    num_heads,
                    head_dim,
                    3
                )
            )

            trunc_normal_(
                self.relative_pos_query_table,
                std=.02
            )

        if rel_key:
            self.relative_pos_key_table = nn.Parameter(
                torch.zeros(
                    2 * quant_grid_length - 1,
                    num_heads,
                    head_dim,
                    3
                )
            )

            trunc_normal_(
                self.relative_pos_key_table,
                std=.02
            )

        if rel_value:
            self.relative_pos_value_table = nn.Parameter(
                torch.zeros(
                    2 * quant_grid_length - 1,
                    num_heads,
                    head_dim,
                    3
                )
            )

            trunc_normal_(
                self.relative_pos_value_table,
                std=.02
            )

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias
        )

        # New PGGA module
        self.pgga = PGGA(
            num_heads=num_heads,
            hidden_dim=32
        )

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

    def map_func(self, relative_position):

        return (
            relative_position
            + self.quant_grid_length
            - 1
        )

    def forward(
        self,
        feats,
        xyz,
        index_0,
        index_0_offsets,
        n_max,
        index_1,
        shift_size,
        normals,
        curvature
    ):

        N, C = feats.shape

        # ----------------------------------------------------
        # Step 1: QKV
        # ----------------------------------------------------

        qkv = self.qkv(feats).reshape(
            N,
            3,
            self.num_heads,
            C // self.num_heads
        ).permute(
            1, 0, 2, 3
        ).contiguous()

        query, key, value = (
            qkv[0],
            qkv[1],
            qkv[2]
        )

        query = query * self.scale

        # ----------------------------------------------------
        # Step 2: Original attention scores
        # ----------------------------------------------------

        attn_flat = pointops.attention_step1_v2(
            query.float(),
            key.float(),
            index_1.int(),
            index_0_offsets.int(),
            n_max
        )

        # ----------------------------------------------------
        # Step 3: Original relative position bias
        # ----------------------------------------------------

        xyz_quant = (
            xyz - xyz.min(0)[0] + shift_size
        ) % self.window_size

        xyz_quant = (
            xyz_quant // self.quant_size
        )

        relative_position = (
            xyz_quant[index_0]
            - xyz_quant[index_1]
        )

        relative_position_index = self.map_func(
            relative_position
        )

        if self.rel_query and self.rel_key:

            relative_position_bias = (
                pointops.dot_prod_with_idx_v3(
                    query.float(),
                    index_0_offsets.int(),
                    n_max,
                    key.float(),
                    index_1.int(),
                    self.relative_pos_query_table.float(),
                    self.relative_pos_key_table.float(),
                    relative_position_index.int()
                )
            )

        elif self.rel_query:

            relative_position_bias = (
                pointops.dot_prod_with_idx(
                    query.float(),
                    index_0.int(),
                    self.relative_pos_query_table.float(),
                    relative_position_index.int()
                )
            )

        elif self.rel_key:

            relative_position_bias = (
                pointops.dot_prod_with_idx(
                    key.float(),
                    index_1.int(),
                    self.relative_pos_key_table.float(),
                    relative_position_index.int()
                )
            )

        else:

            relative_position_bias = 0

        # ----------------------------------------------------
        # Step 4: PGGA geometry bias
        # ----------------------------------------------------

        geometry_bias = self.pgga(
            xyz,
            normals,
            curvature,
            index_0,
            index_1
        )

        # ----------------------------------------------------
        # Step 5: Geometry-aware attention
        # ----------------------------------------------------

        attn_flat = (
            attn_flat
            + relative_position_bias
            + geometry_bias
        )

        softmax_attn_flat = scatter_softmax(
            src=attn_flat,
            index=index_0,
            dim=0
        )

        softmax_attn_flat = self.attn_drop(
            softmax_attn_flat
        )

        # ----------------------------------------------------
        # Step 6: Attention aggregation
        # ----------------------------------------------------

        if self.rel_value:

            x = (
                pointops.attention_step2_with_rel_pos_value_v2(
                    softmax_attn_flat.float(),
                    value.float(),
                    index_0_offsets.int(),
                    n_max,
                    index_1.int(),
                    self.relative_pos_value_table.float(),
                    relative_position_index.int()
                )
            )

        else:

            x = pointops.attention_step2(
                softmax_attn_flat.float(),
                value.float(),
                index_0.int(),
                index_1.int()
            )

        x = x.view(N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# ============================================================
# 6. RSM Transformer Block
# ============================================================

class RSMTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        quant_size,
        rel_query=True,
        rel_key=False,
        rel_value=False,
        drop_path=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mode=4
    ):
        super().__init__()

        self.window_size = window_size
        self.mode = mode

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            quant_size=quant_size,
            rel_query=rel_query,
            rel_key=rel_key,
            rel_value=rel_value,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(
            dim * mlp_ratio
        )

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer
        )

    def forward(
        self,
        feats,
        xyz,
        index_0,
        index_0_offsets,
        n_max,
        index_1,
        shift_size,
        normals,
        curvature
    ):

        shortcut = feats

        feats = self.norm1(feats)

        feats = self.attn(
            feats,
            xyz,
            index_0,
            index_0_offsets,
            n_max,
            index_1,
            shift_size,
            normals,
            curvature
        )

        feats = shortcut + self.drop_path(feats)

        feats = feats + self.drop_path(
            self.mlp(
                self.norm2(feats)
            )
        )

        return feats


# ============================================================
# 7. BasicLayer
# ============================================================

class BasicLayer(nn.Module):

    def __init__(
        self,
        depth,
        channel,
        num_heads,
        window_size,
        grid_size,
        quant_size,
        rel_query=True,
        rel_key=False,
        rel_value=False,
        drop_path=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        downsample=None,
        ratio=0.25,
        k=16,
        out_channels=None
    ):
        super().__init__()

        self.window_size = window_size
        self.depth = depth
        self.grid_size = grid_size

        self.max_window_counts = 64

        # Geometry estimator
        self.geometry_estimator = GeometryEstimator(
            k=16
        )

        self.blocks = nn.ModuleList([
            RSMTransformerBlock(
                channel,
                num_heads,
                window_size,
                quant_size,
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=(
                    drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path
                ),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.downsample = (
            downsample(
                channel,
                out_channels,
                ratio,
                k
            )
            if downsample
            else None
        )

    @staticmethod
    def build_window_indices(p2v_map, counts, device):

        n, k = p2v_map.shape

        mask = (
            torch.arange(
                k,
                device=device
            ).unsqueeze(0)
            < counts.unsqueeze(-1)
        )

        mask_mat = (
            mask.unsqueeze(-1)
            & mask.unsqueeze(-2)
        )

        index_0 = (
            p2v_map.unsqueeze(-1)
            .expand(-1, -1, k)[mask_mat]
        )

        index_1 = (
            p2v_map.unsqueeze(1)
            .expand(-1, k, -1)[mask_mat]
        )

        index_0, indices = torch.sort(index_0)

        index_1 = index_1[indices]

        return index_0, index_1

    @staticmethod
    def build_attention_offsets(index_0, N, device):

        counts = torch.bincount(
            index_0,
            minlength=N
        )

        n_max = int(counts.max().item())

        offsets = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=torch.long,
                    device=device
                ),
                counts.cumsum(dim=0)
            ],
            dim=0
        )

        return offsets, n_max

    def forward(self, feats, xyz, offset):

        device = xyz.device

        N, C = feats.shape

        # ----------------------------------------------------
        # Step 1: Geometry estimation
        # ----------------------------------------------------

        normals, curvature = (
            self.geometry_estimator(
                xyz,
                offset
            )
        )

        # ----------------------------------------------------
        # Step 2: Batch indices
        # ----------------------------------------------------

        counts = torch.diff(
            torch.cat([
                offset.new_zeros(1),
                offset
            ])
        ).long()

        batch = torch.repeat_interleave(
            torch.arange(
                len(counts),
                device=device
            ),
            counts
        )

        window_size = torch.tensor(
            [self.window_size] * 3,
            dtype=xyz.dtype,
            device=device
        )

        # ----------------------------------------------------
        # Step 3: Normal windows
        # ----------------------------------------------------

        _, p2v_map, counts_normal = grid_sample(
            xyz,
            batch,
            window_size,
            start=None
        )

        index_0, index_1 = (
            self.build_window_indices(
                p2v_map,
                counts_normal,
                device
            )
        )

        index_0_offsets, n_max = (
            self.build_attention_offsets(
                index_0,
                N,
                device
            )
        )

        # ----------------------------------------------------
        # Step 4: Shifted windows
        # ----------------------------------------------------

        shift_size = 0.5 * window_size

        _, shift_p2v_map, shift_counts = grid_sample(
            xyz + shift_size,
            batch,
            window_size,
            start=xyz.min(dim=0)[0]
        )

        index_0_shift, index_1_shift = (
            self.build_window_indices(
                shift_p2v_map,
                shift_counts,
                device
            )
        )

        (
            index_0_offsets_shift,
            n_max_shift
        ) = self.build_attention_offsets(
            index_0_shift,
            N,
            device
        )

        # ----------------------------------------------------
        # Step 5: Transformer blocks
        # ----------------------------------------------------

        for i, blk in enumerate(self.blocks):

            if i % 2 == 0:

                index_0_blk = index_0
                index_1_blk = index_1

                index_0_offsets_blk = (
                    index_0_offsets
                )

                n_max_blk = n_max
                shift_size_blk = 0.0

            else:

                index_0_blk = index_0_shift
                index_1_blk = index_1_shift

                index_0_offsets_blk = (
                    index_0_offsets_shift
                )

                n_max_blk = n_max_shift
                shift_size_blk = shift_size

            feats = blk(
                feats,
                xyz,
                index_0_blk,
                index_0_offsets_blk,
                n_max_blk,
                index_1_blk,
                shift_size_blk,
                normals,
                curvature
            )

        # ----------------------------------------------------
        # Step 6: MNAD downsampling
        # ----------------------------------------------------

        if self.downsample is not None:

            (
                feats_down,
                xyz_down,
                offset_down
            ) = self.downsample(
                feats,
                xyz,
                offset
            )

        else:

            feats_down = None
            xyz_down = None
            offset_down = None

        return (
            feats,
            xyz,
            offset,
            feats_down,
            xyz_down,
            offset_down
        )


# ============================================================
# 8. Original Upsample
# ============================================================

class Upsample(nn.Module):

    def __init__(
        self,
        k,
        in_channels,
        out_channels,
        bn_momentum=0.02
    ):
        super().__init__()

        self.k = k

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(
                out_channels,
                out_channels
            )
        )

        self.linear2 = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(
                in_channels,
                out_channels
            )
        )

    def forward(
        self,
        feats,
        xyz,
        support_xyz,
        offset,
        support_offset,
        support_feats=None
    ):

        feats = (
            self.linear1(support_feats)
            +
            pointops.interpolation(
                xyz,
                support_xyz,
                self.linear2(feats),
                offset,
                support_offset
            )
        )

        return (
            feats,
            support_xyz,
            support_offset
        )


# ============================================================
# 9. Original KPConvSimpleBlock
# ============================================================

class KPConvSimpleBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        prev_grid_size,
        sigma=1.0,
        negative_slope=0.2,
        bn_momentum=0.02
    ):
        super().__init__()

        self.kpconv = KPConvLayer(
            in_channels,
            out_channels,
            point_influence=prev_grid_size * sigma,
            add_one=False
        )

        self.bn = FastBatchNorm1d(
            out_channels,
            momentum=bn_momentum
        )

        self.activation = nn.LeakyReLU(
            negative_slope=negative_slope
        )

    def forward(
        self,
        feats,
        xyz,
        batch,
        neighbor_idx
    ):

        feats = self.kpconv(
            xyz,
            xyz,
            neighbor_idx,
            feats
        )

        feats = self.activation(
            self.bn(feats)
        )

        return feats


# ============================================================
# 10. Original KPConvResBlock
# ============================================================

class KPConvResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        prev_grid_size,
        sigma=1.0,
        negative_slope=0.2,
        bn_momentum=0.02
    ):
        super().__init__()

        d_2 = out_channels // 4

        activation = nn.LeakyReLU(
            negative_slope=negative_slope
        )

        self.unary_1 = nn.Sequential(
            nn.Linear(
                in_channels,
                d_2,
                bias=False
            ),
            FastBatchNorm1d(
                d_2,
                momentum=bn_momentum
            ),
            activation
        )

        self.unary_2 = nn.Sequential(
            nn.Linear(
                d_2,
                out_channels,
                bias=False
            ),
            FastBatchNorm1d(
                out_channels,
                momentum=bn_momentum
            ),
            activation
        )

        self.kpconv = KPConvLayer(
            d_2,
            d_2,
            point_influence=prev_grid_size * sigma,
            add_one=False
        )

        self.bn = FastBatchNorm1d(
            out_channels,
            momentum=bn_momentum
        )

        self.activation = activation

        if in_channels != out_channels:

            self.shortcut_op = nn.Sequential(
                nn.Linear(
                    in_channels,
                    out_channels,
                    bias=False
                ),
                FastBatchNorm1d(
                    out_channels,
                    momentum=bn_momentum
                )
            )

        else:

            self.shortcut_op = nn.Identity()

    def forward(
        self,
        feats,
        xyz,
        batch,
        neighbor_idx
    ):

        shortcut = feats

        feats = self.unary_1(feats)

        feats = self.kpconv(
            xyz,
            xyz,
            neighbor_idx,
            feats
        )

        feats = self.unary_2(feats)

        shortcut = self.shortcut_op(
            shortcut
        )

        feats = feats + shortcut

        return feats


# ============================================================
# 11. Complete RSM-Net
# ============================================================

class RSM(nn.Module):

    def __init__(
        self,
        depths,
        channels,
        num_heads,
        window_sizes,
        up_k,
        grid_sizes,
        quant_sizes,
        rel_query=True,
        rel_key=False,
        rel_value=False,
        drop_path_rate=0.2,
        num_layers=4,
        concat_xyz=False,
        num_classes=13,
        ratio=0.25,
        k=16,
        prev_grid_size=0.04,
        sigma=1.0,
        stem_transformer=False
    ):
        super().__init__()

        dpr = [
            x.item()
            for x in torch.linspace(
                0,
                drop_path_rate,
                sum(depths)
            )
        ]

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        if stem_transformer:

            self.stem_layer = nn.ModuleList([
                KPConvSimpleBlock(
                    3 if not concat_xyz else 6,
                    channels[0],
                    prev_grid_size,
                    sigma=sigma
                )
            ])

            self.layer_start = 0

        else:

            self.stem_layer = nn.ModuleList([
                KPConvSimpleBlock(
                    3 if not concat_xyz else 6,
                    channels[0],
                    prev_grid_size,
                    sigma=sigma
                ),

                KPConvResBlock(
                    channels[0],
                    channels[0],
                    prev_grid_size,
                    sigma=sigma
                )
            ])

            # Replaced TransitionDown with MNAD
            self.downsample = MNAD(
                channels[0],
                channels[1],
                ratio,
                k
            )

            self.layer_start = 1

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.layers = nn.ModuleList([
            BasicLayer(
                depths[i],
                channels[i],
                num_heads[i],
                window_sizes[i],
                grid_sizes[i],
                quant_sizes[i],
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=dpr[
                    sum(depths[:i]):
                    sum(depths[:i + 1])
                ],
                downsample=(
                    MNAD
                    if i < num_layers - 1
                    else None
                ),
                ratio=ratio,
                k=k,
                out_channels=(
                    channels[i + 1]
                    if i < num_layers - 1
                    else None
                )
            )
            for i in range(
                self.layer_start,
                num_layers
            )
        ])

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.upsamples = nn.ModuleList([
            Upsample(
                up_k,
                channels[i],
                channels[i - 1]
            )
            for i in range(
                num_layers - 1,
                0,
                -1
            )
        ])

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                channels[0],
                channels[0]
            ),
            nn.BatchNorm1d(
                channels[0]
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                channels[0],
                num_classes
            )
        )

        self.init_weights()

    def forward(
        self,
        feats,
        xyz,
        offset,
        batch,
        neighbor_idx
    ):

        feats_stack = []
        xyz_stack = []
        offset_stack = []

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        for layer in self.stem_layer:

            feats = layer(
                feats,
                xyz,
                batch,
                neighbor_idx
            )

        feats = feats.contiguous()

        # ----------------------------------------------------
        # Initial MNAD
        # ----------------------------------------------------

        if self.layer_start == 1:

            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)

            feats, xyz, offset = self.downsample(
                feats,
                xyz,
                offset
            )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        for layer in self.layers:

            (
                feats,
                xyz,
                offset,
                feats_down,
                xyz_down,
                offset_down
            ) = layer(
                feats,
                xyz,
                offset
            )

            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)

            feats = feats_down
            xyz = xyz_down
            offset = offset_down

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        feats = feats_stack.pop()
        xyz = xyz_stack.pop()
        offset = offset_stack.pop()

        for upsample in self.upsamples:

            feats, xyz, offset = upsample(
                feats,
                xyz,
                xyz_stack.pop(),
                offset,
                offset_stack.pop(),
                support_feats=feats_stack.pop()
            )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        out = self.classifier(feats)

        return out

    def init_weights(self):

        def _init_weights(m):

            if isinstance(m, nn.Linear):

                trunc_normal_(
                    m.weight,
                    std=.02
                )

                if m.bias is not None:

                    nn.init.constant_(
                        m.bias,
                        0
                    )

            elif isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d
                )
            ):

                if m.bias is not None:

                    nn.init.constant_(
                        m.bias,
                        0
                    )

                if m.weight is not None:

                    nn.init.constant_(
                        m.weight,
                        1.0
                    )

        self.apply(_init_weights)