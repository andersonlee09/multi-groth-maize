import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import colorsys
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 参数区
# ============================================================
INPUT_PATH = ""
OUTPUT_DIR = ""

VOXEL_SIZE_FACTOR = 4.0
THETA_DEG         = 20.0
DIST_FACTOR       = 2.0
CURV_THRESH       = 0.10
KNN_NORMAL        = 30
MIN_SV_POINTS     = 3
MIN_CLUSTER_ORIG  = 300      # 大簇最少点数；小簇并入最近大簇
DO_VISUALIZE      = False    # 批量时建议 False；单文件想检查时可改 True
SAVE              = True

STEM_COL4_VALUE   = 1        # 第4列==1 → 茎，不参与聚类
STEM_COL5_VALUE   = 1        # 茎在输出第5列的值
LEAF_COL5_BASE    = 2        # 叶簇号从 2 开始（簇0→2，簇1→3...）



def cluster_leaf_points(leaf_pts, verbose=True):
    # --- 2. 平均点距 ---
    samp_n = min(20000, len(leaf_pts))
    rng = np.random.default_rng(0)
    samp = leaf_pts[rng.choice(len(leaf_pts), samp_n, replace=False)]
    samp_pcd = o3d.geometry.PointCloud()
    samp_pcd.points = o3d.utility.Vector3dVector(samp)
    avg_spacing = float(np.median(np.asarray(samp_pcd.compute_nearest_neighbor_distance())))
    if verbose: print(f"  平均点距: {avg_spacing:.5f}")

    # --- 3. 逐点法向 + 曲率 ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(leaf_pts)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=KNN_NORMAL))
    point_normals = np.asarray(pcd.normals)
    try:
        pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(knn=KNN_NORMAL))
        eigvals = np.linalg.eigvalsh(np.asarray(pcd.covariances))
        point_curv = eigvals[:, 0] / (eigvals.sum(axis=1) + 1e-12)
    except Exception:
        nn = NearestNeighbors(n_neighbors=KNN_NORMAL).fit(leaf_pts)
        _, idxs = nn.kneighbors(leaf_pts)
        point_curv = np.zeros(len(leaf_pts))
        for i in range(0, len(leaf_pts), 5000):
            b = idxs[i:i+5000]
            p = leaf_pts[b] - leaf_pts[b].mean(axis=1, keepdims=True)
            cov = np.einsum('nki,nkj->nij', p, p) / KNN_NORMAL
            ev  = np.linalg.eigvalsh(cov)
            point_curv[i:i+5000] = ev[:, 0] / (ev.sum(axis=1) + 1e-12)

    # --- 4. 超体素 ---
    voxel_size = avg_spacing * VOXEL_SIZE_FACTOR
    keys = np.floor(leaf_pts / voxel_size).astype(np.int64)
    unique_keys, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    N_sv = len(unique_keys)
    if verbose: print(f"  超体素数: {N_sv}   体素边长: {voxel_size:.4f}")

    sums = np.zeros((N_sv, 3)); np.add.at(sums, inverse, leaf_pts)
    centroids = sums / counts[:, None]
    nsum = np.zeros((N_sv, 3)); np.add.at(nsum, inverse, point_normals)
    sv_normals = nsum / counts[:, None]
    nr = np.linalg.norm(sv_normals, axis=1, keepdims=True)
    sv_normals = sv_normals / np.where(nr > 1e-8, nr, 1.0)
    csum = np.zeros(N_sv); np.add.at(csum, inverse, point_curv)
    sv_curv = csum / counts

    # --- 5. 26 邻接图 ---
    key_to_idx = {tuple(k): i for i, k in enumerate(unique_keys)}
    neighbors_26 = [(a, b, c) for a in (-1, 0, 1)
                    for b in (-1, 0, 1)
                    for c in (-1, 0, 1)
                    if not (a == 0 and b == 0 and c == 0)]
    ei, ej = [], []
    for i, k in enumerate(unique_keys):
        for dk in neighbors_26:
            nk = (k[0] + dk[0], k[1] + dk[1], k[2] + dk[2])
            j = key_to_idx.get(nk)
            if j is not None and j > i:
                ei.append(i); ej.append(j)
    ei = np.array(ei); ej = np.array(ej)
    if verbose: print(f"  邻接边数: {len(ei)}")

    # --- 6. 权重 + 硬约束 ---
    dist_th = voxel_size * DIST_FACTOR
    cos_vals = np.abs(np.sum(sv_normals[ei] * sv_normals[ej], axis=1))
    angles   = np.arccos(np.clip(cos_vals, -1.0, 1.0))
    dists    = np.linalg.norm(centroids[ei] - centroids[ej], axis=1)
    curv_d   = np.abs(sv_curv[ei] - sv_curv[ej])
    ok = (angles < np.deg2rad(THETA_DEG)) & (dists < dist_th) & (curv_d < CURV_THRESH)
    if verbose: print(f"  满足硬约束边: {ok.sum()}/{len(ok)}")
    weight = angles + 0.5 * (dists / dist_th) + 3.0 * curv_d
    weight[~ok] = np.inf

    # --- 7. Union-Find 合并 ---
    parent = np.arange(N_sv)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    order = np.argsort(weight)
    for o in order:
        if not np.isfinite(weight[o]): break
        i, j = ei[o], ej[o]
        if find(i) != find(j): union(i, j)
    roots = np.array([find(i) for i in range(N_sv)])
    _, sv_labels = np.unique(roots, return_inverse=True)

    # --- 8. 小簇并入最近大簇 ---
    labels_full = sv_labels[inverse]
    unique_l = np.unique(labels_full)
    sizes = {l: int((labels_full == l).sum()) for l in unique_l}
    big   = [l for l in unique_l if sizes[l] >= MIN_CLUSTER_ORIG]
    small = [l for l in unique_l if sizes[l] <  MIN_CLUSTER_ORIG]
    if len(big) == 0:
        raise RuntimeError(f"没有簇满足 MIN_CLUSTER_ORIG={MIN_CLUSTER_ORIG}")

    def centroid_of(lab):
        return leaf_pts[labels_full == lab].mean(axis=0)
    big_centroids = np.array([centroid_of(l) for l in big])

    labels_merged = labels_full.copy()
    for l in small:
        c  = centroid_of(l)
        d  = np.linalg.norm(big_centroids - c, axis=1)
        nb = big[int(np.argmin(d))]
        labels_merged[labels_full == l] = nb

    final_l = sorted(big)
    remap   = {old: new for new, old in enumerate(final_l)}
    labels_kept = np.array([remap[l] for l in labels_merged])
    n_final = len(final_l)
    if verbose:
        print(f"  最终聚类数: {n_final}, 叶点数: {len(leaf_pts)}")
    return labels_kept, n_final



def process_one(in_txt, out_txt):
    print(f"\n处理: {in_txt}")
    try:
        data = np.loadtxt(in_txt, comments='#', ndmin=2)
    except Exception as e:
        print(f"  [失败] 加载: {e}")
        return False

    if data.shape[1] < 4:
        print(f"  [跳过] 列数不足 4: {data.shape}")
        return False

    xyz    = data[:, :3].astype(np.float64)
    col4   = data[:, 3]
    stem_mask = np.isclose(col4, STEM_COL4_VALUE)
    leaf_mask = ~stem_mask

    print(f"  总点数: {len(data)}  "
          f"茎(第4列=1): {int(stem_mask.sum())}  "
          f"叶(第4列≠1): {int(leaf_mask.sum())}")

    # 茎 → 第5列 = 1
    col5 = np.full(len(data), STEM_COL5_VALUE, dtype=np.int64)
    n_final = 0

    if leaf_mask.sum() >= MIN_CLUSTER_ORIG:
        leaf_pts = xyz[leaf_mask]
        try:
            leaf_labels, n_final = cluster_leaf_points(leaf_pts, verbose=True)
            leaf_global_idx = np.where(leaf_mask)[0]
            col5[leaf_global_idx] = leaf_labels + LEAF_COL5_BASE
        except Exception as e:
            print(f"  [聚类失败] {e}")
            n_final = 0
    else:
        print(f"  [警告] 叶点 {int(leaf_mask.sum())} < MIN_CLUSTER_ORIG={MIN_CLUSTER_ORIG}，跳过聚类")
        leaf_pts = None

    # 组装输出：前 4 列原样 + 第 5 列
    out = np.column_stack([data[:, :4], col5])

    if SAVE:
        os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
        # 第4列是整数则 %d，否则 %.8f；XYZ 统一 %.8f
        col4_is_int = np.allclose(col4, np.round(col4))
        fmt = "%.8f %.8f %.8f %d %d" if col4_is_int else "%.8f %.8f %.8f %.8f %d"
        np.savetxt(out_txt, out, fmt=fmt)
        print(f"  [OK] 已保存: {out_txt}   (n_clusters={n_final})")

    # 可选可视化
    if DO_VISUALIZE and n_final > 0:
        colors = np.zeros((len(data), 3))
        colors[stem_mask] = [0.1216, 0.4667, 0.7059]           # 茎蓝
        leaf_colors = np.zeros((int(leaf_mask.sum()), 3))
        leaf_labels_vis = col5[leaf_mask] - LEAF_COL5_BASE     # 还原 0..K-1
        for i in range(n_final):
            h = i / max(n_final, 1)
            leaf_colors[leaf_labels_vis == i] = colorsys.hsv_to_rgb(h, 0.9, 0.95)
        colors[leaf_mask] = leaf_colors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{os.path.basename(in_txt)}  n={n_final}",
                          width=1280, height=900)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.point_size = 2.0
        vis.run(); vis.destroy_window()

    return True


if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        files = sorted(f for f in os.listdir(INPUT_PATH) if f.lower().endswith(".txt"))
        print(f"发现 {len(files)} 个 txt 文件 → 输出到 {OUTPUT_DIR}\n")
        ok = 0
        for fn in files:
            src = os.path.join(INPUT_PATH, fn)
            dst = os.path.join(OUTPUT_DIR, os.path.splitext(fn)[0] + "_clustered.txt")
            if process_one(src, dst): ok += 1
        print(f"\n==== 完成: {ok}/{len(files)} ====")
    else:
        base = os.path.splitext(os.path.basename(INPUT_PATH))[0]
        dst  = os.path.join(OUTPUT_DIR, base + "_clustered.txt")
        process_one(INPUT_PATH, dst)