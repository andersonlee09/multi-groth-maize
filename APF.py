import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


data = np.loadtxt("you input leaf txt")

xyz = data[:, 0:3]
rgb = data[:, 3:6] / 255.0
label = data[:, 6]
normals = data[:, 7:10]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)
pcd.normals = o3d.utility.Vector3dVector(normals)

print("Original point cloud")
o3d.visualization.draw_geometries([pcd])

voxel_size = 0.002
pcd = pcd.voxel_down_sample(voxel_size)

xyz = np.asarray(pcd.points)
rgb = np.asarray(pcd.colors)
normals = np.asarray(pcd.normals)

print("After voxel downsample")
o3d.visualization.draw_geometries([pcd])


k = 30
nbrs = NearestNeighbors(n_neighbors=k).fit(xyz)
_, idx = nbrs.kneighbors(xyz)


residuals = np.zeros(len(xyz))

for i in range(len(xyz)):
    neigh = xyz[idx[i]]
    centroid = neigh.mean(axis=0)

    n = normals[i]
    n = n / (np.linalg.norm(n) + 1e-8)

    residuals[i] = abs(np.dot(xyz[i] - centroid, n))


plt.figure()
plt.hist(residuals, bins=100)
plt.title("Residual Distribution")
plt.show()


res_norm = (residuals - residuals.min()) / (residuals.max() - residuals.min() + 1e-8)
colors = plt.cm.jet(res_norm)[:, :3]

pcd_res = o3d.geometry.PointCloud()
pcd_res.points = o3d.utility.Vector3dVector(xyz)
pcd_res.colors = o3d.utility.Vector3dVector(colors)

print("Residual heatmap (blue=good, red=ghost)")
o3d.visualization.draw_geometries([pcd_res])


threshold = np.percentile(residuals, 70)
print("Threshold:", threshold)

mask = residuals < threshold

xyz_keep = xyz[mask]
rgb_keep = rgb[mask]

xyz_remove = xyz[~mask]
rgb_remove = rgb[~mask]


pcd_keep = o3d.geometry.PointCloud()
pcd_keep.points = o3d.utility.Vector3dVector(xyz_keep)
pcd_keep.colors = o3d.utility.Vector3dVector(
    np.tile([0.0, 0.0, 1.0], (len(xyz_keep), 1))
)


pcd_remove = o3d.geometry.PointCloud()
pcd_remove.points = o3d.utility.Vector3dVector(xyz_remove)
pcd_remove.colors = o3d.utility.Vector3dVector(
    np.tile([1.0, 0.0, 0.0], (len(xyz_remove), 1))
)


print("🔵 Blue = kept (leaf), 🔴 Red = removed (ghost)")
o3d.visualization.draw_geometries([pcd_keep, pcd_remove])


np.savetxt("kept.txt",
           np.hstack([xyz_keep, rgb_keep * 255]),
           fmt="%.6f")

np.savetxt("removed.txt",
           np.hstack([xyz_remove, rgb_remove * 255]),
           fmt="%.6f")

print("Saved: kept.txt, removed.txt")