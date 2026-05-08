import numpy as np
import torch
from sklearn.neighbors import KDTree

from glnet.utils.data_utils.point_clouds import generate_bev, generate_bev_feats


def voxel_downsample_points(pc, voxel_size, bounds=None):
    """Voxel-grid downsample a point cloud with centroid representatives."""
    pc = np.asarray(pc, dtype=np.float32)
    if pc.size == 0 or voxel_size is None or voxel_size <= 0:
        return pc[:, :3].astype(np.float32)

    xyz = pc[:, :3]
    if bounds is not None:
        origin = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    else:
        origin = xyz.min(axis=0)
    voxel_index = np.floor((xyz - origin) / float(voxel_size)).astype(np.int64)

    unique_index, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    sums = np.zeros((len(unique_index), 3), dtype=np.float64)
    counts = np.zeros((len(unique_index), 1), dtype=np.float64)
    np.add.at(sums, inverse, xyz)
    np.add.at(counts, inverse, 1.0)
    return (sums / np.maximum(counts, 1.0)).astype(np.float32)


def compute_downsampled_point_reliability(pc_ds, k=16, min_neighbors=3, eps=1e-6):
    """Compute point-level LiDAR structure reliability with local PCA."""
    pc_ds = np.asarray(pc_ds, dtype=np.float32)
    n_points = pc_ds.shape[0]
    if n_points < min_neighbors:
        return np.zeros((n_points,), dtype=np.float32)

    k = min(int(k), n_points)
    if k < min_neighbors:
        return np.zeros((n_points,), dtype=np.float32)

    tree = KDTree(pc_ds[:, :3])
    neighbor_indices = tree.query(pc_ds[:, :3], k=k, return_distance=False)
    neighbors = pc_ds[neighbor_indices, :3]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.matmul(centered.transpose(0, 2, 1), centered) / float(max(k - 1, 1))
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    sigma1 = eigvals[:, 2]
    sigma3 = eigvals[:, 0]
    mu = eigvals.sum(axis=1) + eps
    reliability = (sigma1 - sigma3) / mu
    return np.clip(reliability, 0.0, 1.0).astype(np.float32)


def propagate_reliability_to_original_points(pc_valid, pc_ds, rel_ds):
    """Assign each original point the reliability of its nearest downsampled point."""
    pc_valid = np.asarray(pc_valid, dtype=np.float32)
    pc_ds = np.asarray(pc_ds, dtype=np.float32)
    rel_ds = np.asarray(rel_ds, dtype=np.float32)
    if pc_valid.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if pc_ds.shape[0] == 0:
        return np.zeros((pc_valid.shape[0],), dtype=np.float32)

    tree = KDTree(pc_ds[:, :3])
    nearest = tree.query(pc_valid[:, :3], k=1, return_distance=False).reshape(-1)
    return rel_ds[nearest].astype(np.float32)


def aggregate_point_reliability_to_voxels(pc_valid, rel_valid, Z, Y, X, bounds, eps=1e-6):
    """Voxelize point reliability with the same layout helper as generate_bev.

    generate_bev_feats shares Vox_util coordinate conversion and the final
    [Z,Y,X] -> [Z,X,Y] permutation with generate_bev, which keeps reliability
    maps aligned with batch['pc'].
    """
    pc_valid = np.asarray(pc_valid, dtype=np.float32)
    rel_valid = np.asarray(rel_valid, dtype=np.float32)
    if pc_valid.shape[0] == 0:
        return np.zeros((Z, X, Y), dtype=np.float32)

    rel_feats = rel_valid.reshape(-1, 1).astype(np.float32)
    rel_vox = generate_bev_feats(pc_valid[:, :3], rel_feats, Z=Z, Y=Y, X=X, bounds=bounds)
    rel_vox = rel_vox.squeeze(0).squeeze(0)
    if rel_vox.shape != (Z, X, Y):
        raise ValueError(f'Expected reliability voxel shape {(Z, X, Y)}, got {tuple(rel_vox.shape)}')
    return rel_vox.clamp(0.0, 1.0).numpy().astype(np.float32)


def compute_lidar_reliability_bev(
    pc,
    Z,
    Y,
    X,
    bounds,
    downsample_voxel_size=0.3,
    k=16,
    min_neighbors=3,
    eps=1e-6,
):
    """Compute an offline/online LiDAR BEV reliability map.

    The returned layout is [1,H,W], where [H,W] matches generate_bev(...)[-2:].
    Reliability is point-level local PCA propagated back to original occupied
    voxels, then occupancy-weighted across the vertical Z bins.
    """
    if torch.is_tensor(pc):
        pc = pc.detach().cpu().numpy()
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        return np.zeros((1, X, Y), dtype=np.float32)

    xyz = pc[:, :3]
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    mask = (
        (xyz[:, 0] >= x_min) & (xyz[:, 0] < x_max) &
        (xyz[:, 1] >= y_min) & (xyz[:, 1] < y_max) &
        (xyz[:, 2] >= z_min) & (xyz[:, 2] < z_max)
    )
    pc_valid = xyz[mask].astype(np.float32)
    if pc_valid.shape[0] < min_neighbors:
        return np.zeros((1, X, Y), dtype=np.float32)

    pc_bev = generate_bev(pc_valid, Z=Z, Y=Y, X=X, bounds=bounds)
    occ_weight = (pc_bev > 0).float()

    pc_ds = voxel_downsample_points(pc_valid, downsample_voxel_size, bounds=bounds)
    rel_ds = compute_downsampled_point_reliability(pc_ds, k=k, min_neighbors=min_neighbors, eps=eps)
    rel_valid = propagate_reliability_to_original_points(pc_valid, pc_ds, rel_ds)
    rl_vox_np = aggregate_point_reliability_to_voxels(pc_valid, rel_valid, Z=Z, Y=Y, X=X, bounds=bounds, eps=eps)
    rl_vox = torch.from_numpy(rl_vox_np).float()

    ml = (rl_vox * occ_weight).sum(dim=0, keepdim=True) / (occ_weight.sum(dim=0, keepdim=True) + eps)
    return ml.clamp(0.0, 1.0).numpy().astype(np.float32)
