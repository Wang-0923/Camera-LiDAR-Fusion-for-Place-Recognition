import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree

from glnet.config.config import nclt_pc_bev_conf, oxford_pc_bev_conf
from glnet.utils.data_utils.lidar_reliability import compute_lidar_reliability_bev
import glnet.utils.vox_utils.vox as vox


class AdaptiveReliabilityEstimator(nn.Module):
    """Reliability-to-gate module for adaptive visual-LiDAR BEV fusion.

    Visual reliability is expected to be projected by RINGSharpV/BaseLSSFPN
    before entering this module. LiDAR reliability is computed from raw points
    with point-level local PCA, then aggregated into the same voxel layout as
    batch['pc'].
    """

    def __init__(
        self,
        dataset_type='nclt',
        rho=0.5,
        temperature=0.7,
        use_visual_reliability=True,
        use_lidar_reliability=True,
        eps=1e-6,
        lidar_density_tau=5.0,
        lidar_knn=16,
        lidar_pca_chunk_size=4096,
        lidar_reliability_mode='online',
        lidar_downsample_voxel_size=0.3,
        lidar_min_neighbors=3,
    ):
        super().__init__()
        self.rho = float(rho)
        self.temperature = float(temperature)
        self.use_visual_reliability = bool(use_visual_reliability)
        self.use_lidar_reliability = bool(use_lidar_reliability)
        self.eps = float(eps)
        self.lidar_density_tau = float(lidar_density_tau)
        self.lidar_knn = int(lidar_knn)
        self.lidar_pca_chunk_size = int(lidar_pca_chunk_size)
        self.lidar_downsample_voxel_size = float(lidar_downsample_voxel_size)
        self.lidar_min_neighbors = int(lidar_min_neighbors)
        self.lidar_reliability_mode = lidar_reliability_mode.lower()
        if self.lidar_reliability_mode not in ['online', 'offline', 'auto']:
            raise ValueError(
                'lidar_reliability_mode must be one of ["online", "offline", "auto"], '
                f'got {self.lidar_reliability_mode}'
            )

        if dataset_type == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif dataset_type == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        else:
            raise NotImplementedError(f'Unsupported dataset type for adaptive fusion: {dataset_type}')

        self.x_bound = pc_bev_conf['x_bound']
        self.y_bound = pc_bev_conf['y_bound']
        self.z_bound = pc_bev_conf['z_bound']
        self.x_grid = pc_bev_conf['x_grid']
        self.y_grid = pc_bev_conf['y_grid']
        self.z_grid = pc_bev_conf['z_grid']
        self.bounds = (
            self.x_bound[0],
            self.x_bound[1],
            self.y_bound[0],
            self.y_bound[1],
            self.z_bound[0],
            self.z_bound[1],
        )

    def forward(self, batch, visual_bev, lidar_bev, visual_outputs=None, lidar_outputs=None):
        b, _, h, w = visual_bev.shape
        device = visual_bev.device
        dtype = visual_bev.dtype

        with torch.no_grad():
            if self.use_visual_reliability:
                if visual_outputs is None or 'visual_reliability_bev' not in visual_outputs:
                    raise ValueError(
                        'visual_reliability_bev is required when adaptive visual reliability is enabled'
                    )
                mv = visual_outputs['visual_reliability_bev']
                if mv.ndim != 4 or mv.shape[1] != 1:
                    raise ValueError(f'visual_reliability_bev must have shape [B,1,H,W], got {tuple(mv.shape)}')
                if mv.shape[0] != b or mv.shape[-2:] != (h, w):
                    raise ValueError(
                        'visual_reliability_bev must match visual_bev batch and spatial size, got '
                        f'{tuple(mv.shape)} vs {tuple(visual_bev.shape)}'
                    )
                mv = mv.to(device=device, dtype=dtype)
            else:
                mv = torch.ones((b, 1, h, w), device=device, dtype=dtype)

            if self.use_lidar_reliability:
                if self.lidar_reliability_mode == 'offline':
                    ml = self._lidar_reliability_from_cache(batch, lidar_bev, required=True)
                elif self.lidar_reliability_mode == 'auto':
                    if 'lidar_reliability_bev' in batch and batch['lidar_reliability_bev'] is not None:
                        ml = self._lidar_reliability_from_cache(batch, lidar_bev, required=True)
                    else:
                        ml = self._lidar_reliability_from_raw_points_and_pc(batch, lidar_bev)
                else:
                    ml = self._lidar_reliability_from_raw_points_and_pc(batch, lidar_bev)
            else:
                ml = torch.ones((b, 1, lidar_bev.shape[-2], lidar_bev.shape[-1]), device=lidar_bev.device,
                                dtype=lidar_bev.dtype)

            if ml.shape[-2:] != (h, w):
                ml = F.interpolate(ml, size=(h, w), mode='bilinear', align_corners=False)
            ml = ml.to(device=device, dtype=dtype)

            mv = mv.clamp(self.eps, 1.0 - self.eps)
            ml = ml.clamp(self.eps, 1.0 - self.eps)

            scores = torch.cat([mv, ml], dim=1) / max(self.temperature, self.eps)
            weights = torch.softmax(scores, dim=1)
            wv = weights[:, 0:1]
            wl = weights[:, 1:2]
            gv = 1.0 + self.rho * (2.0 * wv - 1.0)
            gl = 1.0 + self.rho * (2.0 * wl - 1.0)

        return {
            'Mv': mv,
            'Ml': ml,
            'Wv': wv,
            'Wl': wl,
            'Gv': gv,
            'Gl': gl,
        }

    def _lidar_reliability_from_cache(self, batch, lidar_bev, required=True):
        if 'lidar_reliability_bev' not in batch or batch['lidar_reliability_bev'] is None:
            if required:
                raise FileNotFoundError(
                    'batch["lidar_reliability_bev"] is required when '
                    "adaptive_lidar_reliability_mode='offline'. Please run "
                    'generate_training_tuples.py / generate_evaluation_sets.py with --lidar_reliability.'
                )
            return None

        ml = batch['lidar_reliability_bev']
        if not torch.is_tensor(ml):
            ml = torch.as_tensor(ml)
        ml = ml.to(device=lidar_bev.device, dtype=lidar_bev.dtype)
        if ml.ndim == 3:
            ml = ml.unsqueeze(1)
        if ml.ndim != 4 or ml.shape[1] != 1:
            raise ValueError(f'batch["lidar_reliability_bev"] must have shape [B,1,H,W] or [B,H,W], got {tuple(ml.shape)}')
        if ml.shape[0] != lidar_bev.shape[0]:
            raise ValueError(
                f'batch["lidar_reliability_bev"] batch size {ml.shape[0]} does not match lidar_bev {lidar_bev.shape[0]}'
            )
        if ml.shape[-2:] != lidar_bev.shape[-2:]:
            ml = F.interpolate(ml, size=lidar_bev.shape[-2:], mode='bilinear', align_corners=False)
        return ml.clamp(0.0, 1.0)

    def _lidar_reliability_from_raw_points_and_pc(self, batch, lidar_bev):
        if 'orig_pc' not in batch or batch['orig_pc'] is None:
            raise ValueError('batch["orig_pc"] is required when adaptive LiDAR reliability is enabled')
        if 'pc' not in batch or batch['pc'] is None or not torch.is_tensor(batch['pc']):
            raise ValueError('batch["pc"] is required when adaptive LiDAR reliability is enabled')

        occ = batch['pc'].to(device=lidar_bev.device)
        if occ.ndim != 4:
            raise ValueError(f'batch["pc"] must have shape [B,Z,H,W], got {tuple(occ.shape)}')
        if occ.shape[0] != lidar_bev.shape[0]:
            raise ValueError(f'batch["pc"] batch size {occ.shape[0]} does not match lidar_bev {lidar_bev.shape[0]}')

        pcs = batch['orig_pc'] if isinstance(batch['orig_pc'], (list, tuple)) else [batch['orig_pc']]
        if len(pcs) != occ.shape[0]:
            raise ValueError(f'batch["orig_pc"] length {len(pcs)} does not match batch size {occ.shape[0]}')

        grid_z, grid_h, grid_w = occ.shape[1], occ.shape[2], occ.shape[3]
        ml_maps = []
        for pc in pcs:
            rel_bev = compute_lidar_reliability_bev(
                pc,
                Z=grid_z,
                Y=grid_w,
                X=grid_h,
                bounds=self.bounds,
                downsample_voxel_size=self.lidar_downsample_voxel_size,
                k=self.lidar_knn,
                min_neighbors=self.lidar_min_neighbors,
                eps=self.eps,
            )
            ml_maps.append(torch.from_numpy(rel_bev))

        ml = torch.stack(ml_maps, dim=0).to(device=lidar_bev.device, dtype=lidar_bev.dtype)
        if ml.shape[-2:] != lidar_bev.shape[-2:]:
            ml = F.interpolate(ml, size=lidar_bev.shape[-2:], mode='bilinear', align_corners=False)
        return ml.clamp(0.0, 1.0)

    def _compute_point_pca_reliability(self, points):
        n_points = points.shape[0]
        if n_points < 3:
            return points.new_zeros((n_points,))

        k = min(self.lidar_knn, n_points)
        if k < 3:
            return points.new_zeros((n_points,))

        points_np = points.detach().cpu().numpy().astype(np.float32)
        tree = KDTree(points_np)
        neighbor_indices = tree.query(points_np, k=k, return_distance=False)
        neighbor_indices = torch.as_tensor(neighbor_indices, device=points.device, dtype=torch.long)

        reliability = points.new_zeros((n_points,))
        for start in range(0, n_points, self.lidar_pca_chunk_size):
            end = min(start + self.lidar_pca_chunk_size, n_points)
            neighbors = points[neighbor_indices[start:end]]
            centered = neighbors - neighbors.mean(dim=1, keepdim=True)
            cov = centered.transpose(1, 2).bmm(centered) / float(max(k - 1, 1))
            eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
            sigma1 = eigvals[:, 2]
            sigma3 = eigvals[:, 0]
            mu = eigvals.sum(dim=1) + self.eps
            reliability[start:end] = ((sigma1 - sigma3) / mu).clamp(0.0, 1.0)
        return reliability

    def _aggregate_point_reliability_to_voxels(self, points, point_reliability, z_grid, x_grid, y_grid):
        if points.shape[0] == 0:
            return points.new_zeros((z_grid, x_grid, y_grid))

        scene_centroid = torch.zeros((1, 3), device=points.device, dtype=points.dtype)
        vox_util = vox.Vox_util(
            z_grid,
            y_grid,
            x_grid,
            scene_centroid=scene_centroid,
            bounds=self.bounds,
            assert_cube=False,
        )
        xyz_mem = vox_util.Ref2Mem(points.unsqueeze(0), z_grid, y_grid, x_grid, assert_cube=False)
        valid = vox_util.get_inbounds(xyz_mem, z_grid, y_grid, x_grid, already_mem=True)
        xyz_zero = vox_util.Ref2Mem(points[0:1].unsqueeze(0) * 0.0, z_grid, y_grid, x_grid, assert_cube=False)
        valid = valid & (torch.norm(xyz_zero - xyz_mem, dim=2) >= 0.1)
        valid = valid.squeeze(0)
        if not torch.any(valid):
            return points.new_zeros((z_grid, x_grid, y_grid))

        xyz_mem = xyz_mem.squeeze(0)[valid]
        rel = point_reliability[valid]
        x = torch.round(xyz_mem[:, 0]).clamp(0, x_grid - 1).long()
        y = torch.round(xyz_mem[:, 1]).clamp(0, y_grid - 1).long()
        z = torch.round(xyz_mem[:, 2]).clamp(0, z_grid - 1).long()
        flat_index = z * (x_grid * y_grid) + y * x_grid + x

        flat_size = z_grid * x_grid * y_grid
        sums = points.new_zeros((flat_size,))
        counts = points.new_zeros((flat_size,))
        sums.scatter_add_(0, flat_index, rel)
        counts.scatter_add_(0, flat_index, torch.ones_like(rel))
        voxel_reliability = sums / (counts + self.eps)
        voxel_reliability = voxel_reliability.reshape(z_grid, y_grid, x_grid)
        # generate_bev returns [Z,X,Y] after permuting the original [Z,Y,X].
        return voxel_reliability.permute(0, 2, 1).contiguous().clamp(0.0, 1.0)
