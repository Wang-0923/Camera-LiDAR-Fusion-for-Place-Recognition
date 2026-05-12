import torch
import torch.nn as nn
import torch.nn.functional as F

from glnet.config.config import nclt_pc_bev_conf, oxford_pc_bev_conf
from glnet.utils.data_utils.lidar_reliability import compute_lidar_reliability_bev


class AdaptiveReliabilityEstimator(nn.Module):
    """Reliability-to-gate module for adaptive visual-LiDAR BEV fusion.

    Visual reliability is projected to BEV by RINGSharpV/BaseLSSFPN. LiDAR
    reliability is either read from an offline cache or computed online from
    raw points with local PCA, then converted to fusion gates with temperature
    softmax.
    """

    def __init__(
        self,
        dataset_type='nclt',
        rho=0.5,
        temperature=0.7,
        eps=1e-6,
        lidar_downsample_voxel_size=0.3,
        lidar_knn=16,
        lidar_min_neighbors=3,
        **kwargs,
    ):
        super().__init__()
        self.rho = float(rho)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.lidar_downsample_voxel_size = float(lidar_downsample_voxel_size)
        self.lidar_knn = int(lidar_knn)
        self.lidar_min_neighbors = int(lidar_min_neighbors)

        if dataset_type == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif dataset_type == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        else:
            raise NotImplementedError(f'Unsupported dataset type for adaptive fusion: {dataset_type}')

        self.bounds = (
            pc_bev_conf['x_bound'][0],
            pc_bev_conf['x_bound'][1],
            pc_bev_conf['y_bound'][0],
            pc_bev_conf['y_bound'][1],
            pc_bev_conf['z_bound'][0],
            pc_bev_conf['z_bound'][1],
        )

    def forward(self, batch, visual_bev, lidar_bev, visual_outputs=None, lidar_outputs=None):
        b, _, h, w = visual_bev.shape
        device = visual_bev.device
        dtype = visual_bev.dtype

        with torch.no_grad():
            if visual_outputs is None or 'visual_reliability_bev' not in visual_outputs:
                raise ValueError('visual_reliability_bev is required when adaptive_fusion=True')
            mv = visual_outputs['visual_reliability_bev']
            if mv.ndim != 4 or mv.shape[1] != 1:
                raise ValueError(f'visual_reliability_bev must have shape [B,1,H,W], got {tuple(mv.shape)}')
            if mv.shape[0] != b or mv.shape[-2:] != (h, w):
                raise ValueError(
                    'visual_reliability_bev must match visual_bev batch and spatial size, got '
                    f'{tuple(mv.shape)} vs {tuple(visual_bev.shape)}'
                )
            mv = mv.to(device=device, dtype=dtype).clamp(0.0, 1.0)

            ml = self._lidar_reliability_from_cache(batch, lidar_bev)
            if ml is None:
                ml = self._lidar_reliability_from_raw_points_and_pc(batch, lidar_bev)
            if ml.shape[-2:] != (h, w):
                ml = F.interpolate(ml, size=(h, w), mode='bilinear', align_corners=False)
            ml = ml.to(device=device, dtype=dtype).clamp(0.0, 1.0)

            mv = mv.clamp(self.eps, 1.0 - self.eps)
            ml = ml.clamp(self.eps, 1.0 - self.eps)
            scores = torch.cat([mv, ml], dim=1) / max(self.temperature, self.eps)
            weights = torch.softmax(scores, dim=1)
            wv = weights[:, 0:1]
            wl = weights[:, 1:2]

            gate_strength = max(0.0, min(1.0, self.rho))
            gv = 1.0 + gate_strength * (2.0 * wv - 1.0)
            gl = 1.0 + gate_strength * (2.0 * wl - 1.0)

        return {
            'Mv': mv,
            'Ml': ml,
            'Wv': wv,
            'Wl': wl,
            'Gv': gv,
            'Gl': gl,
        }

    def _lidar_reliability_from_cache(self, batch, lidar_bev):
        if 'lidar_reliability_bev' not in batch or batch['lidar_reliability_bev'] is None:
            return None

        ml = batch['lidar_reliability_bev']
        if not torch.is_tensor(ml):
            ml = torch.as_tensor(ml)
        ml = ml.to(device=lidar_bev.device, dtype=lidar_bev.dtype)
        if ml.ndim == 3:
            ml = ml.unsqueeze(1)
        if ml.ndim != 4 or ml.shape[1] != 1:
            raise ValueError(
                f'batch["lidar_reliability_bev"] must have shape [B,1,H,W] or [B,H,W], got {tuple(ml.shape)}'
            )
        if ml.shape[0] != lidar_bev.shape[0]:
            raise ValueError(
                f'batch["lidar_reliability_bev"] batch size {ml.shape[0]} does not match lidar_bev {lidar_bev.shape[0]}'
            )
        if ml.shape[-2:] != lidar_bev.shape[-2:]:
            ml = F.interpolate(ml, size=lidar_bev.shape[-2:], mode='bilinear', align_corners=False)
        return ml.clamp(0.0, 1.0)

    def _lidar_reliability_from_raw_points_and_pc(self, batch, lidar_bev):
        if 'orig_pc' not in batch or batch['orig_pc'] is None:
            raise ValueError('batch["orig_pc"] is required when adaptive_fusion=True and no LiDAR reliability cache is present')
        if 'pc' not in batch or batch['pc'] is None or not torch.is_tensor(batch['pc']):
            raise ValueError('batch["pc"] is required when adaptive_fusion=True and no LiDAR reliability cache is present')

        occ = batch['pc']
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
