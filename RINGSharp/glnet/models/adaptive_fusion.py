import torch
import torch.nn as nn


class AdaptiveReliabilityEstimator(nn.Module):
    """Visual-degradation reliability gating for visual-LiDAR BEV fusion.

    This module assumes LiDAR is clean and uses LiDAR only as compensation when
    visual BEV reliability drops. Visual reliability must already be projected
    to the BEV grid by RINGSharpV/BaseLSSFPN.
    """

    def __init__(
        self,
        dataset_type='nclt',
        rho=0.5,
        temperature=0.7,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.rho = float(rho)
        self.temperature = float(temperature)
        self.eps = float(eps)

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

            # LiDAR is assumed undegraded. Its compensation weight is the visual
            # reliability deficit, not an independently estimated reliability.
            wv = mv
            wl = (1.0 - mv).clamp(0.0, 1.0)
            ml = torch.ones_like(mv)

            # Residual gates are centered at 1. Clean vision yields identity
            # gates. As vision reliability falls, visual features are softened
            # and LiDAR compensation is amplified.
            gate_strength = max(0.0, min(1.0, self.rho))
            gv = 1.0 + gate_strength * (wv - 1.0)
            gl = 1.0 + gate_strength * wl

        return {
            'Mv': mv,
            'Ml': ml,
            'Wv': wv,
            'Wl': wl,
            'Gv': gv,
            'Gl': gl,
        }
