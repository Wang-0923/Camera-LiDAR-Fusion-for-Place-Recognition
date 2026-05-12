import os
import sys

import pytest
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from glnet.models.bev_fusion import (  # noqa: E402
    BEVDeformableFusion,
    BEVFeaturePyramid,
    build_bev_reference_points,
    validate_bev_alignment,
)
from glnet.models.adaptive_fusion import AdaptiveReliabilityEstimator  # noqa: E402


def _meta(**overrides):
    meta = {
        'sample_id': 'sample-001',
        'timestamp': 1000.0,
        'ego_pose_id': 'ego-001',
        'x_min': -51.2,
        'x_max': 51.2,
        'y_min': -51.2,
        'y_max': 51.2,
        'dx': 0.8,
        'dy': 0.8,
        'H': 128,
        'W': 128,
        'origin': 'ego_center',
        'row_axis': 'y',
        'col_axis': 'x',
        'x_direction': 'vehicle_forward',
        'y_direction': 'vehicle_left',
        'row_direction': 'y_max_to_y_min',
        'col_direction': 'x_min_to_x_max',
        'aug_meta': {'rotation': 0.0, 'flip_x': False, 'flip_y': False, 'translation': [0.0, 0.0]},
    }
    meta.update(overrides)
    return meta


@torch.no_grad()
def test_bev_fusion_shape_128():
    module = BEVDeformableFusion(
        visual_in_channels=80,
        lidar_in_channels=128,
        embed_dim=128,
        out_channels=128,
        num_heads=4,
        num_points=2,
    ).eval()
    visual_bev = torch.randn(2, 80, 128, 128)
    lidar_bev = torch.randn(2, 128, 128, 128)

    output = module(visual_bev, lidar_bev, strict_meta=False)

    assert output['fused_bev'].shape == (2, 128, 128, 128)


@torch.no_grad()
def test_bev_fusion_out_projection_shape():
    module = BEVDeformableFusion(out_channels=80, num_heads=4, num_points=2).eval()
    visual_bev = torch.randn(1, 80, 32, 32)
    lidar_bev = torch.randn(1, 128, 32, 32)

    output = module(visual_bev, lidar_bev, strict_meta=False)

    assert output['fused_bev'].shape == (1, 80, 32, 32)


@torch.no_grad()
def test_bev_pyramid_shapes():
    pyramid = BEVFeaturePyramid(128).eval()
    levels = pyramid(torch.randn(2, 128, 128, 128))

    assert [tuple(level.shape) for level in levels] == [
        (2, 128, 128, 128),
        (2, 128, 64, 64),
        (2, 128, 32, 32),
        (2, 128, 16, 16),
    ]


def test_reference_points_xy_order_and_range():
    h, w = 128, 128
    ref = build_bev_reference_points(2, h, w, 4)

    assert ref.shape == (2, h * w, 4, 2)
    assert torch.all(ref > 0.0)
    assert torch.all(ref < 1.0)
    assert torch.allclose(ref[0, 0, 0], torch.tensor([0.5 / w, 0.5 / h]))

    center_ref = ref[0, (h // 2) * w + (w // 2), 0]
    assert torch.allclose(center_ref, torch.tensor([0.5, 0.5]), atol=0.5 / h + 1e-6)
    assert torch.allclose(ref[0, 1, 0, 0] - ref[0, 0, 0, 0], torch.tensor(1.0 / w))
    assert torch.allclose(ref[0, w, 0, 1] - ref[0, 0, 0, 1], torch.tensor(1.0 / h))


def test_validate_bev_alignment_accepts_matching_metadata():
    assert validate_bev_alignment(_meta(), _meta(), strict=True)


@pytest.mark.parametrize(
    'field,value,error',
    [
        ('sample_id', 'sample-002', 'sample_id'),
        ('timestamp', 1200.0, 'timestamp'),
        ('x_max', 50.0, 'x_max'),
        ('row_direction', 'y_min_to_y_max', 'row_direction'),
    ],
)
def test_validate_bev_alignment_raises_on_mismatch(field, value, error):
    with pytest.raises(ValueError, match=error):
        validate_bev_alignment(_meta(), _meta(**{field: value}), strict=True, timestamp_tolerance_ms=50.0)


def test_validate_bev_alignment_raises_on_aug_mismatch():
    with pytest.raises(ValueError, match='augmentation'):
        validate_bev_alignment(
            _meta(),
            _meta(aug_meta={'rotation': 10.0, 'flip_x': False, 'flip_y': False, 'translation': [0.0, 0.0]}),
            strict=True,
        )


class _WorkflowProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = BEVDeformableFusion(num_heads=4, num_points=2)
        self.seen_visual = False
        self.seen_lidar = False
        self.downstream_input_shape = None

    def forward(self, visual_bev, lidar_bev):
        self.seen_visual = True
        self.seen_lidar = True
        fused = self.fusion(visual_bev, lidar_bev, strict_meta=False)['fused_bev']
        self.downstream_input_shape = tuple(fused.shape)
        spec_like = fused.mean(dim=1, keepdim=True)
        return {'bev': fused, 'spec': spec_like}


@torch.no_grad()
def test_fusion_workflow_uses_both_modalities_before_downstream():
    model = _WorkflowProbe().eval()
    output = model(torch.randn(1, 80, 32, 32), torch.randn(1, 128, 32, 32))

    assert model.seen_visual
    assert model.seen_lidar
    assert model.downstream_input_shape == (1, 128, 32, 32)
    assert output['bev'].shape == (1, 128, 32, 32)
    assert output['spec'].shape == (1, 1, 32, 32)


@torch.no_grad()
def test_adaptive_reliability_maps_and_gates():
    visual_bev = torch.randn(2, 80, 16, 16)
    lidar_bev = torch.randn(2, 128, 16, 16)
    lidar_reliability = torch.rand(2, 1, 16, 16)
    batch = {'lidar_reliability_bev': lidar_reliability}
    visual_outputs = {'visual_reliability_bev': torch.rand(2, 1, 16, 16)}
    estimator = AdaptiveReliabilityEstimator(dataset_type='nclt', rho=0.5, temperature=0.7)

    adaptive = estimator(batch, visual_bev, lidar_bev, visual_outputs=visual_outputs)

    assert set(['Mv', 'Ml', 'Wv', 'Wl', 'Gv', 'Gl']).issubset(adaptive.keys())
    for key in ['Mv', 'Ml', 'Wv', 'Wl', 'Gv', 'Gl']:
        assert adaptive[key].shape == (2, 1, 16, 16)
    assert torch.allclose(adaptive['Wv'] + adaptive['Wl'], torch.ones_like(adaptive['Wv']), atol=1e-5)
    expected_weights = torch.softmax(torch.cat([adaptive['Mv'], adaptive['Ml']], dim=1) / 0.7, dim=1)
    assert torch.allclose(adaptive['Wv'], expected_weights[:, 0:1], atol=1e-6)
    assert torch.allclose(adaptive['Wl'], expected_weights[:, 1:2], atol=1e-6)
    assert torch.allclose(adaptive['Ml'], lidar_reliability.clamp(1e-6, 1.0 - 1e-6), atol=1e-6)
    assert torch.allclose(adaptive['Gv'], 1.0 + 0.5 * (2.0 * adaptive['Wv'] - 1.0), atol=1e-6)
    assert torch.allclose(adaptive['Gl'], 1.0 + 0.5 * (2.0 * adaptive['Wl'] - 1.0), atol=1e-6)


@torch.no_grad()
def test_adaptive_rho_zero_gates_are_identity_and_fusion_shape():
    visual_bev = torch.randn(2, 80, 16, 16)
    lidar_bev = torch.randn(2, 128, 16, 16)
    batch = {'lidar_reliability_bev': torch.rand(2, 1, 16, 16)}
    visual_outputs = {'visual_reliability_bev': torch.rand(2, 1, 16, 16)}
    estimator = AdaptiveReliabilityEstimator(dataset_type='nclt', rho=0.0, temperature=0.7)
    adaptive = estimator(batch, visual_bev, lidar_bev, visual_outputs=visual_outputs)

    assert torch.allclose(adaptive['Gv'], torch.ones_like(adaptive['Gv']), atol=1e-6)
    assert torch.allclose(adaptive['Gl'], torch.ones_like(adaptive['Gl']), atol=1e-6)

    module = BEVDeformableFusion(embed_dim=32, out_channels=32, num_heads=4, num_points=2).eval()
    output = module(visual_bev, lidar_bev, adaptive=adaptive, strict_meta=False, return_intermediates=False)
    assert output['fused_bev'].shape == (2, 32, 16, 16)
    assert set(['Mv', 'Ml', 'Wv', 'Wl', 'Gv', 'Gl']).issubset(output['adaptive'].keys())
