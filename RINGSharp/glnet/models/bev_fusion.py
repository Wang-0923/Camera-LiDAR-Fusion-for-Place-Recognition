from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(channels):
    return nn.BatchNorm2d(channels)


class ConvNormReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            _make_norm(out_channels),
            nn.ReLU(inplace=True),
        )


class ResDownBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.main = nn.Sequential(
            ConvNormReLU(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm(channels),
        )
        self.skip = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(x)
        return self.out(self.main(x) + self.skip(x))


class BEVFeaturePyramid(nn.Module):
    def __init__(self, channels, num_levels=4):
        super().__init__()
        if num_levels != 4:
            raise ValueError(f'BEVFeaturePyramid currently expects 4 levels, got {num_levels}')
        self.blocks = nn.ModuleList([ResDownBlock(channels) for _ in range(num_levels - 1)])

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(f'BEV H and W must be divisible by 8 for a 4-level pyramid, got H={h}, W={w}')
        pyramid = [x]
        for block in self.blocks:
            pyramid.append(block(pyramid[-1]))
        return pyramid


def build_bev_reference_points(batch_size, height, width, num_levels, device=None, dtype=None):
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
    try:
        ref_y, ref_x = torch.meshgrid(ys, xs, indexing='ij')
    except TypeError:
        ref_y, ref_x = torch.meshgrid(ys, xs)
    ref = torch.stack((ref_x, ref_y), dim=-1).reshape(1, height * width, 1, 2)
    ref = ref.repeat(batch_size, 1, num_levels, 1)
    return ref


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class GridSampleMSDeformCrossAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_levels=4, num_points=4, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= float(i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, value_pyramid):
        b, len_q, c = query.shape
        if c != self.embed_dim:
            raise ValueError(f'Expected query dim {self.embed_dim}, got {c}')
        if len(value_pyramid) != self.num_levels:
            raise ValueError(f'Expected {self.num_levels} value levels, got {len(value_pyramid)}')
        if reference_points.shape != (b, len_q, self.num_levels, 2):
            raise ValueError(
                'reference_points must have shape '
                f'{(b, len_q, self.num_levels, 2)}, got {tuple(reference_points.shape)}'
            )

        spatial_shapes = []
        flat_values = []
        for level, feat in enumerate(value_pyramid):
            if feat.ndim != 4:
                raise ValueError(f'Pyramid level {level} must be BCHW, got {tuple(feat.shape)}')
            if feat.shape[0] != b or feat.shape[1] != self.embed_dim:
                raise ValueError(
                    f'Pyramid level {level} must have shape [B,{self.embed_dim},H,W], got {tuple(feat.shape)}'
                )
            spatial_shapes.append(feat.shape[-2:])
            flat_values.append(feat.flatten(2).transpose(1, 2))

        value = torch.cat(flat_values, dim=1)
        len_in = value.shape[1]
        value = self.value_proj(value).view(b, len_in, self.num_heads, self.head_dim)
        value_levels = value.split([h * w for h, w in spatial_shapes], dim=1)

        offsets = self.sampling_offsets(query).view(
            b, len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        weights = self.attention_weights(query).view(
            b, len_q, self.num_heads, self.num_levels * self.num_points
        )
        weights = F.softmax(weights, dim=-1).view(b, len_q, self.num_heads, self.num_levels, self.num_points)

        sampled_levels = []
        for level, ((h, w), value_level) in enumerate(zip(spatial_shapes, value_levels)):
            normalizer = query.new_tensor([w, h])
            locations = reference_points[:, :, None, level, None, :] + offsets[:, :, :, level] / normalizer
            grid = 2.0 * locations - 1.0
            grid = grid.permute(0, 2, 1, 3, 4).reshape(b * self.num_heads, len_q, self.num_points, 2)

            value_level = value_level.permute(0, 2, 3, 1).reshape(b * self.num_heads, self.head_dim, h, w)
            sampled = F.grid_sample(value_level, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled = sampled.view(b, self.num_heads, self.head_dim, len_q, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)
            sampled_levels.append(sampled)

        sampled = torch.stack(sampled_levels, dim=3)
        output = (sampled * weights.unsqueeze(-1)).sum(dim=(3, 4))
        output = output.reshape(b, len_q, self.embed_dim)
        output = self.output_proj(output)
        return self.dropout(output)


class AutoMSDeformCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        num_levels=4,
        num_points=4,
        dropout=0.0,
        prefer_cuda_ms_deform_attn=True,
    ):
        super().__init__()
        self.grid_attn = GridSampleMSDeformCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.cuda_attn = None
        self.backend = 'grid_sample'

        if prefer_cuda_ms_deform_attn:
            try:
                from glnet.ops.modules import MSDeformAttn

                self.cuda_attn = MSDeformAttn(
                    d_model=embed_dim,
                    n_levels=num_levels,
                    n_heads=num_heads,
                    n_points=num_points,
                )
                self.backend = 'ms_deform_attn_cuda'
            except Exception:
                self.cuda_attn = None

    def forward(self, query, reference_points, value_pyramid):
        if self.cuda_attn is None or not query.is_cuda:
            return self.grid_attn(query, reference_points, value_pyramid)

        spatial_shapes = []
        flat_values = []
        for feat in value_pyramid:
            spatial_shapes.append(feat.shape[-2:])
            flat_values.append(feat.flatten(2).transpose(1, 2))
        input_flatten = torch.cat(flat_values, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=query.device)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        return self.cuda_attn(query, reference_points, input_flatten, spatial_shapes, level_start_index)


class DeformableCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        num_levels=4,
        num_points=4,
        dropout=0.0,
        drop_path=0.0,
        prefer_cuda_ms_deform_attn=True,
    ):
        super().__init__()
        self.attn = AutoMSDeformCrossAttention(
            embed_dim,
            num_heads,
            num_levels,
            num_points,
            dropout,
            prefer_cuda_ms_deform_attn=prefer_cuda_ms_deform_attn,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = StochasticDepth(drop_path)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_feat, value_pyramid, reference_points):
        b, c, h, w = query_feat.shape
        query = query_feat.flatten(2).transpose(1, 2)
        attn_out = self.attn(query, reference_points, value_pyramid)
        query = self.norm1(query + self.drop_path(attn_out))
        query = self.norm2(query + self.drop_path(self.ffn(query)))
        return query.transpose(1, 2).reshape(b, c, h, w)


class BEVDeformableFusion(nn.Module):
    def __init__(
        self,
        visual_in_channels=80,
        lidar_in_channels=128,
        embed_dim=128,
        out_channels=128,
        num_heads=8,
        num_levels=4,
        num_points=4,
        dropout=0.0,
        drop_path=0.0,
        use_deform_attn=True,
        prefer_cuda_ms_deform_attn=True,
    ):
        super().__init__()
        self.visual_in_channels = visual_in_channels
        self.lidar_in_channels = lidar_in_channels
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_deform_attn = use_deform_attn

        self.visual_adapter = ConvNormReLU(visual_in_channels, embed_dim, kernel_size=1, padding=0)
        self.lidar_adapter = ConvNormReLU(lidar_in_channels, embed_dim, kernel_size=1, padding=0)
        self.visual_pyramid = BEVFeaturePyramid(embed_dim, num_levels=num_levels)
        self.lidar_pyramid = BEVFeaturePyramid(embed_dim, num_levels=num_levels)

        self.lidar_to_visual = DeformableCrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            drop_path=drop_path,
            prefer_cuda_ms_deform_attn=prefer_cuda_ms_deform_attn,
        )
        self.visual_to_lidar = DeformableCrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            drop_path=drop_path,
            prefer_cuda_ms_deform_attn=prefer_cuda_ms_deform_attn,
        )

        self.fuse_conv = nn.Sequential(
            ConvNormReLU(embed_dim * 4, embed_dim, kernel_size=3, padding=1),
            ConvNormReLU(embed_dim, embed_dim, kernel_size=3, padding=1),
        )
        self.base_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Identity() if out_channels == embed_dim else nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(
        self,
        visual_bev,
        lidar_bev,
        adaptive=None,
        visual_meta=None,
        lidar_meta=None,
        strict_meta=False,
        timestamp_tolerance_ms=50.0,
        return_intermediates=True,
    ):
        if visual_bev.ndim != 4 or lidar_bev.ndim != 4:
            raise ValueError(
                f'visual_bev and lidar_bev must be BCHW tensors, got {tuple(visual_bev.shape)} and {tuple(lidar_bev.shape)}'
            )
        if visual_bev.shape[0] != lidar_bev.shape[0]:
            raise ValueError(f'Batch size mismatch: visual={visual_bev.shape[0]}, lidar={lidar_bev.shape[0]}')
        if visual_bev.shape[-2:] != lidar_bev.shape[-2:]:
            raise ValueError(
                f'BEV spatial shape mismatch: visual={tuple(visual_bev.shape[-2:])}, '
                f'lidar={tuple(lidar_bev.shape[-2:])}'
            )
        if visual_bev.shape[1] != self.visual_in_channels:
            raise ValueError(f'Expected visual BEV channels={self.visual_in_channels}, got {visual_bev.shape[1]}')
        if lidar_bev.shape[1] != self.lidar_in_channels:
            raise ValueError(f'Expected LiDAR BEV channels={self.lidar_in_channels}, got {lidar_bev.shape[1]}')

        validate_bev_alignment(
            visual_meta,
            lidar_meta,
            visual_shape=visual_bev.shape,
            lidar_shape=lidar_bev.shape,
            strict=strict_meta,
            timestamp_tolerance_ms=timestamp_tolerance_ms,
        )

        _, _, h, w = visual_bev.shape
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(f'BEV H and W must be divisible by 8 for a 4-level pyramid, got H={h}, W={w}')

        fv0 = self.visual_adapter(visual_bev)
        fl0 = self.lidar_adapter(lidar_bev)
        if adaptive is None:
            fv = fv0
            fl = fl0
            gv = None
            gl = None
        else:
            gv = self._resize_gate(adaptive['Gv'], fv0)
            gl = self._resize_gate(adaptive['Gl'], fl0)
            fv = gv * fv0
            fl = gl * fl0

        f_base = self.base_proj(torch.cat([fv, fl], dim=1))
        if self.use_deform_attn:
            visual_pyramid = self.visual_pyramid(fv)
            lidar_pyramid = self.lidar_pyramid(fl)
            ref = build_bev_reference_points(
                batch_size=visual_bev.shape[0],
                height=h,
                width=w,
                num_levels=self.num_levels,
                device=visual_bev.device,
                dtype=visual_bev.dtype,
            )

            ov = self.lidar_to_visual(fv, lidar_pyramid, ref)
            ol = self.visual_to_lidar(fl, visual_pyramid, ref)
            if adaptive is not None:
                # Cross-modal outputs are gated by the reliability of the source
                # modality injected through attention.
                ov = gl * ov
                ol = gv * ol

            f_cat = torch.cat([fv, fl, ov, ol], dim=1)
            f_conv = self.fuse_conv(f_cat)
            fused = self.out_proj(f_conv + f_base)
            attention_backend = self.lidar_to_visual.attn.backend
        else:
            visual_pyramid = None
            lidar_pyramid = None
            ref = None
            ov = None
            ol = None
            fused = self.out_proj(f_base)
            attention_backend = 'disabled_concat'

        output = {
            'fused_bev': fused,
            'attention_backend': attention_backend,
        }
        if adaptive is not None:
            output['adaptive'] = adaptive
        if return_intermediates:
            output.update({
                'visual_adapter_bev': fv0,
                'lidar_adapter_bev': fl0,
                'visual_pyramid': visual_pyramid,
                'lidar_pyramid': lidar_pyramid,
                'reference_points': ref,
                'lidar_to_visual': ov,
                'visual_to_lidar': ol,
            })
            if adaptive is not None:
                output.update({
                    'visual_gated_bev': fv,
                    'lidar_gated_bev': fl,
                })
        return output

    @staticmethod
    def _resize_gate(gate, target):
        if gate.ndim != 4 or gate.shape[1] != 1:
            raise ValueError(f'Adaptive gate must have shape [B,1,H,W], got {tuple(gate.shape)}')
        if gate.shape[0] != target.shape[0]:
            raise ValueError(f'Adaptive gate batch mismatch: gate={gate.shape[0]}, target={target.shape[0]}')
        gate = gate.to(device=target.device, dtype=target.dtype)
        if gate.shape[-2:] != target.shape[-2:]:
            gate = F.interpolate(gate, size=target.shape[-2:], mode='bilinear', align_corners=False)
        return gate


def _is_missing(value):
    return value is None


def _as_item(value, index):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        item = value[index]
        return item.item() if item.ndim == 0 else item
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value[index]
    return value


def _get(meta, key, default=None):
    if meta is None:
        return default
    if isinstance(meta, Mapping):
        return meta.get(key, default)
    return getattr(meta, key, default)


def _get_path(meta, path, default=None):
    cur = meta
    for key in path:
        if isinstance(key, int) and isinstance(cur, Sequence) and not isinstance(cur, (str, bytes, bytearray)):
            cur = cur[key] if len(cur) > key else None
        else:
            cur = _get(cur, key, default=None)
        if cur is None:
            return default
    return cur


def _field(meta, aliases, sample_index=None, required=False, name=None):
    for alias in aliases:
        value = _get_path(meta, alias if isinstance(alias, tuple) else (alias,), default=None)
        if value is not None:
            return _as_item(value, sample_index) if sample_index is not None else value
    if required:
        raise ValueError(f'Missing required BEV alignment field: {name or aliases[0]}')
    return None


def _num(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            return value
        return float(value[0])
    return float(value)


def _values_equal(a, b, tol=0.0):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().tolist()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().tolist()
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not isinstance(a, Mapping) or not isinstance(b, Mapping) or set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[key], b[key], tol=tol) for key in a.keys())
    if isinstance(a, np.ndarray):
        a = a.tolist()
    if isinstance(b, np.ndarray):
        b = b.tolist()
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != len(b):
            return False
        return all(_values_equal(x, y, tol=tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _compare_field(v_meta, l_meta, aliases, field_name, sample_index=None, tol=0.0, required=False):
    visual = _field(v_meta, aliases, sample_index=sample_index, required=required, name=field_name)
    lidar = _field(l_meta, aliases, sample_index=sample_index, required=required, name=field_name)
    if visual is None or lidar is None:
        return
    if not _values_equal(visual, lidar, tol=tol):
        raise ValueError(f'BEV alignment mismatch for {field_name}: visual={visual}, lidar={lidar}')


def _meta_len(meta):
    if isinstance(meta, Sequence) and not isinstance(meta, (str, bytes, bytearray, Mapping)):
        return len(meta)
    sample_id = _get(meta, 'sample_id')
    if isinstance(sample_id, torch.Tensor) and sample_id.ndim > 0:
        return sample_id.shape[0]
    if isinstance(sample_id, Sequence) and not isinstance(sample_id, (str, bytes, bytearray)):
        return len(sample_id)
    return 1


def _sample_meta(meta, index):
    if isinstance(meta, Sequence) and not isinstance(meta, (str, bytes, bytearray, Mapping)):
        return meta[index], None
    return meta, index


def _timestamp_to_ms(value):
    value = _num(value)
    if abs(value) > 1e12:
        return value / 1000.0
    if abs(value) > 1e9:
        return value
    return value * 1000.0


def validate_bev_alignment(
    visual_meta,
    lidar_meta,
    visual_shape=None,
    lidar_shape=None,
    strict=True,
    timestamp_tolerance_ms=50.0,
    range_tolerance=1e-6,
):
    if visual_shape is not None and lidar_shape is not None:
        if tuple(visual_shape[-2:]) != tuple(lidar_shape[-2:]):
            raise ValueError(f'BEV H/W mismatch: visual={tuple(visual_shape[-2:])}, lidar={tuple(lidar_shape[-2:])}')

    if visual_meta is None or lidar_meta is None:
        if strict:
            missing = 'visual_meta' if visual_meta is None else 'lidar_meta'
            raise ValueError(f'Missing required BEV alignment metadata: {missing}')
        return True

    if _meta_len(visual_meta) != _meta_len(lidar_meta):
        raise ValueError(f'Metadata batch size mismatch: visual={_meta_len(visual_meta)}, lidar={_meta_len(lidar_meta)}')

    for idx in range(_meta_len(visual_meta)):
        v_meta, v_index = _sample_meta(visual_meta, idx)
        l_meta, l_index = _sample_meta(lidar_meta, idx)
        sample_index = v_index if v_index is not None else l_index

        _compare_field(v_meta, l_meta, ['sample_id', 'id'], 'sample_id', sample_index, required=strict)

        v_ts = _field(v_meta, ['timestamp', 'ts', ('sensor_timestamps', 'camera'), ('sensor_timestamps', 'image')], sample_index)
        l_ts = _field(l_meta, ['timestamp', 'ts', ('sensor_timestamps', 'lidar')], sample_index)
        if v_ts is None or l_ts is None:
            if strict:
                raise ValueError('Missing required BEV alignment field: timestamp')
        elif abs(_timestamp_to_ms(v_ts) - _timestamp_to_ms(l_ts)) > timestamp_tolerance_ms:
            raise ValueError(f'BEV timestamp mismatch: visual={v_ts}, lidar={l_ts}')

        ego_aliases = ['ego_pose_id', 'ego_pose_timestamp', ('ego_pose', 'timestamp')]
        has_ego = any(_field(v_meta, [alias], sample_index) is not None for alias in ego_aliases)
        has_lidar_ego = any(_field(l_meta, [alias], sample_index) is not None for alias in ego_aliases)
        if strict and (not has_ego or not has_lidar_ego):
            raise ValueError('Missing required BEV alignment field: ego_pose_id or ego_pose timestamp')
        for alias in ego_aliases:
            _compare_field(v_meta, l_meta, [alias], str(alias), sample_index, tol=range_tolerance)

        is_multi_frame = _field(l_meta, ['is_multi_frame', 'multi_frame'], sample_index)
        if bool(is_multi_frame) if is_multi_frame is not None else False:
            compensated = _field(l_meta, ['ego_motion_compensated', 'motion_compensated'], sample_index)
            if not bool(compensated):
                raise ValueError('LiDAR multi-frame BEV is not marked as ego-motion compensated')
            reference_time = _field(l_meta, ['reference_timestamp', 'reference_ts'], sample_index, required=strict)
            if reference_time is not None and l_ts is not None:
                if abs(_timestamp_to_ms(reference_time) - _timestamp_to_ms(l_ts)) > timestamp_tolerance_ms:
                    raise ValueError(f'LiDAR multi-frame reference timestamp mismatch: reference={reference_time}, lidar={l_ts}')

        bev_aliases = [
            (['x_min', ('bev_grid', 'x_min'), ('x_bound', 0), ('bev_grid', 'x_bound', 0)], 'x_min'),
            (['x_max', ('bev_grid', 'x_max'), ('x_bound', 1), ('bev_grid', 'x_bound', 1)], 'x_max'),
            (['y_min', ('bev_grid', 'y_min'), ('y_bound', 0), ('bev_grid', 'y_bound', 0)], 'y_min'),
            (['y_max', ('bev_grid', 'y_max'), ('y_bound', 1), ('bev_grid', 'y_bound', 1)], 'y_max'),
            (['dx', ('bev_grid', 'dx')], 'dx'),
            (['dy', ('bev_grid', 'dy')], 'dy'),
            (['H', 'height', 'y_grid', ('bev_grid', 'H'), ('bev_grid', 'height'), ('bev_grid', 'y_grid')], 'H'),
            (['W', 'width', 'x_grid', ('bev_grid', 'W'), ('bev_grid', 'width'), ('bev_grid', 'x_grid')], 'W'),
            (['origin', ('bev_grid', 'origin')], 'origin'),
            (['row_axis', ('bev_grid', 'row_axis')], 'row_axis'),
            (['col_axis', ('bev_grid', 'col_axis')], 'col_axis'),
            (['x_direction', 'x_axis', ('axis_convention', 'x_direction')], 'x_direction'),
            (['y_direction', 'y_axis', ('axis_convention', 'y_direction')], 'y_direction'),
            (['row_direction', 'row_convention', ('axis_convention', 'row_direction'), ('bev_grid', 'row_convention')], 'row_direction'),
            (['col_direction', ('axis_convention', 'col_direction')], 'col_direction'),
        ]
        for aliases, name in bev_aliases:
            _compare_field(v_meta, l_meta, aliases, name, sample_index, tol=range_tolerance, required=strict)

        visual_aug = _field(v_meta, ['aug_meta', 'augmentation', 'augmentation_meta'], sample_index)
        lidar_aug = _field(l_meta, ['aug_meta', 'augmentation', 'augmentation_meta'], sample_index)
        if visual_aug is None or lidar_aug is None:
            if strict:
                raise ValueError('Missing required BEV alignment field: augmentation metadata')
        elif not _values_equal(visual_aug, lidar_aug, tol=range_tolerance):
            raise ValueError(f'BEV augmentation mismatch: visual={visual_aug}, lidar={lidar_aug}')

    return True
