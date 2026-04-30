import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
for path in (WORKSPACE_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
VIS_PERCENTILES = (1.0, 99.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export paired visual/LiDAR BEV feature maps without running rotation or translation branches'
    )
    parser.add_argument('--dataset_type', required=True, choices=['nclt', 'oxford'])
    parser.add_argument('--dataset_root', required=True, help='Dataset root following the repository README layout')
    parser.add_argument('--sequence', required=True, help='Sequence name to export')
    parser.add_argument('--output_dir', required=True, help='Directory used to save paired BEV feature outputs')
    parser.add_argument('--start_idx', type=int, default=0, help='Inclusive frame index inside the raw sequence iterator')
    parser.add_argument('--end_idx', type=int, default=None, help='Exclusive frame index inside the raw sequence iterator')
    parser.add_argument('--frame_stride', type=int, default=1, help='Export one frame every N frames')
    parser.add_argument('--num_samples', type=int, default=None, help='Optional cap on exported frame count after stride')
    parser.add_argument('--visual_model_config', type=str, default=None, help='Model config for the visual branch')
    parser.add_argument('--lidar_model_config', type=str, default=None, help='Model config for the LiDAR branch')
    parser.add_argument('--visual_weight', type=str, default=None, help='Checkpoint for the visual branch')
    parser.add_argument('--lidar_weight', type=str, default=None, help='Checkpoint for the LiDAR branch')
    parser.add_argument('--device', type=str, default=None, help='Torch device, defaults to cuda if available')
    parser.add_argument(
        '--enable_visual_backbone_pretrain',
        action='store_true',
        help='Keep the visual backbone torchvision pretrain init enabled when no checkpoint is provided',
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing pair directories instead of skipping them')
    return parser.parse_args()


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def default_model_config(dataset_type: str, branch: str) -> str:
    file_name = f'ring_sharp_{branch}_{dataset_type}.txt'
    return os.path.join(REPO_ROOT, 'glnet', 'config', file_name)


def select_indices(num_frames: int, start_idx: int, end_idx: Optional[int], frame_stride: int, num_samples: Optional[int]) -> List[int]:
    start_idx = max(0, start_idx)
    end_idx = num_frames if end_idx is None else min(num_frames, end_idx)
    indices = list(range(start_idx, end_idx, max(1, frame_stride)))
    if num_samples is not None:
        indices = indices[: max(0, num_samples)]
    return indices


def build_grid_alignment_report(lidar_conf: Dict[str, object], visual_grid) -> Dict[str, bool]:
    expected_x = [visual_grid.x_min, visual_grid.x_max]
    expected_y = [visual_grid.y_min, visual_grid.y_max]
    return {
        'x_bound_matches_visual': np.allclose(lidar_conf['x_bound'], expected_x),
        'y_bound_matches_visual': np.allclose(lidar_conf['y_bound'], expected_y),
        'x_grid_matches_visual': int(lidar_conf['x_grid']) == int(visual_grid.x_cells),
        'y_grid_matches_visual': int(lidar_conf['y_grid']) == int(visual_grid.y_cells),
    }


def get_lidar_bev_generation_args(lidar_conf: Dict[str, object]) -> Dict[str, object]:
    return {
        'Z': int(lidar_conf['z_grid']),
        'Y': int(lidar_conf['y_grid']),
        'X': int(lidar_conf['x_grid']),
        'bounds': (
            float(lidar_conf['x_bound'][0]),
            float(lidar_conf['x_bound'][1]),
            float(lidar_conf['y_bound'][0]),
            float(lidar_conf['y_bound'][1]),
            float(lidar_conf['z_bound'][0]),
            float(lidar_conf['z_bound'][1]),
        ),
    }


def normalize_image(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected HxWx3 image, got {tuple(image.shape)}')
    image = image / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(image)).float()


def robust_normalize(values: np.ndarray, percentiles: Tuple[float, float] = VIS_PERCENTILES) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    normalized = np.zeros_like(values, dtype=np.float32)
    mask = np.isfinite(values)
    if not np.any(mask):
        return normalized

    valid = values[mask]
    lo, hi = np.percentile(valid, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        lo = float(valid.min())
        hi = float(valid.max())
    if hi <= lo + 1e-6:
        normalized[mask] = 0.5
        return normalized

    normalized[mask] = np.clip((values[mask] - lo) / (hi - lo), 0.0, 1.0)
    return normalized


def scalar_map_to_rgb(scalar_map: np.ndarray) -> np.ndarray:
    import cv2

    normalized = robust_normalize(scalar_map)
    image = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def feature_map_to_pca_rgb(feature_map: np.ndarray) -> np.ndarray:
    feature_map = np.asarray(feature_map, dtype=np.float32)
    channels, height, width = feature_map.shape
    flat = feature_map.reshape(channels, -1).T
    mask = np.isfinite(flat).all(axis=1)
    rgb = np.zeros((flat.shape[0], 3), dtype=np.float32)

    if int(np.count_nonzero(mask)) < 3:
        return (rgb.reshape(height, width, 3) * 255.0).astype(np.uint8)

    valid = flat[mask]
    valid = valid - np.mean(valid, axis=0, keepdims=True)
    valid = valid / np.clip(np.linalg.norm(valid, axis=1, keepdims=True), 1e-6, None)
    valid = valid - np.mean(valid, axis=0, keepdims=True)

    _, _, vh = np.linalg.svd(valid, full_matrices=False)
    num_components = min(3, vh.shape[0])
    projected = valid @ vh[:num_components].T

    for channel_ndx in range(num_components):
        rgb[mask, channel_ndx] = robust_normalize(projected[:, channel_ndx])

    return np.clip(rgb.reshape(height, width, 3) * 255.0, 0.0, 255.0).astype(np.uint8)


def feature_tensor_to_numpy(feature_tensor: torch.Tensor) -> np.ndarray:
    feature_tensor = feature_tensor.detach().cpu().float()
    if feature_tensor.ndim != 4 or feature_tensor.shape[0] != 1:
        raise ValueError(f'Expected feature tensor with shape (1, C, H, W), got {tuple(feature_tensor.shape)}')
    return feature_tensor.squeeze(0).numpy()


def feature_stats(feature_map: np.ndarray) -> Dict[str, float]:
    return {
        'min': float(np.min(feature_map)),
        'max': float(np.max(feature_map)),
        'mean': float(np.mean(feature_map)),
        'std': float(np.std(feature_map)),
    }


def save_feature_outputs(
    pair_dir: str,
    prefix: str,
    feature_tensor: np.ndarray,
    save_rgb_png,
) -> Dict[str, object]:
    mean_map = feature_tensor.mean(axis=0)
    norm_map = np.linalg.norm(feature_tensor, axis=0)
    pca_rgb = feature_map_to_pca_rgb(feature_tensor)

    np.save(os.path.join(pair_dir, f'{prefix}.npy'), feature_tensor.astype(np.float32))
    save_rgb_png(os.path.join(pair_dir, f'{prefix}_vis.png'), scalar_map_to_rgb(norm_map))
    save_rgb_png(os.path.join(pair_dir, f'{prefix}_mean.png'), scalar_map_to_rgb(mean_map))
    save_rgb_png(os.path.join(pair_dir, f'{prefix}_norm.png'), scalar_map_to_rgb(norm_map))
    save_rgb_png(os.path.join(pair_dir, f'{prefix}_pca.png'), pca_rgb)

    return {
        'shape': list(feature_tensor.shape),
        'stats': feature_stats(feature_tensor),
        'visualizations': {
            'primary': 'channel_l2_norm',
            'extras': ['channel_mean', 'channel_pca_rgb'],
            'normalization': {
                'type': 'percentile',
                'lower': VIS_PERCENTILES[0],
                'upper': VIS_PERCENTILES[1],
                'colormap': 'jet',
            },
        },
    }


def parse_timestamp_from_path(path: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else None


def resolve_frame_paths(dataset_type: str, dataset_root: str, sequence_ds, frame_index: int):
    if dataset_type == 'nclt':
        from glnet.datasets.nclt.nclt_raw import pc2image_file

        lidar_path = os.path.join(dataset_root, sequence_ds.rel_scan_filepath[frame_index])
        image_paths = [pc2image_file(lidar_path, '/velodyne_sync/', cam_num, '.bin') for cam_num in range(1, 6)]
        timestamp = int(sequence_ds.timestamps[frame_index])
        return {
            'image_paths': image_paths,
            'lidar_paths': [lidar_path],
            'sensor_timestamps': {
                'lidar': timestamp,
                'camera_1': timestamp,
                'camera_2': timestamp,
                'camera_3': timestamp,
                'camera_4': timestamp,
                'camera_5': timestamp,
            },
            'sync_policy': 'NCLTSequence uses identical filename timestamps for LiDAR and all five rectified cameras.',
        }

    filepaths = sequence_ds.filepaths[frame_index] if hasattr(sequence_ds, 'filepaths') else sequence_ds.get_filepaths(frame_index)
    return {
        'image_paths': filepaths[2:6],
        'lidar_paths': filepaths[:2],
        'sensor_timestamps': {
            'lidar_left': parse_timestamp_from_path(filepaths[0]),
            'lidar_right': parse_timestamp_from_path(filepaths[1]),
            'mono_left': parse_timestamp_from_path(filepaths[2]),
            'mono_right': parse_timestamp_from_path(filepaths[3]),
            'mono_rear': parse_timestamp_from_path(filepaths[4]),
            'stereo_centre': parse_timestamp_from_path(filepaths[5]),
        },
        'sync_policy': 'OxfordSequence keeps the left LiDAR frame as the anchor timestamp and matches the right LiDAR plus four rectified cameras by nearest timestamp.',
    }


def load_model_weights(model: torch.nn.Module, weight_path: Optional[str], device: torch.device) -> Dict[str, object]:
    if weight_path is None:
        return {'loaded': False, 'path': None}

    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_keys = set(model.state_dict().keys())
    candidates = [state_dict]
    for prefix in ('module.', 'model.'):
        if any(key.startswith(prefix) for key in state_dict.keys()):
            candidates.append({
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in state_dict.items()
            })

    def overlap(candidate):
        return len(model_keys.intersection(candidate.keys()))

    best_state_dict = max(candidates, key=overlap)
    incompatible = model.load_state_dict(best_state_dict, strict=False)
    return {
        'loaded': True,
        'path': weight_path,
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
    }


def main():
    args = parse_args()

    from glnet.utils.common_utils import _ex
    from glnet.utils.params import ModelParams
    from glnet.utils.data_utils.point_clouds import generate_bev
    from glnet.utils.data_utils.bev_common import (
        get_lidar_bev_conf,
        get_sequence_dataset,
        get_visual_bev_grid,
        load_image_meta,
        save_rgb_png,
    )
    from glnet.models.localizer.ring_sharp_v import RINGSharpV
    from glnet.models.localizer.ring_sharp_l import RINGSharpL

    dataset_root = _ex(args.dataset_root)
    output_dir = _ex(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    visual_model_config = _ex(args.visual_model_config or default_model_config(args.dataset_type, 'v'))
    lidar_model_config = _ex(args.lidar_model_config or default_model_config(args.dataset_type, 'l'))
    visual_weight = _ex(args.visual_weight) if args.visual_weight is not None else None
    lidar_weight = _ex(args.lidar_weight) if args.lidar_weight is not None else None

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    visual_grid = get_visual_bev_grid()
    lidar_conf = get_lidar_bev_conf(args.dataset_type)
    lidar_bev_args = get_lidar_bev_generation_args(lidar_conf)
    grid_alignment = build_grid_alignment_report(lidar_conf, visual_grid)
    if not all(grid_alignment.values()):
        raise RuntimeError(f'LiDAR x/y grid is not aligned with the visual grid: {grid_alignment}')

    image_meta_path = os.path.join(dataset_root, 'image_meta.pkl')
    image_meta = load_image_meta(args.dataset_type, dataset_root)
    image_meta_source = 'pickle' if os.path.exists(image_meta_path) else 'calibration_files'

    visual_params = ModelParams(visual_model_config, args.dataset_type, dataset_root)
    if not args.enable_visual_backbone_pretrain:
        visual_params.use_pretrained_model = False
    visual_params.image_meta_path = image_meta_path

    lidar_params = ModelParams(lidar_model_config, args.dataset_type, dataset_root)

    visual_model = RINGSharpV(visual_params).to(device)
    lidar_model = RINGSharpL(lidar_params).to(device)

    visual_load_info = load_model_weights(visual_model, visual_weight, device)
    lidar_load_info = load_model_weights(lidar_model, lidar_weight, device)
    visual_model.eval()
    lidar_model.eval()

    sequence_ds = get_sequence_dataset(args.dataset_type, dataset_root, args.sequence)
    export_indices = select_indices(len(sequence_ds), args.start_idx, args.end_idx, args.frame_stride, args.num_samples)
    print(f'Found {len(sequence_ds)} valid raw frames in {args.sequence}; exporting {len(export_indices)} feature pairs')

    if len(export_indices) == 0:
        print('No frames selected; nothing to export')
        return

    with torch.inference_mode():
        for export_ndx, frame_ndx in enumerate(export_indices):
            pair_dir = os.path.join(output_dir, f'pair_{export_ndx:06d}')
            if os.path.exists(pair_dir) and not args.overwrite:
                print(f'Skipping existing {pair_dir} (use --overwrite to replace it)')
                continue
            os.makedirs(pair_dir, exist_ok=True)

            frame = sequence_ds[frame_ndx]
            frame_paths = resolve_frame_paths(args.dataset_type, dataset_root, sequence_ds, frame_ndx)

            image_tensors = [normalize_image(image) for image in frame['img']]
            visual_batch = {
                'img': torch.stack(image_tensors, dim=0).unsqueeze(0).to(device),
                'image_meta': image_meta,
            }

            lidar_bev_input = generate_bev(
                frame['pc'],
                Z=lidar_bev_args['Z'],
                Y=lidar_bev_args['Y'],
                X=lidar_bev_args['X'],
                bounds=lidar_bev_args['bounds'],
            ).unsqueeze(0).to(device).float()
            lidar_batch = {'pc': lidar_bev_input}

            visual_bev = visual_model.extract_bev_features(visual_batch)['bev']
            lidar_bev = lidar_model.extract_bev_features(lidar_batch)['bev']

            if tuple(visual_bev.shape[-2:]) != tuple(lidar_bev.shape[-2:]):
                raise RuntimeError(
                    f'Visual and LiDAR BEV feature grids differ: visual={tuple(visual_bev.shape[-2:])}, '
                    f'lidar={tuple(lidar_bev.shape[-2:])}. Resize is intentionally forbidden.'
                )

            visual_bev_np = feature_tensor_to_numpy(visual_bev)
            lidar_bev_np = feature_tensor_to_numpy(lidar_bev)

            visual_meta = save_feature_outputs(pair_dir, 'visual_bev_feat', visual_bev_np, save_rgb_png)
            lidar_meta = save_feature_outputs(pair_dir, 'lidar_bev_feat', lidar_bev_np, save_rgb_png)

            meta = {
                'dataset': args.dataset_type,
                'dataset_root': dataset_root,
                'sequence': args.sequence,
                'frame_index': int(frame_ndx),
                'timestamp': int(frame['ts']),
                'pose': np.asarray(frame['pose'], dtype=np.float32).tolist(),
                'data_paths': {
                    'images': frame_paths['image_paths'],
                    'lidar': frame_paths['lidar_paths'],
                },
                'sensor_timestamps': frame_paths['sensor_timestamps'],
                'sync_policy': frame_paths['sync_policy'],
                'visual_model_config': visual_model_config,
                'lidar_model_config': lidar_model_config,
                'visual_weight': visual_load_info,
                'lidar_weight': lidar_load_info,
                'visual_feature_stage': 'RINGSharpV.extract_bev_features: image backbone + depth net + view transform output before yaw/spec and translation branches.',
                'lidar_feature_stage': 'RINGSharpL.extract_bev_features: SteerableCNN BEV encoder output before yaw/spec and translation branches.',
                'visual_feature': visual_meta,
                'lidar_feature': lidar_meta,
                'shared_bev_grid': visual_grid.to_dict(),
                'lidar_bev_conf': {
                    'x_bound': list(lidar_conf['x_bound']),
                    'y_bound': list(lidar_conf['y_bound']),
                    'z_bound': list(lidar_conf['z_bound']),
                    'x_grid': int(lidar_conf['x_grid']),
                    'y_grid': int(lidar_conf['y_grid']),
                    'z_grid': int(lidar_conf['z_grid']),
                },
                'grid_alignment_with_visual': grid_alignment,
                'visual_input_image_shape': list(visual_batch['img'].shape[-3:]),
                'lidar_input_bev_shape': list(lidar_bev_input.shape[1:]),
                'image_meta_path': image_meta_path,
                'image_meta_source': image_meta_source,
                'rotation_translation_applied': False,
                'stopped_at_bev_feature_only': True,
            }
            write_json(os.path.join(pair_dir, 'meta.json'), meta)
            print(f'[{export_ndx + 1}/{len(export_indices)}] exported frame {frame_ndx} (ts={frame["ts"]}) -> {pair_dir}')


if __name__ == '__main__':
    main()
