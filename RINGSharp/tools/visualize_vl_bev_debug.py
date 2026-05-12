import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

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
        description='Export per-frame VL BEV debug visualizations for alignment inspection'
    )
    parser.add_argument('--dataset_type', default='nclt', choices=['nclt', 'oxford'])
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--sequence', required=True, help='Sequence name, e.g. 2012-03-17')
    parser.add_argument('--frame_idx', type=int, default=0, help='Frame index inside the selected raw sequence')
    parser.add_argument('--model_config', required=True, help='VL model config, e.g. glnet/config/ring_sharp_vl_pr_nclt.txt')
    parser.add_argument('--weight', type=str, default=None, help='Optional trained VL checkpoint')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--enable_visual_backbone_pretrain',
        action='store_true',
        help='Keep visual backbone pretrain init enabled when constructing the model',
    )
    parser.add_argument(
        '--force_adaptive_fusion',
        action='store_true',
        help='Override model config and force adaptive_fusion=True for reliability/gate export',
    )
    return parser.parse_args()


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def save_rgb_png(path: str, image: np.ndarray) -> None:
    import cv2

    image = np.asarray(image)
    if image.ndim == 2:
        cv2.imwrite(path, image)
    elif image.shape[-1] == 3:
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif image.shape[-1] == 4:
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    else:
        raise ValueError(f'Unsupported image shape for PNG save: {tuple(image.shape)}')


def save_input_image(path: str, image: np.ndarray) -> None:
    import cv2

    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f'Expected HxWxC input image, got {tuple(image.shape)}')
    if image.shape[2] == 3:
        cv2.imwrite(path, image)
    elif image.shape[2] == 4:
        cv2.imwrite(path, image)
    else:
        raise ValueError(f'Unsupported input image shape: {tuple(image.shape)}')


def normalize_image(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f'Expected HxWxC image, got {tuple(image.shape)}')
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if image.shape[2] != 3:
        raise ValueError(f'Expected 3-channel image, got {tuple(image.shape)}')
    image = image / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(image)).float()


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


def tensor_bchw_to_numpy(tensor: torch.Tensor, name: str) -> np.ndarray:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f'Expected {name} with shape [1,C,H,W], got {tuple(tensor.shape)}')
    return tensor.squeeze(0).numpy()


def save_scalar_output(output_dir: str, prefix: str, scalar_map: np.ndarray) -> Dict[str, object]:
    scalar_map = np.asarray(scalar_map, dtype=np.float32).squeeze()
    np.save(os.path.join(output_dir, f'{prefix}.npy'), scalar_map.astype(np.float32))
    save_rgb_png(os.path.join(output_dir, f'{prefix}.png'), scalar_map_to_rgb(scalar_map))
    return {
        'shape': list(scalar_map.shape),
        'min': float(np.min(scalar_map)),
        'max': float(np.max(scalar_map)),
        'mean': float(np.mean(scalar_map)),
        'std': float(np.std(scalar_map)),
    }


def save_feature_output(output_dir: str, prefix: str, feature_map: np.ndarray) -> Dict[str, object]:
    feature_map = np.asarray(feature_map, dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError(f'Expected feature map [C,H,W], got {tuple(feature_map.shape)}')

    mean_map = feature_map.mean(axis=0)
    norm_map = np.linalg.norm(feature_map, axis=0)
    pca_rgb = feature_map_to_pca_rgb(feature_map)

    np.save(os.path.join(output_dir, f'{prefix}.npy'), feature_map.astype(np.float32))
    save_rgb_png(os.path.join(output_dir, f'{prefix}_mean.png'), scalar_map_to_rgb(mean_map))
    save_rgb_png(os.path.join(output_dir, f'{prefix}_norm.png'), scalar_map_to_rgb(norm_map))
    save_rgb_png(os.path.join(output_dir, f'{prefix}_pca.png'), pca_rgb)
    return {
        'shape': list(feature_map.shape),
        'min': float(np.min(feature_map)),
        'max': float(np.max(feature_map)),
        'mean': float(np.mean(feature_map)),
        'std': float(np.std(feature_map)),
        'visualizations': ['channel_mean', 'channel_l2_norm', 'channel_pca_rgb'],
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

    from glnet.models.model_factory import model_factory
    from glnet.utils.common_utils import _ex
    from glnet.utils.params import ModelParams
    from glnet.utils.data_utils.bev_common import build_bev_alignment_meta, get_lidar_bev_conf, load_image_meta
    from glnet.utils.data_utils.lidar_reliability import compute_lidar_reliability_bev
    from glnet.utils.data_utils.point_clouds import generate_bev
    from glnet.utils.data_utils.bev_common import get_sequence_dataset

    dataset_root = _ex(args.dataset_root)
    model_config = _ex(args.model_config)
    output_dir = _ex(args.output_dir)
    frame_dir = os.path.join(output_dir, f'{args.sequence}_frame_{args.frame_idx:06d}')
    if os.path.exists(frame_dir) and not args.overwrite:
        raise FileExistsError(f'{frame_dir} exists. Use --overwrite to replace it.')
    os.makedirs(frame_dir, exist_ok=True)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    params = ModelParams(model_config, args.dataset_type, dataset_root)
    params.image_meta_path = os.path.join(dataset_root, 'image_meta.pkl')
    if args.force_adaptive_fusion:
        params.adaptive_fusion = True
    if not args.enable_visual_backbone_pretrain:
        params.use_pretrained_model = False

    model = model_factory(params).to(device)
    load_info = load_model_weights(model, _ex(args.weight) if args.weight else None, device)
    model.eval()

    sequence_ds = get_sequence_dataset(args.dataset_type, dataset_root, args.sequence)
    if args.frame_idx < 0 or args.frame_idx >= len(sequence_ds):
        raise IndexError(f'frame_idx={args.frame_idx} is outside sequence length {len(sequence_ds)}')

    frame = sequence_ds[args.frame_idx]
    lidar_conf = get_lidar_bev_conf(args.dataset_type)
    bounds = (
        float(lidar_conf['x_bound'][0]),
        float(lidar_conf['x_bound'][1]),
        float(lidar_conf['y_bound'][0]),
        float(lidar_conf['y_bound'][1]),
        float(lidar_conf['z_bound'][0]),
        float(lidar_conf['z_bound'][1]),
    )
    grid_z = int(lidar_conf['z_grid'])
    grid_y = int(lidar_conf['y_grid'])
    grid_x = int(lidar_conf['x_grid'])

    pc = np.asarray(frame['pc'], dtype=np.float32)[:, :3]
    images = frame['img']
    timestamp = int(frame['ts'])
    pose = np.asarray(frame['pose'])

    rel_scan_filepath = sequence_ds.rel_scan_filepath[args.frame_idx]
    scan_filepath = os.path.join(dataset_root, rel_scan_filepath)
    bev_path = scan_filepath.replace('velodyne_sync', 'bev').replace('bin', 'npy')
    lidar_rel_path = bev_path.replace('bev', 'lidar_reliability_bev')

    if os.path.isfile(bev_path):
        pc_bev = np.load(bev_path).astype(np.float32)
    else:
        pc_bev = generate_bev(pc, Z=grid_z, Y=grid_y, X=grid_x, bounds=bounds).detach().cpu().numpy().astype(np.float32)
        os.makedirs(os.path.dirname(bev_path), exist_ok=True)
        np.save(bev_path, pc_bev)

    if os.path.isfile(lidar_rel_path):
        lidar_reliability = np.load(lidar_rel_path).astype(np.float32)
    else:
        lidar_reliability = compute_lidar_reliability_bev(
            pc,
            Z=grid_z,
            Y=grid_y,
            X=grid_x,
            bounds=bounds,
            downsample_voxel_size=params.adaptive_lidar_reliability_downsample,
            k=params.adaptive_lidar_reliability_k,
            min_neighbors=params.adaptive_lidar_reliability_min_neighbors,
            eps=params.adaptive_eps,
        )
        os.makedirs(os.path.dirname(lidar_rel_path), exist_ok=True)
        np.save(lidar_rel_path, lidar_reliability.astype(np.float32))

    image_tensors = [normalize_image(image) for image in images]
    batch = {
        'orig_pc': pc,
        'pc': torch.from_numpy(pc_bev).unsqueeze(0).float().to(device),
        'img': torch.stack(image_tensors, dim=0).unsqueeze(0).float().to(device),
        'image_meta': load_image_meta(args.dataset_type, dataset_root),
        'bev_meta': [
            build_bev_alignment_meta(
                sample_id=timestamp,
                timestamp=timestamp,
                dataset_type=args.dataset_type,
                pose=pose,
                xyz_aug=False,
            )
        ],
        'lidar_reliability_bev': torch.from_numpy(lidar_reliability).unsqueeze(0).float().to(device),
    }

    with torch.inference_mode():
        output = model(batch)

    # Save input views and offline maps.
    input_dir = os.path.join(frame_dir, 'inputs')
    os.makedirs(input_dir, exist_ok=True)
    for cam_ndx, image in enumerate(images):
        save_input_image(os.path.join(input_dir, f'cam_{cam_ndx + 1}.png'), image)
    save_scalar_output(input_dir, 'lidar_occupancy_sum', pc_bev.sum(axis=0))
    save_scalar_output(input_dir, 'lidar_reliability_bev', lidar_reliability[0])

    # Save model features.
    features_dir = os.path.join(frame_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    output_meta = {}
    for key in ('visual_bev', 'lidar_bev', 'fused_bev', 'spec'):
        if key in output and output[key] is not None:
            feature_np = tensor_bchw_to_numpy(output[key], key)
            output_meta[key] = save_feature_output(features_dir, key, feature_np)

    # Save adaptive maps when adaptive fusion is enabled.
    adaptive_meta = {}
    if 'adaptive' in output and output['adaptive'] is not None:
        adaptive_dir = os.path.join(frame_dir, 'adaptive')
        os.makedirs(adaptive_dir, exist_ok=True)
        for key, value in output['adaptive'].items():
            if value is None:
                continue
            value_np = tensor_bchw_to_numpy(value, f'adaptive_{key}')
            adaptive_meta[key] = save_scalar_output(adaptive_dir, key, value_np[0])
        if 'Mv' in output['adaptive']:
            mv_np = tensor_bchw_to_numpy(output['adaptive']['Mv'], 'adaptive_Mv')
            adaptive_meta['visual_reliability_bev'] = save_scalar_output(
                adaptive_dir, 'visual_reliability_bev', mv_np[0]
            )
            adaptive_meta['visual_reliability_bev_alias'] = 'Mv'
        if 'Ml' in output['adaptive']:
            ml_np = tensor_bchw_to_numpy(output['adaptive']['Ml'], 'adaptive_Ml')
            adaptive_meta['lidar_reliability_bev'] = save_scalar_output(
                adaptive_dir, 'lidar_reliability_bev', ml_np[0]
            )
            adaptive_meta['lidar_reliability_bev_alias'] = 'Ml'

    meta = {
        'dataset_type': args.dataset_type,
        'dataset_root': dataset_root,
        'sequence': args.sequence,
        'frame_idx': int(args.frame_idx),
        'timestamp': timestamp,
        'pose': pose.astype(np.float32).tolist(),
        'scan_filepath': scan_filepath,
        'bev_path': bev_path,
        'lidar_reliability_path': lidar_rel_path,
        'model_config': model_config,
        'weight': load_info,
        'device': str(device),
        'adaptive_fusion': bool(params.adaptive_fusion),
        'input_shapes': {
            'pc': list(pc.shape),
            'pc_bev': list(pc_bev.shape),
            'lidar_reliability_bev': list(lidar_reliability.shape),
            'img': list(batch['img'].shape),
        },
        'outputs': output_meta,
        'adaptive_outputs': adaptive_meta,
        'visualization_notes': {
            'feature_mean': 'channel mean of [C,H,W] feature maps',
            'feature_norm': 'channel L2 norm of [C,H,W] feature maps',
            'feature_pca': 'first 3 PCA components of channel features rendered as RGB',
            'scalar_maps': 'percentile-normalized jet colormap',
        },
    }
    write_json(os.path.join(frame_dir, 'meta.json'), meta)
    print(f'Exported VL BEV debug visualizations to: {frame_dir}')


if __name__ == '__main__':
    main()
