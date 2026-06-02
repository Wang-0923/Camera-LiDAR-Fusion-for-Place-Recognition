import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
for path in (WORKSPACE_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize one PR query and its top-k retrieved database panoramas.'
    )
    parser.add_argument('--dataset_type', default='nclt', choices=['nclt', 'oxford'])
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--eval_set', required=True, help='Evaluation pickle path or file name under dataset_root')
    parser.add_argument('--result_dir', required=True, help='Directory containing pair_dists.npy')
    parser.add_argument('--pair_dists', default=None, help='Optional explicit pair_dists.npy path')
    parser.add_argument('--query_idx', type=int, default=0, help='Query index in eval_set.query_set')
    parser.add_argument('--query_timestamp', type=int, default=None, help='Optional query timestamp, overrides query_idx')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--find_query', action='store_true', help='Automatically select a query with enough correct top-k matches')
    parser.add_argument('--min_correct_topk', type=int, default=None, help='Minimum correct matches required when --find_query is used')
    parser.add_argument('--num_candidates', type=int, default=10, help='Print this many best query candidates when --find_query is used')
    parser.add_argument('--revisit_threshold', type=float, default=5.0)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--save_pointclouds', action='store_true', help='Save raw point cloud visualizations for query and top-k maps')
    parser.add_argument('--max_pointcloud_points', type=int, default=80000, help='Max points drawn in each point cloud PNG; .npy always stores all finite XYZ points')
    parser.add_argument('--model_config', default=None, help='Optional VL model config for adaptive fusion weight export')
    parser.add_argument('--weight', default=None, help='Optional VL checkpoint for adaptive fusion weight export')
    parser.add_argument('--device', default=None, help='Device for optional model forward, defaults to cuda when available')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def resolve_existing_path(path: str, base_dir: str = None) -> str:
    path = expand_path(path)
    if os.path.exists(path):
        return path
    if base_dir is not None:
        candidate = os.path.join(expand_path(base_dir), path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'Cannot access path: {path}')


def load_eval_set(eval_set_path: str):
    from glnet.datasets.base_datasets import EvaluationSet

    eval_set = EvaluationSet()
    eval_set.load(eval_set_path)
    if len(eval_set.query_set) == 0 or len(eval_set.map_set) == 0:
        raise ValueError('Evaluation set must contain non-empty query_set and map_set')
    return eval_set


def find_query_index(eval_set, query_idx: int, query_timestamp: int = None) -> int:
    if query_timestamp is not None:
        matches = [idx for idx, item in enumerate(eval_set.query_set) if int(item.timestamp) == int(query_timestamp)]
        if len(matches) == 0:
            raise ValueError(f'Cannot find query timestamp {query_timestamp} in evaluation set')
        return matches[0]

    if query_idx < 0 or query_idx >= len(eval_set.query_set):
        raise IndexError(f'query_idx={query_idx} outside [0, {len(eval_set.query_set)})')
    return query_idx


def rank_query_candidates(eval_set, pair_dists: np.ndarray, topk: int, revisit_threshold: float):
    map_positions = eval_set.get_map_positions()
    candidates = []
    for query_idx, query_item in enumerate(eval_set.query_set):
        query_pos = np.asarray(query_item.position, dtype=np.float64)
        all_geo_dists = np.linalg.norm(map_positions - query_pos.reshape(1, -1), axis=1)
        num_gt_positives = int(np.count_nonzero(all_geo_dists <= revisit_threshold))
        top_indices = np.argsort(pair_dists[query_idx])[:topk]
        geo_dists = all_geo_dists[top_indices]
        correct_mask = geo_dists <= revisit_threshold
        correct_count = int(np.count_nonzero(correct_mask))
        first_wrong_rank = next((rank for rank, ok in enumerate(correct_mask, start=1) if not bool(ok)), topk + 1)
        correct_prefix_len = 0
        for ok in correct_mask:
            if not bool(ok):
                break
            correct_prefix_len += 1
        mean_correct_dist = float(np.mean(geo_dists[correct_mask])) if correct_count > 0 else float('inf')
        candidates.append({
            'query_index': int(query_idx),
            'timestamp': int(query_item.timestamp),
            'correct_count': correct_count,
            'correct_prefix_len': int(correct_prefix_len),
            'num_gt_positives': num_gt_positives,
            'max_possible_correct': int(min(topk, num_gt_positives)),
            'first_wrong_rank': int(first_wrong_rank),
            'top_indices': top_indices.astype(int).tolist(),
            'top_correct': correct_mask.astype(bool).tolist(),
            'top_geo_dists': geo_dists.astype(float).tolist(),
            'mean_correct_geo_dist': mean_correct_dist,
        })

    candidates.sort(
        key=lambda item: (
            -item['correct_count'],
            -item['correct_prefix_len'],
            -item['max_possible_correct'],
            item['mean_correct_geo_dist'],
            item['query_index'],
        )
    )
    return candidates


def summarize_pair_dists_recall(eval_set, pair_dists: np.ndarray, topk: int, revisit_threshold: float) -> Dict[str, object]:
    map_positions = eval_set.get_map_positions()
    positive_queries = 0
    hits_at_1 = 0
    hits_at_k = 0
    max_gt_positives = 0
    queries_with_k_gt_positives = 0
    for query_idx, query_item in enumerate(eval_set.query_set):
        query_pos = np.asarray(query_item.position, dtype=np.float64)
        all_geo_dists = np.linalg.norm(map_positions - query_pos.reshape(1, -1), axis=1)
        num_gt_positives = int(np.count_nonzero(all_geo_dists <= revisit_threshold))
        max_gt_positives = max(max_gt_positives, num_gt_positives)
        if num_gt_positives >= topk:
            queries_with_k_gt_positives += 1
        if num_gt_positives == 0:
            continue
        positive_queries += 1
        top_indices = np.argsort(pair_dists[query_idx])[:topk]
        top_correct = all_geo_dists[top_indices] <= revisit_threshold
        hits_at_1 += int(bool(top_correct[0]))
        hits_at_k += int(np.any(top_correct))

    denom = max(positive_queries, 1)
    return {
        'positive_queries': int(positive_queries),
        'recall_at_1': float(hits_at_1 / denom),
        f'recall_at_{topk}': float(hits_at_k / denom),
        'max_gt_positives_per_query': int(max_gt_positives),
        f'queries_with_at_least_{topk}_gt_positives': int(queries_with_k_gt_positives),
    }


def load_eval_frame(
    item,
    dataset_type: str,
    dataset_root: str,
    pcim_loader,
):
    if dataset_type == 'nclt':
        scan_path = os.path.join(dataset_root, item.rel_scan_filepath)
        if not os.path.exists(scan_path):
            raise FileNotFoundError(f'Cannot access NCLT scan: {scan_path}')
        pointcloud, images = pcim_loader(scan_path, sph=False)
        return np.asarray(pointcloud), images

    if dataset_type == 'oxford':
        if item.filepaths is None:
            raise ValueError('Oxford evaluation tuple is missing filepaths')
        extrinsics_dir = os.path.join(dataset_root, 'extrinsics')
        pointcloud, images = pcim_loader(item.filepaths, sph=False, extrinsics_dir=extrinsics_dir)
        return np.asarray(pointcloud), images

    raise NotImplementedError(f'Unsupported dataset type: {dataset_type}')


def save_rgb(path: str, image: np.ndarray) -> None:
    import cv2

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected RGB image with shape [H,W,3], got {tuple(image.shape)}')
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def annotate_rgb(image: np.ndarray, title: str, is_correct: bool = None) -> np.ndarray:
    import cv2

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected RGB image with shape [H,W,3], got {tuple(image.shape)}')

    annotated = image.copy()
    h, w = annotated.shape[:2]
    banner_h = max(42, h // 6)
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    if is_correct is True:
        banner[:, :] = np.array([20, 120, 40], dtype=np.uint8)
    elif is_correct is False:
        banner[:, :] = np.array([160, 35, 35], dtype=np.uint8)
    else:
        banner[:, :] = np.array([45, 45, 45], dtype=np.uint8)

    out = np.concatenate([banner, annotated], axis=0)
    cv2.putText(
        out,
        title,
        (12, min(banner_h - 12, 32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_overview(images: List[np.ndarray]) -> np.ndarray:
    import cv2

    heights = [img.shape[0] for img in images]
    target_h = max(heights)
    resized = []
    for img in images:
        if img.shape[0] != target_h:
            scale = target_h / float(img.shape[0])
            target_w = max(1, int(round(img.shape[1] * scale)))
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(img)
    return np.concatenate(resized, axis=1)


def save_pointcloud_views(path_prefix: str, pointcloud: np.ndarray, title: str, max_points: int = 80000) -> Dict[str, object]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pc = np.asarray(pointcloud, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f'Expected point cloud with shape [N,>=3], got {tuple(pc.shape)}')
    pc = pc[:, :3]
    pc = pc[np.isfinite(pc).all(axis=1)]
    if pc.shape[0] == 0:
        raise ValueError('Point cloud contains no finite XYZ points')

    if pc.shape[0] > max_points:
        rng = np.random.default_rng(17)
        keep = rng.choice(pc.shape[0], size=max_points, replace=False)
        pc_vis = pc[keep]
    else:
        pc_vis = pc

    colors = pc_vis[:, 2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    axes[0].scatter(pc_vis[:, 0], pc_vis[:, 1], c=colors, s=0.2, cmap='viridis', linewidths=0)
    axes[0].set_title(f'{title} top-down')
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].grid(True, linewidth=0.2)

    axes[1].scatter(pc_vis[:, 0], pc_vis[:, 2], c=pc_vis[:, 1], s=0.2, cmap='plasma', linewidths=0)
    axes[1].set_title(f'{title} side')
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('z [m]')
    axes[1].grid(True, linewidth=0.2)

    fig.tight_layout()
    png_path = f'{path_prefix}.png'
    fig.savefig(png_path)
    plt.close(fig)

    npy_path = f'{path_prefix}.npy'
    np.save(npy_path, pc.astype(np.float32))
    return {
        'pointcloud_png': os.path.basename(png_path),
        'pointcloud_npy': os.path.basename(npy_path),
        'num_points': int(pc.shape[0]),
        'num_visualized_points': int(pc_vis.shape[0]),
        'xyz_min': pc.min(axis=0).astype(float).tolist(),
        'xyz_max': pc.max(axis=0).astype(float).tolist(),
    }


def item_to_dict(item) -> Dict[str, object]:
    return {
        'timestamp': int(item.timestamp),
        'rel_scan_filepath': item.rel_scan_filepath,
        'position': np.asarray(item.position).astype(float).tolist(),
    }


def load_optional_adaptive_model(args, dataset_root: str, eval_set_path: str):
    if args.model_config is None and args.weight is None:
        return None, None
    if args.model_config is None or args.weight is None:
        raise ValueError('--model_config and --weight must be provided together for adaptive weight export')

    import torch
    from evaluate_pr import PREvaluator, _load_checkpoint
    from glnet.models.model_factory import model_factory
    from glnet.utils.params import ModelParams

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model_params = ModelParams(args.model_config, args.dataset_type, dataset_root)
    if not getattr(model_params, 'adaptive_fusion', False):
        print('Warning: model_config has adaptive_fusion=False; adaptive weights will not be available.')
    model = model_factory(model_params)
    model.to(device)
    weight_path = resolve_existing_path(args.weight)
    _load_checkpoint(model, weight_path, device)
    model.eval()

    evaluator = PREvaluator(
        dataset_root,
        args.dataset_type,
        eval_set_path,
        device=device,
        params=model_params,
        radius=[args.revisit_threshold],
        k=max(args.topk, 1),
        n_samples=None,
    )
    return evaluator, model


def tensor_stats(tensor) -> Optional[Dict[str, object]]:
    if tensor is None:
        return None
    arr = tensor.detach().float().cpu().numpy()
    return {
        'shape': list(arr.shape),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def compute_adaptive_weights(adaptive_evaluator, model, item) -> Optional[Dict[str, object]]:
    if adaptive_evaluator is None or model is None:
        return None
    import torch

    batch = adaptive_evaluator._build_batch(item)
    with torch.no_grad():
        output = model(batch)
    adaptive = output.get('adaptive')
    if adaptive is None:
        return None

    stats = {}
    for key in ('Wv', 'Wl', 'Gv', 'Gl', 'Mv', 'Ml'):
        if key in adaptive:
            stats[key] = tensor_stats(adaptive[key])
    if 'Wv' in stats and 'Wl' in stats:
        wv = stats['Wv']['mean']
        wl = stats['Wl']['mean']
        denom = max(wv + wl, 1e-12)
        stats['mean_weight_ratio'] = {
            'visual': float(wv / denom),
            'lidar': float(wl / denom),
        }
    return stats


def main():
    args = parse_args()

    from glnet.datasets.base_datasets import get_pointcloud_with_image_loader
    from glnet.datasets.panorama import generate_sph_image

    dataset_root = expand_path(args.dataset_root)
    eval_set_path = resolve_existing_path(args.eval_set, base_dir=dataset_root)
    result_dir = resolve_existing_path(args.result_dir)
    pair_dists_path = resolve_existing_path(args.pair_dists or os.path.join(result_dir, 'pair_dists.npy'))
    output_dir = expand_path(args.output_dir)

    eval_set = load_eval_set(eval_set_path)
    pair_dists = np.load(pair_dists_path)
    if pair_dists.shape != (len(eval_set.query_set), len(eval_set.map_set)):
        raise ValueError(
            f'pair_dists shape {tuple(pair_dists.shape)} does not match '
            f'queries/maps {(len(eval_set.query_set), len(eval_set.map_set))}'
        )

    topk = min(int(args.topk), len(eval_set.map_set))
    recall_summary = summarize_pair_dists_recall(eval_set, pair_dists, topk, args.revisit_threshold)
    print(
        f"pair_dists recall sanity: positive_queries={recall_summary['positive_queries']}, "
        f"recall@1={recall_summary['recall_at_1']:.4f}, "
        f"recall@{topk}={recall_summary[f'recall_at_{topk}']:.4f}, "
        f"max_gt_positives_per_query={recall_summary['max_gt_positives_per_query']}, "
        f"queries_with_at_least_{topk}_gt_positives="
        f"{recall_summary[f'queries_with_at_least_{topk}_gt_positives']}"
    )
    if args.find_query:
        candidates = rank_query_candidates(eval_set, pair_dists, topk, args.revisit_threshold)
        min_correct = args.min_correct_topk if args.min_correct_topk is not None else topk
        selected = next((item for item in candidates if item['correct_count'] >= min_correct), None)
        if selected is None:
            selected = candidates[0]
            print(
                f'No query has at least {min_correct}/{topk} correct matches. '
                f'Using best available query_idx={selected["query_index"]} '
                f'with {selected["correct_count"]}/{topk} correct.'
            )
        query_idx = int(selected['query_index'])
        print('Best query candidates:')
        for item in candidates[:max(1, args.num_candidates)]:
            print(
                f"query_idx={item['query_index']} ts={item['timestamp']} "
                f"correct={item['correct_count']}/{topk} "
                f"prefix={item['correct_prefix_len']}/{topk} "
                f"gt_pos={item['num_gt_positives']} "
                f"top_correct={item['top_correct']} "
                f"top_geo_dists={[round(v, 2) for v in item['top_geo_dists']]}"
            )
        print(f'Selected query_idx={query_idx}')
    else:
        query_idx = find_query_index(eval_set, args.query_idx, args.query_timestamp)
    top_indices = np.argsort(pair_dists[query_idx])[:topk]

    query_item = eval_set.query_set[query_idx]
    query_pos = np.asarray(query_item.position, dtype=np.float64)
    query_out_dir = os.path.join(output_dir, f'query_{query_idx:06d}_{int(query_item.timestamp)}')
    if os.path.exists(query_out_dir) and os.listdir(query_out_dir) and not args.overwrite:
        raise FileExistsError(f'{query_out_dir} already exists and is not empty. Use --overwrite to write into it.')
    os.makedirs(query_out_dir, exist_ok=True)

    pcim_loader = get_pointcloud_with_image_loader(args.dataset_type)
    adaptive_evaluator, adaptive_model = load_optional_adaptive_model(args, dataset_root, eval_set_path)

    query_pc, query_images = load_eval_frame(query_item, args.dataset_type, dataset_root, pcim_loader)
    query_sph = generate_sph_image(query_images, args.dataset_type, dataset_root)
    save_rgb(os.path.join(query_out_dir, 'query_panorama.png'), query_sph)
    annotated_images = [annotate_rgb(query_sph, f'QUERY idx={query_idx} ts={int(query_item.timestamp)}')]
    query_extra = {}
    if args.save_pointclouds:
        query_extra.update(save_pointcloud_views(
            os.path.join(query_out_dir, 'query_pointcloud'),
            query_pc,
            f'QUERY idx={query_idx}',
            max_points=args.max_pointcloud_points,
        ))
    query_adaptive = compute_adaptive_weights(adaptive_evaluator, adaptive_model, query_item)
    if query_adaptive is not None:
        query_extra['adaptive_weights'] = query_adaptive

    topk_summary = []
    for rank, map_idx in enumerate(top_indices, start=1):
        map_item = eval_set.map_set[int(map_idx)]
        map_pos = np.asarray(map_item.position, dtype=np.float64)
        geo_dist = float(np.linalg.norm(query_pos - map_pos))
        desc_dist = float(pair_dists[query_idx, map_idx])
        correct = bool(geo_dist <= args.revisit_threshold)

        map_pc, map_images = load_eval_frame(map_item, args.dataset_type, dataset_root, pcim_loader)
        map_sph = generate_sph_image(map_images, args.dataset_type, dataset_root)
        label = 'correct' if correct else 'wrong'
        filename = f'top{rank}_map_{int(map_idx):06d}_{label}.png'
        save_rgb(os.path.join(query_out_dir, filename), map_sph)
        map_extra = {}
        if args.save_pointclouds:
            map_extra.update(save_pointcloud_views(
                os.path.join(query_out_dir, f'top{rank}_map_{int(map_idx):06d}_pointcloud'),
                map_pc,
                f'TOP-{rank} map={int(map_idx)}',
                max_points=args.max_pointcloud_points,
            ))
        map_adaptive = compute_adaptive_weights(adaptive_evaluator, adaptive_model, map_item)
        if map_adaptive is not None:
            map_extra['adaptive_weights'] = map_adaptive

        title = (
            f'TOP-{rank} {label.upper()} map={int(map_idx)} '
            f'desc={desc_dist:.4f} geo={geo_dist:.2f}m'
        )
        annotated_images.append(annotate_rgb(map_sph, title, is_correct=correct))

        topk_summary.append({
            'rank': rank,
            'map_index': int(map_idx),
            'correct': correct,
            'descriptor_distance': desc_dist,
            'geometric_distance_m': geo_dist,
            'revisit_threshold_m': float(args.revisit_threshold),
            'map': item_to_dict(map_item),
            'panorama_file': filename,
            **map_extra,
        })

    overview = make_overview(annotated_images)
    save_rgb(os.path.join(query_out_dir, f'overview_top{topk}.png'), overview)

    summary = {
        'dataset_type': args.dataset_type,
        'dataset_root': dataset_root,
        'eval_set': eval_set_path,
        'result_dir': result_dir,
        'pair_dists': pair_dists_path,
        'pair_dists_recall_sanity': recall_summary,
        'query_index': int(query_idx),
        'query': {**item_to_dict(query_item), **query_extra},
        'topk': topk_summary,
    }
    with open(os.path.join(query_out_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f'Saved PR top-{topk} panorama visualization to: {query_out_dir}')
    for item in topk_summary:
        state = 'correct' if item['correct'] else 'wrong'
        print(
            f"top-{item['rank']}: map_index={item['map_index']} {state}, "
            f"desc_dist={item['descriptor_distance']:.4f}, "
            f"geo_dist={item['geometric_distance_m']:.2f}m"
        )


if __name__ == '__main__':
    main()
