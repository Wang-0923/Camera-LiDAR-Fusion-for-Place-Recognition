import argparse
import json
import os
import sys
from typing import Dict, List

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
    parser.add_argument('--revisit_threshold', type=float, default=5.0)
    parser.add_argument('--output_dir', required=True)
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


def load_eval_images(
    item,
    dataset_type: str,
    dataset_root: str,
    pcim_loader,
) -> List[np.ndarray]:
    if dataset_type == 'nclt':
        scan_path = os.path.join(dataset_root, item.rel_scan_filepath)
        if not os.path.exists(scan_path):
            raise FileNotFoundError(f'Cannot access NCLT scan: {scan_path}')
        _, images = pcim_loader(scan_path, sph=False)
        return images

    if dataset_type == 'oxford':
        if item.filepaths is None:
            raise ValueError('Oxford evaluation tuple is missing filepaths')
        extrinsics_dir = os.path.join(dataset_root, 'extrinsics')
        _, images = pcim_loader(item.filepaths, sph=False, extrinsics_dir=extrinsics_dir)
        return images

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


def item_to_dict(item) -> Dict[str, object]:
    return {
        'timestamp': int(item.timestamp),
        'rel_scan_filepath': item.rel_scan_filepath,
        'position': np.asarray(item.position).astype(float).tolist(),
    }


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

    query_idx = find_query_index(eval_set, args.query_idx, args.query_timestamp)
    topk = min(int(args.topk), len(eval_set.map_set))
    top_indices = np.argsort(pair_dists[query_idx])[:topk]

    query_item = eval_set.query_set[query_idx]
    query_pos = np.asarray(query_item.position, dtype=np.float64)
    query_out_dir = os.path.join(output_dir, f'query_{query_idx:06d}_{int(query_item.timestamp)}')
    if os.path.exists(query_out_dir) and os.listdir(query_out_dir) and not args.overwrite:
        raise FileExistsError(f'{query_out_dir} already exists and is not empty. Use --overwrite to write into it.')
    os.makedirs(query_out_dir, exist_ok=True)

    pcim_loader = get_pointcloud_with_image_loader(args.dataset_type)

    query_images = load_eval_images(query_item, args.dataset_type, dataset_root, pcim_loader)
    query_sph = generate_sph_image(query_images, args.dataset_type, dataset_root)
    save_rgb(os.path.join(query_out_dir, 'query_panorama.png'), query_sph)
    annotated_images = [annotate_rgb(query_sph, f'QUERY idx={query_idx} ts={int(query_item.timestamp)}')]

    topk_summary = []
    for rank, map_idx in enumerate(top_indices, start=1):
        map_item = eval_set.map_set[int(map_idx)]
        map_pos = np.asarray(map_item.position, dtype=np.float64)
        geo_dist = float(np.linalg.norm(query_pos - map_pos))
        desc_dist = float(pair_dists[query_idx, map_idx])
        correct = bool(geo_dist <= args.revisit_threshold)

        map_images = load_eval_images(map_item, args.dataset_type, dataset_root, pcim_loader)
        map_sph = generate_sph_image(map_images, args.dataset_type, dataset_root)
        label = 'correct' if correct else 'wrong'
        filename = f'top{rank}_map_{int(map_idx):06d}_{label}.png'
        save_rgb(os.path.join(query_out_dir, filename), map_sph)

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
        })

    overview = make_overview(annotated_images)
    save_rgb(os.path.join(query_out_dir, f'overview_top{topk}.png'), overview)

    summary = {
        'dataset_type': args.dataset_type,
        'dataset_root': dataset_root,
        'eval_set': eval_set_path,
        'result_dir': result_dir,
        'pair_dists': pair_dists_path,
        'query_index': int(query_idx),
        'query': item_to_dict(query_item),
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
