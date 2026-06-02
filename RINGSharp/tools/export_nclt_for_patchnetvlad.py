import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from glnet.datasets.base_datasets import EvaluationSet
from glnet.datasets.nclt.nclt_raw import load_im_file_for_generate, pc2image_file
from glnet.datasets.panorama import generate_sph_image


def _safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _sph_path(dataset_root, rel_scan_filepath):
    return Path(dataset_root) / rel_scan_filepath.replace('velodyne_sync', 'sph').replace('.bin', '.jpg')


def _generate_sph_if_needed(dataset_root, rel_scan_filepath, overwrite=False):
    scan_path = Path(dataset_root) / rel_scan_filepath
    sph_path = _sph_path(dataset_root, rel_scan_filepath)
    if sph_path.exists() and not overwrite:
        return sph_path

    _safe_mkdir(sph_path.parent)
    images = [
        load_im_file_for_generate(pc2image_file(str(scan_path), '/velodyne_sync/', cam_num, '.bin'), False)
        for cam_num in range(1, 6)
    ]
    sph_img = generate_sph_image(images, 'nclt', dataset_root)
    cv2.imwrite(str(sph_path), sph_img)
    return sph_path


def _write_image_list(path, items, dataset_root, generate_sph=False, overwrite=False):
    lines = []
    for item in items:
        sph_path = _generate_sph_if_needed(dataset_root, item.rel_scan_filepath, overwrite) if generate_sph else _sph_path(dataset_root, item.rel_scan_filepath)
        if not sph_path.exists():
            raise FileNotFoundError(
                f'Missing panorama image: {sph_path}\n'
                'Run this script with --generate_sph, or run generate_evaluation_sets.py --sph first.'
            )
        lines.append(str(sph_path.resolve()))

    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(f'Wrote {path} ({len(lines)} images)')


def main():
    parser = argparse.ArgumentParser(description='Export RINGSharp NCLT EvaluationSet to Patch-NetVLAD file lists and GT')
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--eval_set', required=True, help='RINGSharp EvaluationSet pickle filename')
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--positive_radius', type=float, default=5.0)
    parser.add_argument('--generate_sph', action='store_true', help='Generate missing NCLT panorama jpg files')
    parser.add_argument('--overwrite_sph', action='store_true')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    _safe_mkdir(out_root)

    eval_set = EvaluationSet()
    eval_set.load(str(dataset_root / args.eval_set))

    index_file = out_root / 'nclt_imageNames_index.txt'
    query_file = out_root / 'nclt_imageNames_query.txt'
    gt_file = out_root / f'nclt_gt_radius{args.positive_radius:g}.npz'

    _write_image_list(index_file, eval_set.map_set, dataset_root, args.generate_sph, args.overwrite_sph)
    _write_image_list(query_file, eval_set.query_set, dataset_root, args.generate_sph, args.overwrite_sph)

    np.savez(
        gt_file,
        utmQ=eval_set.get_query_positions().astype(np.float32),
        utmDb=eval_set.get_map_positions().astype(np.float32),
        posDistThr=np.asarray(args.positive_radius, dtype=np.float32),
    )
    query_positions = eval_set.get_query_positions()
    map_positions = eval_set.get_map_positions()
    distances = np.linalg.norm(query_positions[:, None, :] - map_positions[None, :, :], axis=2)
    valid_queries = int(np.sum(np.any(distances <= args.positive_radius, axis=1)))
    print(f'Wrote {gt_file}')
    print(f'Map images: {len(eval_set.map_set)}  Query images: {len(eval_set.query_set)}  Radius: {args.positive_radius} m')
    print(f'Queries with at least one positive: {valid_queries}/{len(eval_set.query_set)}')


if __name__ == '__main__':
    main()
