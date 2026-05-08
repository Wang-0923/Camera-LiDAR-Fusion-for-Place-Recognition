# Test sets for NCLT dataset.

import argparse
from typing import List
import os
import sys
import copy
import tqdm
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from glnet.config.config import *
from glnet.datasets.nclt.nclt_raw import NCLTSequence, load_lidar_file_nclt, load_im_file_for_generate, pc2image_file
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet
from glnet.datasets.dataset_utils import filter_query_elements
from glnet.datasets.panorama import generate_sph_image
from glnet.utils.data_utils.point_clouds import visualize_2d_data, generate_bev
from glnet.utils.data_utils.lidar_reliability import compute_lidar_reliability_bev
import cv2
from glnet.utils.common_utils import _ex

DEBUG = False

PROTOCOL1_MAP_SEQUENCE = '2012-02-04'
PROTOCOL1_QUERY_SEQUENCE = '2012-03-17'
PROTOCOL1_SPLIT = 'test'
PROTOCOL1_MAP_SAMPLING_DISTANCE = 20.0
PROTOCOL1_QUERY_SAMPLING_DISTANCE = 5.0
PROTOCOL1_DIST_THRESHOLD = 50.0

bounds = (nclt_pc_bev_conf['x_bound'][0], nclt_pc_bev_conf['x_bound'][1], nclt_pc_bev_conf['y_bound'][0], \
          nclt_pc_bev_conf['y_bound'][1], nclt_pc_bev_conf['z_bound'][0], nclt_pc_bev_conf['z_bound'][1])
bev_x = nclt_pc_bev_conf['x_grid']
bev_y = nclt_pc_bev_conf['y_grid']
bev_z = nclt_pc_bev_conf['z_grid']

def get_scans(sequence: NCLTSequence) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    for ndx in range(len(sequence)):
        pose = sequence.poses[ndx]
        position = pose[:2, 3]
        item = EvaluationTuple(sequence.timestamps[ndx], sequence.rel_scan_filepath[ndx], position=position, pose=pose)
        elems.append(item)
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence: str, query_sequence: str, split: str = 'test', bev: bool = False, sph: bool = False,
                            map_sampling_distance: float = 0.2, query_sampling_distance: float = 0.2, dist_threshold = 25,
                            lidar_reliability: bool = False, overwrite_lidar_reliability: bool = False,
                            lidar_reliability_downsample: float = 0.3, lidar_reliability_k: int = 16,
                            lidar_reliability_min_neighbors: int = 3) -> EvaluationSet:
    query_bev_folder = os.path.join(dataset_root, query_sequence, 'bev')
    map_bev_folder = os.path.join(dataset_root, map_sequence, 'bev')      
    query_lidar_rel_folder = os.path.join(dataset_root, query_sequence, 'lidar_reliability_bev')
    map_lidar_rel_folder = os.path.join(dataset_root, map_sequence, 'lidar_reliability_bev')
    query_sph_folder = os.path.join(dataset_root, query_sequence, 'sph')
    map_sph_folder = os.path.join(dataset_root, map_sequence, 'sph')     
    map_sequence = NCLTSequence(dataset_root, map_sequence, split=split, sampling_distance=map_sampling_distance)
    query_sequence = NCLTSequence(dataset_root, query_sequence, split=split, sampling_distance=query_sampling_distance)

    map_set = get_scans(map_sequence)
    query_set = get_scans(query_sequence)
    query_count_before_filtering = len(query_set)
    
    if bev or lidar_reliability:
        os.makedirs(query_bev_folder, exist_ok=True)
        os.makedirs(map_bev_folder, exist_ok=True)
        if lidar_reliability:
            os.makedirs(query_lidar_rel_folder, exist_ok=True)
            os.makedirs(map_lidar_rel_folder, exist_ok=True)
        for i in tqdm.tqdm(range(len(query_set))):
            reading_filepath = query_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            pc = None
            if bev:
                bev_filename = reading_filepath.replace('bin', 'npy')
                bev_filename = bev_filename.replace('velodyne_sync', 'bev')
                pc = load_lidar_file_nclt(reading_filepath).astype(np.float32) if pc is None else pc
                pc_bev = generate_bev(pc, Z=bev_z, Y=bev_y, X=bev_x, bounds=bounds).numpy()
                print(f'Generating {bev_filename}')
                np.save(bev_filename, pc_bev)
            if lidar_reliability:
                reliability_filename = reading_filepath.replace('bin', 'npy')
                reliability_filename = reliability_filename.replace('velodyne_sync', 'lidar_reliability_bev')
                if os.path.exists(reliability_filename) and not overwrite_lidar_reliability:
                    pass
                else:
                    pc = load_lidar_file_nclt(reading_filepath).astype(np.float32) if pc is None else pc
                    rel_bev = compute_lidar_reliability_bev(
                        pc,
                        Z=bev_z,
                        Y=bev_y,
                        X=bev_x,
                        bounds=bounds,
                        downsample_voxel_size=lidar_reliability_downsample,
                        k=lidar_reliability_k,
                        min_neighbors=lidar_reliability_min_neighbors,
                    )
                    print(f'Generating {reliability_filename}')
                    np.save(reliability_filename, rel_bev.astype(np.float32))

        for i in tqdm.tqdm(range(len(map_set))):
            reading_filepath = map_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            pc = None
            if bev:
                bev_filename = reading_filepath.replace('bin', 'npy')
                bev_filename = bev_filename.replace('velodyne_sync', 'bev')
                pc = load_lidar_file_nclt(reading_filepath).astype(np.float32) if pc is None else pc
                pc_bev = generate_bev(pc, Z=bev_z, Y=bev_y, X=bev_x, bounds=bounds).numpy()
                print(f'Generating {bev_filename}')
                np.save(bev_filename, pc_bev)
            if lidar_reliability:
                reliability_filename = reading_filepath.replace('bin', 'npy')
                reliability_filename = reliability_filename.replace('velodyne_sync', 'lidar_reliability_bev')
                if os.path.exists(reliability_filename) and not overwrite_lidar_reliability:
                    pass
                else:
                    pc = load_lidar_file_nclt(reading_filepath).astype(np.float32) if pc is None else pc
                    rel_bev = compute_lidar_reliability_bev(
                        pc,
                        Z=bev_z,
                        Y=bev_y,
                        X=bev_x,
                        bounds=bounds,
                        downsample_voxel_size=lidar_reliability_downsample,
                        k=lidar_reliability_k,
                        min_neighbors=lidar_reliability_min_neighbors,
                    )
                    print(f'Generating {reliability_filename}')
                    np.save(reliability_filename, rel_bev.astype(np.float32))
                                
    if sph:
        os.makedirs(query_sph_folder, exist_ok=True)
        os.makedirs(map_sph_folder, exist_ok=True)            
        for i in tqdm.tqdm(range(len(query_set))):
            reading_filepath = query_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            sph_filename = reading_filepath.replace('bin', 'jpg')
            sph_filename = sph_filename.replace('velodyne_sync', 'sph')            
            # if os.path.exists(sph_filename):
            #     pass
            # else:            
            images = [load_im_file_for_generate(pc2image_file(reading_filepath, '/velodyne_sync/', i, '.bin'), False) for i in range(1, 6)]
            sph_img = generate_sph_image(images, 'nclt', dataset_root)
            print(f'Generating {sph_filename}')
            cv2.imwrite(sph_filename, sph_img)

        for i in tqdm.tqdm(range(len(map_set))):
            reading_filepath = map_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            sph_filename = reading_filepath.replace('bin', 'jpg')
            sph_filename = sph_filename.replace('velodyne_sync', 'sph')
            # if os.path.exists(sph_filename):
            #     pass
            # else:            
            images = [load_im_file_for_generate(pc2image_file(reading_filepath, '/velodyne_sync/', i, '.bin'), False) for i in range(1, 6)]
            sph_img = generate_sph_image(images, 'nclt', dataset_root)
            print(f'Generating {sph_filename}')
            cv2.imwrite(sph_filename, sph_img)

    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)

    print(f'Number of map/database elements: {len(map_set)}')
    print(f'Number of query elements before filtering: {query_count_before_filtering}')
    print(f'Number of query elements after filtering: {len(query_set)}')
    return EvaluationSet(query_set, map_set)


def _validate_nclt_inputs(dataset_root, sequences):
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f'Cannot access dataset root: {dataset_root}')
    missing = []
    for seq in sequences:
        seq_dir = os.path.join(dataset_root, seq)
        pose_candidates = [
            os.path.join(seq_dir, 'groundtruth_' + seq + '.csv'),
            os.path.join(seq_dir, 'ground_truth', 'groundtruth_' + seq + '.csv'),
        ]
        if not os.path.isdir(seq_dir):
            missing.append(f'session directory: {seq_dir}')
        if not os.path.isdir(os.path.join(seq_dir, 'velodyne_sync')):
            missing.append(f'lidar directory: {os.path.join(seq_dir, "velodyne_sync")}')
        if not any(os.path.exists(path) for path in pose_candidates):
            missing.append('ground truth file: ' + ' or '.join(pose_candidates))
    if missing:
        raise FileNotFoundError('Missing NCLT inputs:\n  ' + '\n  '.join(missing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the NCLT Protocol 1 final PR evaluation set')
    parser.add_argument('--dataset_root', type=str, default='Data/NCLT')
    parser.add_argument('--bev', action='store_true', help='Generate bevs projected by point clouds')
    parser.add_argument('--sph', action='store_true', help='Generate panorama images')
    parser.add_argument('--lidar_reliability', action='store_true',
                        help='Generate offline LiDAR BEV reliability maps')
    parser.add_argument('--overwrite_lidar_reliability', action='store_true',
                        help='Overwrite existing LiDAR reliability cache files')
    parser.add_argument('--lidar_reliability_downsample', type=float, default=0.3,
                        help='Voxel downsample size before LiDAR PCA reliability computation')
    parser.add_argument('--lidar_reliability_k', type=int, default=16,
                        help='kNN size for LiDAR PCA reliability computation')
    parser.add_argument('--lidar_reliability_min_neighbors', type=int, default=3,
                        help='Minimum neighbors for LiDAR PCA reliability')
    args = parser.parse_args()

    map_sequence = PROTOCOL1_MAP_SEQUENCE
    query_sequence = PROTOCOL1_QUERY_SEQUENCE
    split = PROTOCOL1_SPLIT
    map_sampling_distance = PROTOCOL1_MAP_SAMPLING_DISTANCE
    query_sampling_distance = PROTOCOL1_QUERY_SAMPLING_DISTANCE
    dist_threshold = PROTOCOL1_DIST_THRESHOLD
    
    dataset_root = _ex(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print('Protocol: NCLT Protocol 1')
    print(f'Map sequence: {map_sequence}')
    print(f'Query sequence: {query_sequence}')
    print(f'Split: {split}')
    print(f'Map sampling distance: {map_sampling_distance}')
    print(f'Query sampling distance: {query_sampling_distance}')
    print(f'Distance threshold for query filtering: {dist_threshold}')

    try:
        _validate_nclt_inputs(dataset_root, [map_sequence, query_sequence])
        test_set = generate_evaluation_set(dataset_root, map_sequence, query_sequence, split=split, bev=args.bev, sph=args.sph,
                map_sampling_distance=map_sampling_distance, query_sampling_distance=query_sampling_distance, dist_threshold=dist_threshold,
                lidar_reliability=args.lidar_reliability,
                overwrite_lidar_reliability=args.overwrite_lidar_reliability,
                lidar_reliability_downsample=args.lidar_reliability_downsample,
                lidar_reliability_k=args.lidar_reliability_k,
                lidar_reliability_min_neighbors=args.lidar_reliability_min_neighbors)
    except (AssertionError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))

    pickle_name = f'{split}_{map_sequence}_{query_sequence}_{map_sampling_distance}_{query_sampling_distance}.pickle'
    file_path_name = os.path.join(dataset_root, pickle_name)
    test_set.save(file_path_name)
    print(f'Output pickle path: {file_path_name}')
