# Training tuples generation for NCLT dataset.

import numpy as np
import argparse
import tqdm
import pickle
import os
import sys
import cv2
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from glnet.config.config import *
from glnet.datasets.nclt.nclt_raw import NCLTSequence, NCLTSequences, load_lidar_file_nclt, load_im_file_for_generate, pc2image_file, calculate_T
from glnet.datasets.base_datasets import TrainingTuple
from glnet.datasets.nclt.utils import relative_pose
from glnet.utils.common_utils import _ex
from glnet.utils.data_utils.point_clouds import visualize_2d_data, generate_bev, generate_bev_occ, icp, o3d_icp
from glnet.datasets.panorama import generate_sph_image
from glnet.utils.data_utils.poses import m2ypr

DEBUG = False
ICP_REFINE = False

PROTOCOL1_SEQUENCES = ['2012-02-04', '2012-03-17']
PROTOCOL1_POS_THRESHOLD = 25.0
PROTOCOL1_NEG_THRESHOLD = 50.0
PROTOCOL1_SAMPLING_DISTANCE = 0.2

bounds = (nclt_pc_bev_conf['x_bound'][0], nclt_pc_bev_conf['x_bound'][1], nclt_pc_bev_conf['y_bound'][0], \
          nclt_pc_bev_conf['y_bound'][1], nclt_pc_bev_conf['z_bound'][0], nclt_pc_bev_conf['z_bound'][1])
bev_x = nclt_pc_bev_conf['x_grid']
bev_y = nclt_pc_bev_conf['y_grid']
bev_z = nclt_pc_bev_conf['z_grid']

def generate_training_tuples(ds: NCLTSequences, pos_threshold: float = 25, neg_threshold: float = 50, bev: bool = False, sph: bool = False):
    # displacement: displacement between consecutive anchors (if None all scans are takes as anchors).
    #               Use some small displacement to ensure there's only one scan if the vehicle does not move

    tuples = {}   # Dictionary of training tuples: tuples[ndx] = (sef ot positives, set of non negatives)
    for anchor_ndx in tqdm.tqdm(range(len(ds))):
        if bev:
            reading_filepath = os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx])
            bev_filename = reading_filepath.replace('bin', 'npy')
            bev_filename = bev_filename.replace('velodyne_sync', 'bev')
            if os.path.exists(bev_filename):
                pass
            else: 
                pc = load_lidar_file_nclt(reading_filepath).astype(np.float32)
                pc_bev = generate_bev(pc, Z=bev_z, Y=bev_y, X=bev_x, bounds=bounds).numpy()
                print(f'Generating {bev_filename}')
                np.save(bev_filename, pc_bev)
                # for i in range(20):
                #     visualize_2d_data(pc_bev[i], f'pc_bev_{i}.jpg')
                #     visualize_2d_data(pc_bev_2[i], f'pc_bev_2_{i}.jpg')
                                         
        if sph:
            reading_filepath = os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx])
            sph_filename = reading_filepath.replace('bin', 'jpg')
            sph_filename = sph_filename.replace('velodyne_sync', 'sph')
            # if os.path.exists(sph_filename):
            #     pass
            # else:
            images = [load_im_file_for_generate(pc2image_file(reading_filepath, 'velodyne_sync/', i, '.bin'), False) for i in range(1, 6)]                
            sph_img = generate_sph_image(images, 'nclt', ds.dataset_root)
            print(f'Generating {sph_filename}')
            cv2.imwrite(sph_filename, sph_img)
            query_yaw, pitch, roll = m2ypr(ds.poses[anchor_ndx])

        anchor_pos = ds.get_xy()[anchor_ndx]

        # Find timestamps of positive and negative elements
        positives = ds.find_neighbours_ndx(anchor_pos, pos_threshold)
        non_negatives = ds.find_neighbours_ndx(anchor_pos, neg_threshold)
        # Remove anchor element from positives, but leave it in non_negatives
        positives = positives[positives != anchor_ndx]

        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)
        
        # ICP pose refinement
        fitness_l = []
        inlier_rmse_l = []
        positive_poses = {}
        
        if DEBUG:
            # Use ground truth transform without pose refinement
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:
                positive_pose = ds.poses[positive_ndx]
                # Compute initial relative pose
                m, fitness, inlier_rmse = relative_pose(anchor_pose, positive_pose), 1., 1.
                fitness_l.append(fitness)
                inlier_rmse_l.append(inlier_rmse)
                positive_poses[positive_ndx] = m
        else:
            anchor_pc = load_lidar_file_nclt(os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx])).astype(np.float32)
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:
                positive_pose = ds.poses[positive_ndx]
                transform = relative_pose(anchor_pose, positive_pose)
                if ICP_REFINE:
                    positive_pc = load_lidar_file_nclt(os.path.join(ds.dataset_root, ds.rel_scan_filepath[positive_ndx])).astype(np.float32)
                    # Compute initial relative pose
                    # Refine the pose using ICP
                    m, fitness, inlier_rmse = icp(anchor_pc[:, :3], positive_pc[:, :3], transform)

                    fitness_l.append(fitness)
                    inlier_rmse_l.append(inlier_rmse)
                    positive_poses[positive_ndx] = m
                positive_poses[positive_ndx] = transform

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        tuples[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=ds.timestamps[anchor_ndx],
                                           rel_scan_filepath=ds.rel_scan_filepath[anchor_ndx],
                                           positives=positives, non_negatives=non_negatives, pose=anchor_pose,
                                           positives_poses=positive_poses)

    print(f'{len(tuples)} training tuples generated')
    if ICP_REFINE:
        print('ICP pose refimenement stats:')
        print(f'Fitness - min: {np.min(fitness_l):0.3f}   mean: {np.mean(fitness_l):0.3f}   max: {np.max(fitness_l):0.3f}')
        print(f'Inlier RMSE - min: {np.min(inlier_rmse_l):0.3f}   mean: {np.mean(inlier_rmse_l):0.3f}   max: {np.max(inlier_rmse_l):0.3f}')

    return tuples


def generate_image_meta_pickle(dataset_root: str):
    cam_params_path = dataset_root + '/cam_params/'
    K = []
    T = []    
    factor_x = 224. / 600.
    factor_y = 384. / 900.    
    for cam_num in range(1, 6):
        K_matrix = np.loadtxt(cam_params_path + 'K_cam%d.csv' % (cam_num), delimiter=',')
        fx = K_matrix[0][0]
        fy = K_matrix[1][1]
        cx = K_matrix[0][2]
        cy = K_matrix[1][2]

        cy = 1232. - cy 
        cx -= 400.  # cx
        cy -= 182.  # cy
        cx = cx * factor_x
        cy = cy * factor_y
        
        K_matrix[0][0] = fy * factor_y
        K_matrix[0][2] = cy
        K_matrix[1][1] = fx * factor_x
        K_matrix[1][2] = cx
        K.append(K_matrix)
        
        T_matrix = np.loadtxt(cam_params_path + 'x_lb3_c%d.csv' % (cam_num), delimiter=',')
        T_matrix = calculate_T(T_matrix)
        T.append(T_matrix)
    
    image_meta = {'K': K, 'T': T}
    with open(os.path.join(dataset_root, 'image_meta.pkl'), 'wb') as handle:
        pickle.dump(image_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _sequence_name(sequences):
    return ''.join(['_' + seq for seq in sequences])


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


def _generate_one_split(dataset_root, sequences, split, output_prefix, pos_threshold, neg_threshold, sampling_distance,
                        bev=False, sph=False):
    print('-------- Generating NCLT training tuples --------')
    print(f'Dataset root: {dataset_root}')
    print(f'Sequences: {sequences}')
    print(f'Split: {split}')
    print(f'Positive threshold: {pos_threshold}')
    print(f'Negative threshold: {neg_threshold}')
    print(f'Sampling distance: {sampling_distance}')
    print(f'Output prefix: {output_prefix}')

    try:
        ds = NCLTSequences(dataset_root, sequences, split=split, sampling_distance=sampling_distance)
    except AssertionError as exc:
        raise SystemExit(f'Failed to create NCLT split "{split}": {exc}')

    tuples = generate_training_tuples(ds, pos_threshold, neg_threshold, bev=bev, sph=sph)
    pickle_name = f'{output_prefix}{_sequence_name(sequences)}_{pos_threshold}_{neg_threshold}_{sampling_distance}.pickle'
    tuples_filepath = os.path.join(dataset_root, pickle_name)
    pickle.dump(tuples, open(tuples_filepath, 'wb'))

    print(f'Number of generated tuples: {len(tuples)}')
    print(f'Output pickle path: {tuples_filepath}')
    return tuples_filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate NCLT Protocol 1 train and validation tuples')
    parser.add_argument('--dataset_root', default='Data/NCLT')
    parser.add_argument('--bev', action='store_true', help='Generate bevs projected by point clouds')
    parser.add_argument('--sph', action='store_true', help='Generate panorama images')
    args = parser.parse_args()

    sequences = PROTOCOL1_SEQUENCES
    dataset_root = _ex(args.dataset_root)

    print(f'Dataset root: {dataset_root}')
    print(f'Sequences: {sequences}')
    print('Protocol: NCLT Protocol 1')
    print(f'Positive threshold: {PROTOCOL1_POS_THRESHOLD}')
    print(f'Negative threshold: {PROTOCOL1_NEG_THRESHOLD}')
    print(f'Sampling distance: {PROTOCOL1_SAMPLING_DISTANCE}')

    try:
        _validate_nclt_inputs(dataset_root, sequences)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    # generate image meta pickle
    if not os.path.exists(os.path.join(dataset_root, 'image_meta.pkl')) and os.path.isdir(os.path.join(dataset_root, 'cam_params')):
        generate_image_meta_pickle(dataset_root)

    generated_paths = [
        _generate_one_split(dataset_root, sequences, split='train', output_prefix='train',
                            pos_threshold=PROTOCOL1_POS_THRESHOLD, neg_threshold=PROTOCOL1_NEG_THRESHOLD,
                            sampling_distance=PROTOCOL1_SAMPLING_DISTANCE, bev=args.bev, sph=args.sph),
        _generate_one_split(dataset_root, sequences, split='test', output_prefix='val',
                            pos_threshold=PROTOCOL1_POS_THRESHOLD, neg_threshold=PROTOCOL1_NEG_THRESHOLD,
                            sampling_distance=PROTOCOL1_SAMPLING_DISTANCE, bev=args.bev, sph=args.sph),
    ]
    print('-------- Done --------')
    for path in generated_paths:
        print(f'Generated: {path}')
