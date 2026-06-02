import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.neighbors import KDTree
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from glnet.datasets.base_datasets import EvaluationSet
from glnet.datasets.nclt.nclt_raw import load_lidar_file_nclt, load_im_file_for_generate, pc2image_file


def _safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _tuple_items(training_tuple_dict):
    return [training_tuple_dict[k] for k in sorted(training_tuple_dict.keys())]


def _frame_key(rel_scan_filepath):
    seq = Path(rel_scan_filepath).parts[0]
    timestamp = Path(rel_scan_filepath).stem
    return f'{seq}/{timestamp}'


def _write_frame(dataset_root, out_root, rel_scan_filepath, cam_num=1, overwrite=False):
    key = _frame_key(rel_scan_filepath)
    seq, timestamp = key.split('/')
    image_out = Path(out_root) / seq / 'images' / f'{timestamp}.jpg'
    pc_out = Path(out_root) / seq / 'pointclouds' / f'{timestamp}.bin'
    _safe_mkdir(image_out.parent)
    _safe_mkdir(pc_out.parent)

    scan_path = Path(dataset_root) / rel_scan_filepath
    if overwrite or not image_out.exists():
        image_path = pc2image_file(str(scan_path), '/velodyne_sync/', cam_num, '.bin')
        image = load_im_file_for_generate(image_path, sph=False)
        cv2.imwrite(str(image_out), image)

    if overwrite or not pc_out.exists():
        pc = load_lidar_file_nclt(str(scan_path)).astype(np.float32)
        pc = pc[:, :3]
        pc = pc[pc[:, 2] < -0.5]
        # RINGSharp's NCLT point clouds use negative z for useful LiDAR returns,
        # while AdaFusion's NCLT voxelizer keeps z in [0, 5].
        pc[:, 2] = -pc[:, 2]
        pc = pc[(pc[:, 2] >= 0.0) & (pc[:, 2] <= 5.0)]
        pc.tofile(str(pc_out))

    return key


def export_training(
    dataset_root,
    out_root,
    train_pickle,
    overwrite=False,
    cam_num=1,
    max_pos_per_anchor=None,
    max_neg_per_anchor=None,
    max_train_pairs=None,
    seed=17,
):
    with open(Path(dataset_root) / train_pickle, 'rb') as handle:
        tuples = pickle.load(handle)
    items = _tuple_items(tuples)

    file_indices = []
    id_to_export_index = {}
    for export_index, item in enumerate(items):
        key = _write_frame(dataset_root, out_root, item.rel_scan_filepath, cam_num=cam_num, overwrite=overwrite)
        file_indices.append(key)
        id_to_export_index[item.id] = export_index

    rng = np.random.default_rng(seed)
    pos_pairs = []
    neg_pairs = []
    all_ids = np.array([item.id for item in items], dtype=np.int64)
    for item in items:
        anchor = id_to_export_index[item.id]
        positive_ids = [int(pos_id) for pos_id in item.positives if int(pos_id) in id_to_export_index]
        if max_pos_per_anchor is not None and len(positive_ids) > max_pos_per_anchor:
            positive_ids = sorted(rng.choice(positive_ids, size=max_pos_per_anchor, replace=False).tolist())
        for pos_id in positive_ids:
            if int(pos_id) in id_to_export_index:
                pos_pairs.append((anchor, id_to_export_index[int(pos_id)]))
        negative_ids = all_ids[~np.isin(all_ids, item.non_negatives)]
        if max_neg_per_anchor is not None and len(negative_ids) > max_neg_per_anchor:
            negative_ids = rng.choice(negative_ids, size=max_neg_per_anchor, replace=False)
        for neg_id in negative_ids:
            if int(neg_id) in id_to_export_index:
                neg_pairs.append((anchor, id_to_export_index[int(neg_id)]))

    if max_train_pairs is not None and len(pos_pairs) > max_train_pairs:
        keep = rng.choice(len(pos_pairs), size=max_train_pairs, replace=False)
        pos_pairs = [pos_pairs[i] for i in sorted(keep.tolist())]

    if not pos_pairs:
        raise RuntimeError('No positive pairs were exported for AdaFusion training')
    if not neg_pairs:
        raise RuntimeError('No negative pairs were exported for AdaFusion training')

    pairs = {
        'file_indices': file_indices,
        'pos_pairs': np.asarray(pos_pairs, dtype=np.int64),
        'neg_pairs': np.asarray(neg_pairs, dtype=np.int64),
    }
    out_path = Path(out_root) / 'train_pairs_ringsharp.pickle'
    with open(out_path, 'wb') as handle:
        pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {out_path}')
    print(f'Frames: {len(file_indices)}  Pos pairs: {len(pos_pairs)}  Neg pairs: {len(neg_pairs)}')


def export_eval(dataset_root, out_root, eval_pickle, overwrite=False, cam_num=1, positive_radius=20.0, output_name=None):
    eval_set = EvaluationSet()
    eval_set.load(str(Path(dataset_root) / eval_pickle))

    file_indices = []
    for item in eval_set.map_set:
        key = _write_frame(dataset_root, out_root, item.rel_scan_filepath, cam_num=cam_num, overwrite=overwrite)
        file_indices.append(key)
    for item in eval_set.query_set:
        key = _write_frame(dataset_root, out_root, item.rel_scan_filepath, cam_num=cam_num, overwrite=overwrite)
        file_indices.append(key)

    map_positions = eval_set.get_map_positions()
    query_positions = eval_set.get_query_positions()
    tree = KDTree(map_positions)
    pos_items = []
    for query_pos in query_positions:
        positives = tree.query_radius(query_pos.reshape(1, -1), positive_radius)[0]
        pos_items.append(positives.astype(np.int64))

    test_items = {
        'file_indices': [file_indices[:len(eval_set.map_set)], file_indices[len(eval_set.map_set):]],
        'pos_items': {(0, 1): pos_items},
    }
    if output_name is None:
        output_name = 'test_items_ringsharp.pickle'
    out_path = Path(out_root) / output_name
    with open(out_path, 'wb') as handle:
        pickle.dump(test_items, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {out_path}')
    print(f'Map frames: {len(eval_set.map_set)}  Query frames: {len(eval_set.query_set)}')


def write_config(
    out_root,
    adafusion_root,
    template=None,
    test_pickle='test_items_ringsharp.pickle',
    epochs=None,
    eval_epoch_freq=2,
    ckpt_name='adafusion_baseline.pth',
    train_batch_size=None,
    test_batch_size=None,
    workers=None,
    image_size=None,
    voxel_shape=None,
    augmentation=True,
    loss_a=None,
    loss_m=None,
    loss_lp=None,
):
    if template is None:
        template = Path(adafusion_root) / 'experiments' / 'aavislidar3_pairwise' / 'config.yaml'
    with open(template, 'r') as handle:
        config = yaml.safe_load(handle)
    config['data_path'] = str(Path(out_root).parent)
    config['ckpt_name'] = ckpt_name
    config['dataset']['name'] = 'nclt'
    config['dataset']['base_path'] = str(out_root)
    config['dataset']['train_pickle'] = 'train_pairs_ringsharp.pickle'
    config['dataset']['test_pickle'] = test_pickle
    config['dataset']['image_size'] = image_size if image_size is not None else [300, 400]
    config['dataset']['voxel_shape'] = voxel_shape if voxel_shape is not None else [72, 72, 48]
    config['dataset']['augmentation'] = bool(augmentation)
    config['eval_epoch_freq'] = int(eval_epoch_freq)
    if train_batch_size is not None:
        config['train_batch_size'] = int(train_batch_size)
    if test_batch_size is not None:
        config['test_batch_size'] = int(test_batch_size)
    if workers is not None:
        config['workers'] = int(workers)
    if epochs is not None:
        config['epochs'] = int(epochs)
    else:
        config['epochs'] = min(int(config.get('epochs', 50)), 50)
    if loss_a is not None:
        config['loss']['a'] = float(loss_a)
    if loss_m is not None:
        config['loss']['m'] = float(loss_m)
    if loss_lp is not None:
        config['loss']['Lp'] = int(loss_lp)
    work_path = Path(adafusion_root) / 'experiments' / 'nclt_ringsharp_pairwise'
    _safe_mkdir(work_path)
    config_path = work_path / 'config.yaml'
    with open(config_path, 'w') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f'Wrote {config_path}')


def main():
    parser = argparse.ArgumentParser(description='Export RINGSharp NCLT split to AdaFusion data format')
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--adafusion_root', default='/autodl-fs/ringsharp/AdaFusion')
    parser.add_argument('--train_pickle', default='train_2012-02-04_2012-03-17_5.0_7.5_0.2.pickle')
    parser.add_argument(
        '--eval_pickle',
        nargs='+',
        default=['test_2012-02-04_2012-03-17_20.0_5.0.pickle'],
        help='One or more RINGSharp EvaluationSet pickle files. The first one is used in the generated AdaFusion config.',
    )
    parser.add_argument('--cam_num', type=int, default=1, help='NCLT camera number, 1 means Cam1')
    parser.add_argument('--positive_radius', type=float, default=20.0)
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs in generated AdaFusion config')
    parser.add_argument('--eval_epoch_freq', type=int, default=2, help='Run AdaFusion validation every N epochs')
    parser.add_argument('--ckpt_name', default='adafusion_baseline.pth', help='Checkpoint name written to AdaFusion config')
    parser.add_argument('--max_pos_per_anchor', type=int, default=None, help='Limit exported positive training pairs per anchor')
    parser.add_argument('--max_neg_per_anchor', type=int, default=None, help='Limit exported negative training pairs per anchor')
    parser.add_argument('--max_train_pairs', type=int, default=None, help='Limit total exported positive training pairs')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--image_size', type=int, nargs=2, default=None, metavar=('H', 'W'))
    parser.add_argument('--voxel_shape', type=int, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--loss_a', type=float, default=None)
    parser.add_argument('--loss_m', type=float, default=None)
    parser.add_argument('--loss_lp', type=int, default=None)
    parser.add_argument('--disable_augmentation', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_eval', action='store_true')
    args = parser.parse_args()

    _safe_mkdir(args.out_root)
    if not args.skip_train:
        export_training(
            args.dataset_root,
            args.out_root,
            args.train_pickle,
            overwrite=args.overwrite,
            cam_num=args.cam_num,
            max_pos_per_anchor=args.max_pos_per_anchor,
            max_neg_per_anchor=args.max_neg_per_anchor,
            max_train_pairs=args.max_train_pairs,
            seed=args.seed,
        )
    exported_eval_pickles = []
    if not args.skip_eval:
        for ndx, eval_pickle in enumerate(args.eval_pickle):
            stem = Path(eval_pickle).stem
            output_name = 'test_items_ringsharp.pickle' if ndx == 0 else f'test_items_{stem}.pickle'
            exported_eval_pickles.append(output_name)
            export_eval(
                args.dataset_root,
                args.out_root,
                eval_pickle,
                overwrite=args.overwrite,
                cam_num=args.cam_num,
                positive_radius=args.positive_radius,
                output_name=output_name,
            )
    config_test_pickle = exported_eval_pickles[0] if exported_eval_pickles else 'test_items_ringsharp.pickle'
    write_config(
        args.out_root,
        args.adafusion_root,
        test_pickle=config_test_pickle,
        epochs=args.epochs,
        eval_epoch_freq=args.eval_epoch_freq,
        ckpt_name=args.ckpt_name,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        workers=args.workers,
        image_size=args.image_size,
        voxel_shape=args.voxel_shape,
        augmentation=not args.disable_augmentation,
        loss_a=args.loss_a,
        loss_m=args.loss_m,
        loss_lp=args.loss_lp,
    )


if __name__ == '__main__':
    main()
