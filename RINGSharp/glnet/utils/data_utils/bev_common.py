import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from glnet.config.config import backbone_conf, nclt_pc_bev_conf, oxford_pc_bev_conf
from glnet.datasets.nclt.nclt_raw import NCLTSequence, calculate_T
from glnet.datasets.oxford.oxford_raw import OxfordSequence
from glnet.datasets.oxford.utils import build_se3_transform
from glnet.utils.common_utils import _ex


@dataclass(frozen=True)
class BEVGrid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_cells: int
    y_cells: int
    row_convention: str = 'y_max_to_y_min'

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / float(self.x_cells)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / float(self.y_cells)

    @property
    def x_centers(self) -> np.ndarray:
        return self.x_min + (np.arange(self.x_cells, dtype=np.float32) + 0.5) * self.dx

    @property
    def y_centers(self) -> np.ndarray:
        return self.y_max - (np.arange(self.y_cells, dtype=np.float32) + 0.5) * self.dy

    def to_dict(self) -> Dict[str, object]:
        return {
            'x_bound': [self.x_min, self.x_max],
            'y_bound': [self.y_min, self.y_max],
            'x_grid': self.x_cells,
            'y_grid': self.y_cells,
            'dx': self.dx,
            'dy': self.dy,
            'row_convention': self.row_convention,
        }

    def world_to_pixel(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim == 1:
            xy = xy.reshape(1, 2)

        cols = np.floor((xy[:, 0] - self.x_min) / self.dx).astype(np.int32)
        rows = np.floor((self.y_max - xy[:, 1]) / self.dy).astype(np.int32)
        valid = (
            (cols >= 0)
            & (cols < self.x_cells)
            & (rows >= 0)
            & (rows < self.y_cells)
        )
        return rows, cols, valid

    def nearest_center(self, x: float, y: float) -> Tuple[float, float]:
        x_ndx = int(np.argmin(np.abs(self.x_centers - x)))
        y_ndx = int(np.argmin(np.abs(self.y_centers - y)))
        return float(self.x_centers[x_ndx]), float(self.y_centers[y_ndx])


def get_visual_bev_grid() -> BEVGrid:
    x_bound = backbone_conf['x_bound']
    y_bound = backbone_conf['y_bound']
    x_cells = int((x_bound[1] - x_bound[0]) / x_bound[2])
    y_cells = int((y_bound[1] - y_bound[0]) / y_bound[2])
    return BEVGrid(
        x_min=float(x_bound[0]),
        x_max=float(x_bound[1]),
        y_min=float(y_bound[0]),
        y_max=float(y_bound[1]),
        x_cells=x_cells,
        y_cells=y_cells,
    )


def get_lidar_bev_conf(dataset_type: str) -> Dict[str, object]:
    dataset_type = dataset_type.lower()
    if dataset_type == 'nclt':
        return nclt_pc_bev_conf
    if dataset_type == 'oxford':
        return oxford_pc_bev_conf
    raise ValueError(f'Unsupported dataset type: {dataset_type}')


def build_bev_alignment_meta(sample_id, timestamp, dataset_type: str, pose=None, xyz_aug: bool = False) -> Dict[str, object]:
    visual_grid = get_visual_bev_grid()
    lidar_conf = get_lidar_bev_conf(dataset_type)
    lidar_dx = (lidar_conf['x_bound'][1] - lidar_conf['x_bound'][0]) / float(lidar_conf['x_grid'])
    lidar_dy = (lidar_conf['y_bound'][1] - lidar_conf['y_bound'][0]) / float(lidar_conf['y_grid'])

    if xyz_aug:
        raise ValueError(
            'BEV fusion requires visual and LiDAR BEV to share the same frame. '
            'Current dataset pipeline only augments LiDAR BEV for xyz_aug=True, so disable xyz_aug for fusion.'
        )

    if abs(visual_grid.x_min - lidar_conf['x_bound'][0]) > 1e-6 or abs(visual_grid.x_max - lidar_conf['x_bound'][1]) > 1e-6:
        raise ValueError('Visual and LiDAR x_bound configs are not aligned')
    if abs(visual_grid.y_min - lidar_conf['y_bound'][0]) > 1e-6 or abs(visual_grid.y_max - lidar_conf['y_bound'][1]) > 1e-6:
        raise ValueError('Visual and LiDAR y_bound configs are not aligned')
    if visual_grid.x_cells != lidar_conf['x_grid'] or visual_grid.y_cells != lidar_conf['y_grid']:
        raise ValueError('Visual and LiDAR BEV grid sizes are not aligned')
    if abs(visual_grid.dx - lidar_dx) > 1e-6 or abs(visual_grid.dy - lidar_dy) > 1e-6:
        raise ValueError('Visual and LiDAR BEV resolutions are not aligned')

    return {
        'sample_id': int(sample_id) if isinstance(sample_id, (int, np.integer)) else sample_id,
        'timestamp': int(timestamp) if timestamp is not None else None,
        'ego_pose_id': int(timestamp) if timestamp is not None else None,
        'pose': pose,
        'x_min': visual_grid.x_min,
        'x_max': visual_grid.x_max,
        'y_min': visual_grid.y_min,
        'y_max': visual_grid.y_max,
        'dx': visual_grid.dx,
        'dy': visual_grid.dy,
        'H': visual_grid.y_cells,
        'W': visual_grid.x_cells,
        'origin': 'ego_vehicle_frame',
        'row_axis': 'y',
        'col_axis': 'x',
        'x_direction': 'vehicle_forward',
        'y_direction': 'vehicle_left',
        'row_direction': visual_grid.row_convention,
        'col_direction': 'x_min_to_x_max',
        'aug_meta': {'xyz_aug': False},
    }


def get_sequence_dataset(dataset_type: str, dataset_root: str, sequence_name: str):
    dataset_root = _ex(dataset_root)
    dataset_type = dataset_type.lower()
    if dataset_type == 'nclt':
        return NCLTSequence(dataset_root, sequence_name, split='all', sampling_distance=-1.0)
    if dataset_type == 'oxford':
        return OxfordSequence(dataset_root, sequence_name, split='all', sampling_distance=-1.0)
    raise ValueError(f'Unsupported dataset type: {dataset_type}')


def _load_cached_image_meta(dataset_root: str) -> Optional[Dict[str, List[np.ndarray]]]:
    meta_path = os.path.join(_ex(dataset_root), 'image_meta.pkl')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'rb') as handle:
        image_meta = pickle.load(handle)
    image_meta['K'] = [np.asarray(k, dtype=np.float32) for k in image_meta['K']]
    image_meta['T'] = [np.asarray(t, dtype=np.float32) for t in image_meta['T']]
    return image_meta


def _build_nclt_image_meta(dataset_root: str) -> Dict[str, List[np.ndarray]]:
    cam_params_path = os.path.join(_ex(dataset_root), 'cam_params')
    if not os.path.isdir(cam_params_path):
        raise FileNotFoundError(f'Cannot find NCLT camera params: {cam_params_path}')

    k_list = []
    t_list = []
    factor_x = 224.0 / 600.0
    factor_y = 384.0 / 900.0
    for cam_num in range(1, 6):
        k_matrix = np.loadtxt(os.path.join(cam_params_path, f'K_cam{cam_num}.csv'), delimiter=',').astype(np.float32)
        fx = k_matrix[0, 0]
        fy = k_matrix[1, 1]
        cx = k_matrix[0, 2]
        cy = k_matrix[1, 2]

        cy = 1232.0 - cy
        cx -= 400.0
        cy -= 182.0
        cx = cx * factor_x
        cy = cy * factor_y

        k_matrix[0, 0] = fy * factor_y
        k_matrix[0, 2] = cy
        k_matrix[1, 1] = fx * factor_x
        k_matrix[1, 2] = cx
        k_list.append(k_matrix[:3, :3].astype(np.float32))

        t_matrix = np.loadtxt(os.path.join(cam_params_path, f'x_lb3_c{cam_num}.csv'), delimiter=',').astype(np.float32)
        t_list.append(calculate_T(t_matrix).astype(np.float32))

    return {'K': k_list, 'T': t_list}


def _build_oxford_image_meta(dataset_root: str) -> Dict[str, List[np.ndarray]]:
    dataset_root = _ex(dataset_root)
    models = ['mono_left', 'mono_right', 'mono_rear', 'stereo']
    intrinsic_root = os.path.join(dataset_root, 'models')
    extrinsic_root = os.path.join(dataset_root, 'extrinsics')
    if not os.path.isdir(intrinsic_root):
        raise FileNotFoundError(f'Cannot find Oxford models directory: {intrinsic_root}')
    if not os.path.isdir(extrinsic_root):
        raise FileNotFoundError(f'Cannot find Oxford extrinsics directory: {extrinsic_root}')

    factor_x_mono = 320.0 / 512.0
    factor_y_mono = 320.0 / 512.0
    factor_x_stereo = 640.0 / 1280.0
    factor_y_stereo = 320.0 / 640.0

    k_list = []
    t_list = []
    for model in models:
        model_name = 'stereo_narrow_left' if model == 'stereo' else model
        if model == 'mono_left':
            factor_x = factor_x_mono
            factor_y = factor_y_mono
            crop_offset_y = 200.0
        else:
            factor_x = factor_x_stereo
            factor_y = factor_y_stereo
            crop_offset_y = 160.0

        k_matrix = np.eye(3, dtype=np.float32)
        t_cam_matrix = []
        with open(os.path.join(intrinsic_root, f'{model_name}.txt')) as intrinsics_file:
            vals = [float(x) for x in next(intrinsics_file).split()]
            k_matrix[0, 0] = vals[0] * factor_x
            k_matrix[1, 1] = vals[1] * factor_y
            k_matrix[0, 2] = vals[2] * factor_x
            k_matrix[1, 2] = (vals[3] - crop_offset_y) * factor_y
            for line in intrinsics_file:
                t_cam_matrix.append([float(x) for x in line.split()])

        with open(os.path.join(extrinsic_root, f'{model}.txt')) as extrinsics_file:
            extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
            t_matrix = build_se3_transform(extrinsics)

        k_list.append(k_matrix.astype(np.float32))
        t_list.append(np.dot(np.linalg.inv(np.asarray(t_cam_matrix, dtype=np.float32)), t_matrix).astype(np.float32))

    return {'K': k_list, 'T': t_list}


def load_image_meta(dataset_type: str, dataset_root: str) -> Dict[str, List[np.ndarray]]:
    cached = _load_cached_image_meta(dataset_root)
    if cached is not None:
        return cached

    dataset_type = dataset_type.lower()
    if dataset_type == 'nclt':
        return _build_nclt_image_meta(dataset_root)
    if dataset_type == 'oxford':
        return _build_oxford_image_meta(dataset_root)
    raise ValueError(f'Unsupported dataset type: {dataset_type}')


def save_rgb_png(path: str, image: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
