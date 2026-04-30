import argparse
import os
import random
import sys
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.neighbors import KDTree
from torchvision import transforms as transforms

from glnet.config.config import nclt_pc_bev_conf, oxford_pc_bev_conf
from glnet.datasets.base_datasets import EvaluationTuple, get_pointcloud_loader, get_pointcloud_with_image_loader
from glnet.models.model_factory import model_factory
from glnet.utils.common_utils import _ex, to_torch
from glnet.utils.data_utils.bev_common import build_bev_alignment_meta
from glnet.utils.data_utils.point_clouds import generate_bev
from glnet.utils.params import ModelParams
try:
    from tools.evaluator import Evaluator
except ModuleNotFoundError:
    from evaluator import Evaluator


class PREvaluator(Evaluator):
    """PR-only evaluator for global descriptors."""

    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float] = None, k: int = 10, n_samples=None, debug: bool = False):
        radius = [10.0] if radius is None else radius
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, params, radius, k, n_samples, debug=debug)
        if dataset_type == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif dataset_type == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        else:
            raise NotImplementedError(f'Unsupported dataset type: {dataset_type}')

        self.bounds = (pc_bev_conf['x_bound'][0], pc_bev_conf['x_bound'][1],
                       pc_bev_conf['y_bound'][0], pc_bev_conf['y_bound'][1],
                       pc_bev_conf['z_bound'][0], pc_bev_conf['z_bound'][1])
        self.X = pc_bev_conf['x_grid']
        self.Y = pc_bev_conf['y_grid']
        self.Z = pc_bev_conf['z_grid']

    def _build_batch(self, e: EvaluationTuple):
        sph = self.params.use_panorama

        if self.dataset_type == 'nclt':
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            assert os.path.exists(scan_filepath), f'Cannot access point cloud file: {scan_filepath}'
            orig_pc, orig_imgs = self.pcim_loader(scan_filepath, sph)
            bev_path = scan_filepath.replace('velodyne_sync', 'bev').replace('bin', 'npy')
        else:
            extrinsics_dir = os.path.join(self.params.dataset_folder, 'extrinsics')
            scan_filepath = e.filepaths[0]
            assert os.path.exists(scan_filepath), f'Cannot access point cloud file: {scan_filepath}'
            orig_pc, orig_imgs = self.pcim_loader(e.filepaths, sph, extrinsics_dir)
            bev_path = scan_filepath.replace('velodyne_left', 'bev').replace('png', 'npy').replace('bin', 'npy')

        orig_pc = np.asarray(orig_pc)
        pc = None
        if self.params.use_bev:
            bev_folder = os.path.dirname(bev_path)
            os.makedirs(bev_folder, exist_ok=True)
            if os.path.isfile(bev_path):
                bev = to_torch(np.load(bev_path)).float()
            else:
                bev = generate_bev(orig_pc, Z=self.Z, Y=self.Y, X=self.X, bounds=self.bounds)
                np.save(bev_path, bev.detach().cpu().numpy())
            pc = bev.unsqueeze(0).to(self.device)

        imgs = None
        if self.params.use_rgb:
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            imgs = [to_tensor(img) for img in orig_imgs]
            imgs = torch.stack(imgs).float()
            if not sph:
                imgs = imgs.unsqueeze(0)
            imgs = imgs.to(self.device)

        batch = {'pc': pc, 'img': imgs, 'orig_pc': orig_pc}
        if self.params.enable_bev_fusion:
            batch['bev_meta'] = [
                build_bev_alignment_meta(
                    sample_id=e.timestamp,
                    timestamp=e.timestamp,
                    dataset_type=self.dataset_type,
                    pose=e.pose,
                    xyz_aug=False,
                )
            ]

        return batch

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        model.eval()
        embeddings = None

        for ndx, e in tqdm.tqdm(enumerate(eval_subset), total=len(eval_subset)):
            batch = self._build_batch(e)
            with torch.no_grad():
                y = model(batch)
                desc = y['global'].detach().cpu().numpy()

            if desc.ndim == 1:
                desc = desc.reshape(1, -1)
            if embeddings is None:
                embeddings = np.zeros((len(eval_subset), desc.shape[1]), dtype=desc.dtype)
            embeddings[ndx] = desc[0]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings

    def evaluate(self, model, exp_name=None, *args, **kwargs):
        if exp_name is None:
            exp_name = self.params.model

        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)
        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()
        map_tree = KDTree(map_positions)

        num_maps = len(map_positions)
        num_queries = len(query_positions)
        print(f'{num_maps} database elements, {num_queries} query elements')

        if self.n_samples is None or len(query_embeddings) <= self.n_samples:
            query_indexes = list(range(len(query_embeddings)))
        else:
            query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        max_k = min(self.k, num_maps)
        pair_dists = np.zeros((len(query_embeddings), len(map_embeddings)), dtype=np.float32)
        recalls = {r: np.zeros(max_k, dtype=np.float64) for r in self.radius}
        positives = {r: 0 for r in self.radius}

        for query_ndx in tqdm.tqdm(query_indexes):
            query_embedding = query_embeddings[query_ndx]
            dists = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            pair_dists[query_ndx] = dists
            idxs_sorted = np.argsort(dists)[:max_k]

            for revisit_threshold in self.radius:
                nn_ndx = map_tree.query_radius(query_positions[query_ndx].reshape(1, -1), revisit_threshold)[0]
                if len(nn_ndx) == 0:
                    continue
                positives[revisit_threshold] += 1
                nn_set = set(nn_ndx.tolist())
                for k in range(max_k):
                    if any(idx in nn_set for idx in idxs_sorted[:k + 1]):
                        recalls[revisit_threshold][k:] += 1
                        break

        eval_setting = os.path.splitext(os.path.basename(self.eval_set_filepath))[0]
        folder_path = _ex(f'./results/{exp_name}/{eval_setting}/pr_only')
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'pair_dists.npy'), pair_dists)
        np.save(os.path.join(folder_path, 'map_embeddings.npy'), map_embeddings)
        np.save(os.path.join(folder_path, 'query_embeddings.npy'), query_embeddings)

        metrics = {
            'exp_name': exp_name,
            'dataset_type': self.params.dataset_type,
            'eval_setting': eval_setting,
            'num_queries': num_queries,
            'num_maps': num_maps,
            'topk': max_k,
            'radius': self.radius,
        }

        for revisit_threshold in self.radius:
            denom = max(positives[revisit_threshold], 1)
            recall_curve = recalls[revisit_threshold] / denom
            metrics[f'{revisit_threshold}m'] = {
                'num_positives': positives[revisit_threshold],
                'recall_pr': recall_curve,
                'recall_at_1': recall_curve[0] if max_k >= 1 else 0.0,
                'recall_at_5': recall_curve[4] if max_k >= 5 else recall_curve[-1],
                'recall_at_10': recall_curve[9] if max_k >= 10 else recall_curve[-1],
            }
            print(f'-------- Revisit threshold: {revisit_threshold} m --------')
            print(f"Recall@1: {metrics[f'{revisit_threshold}m']['recall_at_1']}")
            print(f"Recall@5: {metrics[f'{revisit_threshold}m']['recall_at_5']}")
            print(f"Recall@10: {metrics[f'{revisit_threshold}m']['recall_at_10']}")

        return metrics


def _load_checkpoint(model, weight, device):
    checkpoint = torch.load(weight, map_location=device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model_state = model.state_dict()
    checkpoint_has_module = any(k.startswith('module.') for k in state_dict.keys())
    model_has_module = any(k.startswith('module.') for k in model_state.keys())
    if checkpoint_has_module and not model_has_module:
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    elif model_has_module and not checkpoint_has_module:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PR-only global descriptors')
    parser.add_argument('--dataset_root', type=str, default='Data/NCLT')
    parser.add_argument('--dataset_type', type=str, default='nclt', choices=['nclt', 'oxford'])
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--eval_set', type=str, default='test_2012-02-04_2012-03-17_20.0_5.0.pickle')
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--revisit_threshold', type=float, default=10.0)
    parser.add_argument('--n_samples', type=int, default=None)
    args = parser.parse_args()

    dataset_root = _ex(args.dataset_root)
    weight = _ex(args.weight) if args.weight is not None else None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Dataset root: {dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Model config path: {args.model_config}')
    print(f'Weight: {weight}')
    print(f'Revisit threshold: {args.revisit_threshold}')
    print(f'Device: {device}')

    model_params = ModelParams(args.model_config, args.dataset_type, dataset_root)
    exp_name = args.exp_name or model_params.model
    model = model_factory(model_params)
    model.to(device)

    if weight is not None:
        assert os.path.exists(weight), f'Cannot open network weight: {weight}'
        _load_checkpoint(model, weight, device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    evaluator = PREvaluator(dataset_root, args.dataset_type, args.eval_set, device=device,
                            params=model_params, radius=[args.revisit_threshold], k=10,
                            n_samples=args.n_samples)
    metrics = evaluator.evaluate(model, exp_name=exp_name)
    metrics['weight'] = weight
    evaluator.print_results(metrics)
    evaluator.export_eval_stats(f'./results/{exp_name}/eval_pr_results_{args.dataset_type}.txt', metrics)
