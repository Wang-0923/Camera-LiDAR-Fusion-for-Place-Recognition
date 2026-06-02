import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from glnet.models.localizer.netvlad import OfficialNetVLAD
from glnet.utils.params import ModelParams


VGG_CONV_TO_TORCH_INDEX = {
    'conv1_1': 0,
    'conv1_2': 2,
    'conv2_1': 5,
    'conv2_2': 7,
    'conv3_1': 10,
    'conv3_2': 12,
    'conv3_3': 14,
    'conv4_1': 17,
    'conv4_2': 19,
    'conv4_3': 21,
    'conv5_1': 24,
    'conv5_2': 26,
    'conv5_3': 28,
}


def _load_mat(path):
    try:
        import scipy.io
    except ImportError as exc:
        raise SystemExit('scipy is required: pip install scipy') from exc

    try:
        return scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError as exc:
        raise SystemExit(
            'This .mat appears to be MATLAB v7.3/HDF5. This converter currently expects the official '
            'Relja NetVLAD v7 .mat files readable by scipy.io.loadmat.'
        ) from exc


def _field(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [v for v in value.reshape(-1)]
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _string(value):
    if isinstance(value, np.ndarray):
        value = value.squeeze()
        if value.dtype.kind in {'U', 'S'}:
            return ''.join(value.tolist()) if value.ndim > 0 else str(value.item())
    return str(value)


def _weights(layer):
    return _as_list(_field(layer, 'weights'))


def _to_numpy(value):
    value = np.asarray(value)
    if value.dtype == object and value.size == 1:
        value = np.asarray(value.item())
    return value.astype(np.float32, copy=False)


def _squeeze_hwio(value):
    value = _to_numpy(value)
    return np.squeeze(value)


def _numeric_arrays(obj, path='', depth=0, seen=None):
    if seen is None:
        seen = set()
    if depth > 8:
        return
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            for field_name in obj.dtype.names:
                yield from _numeric_arrays(obj[field_name], f'{path}.{field_name}', depth + 1, seen)
            return
        if obj.dtype == object:
            for index, value in np.ndenumerate(obj):
                yield from _numeric_arrays(value, f'{path}[{index}]', depth + 1, seen)
        elif np.issubdtype(obj.dtype, np.number):
            yield path, np.squeeze(obj).astype(np.float32, copy=False)
        return

    if isinstance(obj, np.void) and obj.dtype.names:
        for field_name in obj.dtype.names:
            yield from _numeric_arrays(obj[field_name], f'{path}.{field_name}', depth + 1, seen)
        return

    field_names = getattr(obj, '_fieldnames', None)
    if field_names:
        for field_name in field_names:
            yield from _numeric_arrays(getattr(obj, field_name), f'{path}.{field_name}', depth + 1, seen)
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _numeric_arrays(value, f'{path}.{key}', depth + 1, seen)
        return

    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from _numeric_arrays(value, f'{path}[{index}]', depth + 1, seen)


def _looks_like_512x64(value):
    value = np.squeeze(value)
    return value.ndim == 2 and sorted(value.shape) == [64, 512]


def _to_512x64(value):
    value = np.squeeze(value).astype(np.float32, copy=False)
    if value.shape == (64, 512):
        value = value.T
    if value.shape != (512, 64):
        raise ValueError(f'Expected a [512,64] VLAD matrix, got {value.shape}')
    return value


def _extract_layers(mat):
    if 'net' not in mat:
        raise SystemExit('Input .mat does not contain variable "net"')
    net = mat['net']
    layers = _field(net, 'layers')
    layers = _as_list(layers)
    if not layers:
        raise SystemExit('Could not find net.layers in the .mat file')
    return layers


def _copy_vgg_layers(layers, state):
    found = set()
    for layer in layers:
        name = _string(_field(layer, 'name', ''))
        if name not in VGG_CONV_TO_TORCH_INDEX:
            continue
        weights = _weights(layer)
        if len(weights) < 2:
            continue
        weight = _to_numpy(weights[0])
        bias = _to_numpy(weights[1]).reshape(-1)
        if weight.ndim != 4:
            raise SystemExit(f'Unexpected VGG weight shape for {name}: {weight.shape}')
        torch_index = VGG_CONV_TO_TORCH_INDEX[name]
        state[f'encoder.{torch_index}.weight'] = torch.from_numpy(weight.transpose(3, 2, 0, 1).copy())
        state[f'encoder.{torch_index}.bias'] = torch.from_numpy(bias.copy())
        found.add(name)

    missing = sorted(set(VGG_CONV_TO_TORCH_INDEX.keys()) - found)
    if missing:
        raise SystemExit('Missing VGG convolution layers in .mat: ' + ', '.join(missing))


def _looks_like_vlad_weights(weights):
    if len(weights) != 2:
        return False
    a = _squeeze_hwio(weights[0])
    b = _squeeze_hwio(weights[1])
    return a.shape == b.shape and sorted(a.shape) == [64, 512]


def _copy_vlad_from_arrays(state, assign, centroids, source='external'):
    assign = _to_512x64(assign)
    centroids = _to_512x64(centroids)
    state['pool.conv.weight'] = torch.from_numpy(assign.T[:, :, None, None].copy())
    state['pool.centroids'] = torch.from_numpy(centroids.T.copy())
    print(f'Using VLAD parameters from {source}')


def _copy_vlad_from_helper_mat(helper_mat, state):
    mat = _load_mat(helper_mat)
    assign = None
    centroids = None
    for key in ('clstsAssign', 'assign', 'assignment', 'assignments'):
        if key in mat:
            assign = mat[key]
            break
    for key in ('clsts', 'centroids', 'centers', 'clusters'):
        if key in mat:
            centroids = mat[key]
            break
    if assign is None or centroids is None:
        raise SystemExit(
            f'Helper MAT {helper_mat} must contain clstsAssign/assign and clsts/centroids arrays'
        )
    _copy_vlad_from_arrays(state, assign, centroids, source=helper_mat)


def _copy_vlad_layer(layers, state):
    candidates = []
    for layer in layers:
        name = _string(_field(layer, 'name', '')).lower()
        weights = _weights(layer)
        if _looks_like_vlad_weights(weights):
            candidates.append((name, weights))

    if not candidates:
        recursive_candidates = []
        for ndx, layer in enumerate(layers):
            name = _string(_field(layer, 'name', '')).lower()
            arrays = [
                (path, value)
                for path, value in _numeric_arrays(layer, f'layer{ndx}:{name}')
                if _looks_like_512x64(value)
            ]
            if len(arrays) >= 2:
                recursive_candidates.append((ndx, name, arrays))

        if not recursive_candidates:
            raise SystemExit(
                'Could not find a VLAD layer with [512,64] assignment/centroid weights, '
                'including recursive MATLAB opaque/function-handle search. '
                'This usually means the Relja .mat stores VLAD clsts/clstsAssign inside a MATLAB function handle '
                'that scipy cannot decode directly.'
            )

        preferred = [
            item for item in recursive_candidates
            if 'vlad' in item[1] or any('vlad' in path.lower() for path, _ in item[2])
        ]
        ndx, name, arrays = preferred[-1] if preferred else recursive_candidates[-1]

        def _path_score(path, keywords):
            path = path.lower()
            return sum(1 for keyword in keywords if keyword in path)

        assign_items = [
            item for item in arrays
            if _path_score(item[0], ('assign', 'assignment', 'trafo', 'soft'))
        ]
        centroid_items = [
            item for item in arrays
            if _path_score(item[0], ('clst', 'cluster', 'center', 'centroid')) and 'assign' not in item[0].lower()
        ]
        assign_path, assign_value = assign_items[-1] if assign_items else arrays[0]
        centroid_path, centroid_value = centroid_items[-1] if centroid_items else next(
            item for item in arrays if item[0] != assign_path
        )

        assign = _to_512x64(assign_value)
        centroids = _to_512x64(centroid_value)
        if 'offset' in centroid_path.lower():
            centroids = -centroids

        print(
            f'Using VLAD parameters from layer {ndx} "{name}": '
            f'assignment={assign_path}, centroids={centroid_path}'
        )
        _copy_vlad_from_arrays(state, assign, centroids, source=f'layer {ndx}')
        return

    preferred = [item for item in candidates if 'vlad' in item[0]]
    name, weights = preferred[-1] if preferred else candidates[-1]
    assign = _squeeze_hwio(weights[0])
    stored_offset = _squeeze_hwio(weights[1])
    if assign.shape == (64, 512):
        assign = assign.T
        stored_offset = stored_offset.T
    if assign.shape != (512, 64):
        raise SystemExit(f'Unexpected VLAD assignment shape in {name}: {assign.shape}')

    centroids = -stored_offset
    _copy_vlad_from_arrays(state, assign, centroids, source=f'layer "{name}"')


def _find_whitening(layers, pca_dim):
    input_dim = 64 * 512
    candidates = []
    for layer in layers:
        name = _string(_field(layer, 'name', '')).lower()
        weights = _weights(layer)
        if len(weights) < 2:
            continue
        weight = _squeeze_hwio(weights[0])
        bias = _squeeze_hwio(weights[1]).reshape(-1)
        if weight.ndim != 2:
            continue
        if set(weight.shape) == {input_dim, pca_dim} and bias.size == pca_dim:
            candidates.append((name, weight, bias))

    if not candidates:
        raise SystemExit(
            f'Could not find PCA/whitening layer with input dim {input_dim} and output dim {pca_dim}. '
            'Make sure you downloaded the VGG-16 + NetVLAD + whitening model.'
        )

    preferred = [item for item in candidates if 'pca' in item[0] or 'white' in item[0]]
    return preferred[-1] if preferred else candidates[-1]


def _copy_whitening(layers, state, pca_dim):
    name, weight, bias = _find_whitening(layers, pca_dim)
    if weight.shape == (pca_dim, 64 * 512):
        torch_weight = weight
    else:
        torch_weight = weight.T
    if torch_weight.shape != (pca_dim, 64 * 512):
        raise SystemExit(f'Unexpected whitening weight shape in {name}: {weight.shape}')
    state['whiten.weight'] = torch.from_numpy(torch_weight.copy())
    state['whiten.bias'] = torch.from_numpy(bias.copy())


def main():
    parser = argparse.ArgumentParser(description='Convert official Relja NetVLAD .mat to a RINGSharp PyTorch checkpoint')
    parser.add_argument('--mat', required=True, help='Path to official .mat, e.g. vd16_pitts30k_conv5_3_vlad_preL2_intra_white.mat')
    parser.add_argument('--output', required=True, help='Output .pth path')
    parser.add_argument('--model_config', default='glnet/config/official_netvlad_nclt.txt')
    parser.add_argument('--dataset_root', default='Data/NCLT')
    parser.add_argument('--dataset_type', default='nclt')
    parser.add_argument('--pca_dim', type=int, default=4096)
    parser.add_argument(
        '--vlad_params_mat',
        default=None,
        help='Optional helper .mat containing clstsAssign and clsts exported from MATLAB/Octave',
    )
    parser.add_argument('--dump_layers', action='store_true', help='Print layer names and weight shapes before converting')
    args = parser.parse_args()

    mat = _load_mat(args.mat)
    layers = _extract_layers(mat)
    if args.dump_layers:
        for ndx, layer in enumerate(layers):
            shapes = [tuple(np.asarray(w).shape) for w in _weights(layer)]
            print(f'{ndx:03d} {_string(_field(layer, "name", ""))}: {shapes}')

    params = ModelParams(args.model_config, args.dataset_type, args.dataset_root)
    model = OfficialNetVLAD(params, pca_dim=args.pca_dim)
    state = model.state_dict()

    _copy_vgg_layers(layers, state)
    if args.vlad_params_mat is None:
        _copy_vlad_layer(layers, state)
    else:
        _copy_vlad_from_helper_mat(args.vlad_params_mat, state)
    _copy_whitening(layers, state, args.pca_dim)

    model.load_state_dict(state, strict=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({'model': model.state_dict(), 'source_mat': os.path.abspath(args.mat), 'pca_dim': args.pca_dim}, args.output)
    print(f'Saved converted checkpoint: {args.output}')


if __name__ == '__main__':
    main()
