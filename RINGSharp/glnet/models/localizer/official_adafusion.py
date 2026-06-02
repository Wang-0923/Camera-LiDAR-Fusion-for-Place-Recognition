import importlib
import os
import sys
from types import SimpleNamespace

import torch.nn as nn


class OfficialAdaFusion(nn.Module):
    """Runtime wrapper for MetaSLAM/AdaFusion AAVisLidarNet3."""

    def __init__(self, model_params):
        super().__init__()
        root = getattr(model_params, 'official_adafusion_root', None) or os.environ.get('ADAFUSION_ROOT')
        if not root:
            raise ValueError('OfficialAdaFusion requires official_adafusion_root in config or ADAFUSION_ROOT env')
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f'Cannot access AdaFusion root: {root}')
        self.official_adafusion_root = root
        if root not in sys.path:
            sys.path.insert(0, root)

        models = importlib.import_module('models')
        config = SimpleNamespace(
            architecture='aavislidarnet3',
            fusion_method=getattr(model_params, 'adafusion_fusion_method', 'cat'),
            loss=SimpleNamespace(Lp=getattr(model_params, 'adafusion_lp', 1)),
        )
        self.model = models.get_model(config)

    def load_official_state_dict(self, checkpoint, strict=True):
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key[len('module.'):] if key.startswith('module.') else key] = value
        return self.model.load_state_dict(cleaned, strict=strict)

    def forward(self, batch):
        return {'global': self.model(batch['img'], batch['pc'])}

    def print_info(self):
        print('Model class: Official AdaFusion')
        print(f'Official repository: {self.official_adafusion_root}')
