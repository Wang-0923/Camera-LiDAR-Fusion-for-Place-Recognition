import importlib
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn


class OfficialMinkLocMultimodal(nn.Module):
    """Wrapper around jac99/MinkLocMultimodal for RINGSharp evaluation.

    The official repository is imported at runtime so its pretrained state dict
    can be loaded without renaming all architecture keys. Set
    official_minkloc_root in the model config or export MINKLOC_MULTIMODAL_ROOT.
    """

    def __init__(self, model_params):
        super().__init__()
        root = getattr(model_params, 'official_minkloc_root', None)
        if root is None:
            root = os.environ.get('MINKLOC_MULTIMODAL_ROOT')
        if not root:
            raise ValueError(
                'OfficialMinkLocMultimodal requires official_minkloc_root in the model config '
                'or MINKLOC_MULTIMODAL_ROOT in the environment.'
            )
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f'Cannot access official MinkLocMultimodal root: {root}')

        self.official_minkloc_root = root
        self._add_official_root(root)
        self.model = self._build_official_model()

    @staticmethod
    def _add_official_root(root):
        if root not in sys.path:
            sys.path.insert(0, root)

    @staticmethod
    def _build_official_model():
        # The official model constructs ResNet18 with pretrained=True. The
        # official checkpoint overwrites those weights, so avoid network access.
        import torchvision.models as tv_models

        original_resnet18 = tv_models.resnet18

        def resnet18_no_download(*args, **kwargs):
            kwargs['pretrained'] = False
            return original_resnet18(*args, **kwargs)

        tv_models.resnet18 = resnet18_no_download
        try:
            factory = importlib.import_module('models.model_factory')
            params = SimpleNamespace(model_params=SimpleNamespace(model='MinkLocMultimodal'))
            return factory.model_factory(params)
        finally:
            tv_models.resnet18 = original_resnet18

    def load_official_state_dict(self, state_dict, strict=True):
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        return self.model.load_state_dict(state_dict, strict=strict)

    def forward(self, batch):
        out = self.model(batch)
        return {
            'global': out['embedding'],
            'cloud_embedding': out.get('cloud_embedding'),
            'image_embedding': out.get('image_embedding'),
        }

    def print_info(self):
        print('Model class: Official MinkLoc++ multimodal')
        print(f'Official repository: {self.official_minkloc_root}')
        if hasattr(self.model, 'print_info'):
            self.model.print_info()
