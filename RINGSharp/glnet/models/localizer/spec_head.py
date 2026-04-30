import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecGlobalDescriptorHead(nn.Module):
    """Convert a RING# spectrum tensor into a fixed-size global descriptor."""

    def __init__(self, input_channels: int = 1, output_dim: int = 256, pool_size=(4, 16), hidden_dim: int = 512):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.pool_size = pool_size
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.max_pool = nn.AdaptiveMaxPool2d(pool_size)
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(2 * input_channels * pool_size[0] * pool_size[1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
        if spec.dim() != 4:
            raise ValueError(f'Expected spec with shape (B, C, H, W), got {tuple(spec.shape)}')
        if spec.shape[1] != self.input_channels:
            if self.input_channels == 1:
                spec = spec.mean(dim=1, keepdim=True)
            else:
                raise ValueError(f'Expected spec with {self.input_channels} channels, got {spec.shape[1]}')

        pooled = torch.cat((self.avg_pool(spec), self.max_pool(spec)), dim=1)
        desc = self.proj(pooled)
        return F.normalize(desc, p=2, dim=1)
