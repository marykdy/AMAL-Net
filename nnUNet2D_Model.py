"""Two explicitly distinguished nnU-Net routes.

1. NNUNet2DFromPlans: instantiate the ACTUAL official network from nnU-Net plans.
   Use the official CLI for preprocessing/training/inference to claim 'nnU-Net 2D'.
2. NNUNet2DArchitecture: dependency-light PlainConvUNet-style architecture
   reproduction for controlled architectural comparisons, NOT the full pipeline.

No silent fallback is used when the official dependency is missing.
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class PlainStage(nn.Sequential):
    def __init__(self, cin, cout, stride=1, count=2):
        modules = []
        for i in range(count):
            modules.extend([nn.Conv2d(cin if i == 0 else cout, cout, 3,
                                      stride=stride if i == 0 else 1, padding=1, bias=True),
                            nn.InstanceNorm2d(cout, eps=1e-5, affine=True),
                            nn.LeakyReLU(negative_slope=1e-2, inplace=False)])
        super().__init__(*modules)


class NNUNet2DArchitecture(nn.Module):
    """Fixed proposed architecture; not a data-derived official nnU-Net plan."""
    def __init__(self, in_channels=1, num_classes=3,
                 features=(32, 64, 128, 256, 320, 320), deep_supervision=True):
        super().__init__()
        if len(features) < 3:
            raise ValueError('at least three stages are required')
        self.divisor = 2 ** (len(features) - 1)
        self.deep_supervision = deep_supervision
        self.encoder = nn.ModuleList([PlainStage(in_channels if i == 0 else features[i - 1],
                                                 ch, 1 if i == 0 else 2) for i, ch in enumerate(features)])
        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.heads = nn.ModuleList()
        for i in range(len(features) - 2, -1, -1):
            self.ups.append(nn.ConvTranspose2d(features[i + 1], features[i], 2, 2, bias=True))
            self.decoder.append(PlainStage(2 * features[i], features[i]))
            self.heads.append(nn.Conv2d(features[i], num_classes, 1))
        self.apply(self._init)

    @staticmethod
    def _init(layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(layer.weight, a=1e-2)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        if h < 2 * self.divisor or w < 2 * self.divisor:
            raise ValueError(f'InstanceNorm requires >=2x2 bottleneck here; input >= {2*self.divisor}')
        if h % self.divisor or w % self.divisor:
            raise ValueError('Use planned/divisible patch sizes; no silent resizing is performed.')
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        preds = []
        for up, conv, head, skip in zip(self.ups, self.decoder, self.heads, reversed(skips[:-1])):
            x = conv(torch.cat([up(x), skip], 1))
            preds.append(head(x))
        out = {'stage2_final': preds[-1]}
        if self.deep_supervision:
            out['deep_supervision'] = list(reversed(preds[:-1]))
        return out


class NNUNet2DFromPlans(nn.Module):
    """Official architecture construction with no alteration of the selected plan.

    This wrapper accepts preprocessed 2D patches. Normalization, resampling,
    sliding-window inference and inverse transforms are provided by official CLI,
    NOT by this wrapper.
    """
    def __init__(self, plans_path, dataset_json_path, deep_supervision=True):
        super().__init__()
        import json
        try:
            from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
            from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        except ImportError as exc:
            raise ImportError('Install nnunetv2 and its dependencies; do not substitute a generic U-Net.') from exc
        with open(dataset_json_path, encoding='utf-8') as f:
            dataset = json.load(f)
        manager = PlansManager(str(Path(plans_path)))
        config = manager.get_configuration('2d')
        if len(config.patch_size) != 2:
            raise ValueError('selected configuration is not 2D')
        label_manager = manager.get_label_manager(dataset)
        if label_manager.has_regions or label_manager.has_ignore_label:
            raise ValueError('The common wrapper supports ordinary exclusive class labels only. Use native trainer for regions/ignore.')
        in_channels = len(dataset['channel_names'])
        self.network = get_network_from_plans(
            config.network_arch_class_name,
            config.network_arch_init_kwargs,
            config.network_arch_init_kwargs_req_import,
            in_channels, label_manager.num_segmentation_heads,
            allow_init=True, deep_supervision=deep_supervision)
        self.patch_size = tuple(config.patch_size)
        self.deep_supervision = deep_supervision

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.patch_size:
            raise ValueError(f'Use official planned patch size {self.patch_size}')
        predictions = self.network(x)
        if isinstance(predictions, (tuple, list)):
            return {'stage2_final': predictions[0], 'deep_supervision': list(predictions[1:])}
        return {'stage2_final': predictions}


def official_plain_architecture(in_channels=1, num_classes=3,
                                features=(32, 64, 128, 256, 320, 320), deep_supervision=True):
    """Actual dynamic-network-architectures class; still NOT the nnU-Net pipeline."""
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
    except ImportError as exc:
        raise ImportError('Install dynamic-network-architectures to use this official architecture.') from exc
    n = len(features)
    network = PlainConvUNet(input_channels=in_channels, n_stages=n, features_per_stage=list(features),
                           conv_op=nn.Conv2d, kernel_sizes=[(3, 3)] * n,
                           strides=[(1, 1)] + [(2, 2)] * (n - 1), n_conv_per_stage=[2] * n,
                           num_classes=num_classes, n_conv_per_stage_decoder=[2] * (n - 1),
                           conv_bias=True, norm_op=nn.InstanceNorm2d,
                           norm_op_kwargs={'eps': 1e-5, 'affine': True},
                           dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
                           nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
                           deep_supervision=deep_supervision)
    if hasattr(network, 'initialize'):
        network.apply(network.initialize)
    return network
