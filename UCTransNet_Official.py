"""Load an explicit local checkout of the authors' UCTransNet (no source substitution)."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import hashlib
import sys
from torch import nn


class UCTransNetOfficial(nn.Module):
    def __init__(self, repo_dir, in_channels=1, num_classes=3, image_size=256):
        super().__init__()
        root = Path(repo_dir).resolve()
        source = root / 'nets' / 'UCTransNet.py'
        if not source.is_file() or not (root / 'nets' / 'CTrans.py').is_file():
            raise FileNotFoundError('Expected the official UCTransNet checkout with nets/UCTransNet.py and nets/CTrans.py')
        if not isinstance(image_size, int) or image_size % 16:
            raise ValueError('The official wrapper supports square inputs divisible by 16')
        namespace = '_amal_official_uct_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
        if namespace not in sys.modules:
            package = ModuleType(namespace)
            package.__path__ = [str(root / 'nets')]
            sys.modules[namespace] = package
        spec = importlib.util.spec_from_file_location(namespace + '.UCTransNet', source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        # Build only architecture config; do NOT import upstream Config.py because
        # it changes CUDA_VISIBLE_DEVICES and global random-state environment settings.
        config = SimpleNamespace(base_channel=64, KV_size=960, expand_ratio=4,
                                 patch_sizes=[16, 8, 4, 2], n_classes=num_classes,
                                 transformer={'num_heads': 4, 'num_layers': 4,
                                     'embeddings_dropout_rate': .1,
                                     'attention_dropout_rate': .1, 'dropout_rate': 0.})
        self.network = module.UCTransNet(config, n_channels=in_channels,
                                        n_classes=num_classes, img_size=image_size, vis=False)
        # Official one-channel model returns sigmoid probabilities. Make that
        # path return logits for our unified interface without changing its weights.
        if num_classes == 1:
            self.network.last_activation = nn.Identity()
        self.image_size = image_size

    def forward(self, x):
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError('Unexpected input resolution for official UCTransNet')
        return {'stage2_final': self.network(x)}
