"""COMA-Net paper-based reproduction with an explicit multiclass adaptation.

Ahmed & Hasan, BSPC 86 (2023) 105198, DOI 10.1016/j.bspc.2023.105198.
Preserved: independent ResNet34 encoders, PRM/NRM on E1-E3, four-way zero-
padded feature shifts (7,5,3), dilations (1,2,4,8), additive/subtractive masks,
and positive-mask plus complementary Dice losses (alpha=.95, beta=.05).

This is NOT author code or a claim of numerical reproduction. Where accessible
paper text did not specify a width or channel-shift fraction, explicit engineering
choices are recorded in docs/BASELINE_PROVENANCE.md and attributes below.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                         nn.BatchNorm2d(cout)) if stride != 1 or cin != cout else None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + identity)


class ResNet34Encoder(nn.Module):
    """Standard 3/4/6/3 ResNet34 feature extractor; torchvision key-compatible."""
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cin = 64
        for i, (width, count) in enumerate(zip((64, 128, 256, 512), (3, 4, 6, 3)), 1):
            blocks = [BasicBlock(cin, width, 1 if i == 1 else 2)]
            blocks.extend(BasicBlock(width, width) for _ in range(count - 1))
            setattr(self, f'layer{i}', nn.Sequential(*blocks))
            cin = width
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        result = []
        for i in range(1, 5):
            x = getattr(self, f'layer{i}')(x)
            result.append(x)
        return result

    def load_imagenet(self, path):
        state = torch.load(Path(path), map_location='cpu', weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        state.pop('fc.weight', None)
        state.pop('fc.bias', None)
        if self.conv1.in_channels == 1 and state['conv1.weight'].shape[1] == 3:
            # Sum exactly matches applying the RGB kernel to three repeated gray channels.
            state['conv1.weight'] = state['conv1.weight'].sum(1, keepdim=True)
        if state['conv1.weight'].shape != self.conv1.weight.shape:
            raise ValueError('Pretrained input stem has incompatible shape; no hidden remapping is made.')
        self.load_state_dict(state, strict=True)


def shift_zero(x, dy=0, dx=0):
    """Translate with zero padding, not torch.roll's wrap-around."""
    h, w = x.shape[-2:]
    if abs(dy) >= h or abs(dx) >= w:
        return x * 0
    # destination[y,x] = source[y-dy,x-dx]
    padded = F.pad(x, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)))
    sy, sx = max(-dy, 0), max(-dx, 0)
    return padded[..., sy:sy+h, sx:sx+w]


def partial_four_way_shift(x, distance, fraction=.5):
    group = int(x.shape[1] * fraction) // 4
    if group < 1:
        return x
    split = list(torch.split(x[:, :4 * group], group, dim=1))
    moved = [shift_zero(split[0], dy=-distance), shift_zero(split[1], dy=distance),
             shift_zero(split[2], dx=distance), shift_zero(split[3], dx=-distance)]
    return torch.cat([*moved, x[:, 4 * group:]], 1)


def cbr(cin, cout, kernel=3, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel//2*dilation,
                                   dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=False))


class BipolarRefinement(nn.Module):
    def __init__(self, channels, distance, positive, shift_fraction=.5):
        super().__init__()
        self.distance, self.positive, self.shift_fraction = distance, positive, shift_fraction
        hidden = channels // 4  # explicit reproduction choice, not a verified author setting
        self.context = nn.ModuleList([cbr(channels, hidden, dilation=d) for d in (1, 2, 4, 8)])
        self.to_mask = nn.Sequential(cbr(4 * hidden, hidden), nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        shifted = partial_four_way_shift(x, self.distance, self.shift_fraction)
        mask = self.to_mask(torch.cat([layer(shifted) for layer in self.context], 1)).sigmoid()
        # Paper uses a repeated mask ADDITION/SUBTRACTION, not multiplicative SE.
        return (x + mask if self.positive else x - mask), mask


class MergeUp(nn.Module):
    def __init__(self, cin, skip_channels, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.merge = nn.Sequential(cbr(cout + 2 * skip_channels, cout, kernel=1),
                                    cbr(cout, cout), cbr(cout, cout))

    def forward(self, x, positive, negative):
        x = self.up(x)
        if x.shape[-2:] != positive.shape[-2:]:
            x = F.interpolate(x, size=positive.shape[-2:], mode='bilinear', align_corners=False)
        return self.merge(torch.cat([x, positive, negative], 1))


class COMANetPaperReproduction(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, imagenet_weights=None, shift_fraction=.5):
        super().__init__()
        if not 0 <= shift_fraction <= 1:
            raise ValueError('shift_fraction must be in [0,1]')
        self.num_classes = num_classes
        self.positive_encoder = ResNet34Encoder(in_channels)
        self.negative_encoder = ResNet34Encoder(in_channels)
        if imagenet_weights is not None:
            self.positive_encoder.load_imagenet(imagenet_weights)
            self.negative_encoder.load_imagenet(imagenet_weights)
        self.imagenet_initialized = imagenet_weights is not None
        self.prm = nn.ModuleList([BipolarRefinement(c, d, True, shift_fraction)
                                  for c, d in zip((64, 128, 256), (7, 5, 3))])
        self.nrm = nn.ModuleList([BipolarRefinement(c, d, False, shift_fraction)
                                  for c, d in zip((64, 128, 256), (7, 5, 3))])
        self.bottleneck = nn.Sequential(cbr(1024, 512), cbr(512, 512))
        self.decode = nn.ModuleList([MergeUp(512, 256, 256), MergeUp(256, 128, 128),
                                    MergeUp(128, 64, 64)])
        self.final_up = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2), cbr(32, 32),
                                      nn.ConvTranspose2d(32, 16, 2, 2), cbr(16, 16))
        self.head = nn.Conv2d(16, num_classes, 1)
        self.reproduction_choices = {'shift_fraction': shift_fraction,
            'dilated_branch_width': 'C/4', 'decoder_widths': [512, 256, 128, 64, 32, 16],
            'multi_class_head': num_classes != 1, 'paper_result_reproduced': False}

    def forward(self, x):
        shape = x.shape[-2:]
        positive = self.positive_encoder(x)
        negative = self.negative_encoder(x)
        refined_p, refined_n, masks_p, masks_n = [], [], [], []
        for p, n, prm, nrm in zip(positive[:3], negative[:3], self.prm, self.nrm):
            p, mp = prm(p)
            n, mn = nrm(n)
            refined_p.append(p); refined_n.append(n)
            masks_p.append(mp); masks_n.append(mn)
        y = self.bottleneck(torch.cat([positive[-1], negative[-1]], 1))
        for block, p, n in zip(self.decode, reversed(refined_p), reversed(refined_n)):
            y = block(y, p, n)
        y = self.head(self.final_up(y))
        if y.shape[-2:] != shape:
            y = F.interpolate(y, size=shape, mode='bilinear', align_corners=False)
        return {'stage2_final': y, 'positive_masks': masks_p, 'negative_masks': masks_n}


def soft_dice_loss(probability, target, valid=None):
    if valid is not None:
        probability = probability * valid
        target = target * valid
    dims = tuple(range(2, probability.ndim))
    tp = (probability * target).sum(dims)
    denom = probability.sum(dims) + target.sum(dims)
    return (1 - (2 * tp + 1e-6) / (denom + 1e-6)).mean()


class COMANetLoss(nn.Module):
    """Native auxiliary mechanisms preserved; multiclass final Dice is an adaptation."""
    def __init__(self, foreground_ids=(1, 2), alpha=.95, beta=.05, ignore_index=255):
        super().__init__()
        self.foreground_ids, self.alpha, self.beta, self.ignore_index = foreground_ids, alpha, beta, ignore_index

    def forward(self, output, target):
        logits = output['stage2_final'].float()
        valid = target.ne(self.ignore_index)[:, None].float()
        safe = target.masked_fill(target.eq(self.ignore_index), 0)
        foreground = torch.zeros_like(safe, dtype=torch.bool)
        for k in self.foreground_ids:
            foreground |= safe.eq(k)
        fg = foreground[:, None].float()
        if logits.shape[1] == 1:
            final = soft_dice_loss(logits.sigmoid(), fg, valid)
        else:
            onehot = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).float()
            final = soft_dice_loss(logits.softmax(1)[:, 1:], onehot[:, 1:], valid)
        positive_loss = final * 0
        complementary_loss = final * 0
        for positive, negative in zip(output['positive_masks'], output['negative_masks']):
            p = F.interpolate(positive.float(), size=target.shape[-2:], mode='bilinear', align_corners=False)
            n = F.interpolate(negative.float(), size=target.shape[-2:], mode='bilinear', align_corners=False)
            positive_loss = positive_loss + soft_dice_loss(p, fg, valid)
            # On ordinary fully observed masks, compute at native feature resolution
            # as in Eq. (4). For ignored pixels, use the valid full-resolution mask.
            if valid.all():
                complementary_loss = complementary_loss + soft_dice_loss(positive.float(), 1 - negative.float())
            else:
                complementary_loss = complementary_loss + soft_dice_loss(p, 1 - n, valid)
        total = final + self.alpha * positive_loss + self.beta * complementary_loss
        return {'segmentation': final, 'positive_supervision': positive_loss,
                'complementary': complementary_loss, 'total': total}
