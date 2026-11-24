"""AMAL-Net: revised two-stage model, self-contained PyTorch implementation.

Includes AMALConfig, LossConfig, AMALNet, ImprovedTwoStageSegmentation,
AMALLoss, StandardSegLoss, and helper modules. No local-package imports.
The segmentation/presence losses use training labels; model forward does not.
This revision must be trained anew; original checkpoints are not compatible.
"""
from __future__ import annotations


# ---------- CONFIG ----------
"""Explicit settings. Defaults are proposed starting points, NOT measured optima."""
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class AMALConfig:
    in_channels: int = 1
    num_classes: int = 3
    lesion_ids: tuple[int, ...] = (2,)
    foreground_ids: tuple[int, ...] = (1, 2)
    base_channels: int = 32
    norm: str = 'group'
    grid: tuple[int, int] = (3, 3)
    top_k: int = 4
    routing: str = 'learned_topk'  # learned_topk, random_topk, all
    use_scores: bool = True
    score_modulation: bool = True
    score_floor: float = 0.25
    enlargement: float = 2.0
    halo: int = 4
    num_prototypes: int = 4
    temperature_init: float = 0.5
    stage2: bool = True
    patch_refinement: bool = True
    foreground_branch: bool = True
    background_branch: bool = True
    cluster_attention: bool = True
    density_attention: bool = True
    deep_supervision: bool = True
    channel_spatial_attention: bool = True
    skip_attention: bool = True
    residual: bool = True
    channel_gate: bool = False  # optional uploaded-code extension, not UCTransNet CCT
    dropout: float = 0.05
    patch_chunk_size: int = 0  # 0 = process all selected patches together

    def __post_init__(self):
        if self.num_classes < 2 or self.in_channels < 1 or self.base_channels < 4:
            raise ValueError('Use >=2 mutually exclusive classes, positive input channels, base_channels>=4.')
        if self.norm not in ('group', 'batch', 'instance'):
            raise ValueError('norm must be group/batch/instance')
        if len(self.grid) != 2 or min(self.grid) < 1:
            raise ValueError('grid must contain two positive integers')
        if not 1 <= self.top_k <= self.grid[0] * self.grid[1]:
            raise ValueError('top_k must lie in [1, number of grid patches]')
        if self.routing not in ('learned_topk', 'random_topk', 'all'):
            raise ValueError('unknown routing')
        if self.stage2 and self.patch_refinement and self.routing == 'learned_topk' and not self.use_scores:
            raise ValueError('learned_topk requires a scoring head')
        if self.score_modulation and not self.use_scores:
            raise ValueError('score_modulation requires scores')
        if not (self.foreground_branch or self.background_branch):
            raise ValueError('at least one Stage-1 branch must remain')
        if self.enlargement < 1 or self.halo < 0 or not 0 <= self.score_floor < 1:
            raise ValueError('invalid enlargement/halo/score_floor')
        if self.temperature_init <= 0.05 or self.num_prototypes < 1:
            raise ValueError('temperature_init must exceed .05 and prototypes must be positive')
        if not 0 <= self.dropout < 1 or self.patch_chunk_size < 0:
            raise ValueError('invalid dropout/chunk size')
        for ids in (self.lesion_ids, self.foreground_ids):
            if not ids or len(set(ids)) != len(ids) or any(i <= 0 or i >= self.num_classes for i in ids):
                raise ValueError('foreground/lesion ids must be unique valid non-background labels')
        if not set(self.lesion_ids).issubset(self.foreground_ids):
            raise ValueError('foreground target must include the lesion ids')

@dataclass(frozen=True)
class LossConfig:
    omega: float = 0.5
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.3  # false positives
    tversky_beta: float = 0.7   # false negatives
    auxiliary_weight: float = 0.3
    deep_weights: tuple[float, ...] = (0.2, 0.1)  # half, quarter resolution
    global_weight: float = 0.25
    local_weight: float = 0.25
    score_weight: float = 0.2
    ignore_index: int = 255

    def __post_init__(self):
        if not 0 <= self.omega <= 1 or self.focal_gamma < 0:
            raise ValueError('invalid pixel-loss settings')
        weights = (self.tversky_alpha, self.tversky_beta, self.auxiliary_weight,
                   self.global_weight, self.local_weight, self.score_weight, *self.deep_weights)
        if min(weights) < 0:
            raise ValueError('loss weights must be nonnegative')


def dataset_config(name: str, **overrides) -> AMALConfig:
    """Use the actual dataset.json label mapping; never synthesize an organ label."""
    name = name.lower()
    if name in ('lits', 'pancreas', 'msd_pancreas'):
        cfg = AMALConfig()
    elif name in ('lung', 'msd_lung'):
        cfg = AMALConfig(num_classes=2, lesion_ids=(1,), foreground_ids=(1,))
    else:
        raise ValueError('Unknown dataset. Construct AMALConfig with explicit class ids.')
    return replace(cfg, **overrides)


# ---------- BLOCKS ----------
"""AMAL building blocks. Not used to silently alter comparison methods."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def norm_layer(channels, kind='group'):
    if kind == 'batch':
        return nn.BatchNorm2d(channels)
    if kind == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    groups = min(8, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1, dilation=1, norm='group', groups=1):
        super().__init__(nn.Conv2d(cin, cout, kernel, stride,
                                  padding=(kernel // 2) * dilation, dilation=dilation,
                                  groups=groups, bias=False),
                         norm_layer(cout, norm), nn.ReLU(inplace=False))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, norm='group', residual=True, dropout=0.):
        super().__init__()
        self.body = nn.Sequential(ConvNormAct(cin, cout, norm=norm), nn.Dropout2d(dropout),
                                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), norm_layer(cout, norm))
        self.residual = residual
        self.short = (nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), norm_layer(cout, norm))) if residual else None

    def forward(self, x):
        y = self.body(x)
        return F.relu(y + self.short(x) if self.residual else y)


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        mid = max(4, ch // 16)
        self.channel = nn.Sequential(nn.Conv2d(ch, mid, 1), nn.ReLU(), nn.Conv2d(mid, ch, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        gate = torch.sigmoid(self.channel(x.mean((2, 3), keepdim=True)) +
                             self.channel(x.amax((2, 3), keepdim=True)))
        y = x * gate
        return y * torch.sigmoid(self.spatial(torch.cat([y.mean(1, keepdim=True),
                                                        y.amax(1, keepdim=True)], 1)))


class AttentionGate(nn.Module):
    def __init__(self, ch, norm):
        super().__init__()
        mid = max(4, ch // 2)
        self.g = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), norm_layer(mid, norm))
        self.x = nn.Sequential(nn.Conv2d(ch, mid, 1, bias=False), norm_layer(mid, norm))
        self.mask = nn.Conv2d(mid, 1, 1)

    def forward(self, decoder, skip):
        return skip * torch.sigmoid(self.mask(F.relu(self.g(decoder) + self.x(skip))))


class UpBlock(nn.Module):
    def __init__(self, cin, cout, cfg):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.gate = AttentionGate(cout, cfg.norm) if cfg.skip_attention else None
        self.conv = ResBlock(cout * 2, cout, cfg.norm, cfg.residual, cfg.dropout)
        self.cbam = CBAM(cout) if cfg.channel_spatial_attention else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        if self.gate is not None:
            skip = self.gate(x, skip)
        return self.cbam(self.conv(torch.cat([x, skip], 1)))


class ClusterAttention(nn.Module):
    def __init__(self, ch, cfg):
        super().__init__()
        d = max(ch // 4, 8)
        self.project = nn.Conv2d(ch, d, 1, bias=False)
        self.prototypes = nn.Parameter(torch.randn(cfg.num_prototypes, d) * 0.1)
        # Positive, differentiable temperature, instead of a clamped dead parameter.
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(cfg.temperature_init - .05))))
        self.mask = nn.Sequential(ConvNormAct(cfg.num_prototypes, max(ch // 2, 4), 1, norm=cfg.norm),
                                  nn.Conv2d(max(ch // 2, 4), 1, 1))
        self.enhance = ConvNormAct(ch, ch, norm=cfg.norm, groups=ch)

    def forward(self, x):
        q = F.normalize(self.project(x), dim=1)
        centers = F.normalize(self.prototypes, dim=1)
        assignment = torch.einsum('bdhw,kd->bkhw', q, centers)
        assignment = (assignment / (F.softplus(self.raw_temperature) + .05)).softmax(1)
        a = torch.sigmoid(self.mask(assignment))
        return x + a * self.enhance(x), a


class DensityAttention(nn.Module):
    def __init__(self, ch, cfg):
        super().__init__()
        d = max(ch // 4, 8)
        # Actual depthwise spatial filtering followed by pointwise channel reduction.
        self.scales = nn.ModuleList([nn.Sequential(
            ConvNormAct(ch, ch, k, norm=cfg.norm, groups=ch),
            ConvNormAct(ch, d, 1, norm=cfg.norm)) for k in (3, 5, 7)])
        self.mask = nn.Sequential(ConvNormAct(3 * d, max(ch // 2, 4), 1, norm=cfg.norm),
                                  nn.Conv2d(max(ch // 2, 4), 1, 1))
        self.enhance = ConvNormAct(ch, ch, norm=cfg.norm, groups=ch)

    def forward(self, x):
        a = torch.sigmoid(self.mask(torch.cat([layer(x) for layer in self.scales], 1)))
        return x + a * self.enhance(x), a


class SpatialPooledChannelGate(nn.Module):
    """Optional uploaded-code idea; NOT a UCTransNet CCT reproduction."""
    def __init__(self, ch):
        super().__init__()
        self.project = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.gate = nn.Linear(32, 1)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        t = self.project(F.adaptive_avg_pool2d(x, (4, 4)).flatten(2))
        t = self.norm(t)
        t, _ = self.attention(t, t, t, need_weights=False)
        return x * (1 + self.scale.tanh() * self.gate(t).tanh()[..., None])


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        widths = [cfg.base_channels * 2 ** i for i in range(4)]
        self.levels = nn.ModuleList([ResBlock(cfg.in_channels if i == 0 else widths[i - 1], w,
                                             cfg.norm, cfg.residual, cfg.dropout)
                                     for i, w in enumerate(widths)])
        self.channel_gates = nn.ModuleList([SpatialPooledChannelGate(w) if cfg.channel_gate and i >= 2
                                           else nn.Identity() for i, w in enumerate(widths)])

    def forward(self, x):
        feats = []
        for i, (block, gate) in enumerate(zip(self.levels, self.channel_gates)):
            x = block(F.max_pool2d(x, 2) if i else x)
            x = gate(x)
            feats.append(x)
        return feats


class Decoder(nn.Module):
    def __init__(self, cfg, kind, deep=False):
        super().__init__()
        c = cfg.base_channels
        self.attn = (ClusterAttention(8 * c, cfg) if kind == 'cluster' else
                     DensityAttention(8 * c, cfg) if kind == 'density' else None)
        self.up = nn.ModuleList([UpBlock(8*c, 4*c, cfg), UpBlock(4*c, 2*c, cfg), UpBlock(2*c, c, cfg)])
        self.deep_heads = nn.ModuleList([nn.Conv2d(2*c, cfg.num_classes, 1),
                                        nn.Conv2d(4*c, cfg.num_classes, 1)]) if deep else None

    def forward(self, feats):
        x, a = self.attn(feats[-1]) if self.attn is not None else (feats[-1], None)
        maps = []
        for up, skip in zip(self.up, reversed(feats[:-1])):
            x = up(x, skip)
            maps.append(x)
        deep = [self.deep_heads[0](maps[1]), self.deep_heads[1](maps[0])] if self.deep_heads is not None and self.training else []
        return x, a, deep


class ContextFusion(nn.Module):
    def __init__(self, ch, ncls, cfg):
        super().__init__()
        self.pre = ResBlock(ch + ncls, ch, cfg.norm, cfg.residual, cfg.dropout)
        self.aspp = nn.ModuleList([ConvNormAct(ch, ch, dilation=d, norm=cfg.norm) for d in (1, 2, 4)])
        self.reduce = ConvNormAct(3 * ch, ch, 1, norm=cfg.norm)
        self.attn = CBAM(ch) if cfg.channel_spatial_attention else nn.Identity()
        self.out = nn.Conv2d(ch, ncls, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat, delta):
        x = self.pre(torch.cat([feat, delta], 1))
        x = self.reduce(torch.cat([branch(x) for branch in self.aspp], 1))
        return self.out(self.attn(x))


# ---------- PATCHES ----------
"""Exact grid geometry: padding is cropped away, never silently dropped."""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class GridMeta:
    height: int
    width: int
    rows: int
    cols: int
    patch_h: int
    patch_w: int

    @classmethod
    def create(cls, height, width, grid):
        r, c = grid
        if min(height, width, r, c) < 1:
            raise ValueError('positive spatial sizes required')
        return cls(height, width, r, c, math.ceil(height / r), math.ceil(width / c))

    @property
    def count(self):
        return self.rows * self.cols

    @property
    def padded_shape(self):
        return self.rows * self.patch_h, self.cols * self.patch_w


def extract_grid(x: torch.Tensor, meta: GridMeta, halo: int = 0,
                 pad_mode: str = 'replicate', pad_value: float = 0) -> torch.Tensor:
    """[B,C,H,W] -> [B,R,C,ph+2h,pw+2h], including valid image borders."""
    if x.ndim != 4 or tuple(x.shape[-2:]) != (meta.height, meta.width):
        raise ValueError('grid geometry and tensor shape do not match')
    h, w = meta.padded_shape
    pad = (0, w - meta.width, 0, h - meta.height)
    x = F.pad(x, pad, mode=pad_mode, **({'value': pad_value} if pad_mode == 'constant' else {}))
    if halo:
        x = F.pad(x, (halo, halo, halo, halo), mode=pad_mode,
                  **({'value': pad_value} if pad_mode == 'constant' else {}))
    kh, kw = meta.patch_h + 2 * halo, meta.patch_w + 2 * halo
    b, ch = x.shape[:2]
    patches = F.unfold(x, kernel_size=(kh, kw), stride=(meta.patch_h, meta.patch_w))
    return patches.transpose(1, 2).reshape(b, meta.count, ch, kh, kw)


def assemble_grid(patches: torch.Tensor, meta: GridMeta) -> torch.Tensor:
    b, n, ch, ph, pw = patches.shape
    if (n, ph, pw) != (meta.count, meta.patch_h, meta.patch_w):
        raise ValueError('expected core patches with the configured grid shape')
    x = patches.reshape(b, meta.rows, meta.cols, ch, ph, pw)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, ch, *meta.padded_shape)
    return x[..., :meta.height, :meta.width]


def gather_patches(patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(patches.shape[0], device=patches.device)[:, None]
    return patches[batch, indices]


def scatter_patches(patches: torch.Tensor, indices: torch.Tensor, meta: GridMeta) -> torch.Tensor:
    b, _, ch, ph, pw = patches.shape
    out = patches.new_zeros(b, meta.count, ch, ph, pw)
    ix = indices[:, :, None, None, None].expand_as(patches)
    return out.scatter(1, ix, patches)


def label_patches(labels: torch.Tensor, meta: GridMeta, ignore_index: int = 255):
    return extract_grid(labels[:, None].float(), meta, pad_mode='constant',
                        pad_value=ignore_index).squeeze(2).long()


def presence_targets(labels, meta, lesion_ids, ignore_index=255):
    p = label_patches(labels, meta, ignore_index)
    valid = p.ne(ignore_index).any(dim=(-1, -2))
    lesion = torch.zeros_like(p, dtype=torch.bool)
    for class_id in lesion_ids:
        lesion |= p.eq(class_id)
    return lesion.any(dim=(-1, -2)).float(), valid


# ---------- MODEL ----------
"""AMAL-Net revision: dual priors -> pre-refinement scoring -> real sparse refinement.

The hard top-k indices are NOT differentiable. The scorer receives (a) a continuous
selected-patch modulation gradient and (b) derived-mask presence supervision in
losses.py for ALL patches. No ground-truth labels enter inference or routing.
"""
from dataclasses import replace
import warnings
import torch
from torch import nn
import torch.nn.functional as F


class StageOne(nn.Module):
    def __init__(self, cfg: AMALConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels
        self.encoder = Encoder(cfg)
        self.fg = Decoder(cfg, 'cluster' if cfg.cluster_attention else 'none', cfg.deep_supervision) if cfg.foreground_branch else None
        # When foreground is removed, the surviving branch still gets the same
        # multiclass deep-supervision budget; the auxiliary binary target differs.
        self.bg = Decoder(cfg, 'density' if cfg.density_attention else 'none',
                          cfg.deep_supervision and not cfg.foreground_branch) if cfg.background_branch else None
        self.fg_head = nn.Conv2d(c, 2, 1) if self.fg is not None else None
        self.bg_head = nn.Conv2d(c, 2, 1) if self.bg is not None else None
        self.out_channels = c * (int(cfg.foreground_branch) + int(cfg.background_branch))
        # A real multiclass predictor for the Stage-1-only ablation, not its binary prior.
        self.task_head = nn.Conv2d(self.out_channels, cfg.num_classes, 1) if not cfg.stage2 else None

    def forward(self, x):
        enc = self.encoder(x)
        result = {'attn_weights_stage1': {}, 'deep_supervision': []}
        features = []
        for name, decoder, head in (('fg', self.fg, self.fg_head), ('bg', self.bg, self.bg_head)):
            if decoder is None:
                continue
            feat, attention, deep = decoder(enc)
            result[f'{name}_feat'] = feat
            result[f'stage1_{name}'] = head(feat)
            if attention is not None:
                result['attn_weights_stage1'][f'{name}_attn'] = attention
            features.append(feat)
            result['deep_supervision'].extend(deep)
        prior_features = torch.cat(features, 1)
        if self.task_head is not None:
            result['stage2_final'] = self.task_head(prior_features)
        return prior_features, result


class PatchScorer(nn.Module):
    """Cheap scoring BEFORE the expensive branch; max pooling retains tiny cues."""
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels * 2, max(16, channels)), nn.ReLU(),
                                  nn.Linear(max(16, channels), 1))

    def forward(self, patches):
        summary = torch.cat([patches.mean((-1, -2)), patches.amax((-1, -2))], -1)
        return self.mlp(summary).squeeze(-1)


class SparsePatchRefiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c, p = cfg.base_channels, max(8, cfg.base_channels // 2)
        self.scorer = PatchScorer(c) if cfg.use_scores else None
        self.refiner = nn.Sequential(ResBlock(c, p, cfg.norm, cfg.residual, cfg.dropout),
                                      ResBlock(p, p, cfg.norm, cfg.residual, cfg.dropout),
                                      nn.Conv2d(p, cfg.num_classes, 1))
        # Small but nonzero: initial residual is modest AND upstream gradients exist.
        nn.init.normal_(self.refiner[-1].weight, std=0.01)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, features, global_logits, *, routing=None, top_k=None,
                generator=None, warmup_all=False):
        cfg = self.cfg
        b, _, h, w = features.shape
        meta = GridMeta.create(h, w, cfg.grid)
        cores = extract_grid(features, meta)
        scores = self.scorer(cores) if self.scorer is not None else None
        mode = cfg.routing if routing is None else routing
        k = cfg.top_k if top_k is None else top_k
        if mode not in ('learned_topk', 'random_topk', 'all') or not 1 <= k <= meta.count:
            raise ValueError('invalid routing mode or k')
        if warmup_all:
            if not self.training:
                raise ValueError('warmup_all is a training-only option')
            mode = 'all'
        if mode == 'all':
            indices = torch.arange(meta.count, device=features.device)[None].expand(b, -1)
        elif mode == 'random_topk':
            ranks = torch.rand(b, meta.count, device=features.device, generator=generator)
            indices = ranks.argsort(dim=1, stable=True)[:, :k]
        else:
            if scores is None:
                raise ValueError('learned_topk requires a trained scoring head')
            # Deterministic tie-breaking. We do NOT claim gradients through argsort.
            indices = scores.argsort(dim=1, descending=True, stable=True)[:, :k]
        selected = gather_patches(extract_grid(features, meta, halo=cfg.halo), indices)
        k_eff = indices.shape[1]
        flat = selected.reshape(b * k_eff, *selected.shape[2:])
        # Enlarge feature patches, not the original CT. No new acquisition detail is created.
        enlarged_size = tuple(max(1, round(v * cfg.enlargement)) for v in flat.shape[-2:])
        flat = F.interpolate(flat, size=enlarged_size, mode='bilinear', align_corners=False)
        chunk = cfg.patch_chunk_size or flat.shape[0]
        if cfg.norm == 'batch' and cfg.patch_chunk_size:
            warnings.warn('Chunked BatchNorm changes statistics. Prefer GroupNorm for chunked refinement.', stacklevel=2)
        refined = torch.cat([self.refiner(part) for part in flat.split(chunk)], 0)
        refined = F.interpolate(refined, size=selected.shape[-2:], mode='bilinear', align_corners=False)
        m = cfg.halo
        refined = refined[..., m:m + meta.patch_h, m:m + meta.patch_w]
        delta_selected = refined.reshape(b, k_eff, cfg.num_classes, meta.patch_h, meta.patch_w)
        coarse_selected = gather_patches(extract_grid(global_logits, meta), indices)
        local_logits = coarse_selected + delta_selected
        if cfg.score_modulation and scores is not None:
            weights = cfg.score_floor + (1 - cfg.score_floor) * scores.sigmoid().gather(1, indices)
        else:
            weights = features.new_ones(b, k_eff)
        weighted_delta = delta_selected * weights[:, :, None, None, None]
        delta_map = assemble_grid(scatter_patches(weighted_delta, indices, meta), meta)
        selected_ones = features.new_ones(b, k_eff, 1, meta.patch_h, meta.patch_w)
        selection_map = assemble_grid(scatter_patches(selected_ones, indices, meta), meta)
        result = {
            'grid_meta': meta,
            'selected_indices': indices,
            'selected_patch_logits': local_logits,
            'region_outputs': local_logits,  # Compatibility: selected K, not always nine.
            'selection_mask': selection_map,
            'num_refined_patches': k_eff,
            'routing_used': mode,
            'patch_delta': delta_map,
        }
        if scores is not None:
            score_patches = scores.sigmoid()[:, :, None, None, None].expand(b, meta.count, 1, meta.patch_h, meta.patch_w)
            result.update(importance_logit=scores, importance_scores=scores.sigmoid(),
                          imp_map=assemble_grid(score_patches, meta))
        return delta_map, result


class StageTwo(nn.Module):
    def __init__(self, prior_channels, cfg):
        super().__init__()
        c = cfg.base_channels
        self.global_encoder = nn.Sequential(
            ResBlock(cfg.in_channels + prior_channels, c, cfg.norm, cfg.residual, cfg.dropout),
            CBAM(c) if cfg.channel_spatial_attention else nn.Identity(),
            ResBlock(c, c, cfg.norm, cfg.residual, cfg.dropout))
        self.global_head = nn.Conv2d(c, cfg.num_classes, 1)
        self.patches = SparsePatchRefiner(cfg) if cfg.patch_refinement else None
        self.fusion = ContextFusion(c, cfg.num_classes, cfg)

    def forward(self, image, priors, **routing_kwargs):
        feat = self.global_encoder(torch.cat([image, priors], 1))
        coarse = self.global_head(feat)
        out = {'global_logits': coarse}
        if self.patches is not None:
            delta, debug = self.patches(feat, coarse, **routing_kwargs)
            out.update(debug)
        else:
            delta = torch.zeros_like(coarse)
            out.update(num_refined_patches=0, routing_used='disabled', patch_delta=delta)
        # Non-selected regions retain a trained global path. Fusion uses global
        # context to smooth seams, so final predictions need not equal coarse
        # predictions outside the selected patches.
        out['refined_logits'] = coarse + delta
        out['stage2_final'] = coarse + delta + self.fusion(feat, delta)
        return out


class AMALNet(nn.Module):
    def __init__(self, config: AMALConfig | None = None):
        super().__init__()
        self.config = config or AMALConfig()
        self.stage1 = StageOne(self.config)
        self.stage2 = StageTwo(self.stage1.out_channels, self.config) if self.config.stage2 else None

    def forward(self, x, labels=None, *, routing=None, top_k=None, generator=None, warmup_all=False):
        """labels accepted for old call sites but deliberately NEVER used here."""
        if x.ndim != 4 or x.shape[1] != self.config.in_channels or min(x.shape[-2:]) < 16:
            raise ValueError(f'Expected [B,{self.config.in_channels},H,W], H,W>=16')
        if not x.is_floating_point():
            raise TypeError('input must be a floating-point normalized image')
        priors, out = self.stage1(x)
        if self.stage2 is not None:
            out.update(self.stage2(x, priors, routing=routing, top_k=top_k,
                                   generator=generator, warmup_all=warmup_all))
        else:
            out.update(num_refined_patches=0, routing_used='stage1_only')
        return out


class ImprovedTwoStageSegmentation(AMALNet):
    """Explicit compatibility constructor for the uploaded class name.

    Old state_dicts are NOT shape-compatible; train the revision from scratch.
    Unsupported old options fail loudly rather than being silently discarded.
    """
    def __init__(self, in_channels=3, base_channels=32, num_classes=3,
                 deep_supervision=True, dropout=.05, imp_modulation_gamma=.25,
                 use_channel_transformer=False, use_patch_refinement=True,
                 lesion_ids=None, foreground_ids=None, **kwargs):
        if kwargs:
            raise TypeError(f'Unsupported legacy arguments {sorted(kwargs)}. Use AMALConfig explicitly.')
        cfg = AMALConfig(in_channels=in_channels, base_channels=base_channels, num_classes=num_classes,
                         lesion_ids=tuple(lesion_ids or (num_classes - 1,)),
                         foreground_ids=tuple(foreground_ids or range(1, num_classes)),
                         deep_supervision=deep_supervision, dropout=dropout,
                         score_floor=imp_modulation_gamma, channel_gate=use_channel_transformer,
                         patch_refinement=use_patch_refinement)
        super().__init__(cfg)


# ---------- LOSSES ----------
"""Losses using only the supplied training segmentation masks.

Explicit derived patch supervision is NEW relative to the submitted manuscript.
It is not an extra manual annotation and is not implicit gradient supervision.
"""
import torch
from torch import nn
import torch.nn.functional as F


def validate_target(target, num_classes, ignore_index=255):
    if target.ndim != 3 or target.dtype != torch.long:
        raise ValueError('target must be int64 [B,H,W] class indices, not RGB/one-hot masks')
    valid = target.ne(ignore_index)
    if ((target[valid] < 0) | (target[valid] >= num_classes)).any():
        raise ValueError('target contains an unknown class; verify dataset.json and label mapping')
    return valid


def membership(labels, ids):
    target = torch.zeros_like(labels, dtype=torch.bool)
    for k in ids:
        target |= labels.eq(k)
    return target


class PixelLoss(nn.Module):
    """Paper composite CE/Focal/Dice/Tversky. Computes in float32 under AMP."""
    def __init__(self, config: LossConfig | None = None):
        super().__init__()
        self.config = config or LossConfig()

    def forward(self, logits, target):
        cfg = self.config
        if logits.shape[0] != target.shape[0]:
            raise ValueError('batch size mismatch')
        if logits.shape[-2:] != target.shape[-2:]:
            # Upsample logits rather than downsample labels: tiny GT lesions survive.
            logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
        logits = logits.float()
        ncls = logits.shape[1]
        valid = validate_target(target, ncls, cfg.ignore_index)
        safe = target.masked_fill(~valid, 0)
        ce_map = F.cross_entropy(logits, safe, reduction='none')
        weight = valid.float()
        denom = weight.sum().clamp_min(1)
        ce = (ce_map * weight).sum() / denom
        focal = ((1 - torch.exp(-ce_map)).pow(cfg.focal_gamma) * ce_map * weight).sum() / denom
        probabilities = logits.softmax(1) * weight[:, None]
        onehot = F.one_hot(safe, ncls).permute(0, 3, 1, 2).float() * weight[:, None]
        p, y = probabilities[:, 1:], onehot[:, 1:]
        dims = (0, 2, 3)
        tp = (p * y).sum(dims)
        fp = (p * (1 - y)).sum(dims)
        fn = ((1 - p) * y).sum(dims)
        eps = 1e-6
        dice = 1 - ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean()
        tversky = 1 - ((tp + eps) / (tp + cfg.tversky_alpha * fp + cfg.tversky_beta * fn + eps)).mean()
        result = (1 - cfg.omega) * .5 * (ce + focal) + cfg.omega * (.6 * dice + .4 * tversky)
        # Entirely ignored targets contribute zero, including the overlap terms.
        return result * valid.any().to(result.dtype)


def masked_binary_ce(logits, target, valid):
    ce = F.cross_entropy(logits.float(), target.long(), reduction='none')
    return (ce * valid).sum() / valid.sum().clamp_min(1)


def balanced_presence_bce(logits, target, valid):
    """No variance-maximization term: all-negative patches may ALL have low scores."""
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
    positive = target.bool() & valid
    negative = ~target.bool() & valid
    terms = []
    if positive.any():
        terms.append(loss[positive].mean())
    if negative.any():
        terms.append(loss[negative].mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


class AMALLoss(nn.Module):
    def __init__(self, model_config: AMALConfig, config: LossConfig | None = None):
        super().__init__()
        self.model_config = model_config
        self.config = config or LossConfig()
        self.pixel = PixelLoss(self.config)

    def forward(self, output, target):
        cfg = self.config
        valid = validate_target(target, self.model_config.num_classes, cfg.ignore_index)
        terms = {'segmentation': self.pixel(output['stage2_final'], target)}
        total = terms['segmentation']
        foreground = membership(target, self.model_config.foreground_ids)
        aux = []
        for key, label in (('stage1_fg', foreground), ('stage1_bg', ~foreground)):
            if key in output:
                aux.append(masked_binary_ce(output[key], label, valid))
        if aux:
            terms['binary_priors'] = torch.stack(aux).mean()
            total = total + cfg.auxiliary_weight * terms['binary_priors']
        for i, logits in enumerate(output.get('deep_supervision', [])):
            if i >= len(cfg.deep_weights):
                raise ValueError('deep_weights must cover every deep-supervision head')
            terms[f'deep_{i}'] = self.pixel(logits, target)
            total = total + cfg.deep_weights[i] * terms[f'deep_{i}']
        if 'global_logits' in output:
            terms['global'] = self.pixel(output['global_logits'], target)
            total = total + cfg.global_weight * terms['global']
        if 'selected_patch_logits' in output and cfg.local_weight:
            target_grid = label_patches(target, output['grid_meta'], cfg.ignore_index)
            selected_target = gather_patches(target_grid, output['selected_indices'])
            logits = output['selected_patch_logits']
            terms['local'] = self.pixel(logits.flatten(0, 1), selected_target.flatten(0, 1))
            total = total + cfg.local_weight * terms['local']
        if 'importance_logit' in output and cfg.score_weight:
            patch_target, patch_valid = presence_targets(target, output['grid_meta'],
                                                         self.model_config.lesion_ids, cfg.ignore_index)
            terms['patch_presence'] = balanced_presence_bce(output['importance_logit'], patch_target, patch_valid)
            total = total + cfg.score_weight * terms['patch_presence']
        terms['total'] = total
        return terms


class StandardSegLoss(nn.Module):
    """Common protocol loss for UCTransNet / nnU-Net architecture-only runs."""
    def __init__(self, config=None, deep_supervision=True):
        super().__init__()
        self.pixel = PixelLoss(config)
        self.deep_supervision = deep_supervision

    def forward(self, output, target):
        logits = output['stage2_final'] if isinstance(output, dict) else output
        terms = {'segmentation': self.pixel(logits, target)}
        total = terms['segmentation']
        deep = output.get('deep_supervision', []) if isinstance(output, dict) else []
        if self.deep_supervision and deep:
            # All active output heads receive gradients. This common protocol is NOT
            # the official nnU-Net trainer's complete training recipe.
            weights = [1 / (2 ** (i + 1)) for i in range(len(deep))]
            for i, (p, weight) in enumerate(zip(deep, weights)):
                terms[f'deep_{i}'] = self.pixel(p, target)
                total = total + weight * terms[f'deep_{i}']
            total = total / (1 + sum(weights))
        terms['total'] = total
        return terms
