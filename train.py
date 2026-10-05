#!/usr/bin/env python3
"""CGFENet-v2: joint prediction of left-ventricular pressure and wall displacement.

Pressure and displacement are two projections of one set of time-varying
generalized coordinates q(t).  Mesh identity enters the temporal branch through
exactly one channel -- cross-attention over the graph-fused node features -- so
there is no second, redundant copy of a pooled global-mesh vector anywhere else.

    Mesh graph
       |
       v
    GraphFusionEncoder (3 residual GATv2 blocks + one global fusion)
       +-- graph-fused node features H --------------------------+
                                                                 |
    volume / stiffness / Tmax --> Fourier --> physiology p --+    |
    time --> circular phase Fourier -------------------------+    |
                                                             |    |
                          Query Q(t) = MLP(phase(t), p)      |    |
                                                             |    |
                        Cross-attention: Q attends to H <---------+
                                                             |
                        Z(t) = LayerNorm(Q(t) + Cross(t))
                                                             |
            Pointwise residual MLP (no mixing between time tokens)
                     |                    |                   |
                     v                    v                   v
              Pressure head         Coordinate head       Mode head
                     |                    |                   |
                     v                    v                   v
             softplus pressure      q_1..q_K(t)        phi_k(n): [N, K+1, 3]
                     |                    |            (time-independent)
       q_0(t) = stopgrad(pressure) / scale |                   |
                     +--------------------+                   |
                              q(t) = [q_0, q_1, ..., q_K]      |
                                          |                    |
                                          v                    v
                        Modal synthesis:  u(n, t) = sum_k q_k(t) * phi_k(n)

The time representation is strictly periodic and every time point is refined
independently, so the learned trajectory is a continuous function of phase and
physiological conditions rather than of sequence length or sampling order.
This is physically motivated: in a periodic steady-state cardiac cycle,
pressure is a function of phase, geometry and physiology alone.

Training writes exactly two files into --output-dir:

    best_model.pt    self-contained checkpoint (weights + model config +
                     normalization statistics + geometry split)
    loss_curve.png   training / validation loss and learning-rate schedule

Progress and validation checks go to stdout.  See README.md.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"

ARCHITECTURE_NAME = "CGFENet-v2"
HIDDEN_CHANNELS = 128
DROPOUT_RATE = 0.1
GAT_HEADS = 4
# Volume, stiffness and Tmax: the per-case physiological conditioning.
NUM_PHYSIOLOGICAL_PARAMS = 3
# Checkpoint selection runs on the smoothed validation loss, not the raw value;
# the name is recorded in logs and checkpoints, so it must say which one.
BEST_MODEL_METRIC = "val_loss_median"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Colour-blind-safe Okabe-Ito palette, shared by the figures of both scripts.
NATURE_BLUE = "#0072B2"
NATURE_VERMILLION = "#D55E00"
NATURE_GREY = "#4D4D4D"


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    data = parser.add_argument_group("data")
    data.add_argument("--cache-path", type=Path, default=CACHE_DIR / "dataset_cache.pt",
                      help="The dataset cache to train on; the only input this script reads.")
    data.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    run = parser.add_argument_group("run")
    run.add_argument("--device", type=str, default=str(DEVICE),
                     help="Torch device, for example cuda:0 or cpu.")
    run.add_argument("--seed", type=int, default=42,
                     help="Training seed (initialization, dropout and loader order).")
    run.add_argument("--split-seed", type=int, default=42,
                     help="Frozen geometry-split seed; keep fixed when comparing training seeds.")
    run.add_argument("--epochs", type=int, default=800)
    run.add_argument("--batch-size", type=int, default=32)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--lr", type=float, default=1e-3)
    optim.add_argument("--min-lr", type=float, default=1e-6)
    optim.add_argument("--lr-factor", type=float, default=0.5)
    optim.add_argument("--lr-patience", type=int, default=5,
                       help="Unimproved validation checks before ReduceLROnPlateau lowers the LR.")
    optim.add_argument("--lr-threshold-rel", type=float, default=0.005,
                       help="Relative improvement required by the LR scheduler (0.005 = 0.5%%).")
    optim.add_argument("--weight-decay", type=float, default=1e-3)
    optim.add_argument("--max-grad-norm", type=float, default=2.0)

    sched = parser.add_argument_group("validation and early stopping")
    sched.add_argument("--val-every", type=int, default=5)
    sched.add_argument("--val-smoothing-window", type=int, default=3,
                       help="Odd running-median window used by LR scheduling and early stopping.")
    sched.add_argument("--plot-every", type=int, default=10)
    sched.add_argument("--early-stop-patience", type=int, default=21,
                       help="Validation checks without a meaningful improvement before stopping.")
    sched.add_argument("--early-stop-min-delta-rel", type=float, default=0.005,
                       help="Relative validation-loss improvement required to reset early stopping.")
    sched.add_argument("--early-stop-warmup-validations", type=int, default=10,
                       help="Validation checks observed before early stopping may fire.")
    sched.add_argument("--early-stop-lr-grace-validations", type=int, default=3,
                       help="Validation checks guaranteed after an LR reduction before stopping.")

    arch = parser.add_argument_group("architecture")
    arch.add_argument("--phase-fourier-modes", type=int, default=8)
    arch.add_argument("--param-fourier-modes", type=int, default=4)
    arch.add_argument("--cross-attention-heads", type=int, default=4)
    arch.add_argument("--pointwise-layers", type=int, default=2,
                      help="Depth of the pointwise temporal refiner. The default matches the "
                           "two-layer temporal encoder of the previous architecture, so that "
                           "removing temporal mixing is the only change between them.")
    arch.add_argument("--num-modes", type=int, default=6,
                      help="Number of correction modes q_1..q_K (q_0 is the reused pressure coordinate).")

    loss = parser.add_argument_group("losses and units")
    loss.add_argument("--pressure-loss-weight", type=float, default=1.0)
    loss.add_argument("--pressure-periodic-weight", type=float, default=0.05)
    loss.add_argument("--pressure-peak-window-weight", type=float, default=0.15)
    loss.add_argument("--pressure-tail-weight", type=float, default=0.15)
    loss.add_argument("--pressure-tail-start", type=float, default=600.0)
    loss.add_argument("--pressure-peak-sigma", type=float, default=14.0)
    loss.add_argument("--disp-loss-weight", type=float, default=10000.0,
                      help="Weight on the displacement MSE term (displacement normalized by coord_std).")
    loss.add_argument("--disp-time-gap", type=int, default=1,
                      help="Temporal subsampling stride for cached displacement targets.")
    loss.add_argument("--mode-ortho-weight", type=float, default=0.01,
                      help="Weight on the spatial-mode orthogonality regularizer.")
    loss.add_argument("--normalize-pressure", action="store_true", default=False)
    loss.add_argument("--pressure-scale", type=float, default=300.0)
    loss.add_argument("--time-period", type=float, default=800.0)
    loss.add_argument("--train-ratio", type=float, default=0.75)
    loss.add_argument("--val-ratio", type=float, default=0.125)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch size must be at least 1")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.lr <= 0.0 or args.min_lr <= 0.0 or args.min_lr > args.lr:
        raise ValueError("learning rates must satisfy 0 < min_lr <= lr")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("lr_factor must be in (0, 1)")
    if args.val_every < 1:
        raise ValueError("val_every must be at least 1")
    if args.val_smoothing_window < 1 or args.val_smoothing_window % 2 == 0:
        raise ValueError("val_smoothing_window must be a positive odd integer")
    if args.plot_every < 1:
        raise ValueError("plot_every must be at least 1")
    if args.pointwise_layers < 1:
        raise ValueError("pointwise_layers must be at least 1")
    if args.lr_patience < 0:
        raise ValueError("lr_patience must be non-negative")
    if not 0.0 <= args.lr_threshold_rel < 1.0:
        raise ValueError("lr_threshold_rel must be in [0, 1)")
    if args.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be at least 1")
    if not 0.0 <= args.early_stop_min_delta_rel < 1.0:
        raise ValueError("early_stop_min_delta_rel must be in [0, 1)")
    if args.early_stop_warmup_validations < 0 or args.early_stop_lr_grace_validations < 0:
        raise ValueError("early-stopping warmup and LR grace must be non-negative")
    if args.cross_attention_heads < 1 or HIDDEN_CHANNELS % args.cross_attention_heads != 0:
        raise ValueError(f"cross_attention_heads must divide hidden size {HIDDEN_CHANNELS}")
    if args.phase_fourier_modes < 1 or args.param_fourier_modes < 0:
        raise ValueError("phase_fourier_modes must be positive and param_fourier_modes non-negative")
    if args.num_modes < 0 or args.disp_time_gap < 1:
        raise ValueError("num_modes must be non-negative and disp_time_gap positive")
    if args.pressure_scale <= 0.0 or args.time_period <= 0.0:
        raise ValueError("pressure_scale and time_period must be positive")
    if not 0.0 < args.train_ratio < 1.0 or not 0.0 < args.val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be in (0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    nonnegative = (
        args.weight_decay,
        args.pressure_loss_weight,
        args.pressure_periodic_weight,
        args.pressure_peak_window_weight,
        args.pressure_tail_weight,
        args.mode_ortho_weight,
        args.disp_loss_weight,
    )
    if any(value < 0.0 for value in nonnegative) or args.max_grad_norm <= 0.0:
        raise ValueError("loss weights and weight decay must be non-negative; "
                         "max_grad_norm must be positive")


def jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}


def setup_logger() -> logging.Logger:
    """Log to stdout only, so that --output-dir holds nothing but the two results."""
    logger = logging.getLogger(ARCHITECTURE_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Dataset
#
# One "case" is one finite-element simulation directory, named
# <geometry>_<passive stiffness>_<Tmax>, e.g. 015_150_60k.  A case holds many
# "samples": each sample is one full cardiac cycle simulated at a fixed
# end-diastolic volume, so the volume label selects the sequence and pressure
# and displacement vary along it.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CaseMeta:
    case_name: str
    geo_id: str
    stiff: float
    tmax: float
    vtu_path: Path
    h5_path: Path


@dataclass(frozen=True)
class SampleMeta:
    case_name: str
    geo_id: str
    stiff: float
    tmax: float
    sample_index: int
    target_volume: float


@dataclass
class NormalizationStats:
    coord_mean: np.ndarray
    coord_std: np.ndarray
    volume_mean: float
    volume_std: float
    pressure_scale: float
    stiff_mean: float
    stiff_std: float
    tmax_mean: float
    tmax_std: float

    def as_dict(self) -> Dict[str, object]:
        """Plain-Python view, so a checkpoint stays readable without this module."""
        return {
            "coord_mean": np.asarray(self.coord_mean, dtype=np.float64).tolist(),
            "coord_std": np.asarray(self.coord_std, dtype=np.float64).tolist(),
            "volume_mean": float(self.volume_mean),
            "volume_std": float(self.volume_std),
            "pressure_scale": float(self.pressure_scale),
            "stiff_mean": float(self.stiff_mean),
            "stiff_std": float(self.stiff_std),
            "tmax_mean": float(self.tmax_mean),
            "tmax_std": float(self.tmax_std),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "NormalizationStats":
        return cls(
            coord_mean=np.asarray(payload["coord_mean"], dtype=np.float32),
            coord_std=np.asarray(payload["coord_std"], dtype=np.float32),
            volume_mean=float(payload["volume_mean"]),
            volume_std=float(payload["volume_std"]),
            pressure_scale=float(payload["pressure_scale"]),
            stiff_mean=float(payload["stiff_mean"]),
            stiff_std=float(payload["stiff_std"]),
            tmax_mean=float(payload["tmax_mean"]),
            tmax_std=float(payload["tmax_std"]),
        )


class GeometryCache:
    """Placeholder for the mesh reader that built the cache.

    The cache is a pickle, and a pickle names the classes it was written from.
    This class, CaseMeta and SampleMeta exist so those names still resolve when
    the cache is loaded; their instances carry only inert bookkeeping, and no
    method here touches a file.
    """


class MeshSplitSequenceDataset(Dataset):
    """One cardiac cycle of one case at one fixed volume, per item.

    Instances are never constructed here: they are restored from the cache,
    which already holds every graph, conditioning tensor, pressure target and
    displacement target (the last as fp16). That is the whole point of the
    cache -- a training run performs no file access at all after loading it.

    Each item is a dict with `graph`, `dynamic` [T, 3], `pressure_target` [T],
    `disp_target` [N, T, 3], `time_values` [T] and `volume_values` [T].
    """

    def __len__(self) -> int:
        return len(self.preprocessed)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.preprocessed[idx]
        if "disp_target" not in item:
            raise RuntimeError(
                f"Cached sample {idx} carries no displacement target. This cache was "
                "written for lazy loading and cannot be trained on offline."
            )
        return item


def concatenate_displacement(targets: Sequence[torch.Tensor]) -> torch.Tensor:
    """Join per-graph displacement targets along the node axis.

    This moves ~70 MB per mini-batch and is the single most expensive step of an
    iteration, so it goes through numpy rather than torch.cat: torch parallelizes
    the copy across its whole intra-op thread pool, which on a memory-bound copy
    only adds synchronization -- measured at 640 ms with 18 threads against 25 ms
    for the plain memcpy, which is thread-count independent.  Both views share
    the same buffers, so no data is copied twice.
    """
    return torch.from_numpy(np.concatenate([target.numpy() for target in targets], axis=0))


def collate_sequences(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "graph_batch": Batch.from_data_list([item["graph"] for item in batch]),
        "dynamic": torch.stack([item["dynamic"] for item in batch], dim=0),
        "pressure_target": torch.stack([item["pressure_target"] for item in batch], dim=0),
        # Node counts differ per graph, so displacement is concatenated along the
        # node axis, in the same order as graph_batch.
        "disp_target": concatenate_displacement([item["disp_target"] for item in batch]),
        "time_values": torch.stack([item["time_values"] for item in batch], dim=0),
        "volume_values": torch.stack([item["volume_values"] for item in batch], dim=0),
        "case_name": [str(item["case_name"]) for item in batch],
        "geo_id": [str(item["geo_id"]) for item in batch],
        "sample_index": [int(item["sample_index"]) for item in batch],
        "target_volume": [float(item["target_volume"]) for item in batch],
    }


def make_loader(dataset: MeshSplitSequenceDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    # The whole dataset already lives in memory, so a single-process loader has
    # nothing to prefetch and page-locking would only add a serialized copy of
    # the batch before a transfer that is not overlapped with anything.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_sequences,
    )


# --------------------------------------------------------------------------- #
# Dataset cache
#
# Building the dataset from the raw HDF5 files is by far the slowest part of a
# run, so the training and validation splits are cached.  The cache carries a
# signature of every setting that changes its contents; a mismatch forces a
# rebuild.
# --------------------------------------------------------------------------- #

DatasetBundle = Tuple[
    NormalizationStats,
    MeshSplitSequenceDataset,
    MeshSplitSequenceDataset,
    Dict[str, List[str]],
]

# Settings that change what the cache contains.  They are compared against the
# signature stored inside it, so a run cannot silently train on a dataset built
# under different assumptions.  Older caches spell the split seed "seed"; the
# meaning is unchanged.
SIGNATURE_FIELDS = {
    "train_ratio": ("train_ratio",),
    "val_ratio": ("val_ratio",),
    "split_seed": ("split_seed", "seed"),
    "normalize_pressure": ("normalize_pressure",),
    "pressure_scale": ("pressure_scale",),
    "time_period": ("time_period",),
}


def signature_mismatch(signature: object, args: argparse.Namespace) -> Optional[str]:
    """The first disagreement between a cache and these settings, or None."""
    if not isinstance(signature, dict):
        return "the cache carries no configuration signature"
    for attribute, names in SIGNATURE_FIELDS.items():
        expected = getattr(args, attribute)
        stored = next((signature[name] for name in names if name in signature), None)
        if stored is None:
            return f"the cache does not record {attribute}"
        if isinstance(expected, float):
            if not isinstance(stored, (int, float)) or not math.isclose(float(stored), expected):
                return f"{attribute}: cache has {stored}, this run wants {expected}"
        elif stored != expected:
            return f"{attribute}: cache has {stored!r}, this run wants {expected!r}"
    return None


def load_dataset(args: argparse.Namespace, logger: logging.Logger) -> DatasetBundle:
    """Restore the training and validation splits from the cache.

    This is the only input the script has. Nothing else is read, before or
    after; see README.md for how the cache is produced.
    """
    cache_path = args.cache_path.expanduser().resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"No dataset cache at {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")

    missing = {"stats", "train_dataset", "val_dataset", "splits"} - set(cache)
    if missing:
        raise RuntimeError(f"Cache at {cache_path} is missing {', '.join(sorted(missing))}")
    mismatch = signature_mismatch(cache.get("config_signature"), args)
    if mismatch is not None:
        raise RuntimeError(f"Cache at {cache_path} does not match this run -- {mismatch}")

    logger.info(f"Loaded dataset cache from {cache_path}")
    return cache["stats"], cache["train_dataset"], cache["val_dataset"], cache["splits"]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class ResidualGraphBlock(nn.Module):
    """Pre-norm GATv2 block with a residual connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = GraphNorm(in_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT_RATE)
        self.conv = GATv2Conv(
            in_channels,
            out_channels,
            heads=GAT_HEADS,
            concat=False,
            dropout=DROPOUT_RATE,
            add_self_loops=False,
        )
        self.proj = None if in_channels == out_channels else nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.conv(self.drop(self.act(self.norm(x, batch))), edge_index)
        residual = x if self.proj is None else self.proj(x)
        return residual + h


class GlobalFusionBlock(nn.Module):
    """Give every node the pooled context of the mesh it belongs to."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.local_proj = nn.Linear(feature_dim, feature_dim)
        self.global_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features
        global_feat = global_mean_pool(features, batch_idx)
        # Pooling per owning mesh keeps samples in a mini-batch independent.
        global_per_node = self.global_proj(global_feat)[batch_idx]
        return self.norm(self.local_proj(features) + global_per_node)


class GraphFusionEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.block1a = ResidualGraphBlock(in_channels, hidden)
        self.block1b = ResidualGraphBlock(hidden, hidden)
        self.block1c = ResidualGraphBlock(hidden, hidden)
        self.fusion1 = GlobalFusionBlock(hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        x = self.block1a(x, edge_index, batch_idx)
        x = self.block1b(x, edge_index, batch_idx)
        x = self.block1c(x, edge_index, batch_idx)
        return self.fusion1(x, batch_idx)


class FourierEmbedding(nn.Module):
    """Octave-spaced Fourier features for unbounded scalars, plus the raw value."""

    def __init__(self, num_bands: int) -> None:
        super().__init__()
        self.num_bands = int(max(0, num_bands))
        freq = 2.0 ** torch.arange(self.num_bands, dtype=torch.float32) * math.pi
        self.register_buffer("freq", freq, persistent=False)

    @property
    def out_dim(self) -> int:
        return 1 + 2 * self.num_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        if self.num_bands == 0:
            return x
        angle = x * self.freq.view(*([1] * (x.dim() - 1)), -1)
        return torch.cat([x, torch.sin(angle), torch.cos(angle)], dim=-1)


class CircularFourierEmbedding(nn.Module):
    """Strictly periodic phase embedding: phase 0 and phase 1 are identical."""

    def __init__(self, num_modes: int) -> None:
        super().__init__()
        self.num_modes = int(max(0, num_modes))
        freq = 2.0 * math.pi * torch.arange(1, self.num_modes + 1, dtype=torch.float32)
        self.register_buffer("freq", freq, persistent=False)

    @property
    def out_dim(self) -> int:
        return 2 * self.num_modes

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        phase = phase.unsqueeze(-1) if phase.dim() == 1 else phase
        if self.num_modes == 0:
            return phase.new_empty(*phase.shape[:-1], 0)
        angle = phase * self.freq.view(*([1] * (phase.dim() - 1)), -1)
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class PointwiseResidualBlock(nn.Module):
    """Transformer-style FFN with no communication between time points."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(DROPOUT_RATE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear layers act on the last dimension only, so the output at time t
        # cannot depend on any input token at time s != t.
        return x + self.ffn(self.norm(x))


class CGFENet(nn.Module):
    """Cardiac Geometry-Fused Encoder network.

    forward() returns
        pressure  [B, T]            left-ventricular pressure, mmHg
        disp      [sum(N), T, 3]    nodal displacement, coord_std-normalized
        q         [B, T, K+1]       generalized coordinates
        phi       [sum(N), K+1, 3]  spatial modes
        ortho     scalar            mode-Gram penalty
    """

    def __init__(
        self,
        node_in: int,
        hidden: int = HIDDEN_CHANNELS,
        time_period: float = 800.0,
        phase_fourier_modes: int = 8,
        param_fourier_modes: int = 4,
        cross_attention_heads: int = 4,
        pointwise_layers: int = 2,
        num_modes: int = 6,
        pressure_scale: float = 300.0,
        normalize_pressure: bool = False,
    ) -> None:
        super().__init__()
        if phase_fourier_modes < 1:
            raise ValueError("CGFENet requires at least one circular phase Fourier mode")
        if pointwise_layers < 1:
            raise ValueError("pointwise_layers must be at least 1")

        self.time_period = float(time_period)
        self.num_modes = int(max(0, num_modes))
        # q_0, the pressure coordinate, is rescaled to O(1) so that it shares the
        # magnitude of the learned corrections q_1..q_K.
        self.q0_scale = 1.0 if normalize_pressure else (1.0 / float(pressure_scale) if pressure_scale > 0 else 1.0)

        self.encoder = GraphFusionEncoder(node_in, hidden)
        self.phase_embedding = CircularFourierEmbedding(phase_fourier_modes)
        # One stateless embedding shared by volume, stiffness and Tmax: it holds
        # no learnable weights, so three copies would be three identical buffers.
        self.param_embedding = FourierEmbedding(param_fourier_modes)

        self.param_encoder = nn.Sequential(
            nn.Linear(NUM_PHYSIOLOGICAL_PARAMS * self.param_embedding.out_dim, hidden),
            nn.SiLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden + self.phase_embedding.out_dim, hidden),
            nn.SiLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden, hidden),
        )
        self.query_attn = nn.MultiheadAttention(
            hidden, cross_attention_heads, dropout=DROPOUT_RATE, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden)
        self.pointwise_refiner = nn.Sequential(
            *(PointwiseResidualBlock(hidden) for _ in range(pointwise_layers))
        )
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(DROPOUT_RATE), nn.Linear(hidden, 1)
        )
        # Spatial modes phi_k(n): per node, time-independent, (K+1) modes x 3 components.
        self.mode_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden, (self.num_modes + 1) * 3),
        )
        # Correction coordinates q_1..q_K(t): from the temporal context, linear (may be negative).
        self.coord_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden, self.num_modes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.param_encoder, self.query_mlp, self.pressure_head,
                       self.mode_head, self.coord_head):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.75)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    @staticmethod
    def _physiology_columns(num_channels: int) -> Tuple[int, int, int]:
        if num_channels == 3:
            return 0, 1, 2                       # native layout [volume, stiffness, Tmax]
        if num_channels >= 5:
            return 0, 3, 4                       # legacy cache [volume, sin t, cos t, stiffness, Tmax]
        raise ValueError(f"Expected 3 or at least 5 dynamic feature channels, got {num_channels}")

    def _query_tokens(self, param_context: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Q(t): a function of phase and physiology only.

        Mesh identity enters the model exactly once, through the cross-attention
        that consumes these tokens.
        """
        phase_embed = self.phase_embedding(torch.remainder(phase, 1.0).unsqueeze(-1))
        context = param_context.unsqueeze(1).expand(-1, phase_embed.size(1), -1)
        return self.query_mlp(torch.cat([phase_embed, context], dim=-1))

    def _mesh_memory(self, node_features: torch.Tensor, ptr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad the per-graph node features into an attention memory plus its mask."""
        segments = [node_features[int(ptr[i]):int(ptr[i + 1])] for i in range(ptr.numel() - 1)]
        memory = pad_sequence(segments, batch_first=True)                      # [B, max_N, H]
        lengths = ptr[1:] - ptr[:-1]
        positions = torch.arange(memory.size(1), device=memory.device).unsqueeze(0)
        return memory, positions >= lengths.unsqueeze(1)

    def forward(
        self,
        graph_batch: Batch,
        dynamic_feats: torch.Tensor,
        time_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_idx = graph_batch.batch
        ptr = graph_batch.ptr
        node_features = self.encoder(graph_batch.x, graph_batch.edge_index, batch_idx)

        # Physiological parameters are constant over time; read them at t = 0.
        # Shape [B, P, 1] keeps the embedding's per-scalar broadcast, so the
        # flattened result equals a per-parameter embed-then-concatenate.
        columns = self._physiology_columns(dynamic_feats.size(-1))
        physiology = dynamic_feats[:, 0, columns].unsqueeze(-1)
        param_context = self.param_encoder(self.param_embedding(physiology).flatten(1))

        phase = torch.remainder(time_values, self.time_period) / max(self.time_period, 1e-6)
        tokens = self._query_tokens(param_context, phase)

        # Batched attention is mathematically equivalent to processing each graph
        # separately; the padding mask prevents any cross-graph interaction.
        memory, memory_padding_mask = self._mesh_memory(node_features, ptr)
        attended, _ = self.query_attn(
            tokens, memory, memory, key_padding_mask=memory_padding_mask, need_weights=False
        )

        # Standard post-norm residual cross-attention block: the query plus what
        # it read from the mesh, normalized. No separate global-context shortcut.
        refined = self.pointwise_refiner(self.fusion_norm(tokens + attended))
        pressure = F.softplus(self.pressure_head(refined).squeeze(-1))          # [B, T]

        # --- modal displacement -------------------------------------------- #
        phi = self.mode_head(node_features).view(-1, self.num_modes + 1, 3)      # [sum(N), K+1, 3]
        q0 = (pressure.detach() * self.q0_scale).unsqueeze(-1)                   # [B, T, 1]
        q_full = torch.cat([q0, self.coord_head(refined)], dim=-1)               # [B, T, K+1]

        disp_per_graph: List[torch.Tensor] = []
        ortho_accum = torch.zeros((), device=node_features.device)
        eye = torch.eye(self.num_modes + 1, device=node_features.device)
        for graph_idx in range(graph_batch.num_graphs):
            phi_g = phi[int(ptr[graph_idx]):int(ptr[graph_idx + 1])]             # [N_g, K+1, 3]
            # u(n, t) = sum_k q_k(t) * phi_k(n)
            disp_per_graph.append(torch.einsum("tk,nkd->ntd", q_full[graph_idx], phi_g))
            # Gram matrix of the unit-normalized mode fields: keeps modes distinct.
            v = F.normalize(phi_g.permute(1, 0, 2).reshape(self.num_modes + 1, -1), dim=1)
            ortho_accum = ortho_accum + (v @ v.t() - eye).pow(2).sum() / max(
                self.num_modes * (self.num_modes + 1), 1
            )

        return {
            "pressure": pressure,
            "disp": torch.cat(disp_per_graph, dim=0),                            # [sum(N), T, 3]
            "ortho": ortho_accum / max(graph_batch.num_graphs, 1),
            # q and phi allow compact modal-contribution diagnostics without a
            # second forward pass.
            "q": q_full,
            "phi": phi,
        }


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def mse_peak_tail_pressure_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    time_values: torch.Tensor,
    peak_window_weight: float,
    tail_weight: float,
    peak_sigma: float,
    tail_start: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full-cycle MSE plus two clinically weighted windows.

    The peak window is a Gaussian centred on the ground-truth pressure peak, so
    systolic accuracy is not averaged away by the long diastolic tail; the tail
    term protects late filling, where absolute pressures are small.
    """
    mse = F.mse_loss(pred, target)

    steps = torch.arange(pred.size(1), device=pred.device, dtype=pred.dtype).unsqueeze(0)
    sigma = max(float(peak_sigma), 1e-3)
    peak_idx = target.argmax(dim=1).unsqueeze(1).to(pred.dtype)
    peak_weights = torch.exp(-0.5 * ((steps - peak_idx) / sigma).pow(2))
    peak_weights = peak_weights / (peak_weights.sum(dim=1, keepdim=True) + 1e-8)
    peak_window = torch.mean(torch.sum(peak_weights * (pred - target).pow(2), dim=1))

    tail_mask = (time_values >= float(tail_start)).to(pred.dtype)
    tail_weights = tail_mask / tail_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    tail = torch.mean(torch.sum(tail_weights * (pred - target).pow(2), dim=1))

    total = mse + float(peak_window_weight) * peak_window + float(tail_weight) * tail
    return total, {"mse": mse, "peak_window": peak_window, "tail": tail}


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred = outputs["pressure"]
    target = batch["pressure_target"].to(pred.device)
    time_values = batch["time_values"].to(pred.device)

    pressure_loss, parts = mse_peak_tail_pressure_loss(
        pred,
        target,
        time_values,
        peak_window_weight=args.pressure_peak_window_weight,
        tail_weight=args.pressure_tail_weight,
        peak_sigma=args.pressure_peak_sigma,
        tail_start=args.pressure_tail_start,
    )
    # A periodic steady-state cycle must start where it ends.
    periodic_loss = F.mse_loss(pred[:, :1], pred[:, -1:])
    total_loss = (args.pressure_loss_weight * pressure_loss
                  + args.pressure_periodic_weight * periodic_loss)

    disp_pred = outputs["disp"]
    disp_target = batch["disp_target"].to(disp_pred.device).float()          # fp16 cache -> fp32
    if disp_target.size(1) != disp_pred.size(1):
        stride = max(1, int(args.disp_time_gap))
        disp_pred = disp_pred[:, ::stride, :]
        if disp_pred.size(1) != disp_target.size(1):
            raise RuntimeError(
                "Displacement cache/time-gap mismatch: prediction has "
                f"{disp_pred.size(1)} frames after stride {stride}, "
                f"target has {disp_target.size(1)}"
            )
    disp_loss = F.mse_loss(disp_pred, disp_target)
    total_loss = total_loss + args.disp_loss_weight * disp_loss
    if args.mode_ortho_weight > 0.0:
        total_loss = total_loss + args.mode_ortho_weight * outputs["ortho"]

    with torch.no_grad():
        scale = float(args.pressure_scale) if args.normalize_pressure else 1.0
        pressure_rmse = torch.sqrt(torch.mean(((pred - target) * scale).pow(2), dim=1) + 1e-12).mean()
        pred_center = pred - pred.mean(dim=1, keepdim=True)
        target_center = target - target.mean(dim=1, keepdim=True)
        denom = torch.sqrt(torch.mean(pred_center.pow(2), dim=1)
                           * torch.mean(target_center.pow(2), dim=1) + 1e-12)
        corr = torch.where(denom > 1e-8,
                           torch.mean(pred_center * target_center, dim=1) / denom,
                           torch.zeros_like(denom))
        ss_res = torch.sum((pred - target).pow(2), dim=1)
        ss_tot = torch.sum(target_center.pow(2), dim=1)
        r2 = torch.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, torch.zeros_like(ss_tot))

    metrics = {
        "loss": float(total_loss.item()),
        "pressure_loss": float(pressure_loss.item()),
        "pressure_periodic_loss": float(periodic_loss.item()),
        "pressure_mse": float(parts["mse"].item()),
        "pressure_peak_window_loss": float(parts["peak_window"].item()),
        "pressure_tail_loss": float(parts["tail"].item()),
        "pressure_rmse": float(pressure_rmse.item()),
        "pressure_r": float(corr.mean().item()),
        "pressure_r2": float(r2.mean().item()),
        "disp_loss": float(disp_loss.item()),
        "disp_rmse": float(torch.sqrt(disp_loss + 1e-12).item()),
        "ortho": float(outputs["ortho"].item()),
    }
    return total_loss, metrics


# --------------------------------------------------------------------------- #
# Model selection and early stopping
#
# Single-check validation loss on this task is volatile: consecutive checks
# differ by tens of percent and isolated spikes exceed the surrounding level
# several-fold.  Every progress decision therefore runs on a running median of
# the last few validation losses, which rejects spikes without the lag of an
# exponential average.  The raw loss is still logged and plotted.
# --------------------------------------------------------------------------- #

def running_median(values: Sequence[float], window: int) -> float:
    if not values:
        raise ValueError("running_median requires at least one value")
    if window < 1:
        raise ValueError("running_median window must be positive")
    return float(statistics.median(float(value) for value in values[-window:]))


@dataclass
class EarlyStoppingState:
    """Decides when progress has stalled.

    A reset requires a relative improvement large enough to be scientifically
    meaningful, so a run of tiny improvements cannot keep patience alive
    indefinitely.  The numerically lowest validation loss is checkpointed
    separately, independent of this state.
    """

    validations_seen: int = 0
    bad_validations: int = 0
    reference_metric: float = math.inf
    lr_grace_remaining: int = 0
    lr_reductions: int = 0

    def update(
        self,
        metric: float,
        *,
        lr_reduced: bool,
        min_delta_rel: float,
        patience: int,
        warmup_validations: int,
        lr_grace_validations: int,
    ) -> Tuple[bool, bool]:
        if not math.isfinite(metric):
            raise ValueError(f"Early-stopping metric must be finite, got {metric}")
        if not 0.0 <= min_delta_rel < 1.0:
            raise ValueError("early_stop_min_delta_rel must be in [0, 1)")
        if patience < 1:
            raise ValueError("early_stop_patience must be at least 1")

        self.validations_seen += 1
        required_delta = abs(self.reference_metric) * min_delta_rel
        meaningful_improvement = (
            not math.isfinite(self.reference_metric)
            or metric < self.reference_metric - max(required_delta, 1e-12)
        )
        if meaningful_improvement:
            self.reference_metric = metric
            self.bad_validations = 0
        else:
            self.bad_validations += 1

        if lr_reduced:
            self.lr_reductions += 1
            self.lr_grace_remaining = max(0, int(lr_grace_validations))
        elif self.lr_grace_remaining > 0:
            self.lr_grace_remaining -= 1

        should_stop = (
            self.validations_seen >= max(0, int(warmup_validations))
            and self.bad_validations >= patience
            and self.lr_grace_remaining == 0
        )
        return meaningful_improvement, should_stop

    def state_dict(self) -> Dict[str, object]:
        return {
            "validations_seen": self.validations_seen,
            "bad_validations": self.bad_validations,
            "reference_metric": self.reference_metric,
            "lr_grace_remaining": self.lr_grace_remaining,
            "lr_reductions": self.lr_reductions,
        }


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #

def mean_metrics(collected: Dict[str, List[float]]) -> Dict[str, float]:
    return {key: float(sum(values) / max(1, len(values))) for key, values in collected.items()}


def train_one_epoch(
    model: CGFENet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, float]:
    model.train()
    collected: Dict[str, List[float]] = defaultdict(list)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["graph_batch"].to(DEVICE),
            batch["dynamic"].to(DEVICE),
            batch["time_values"].to(DEVICE),
        )
        total_loss, metrics = compute_losses(outputs, batch, args)
        if not torch.isfinite(total_loss):
            logger.warning("Encountered non-finite loss; skipping batch")
            continue
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        for key, value in metrics.items():
            collected[key].append(value)
    return mean_metrics(collected)


@torch.no_grad()
def evaluate(model: CGFENet, loader: DataLoader, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    collected: Dict[str, List[float]] = defaultdict(list)
    for batch in loader:
        outputs = model(
            batch["graph_batch"].to(DEVICE),
            batch["dynamic"].to(DEVICE),
            batch["time_values"].to(DEVICE),
        )
        _, metrics = compute_losses(outputs, batch, args)
        for key, value in metrics.items():
            collected[key].append(value)
    return mean_metrics(collected)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

def plot_loss_curve(history: Dict[str, List[float]], out_path: Path, best_epoch: int) -> None:
    """The training loss, with the learning-rate schedule underneath it.

    Validation drives checkpoint selection, the scheduler and early stopping,
    but it is not drawn: it is measured every few epochs and is volatile enough
    that plotting it obscures the curve it sits next to. The epoch whose
    checkpoint was kept is marked instead.
    """
    fig, (ax_loss, ax_lr) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    train_loss = np.asarray(history["train_loss"], dtype=np.float64)
    epochs = np.arange(1, train_loss.size + 1)
    ax_loss.plot(epochs, train_loss, color=NATURE_BLUE, linewidth=1.7, label="Training loss")
    if 1 <= best_epoch <= train_loss.size:
        ax_loss.axvline(best_epoch, color=NATURE_GREY, linestyle="--", linewidth=1.0,
                        alpha=0.8, label=f"Selected checkpoint (epoch {best_epoch})")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.25)

    learning_rate = np.asarray(history["lr"], dtype=np.float64)
    if learning_rate.size:
        ax_lr.plot(np.arange(1, learning_rate.size + 1), learning_rate,
                   color=NATURE_GREY, label="Learning rate")
        ax_lr.set_yscale("log")
        ax_lr.legend()
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: CGFENet,
    model_config: Dict[str, object],
    stats: NormalizationStats,
    splits: Dict[str, List[str]],
    args: argparse.Namespace,
    *,
    epoch: int,
    best_metric: float,
    validation_metrics: Dict[str, float],
) -> None:
    """Write the single, self-contained model file.

    Everything inference needs travels with the weights: the constructor
    arguments, the normalization statistics computed on the training split and
    the frozen geometry split.  example/example.py needs nothing else.
    """
    torch.save(
        {
            "architecture": ARCHITECTURE_NAME,
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "model_config": model_config,
            "normalization": stats.as_dict(),
            "split": splits,
            "best_metric": float(best_metric),
            "best_metric_name": BEST_MODEL_METRIC,
            "selection_policy": "minimum running-median validation loss",
            "validation_metrics": validation_metrics,
            "args": jsonable_args(args),
        },
        path,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {name} requested, but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )
    return device


def main() -> int:
    global DEVICE
    args = parse_args()
    validate_args(args)
    DEVICE = resolve_device(args.device)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger()
    set_random_seed(args.seed)
    logger.info(f"{ARCHITECTURE_NAME} | device={DEVICE} | "
                f"train_seed={args.seed} | split_seed={args.split_seed}")

    stats, train_dataset, val_dataset, splits = load_dataset(args, logger)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False)
    logger.info(
        f"Geometries | train={len(splits['train'])} val={len(splits['val'])} | "
        f"held out and unused={len(splits['test'])} | "
        f"Sequences | train={len(train_dataset)} val={len(val_dataset)}"
    )

    node_features_dim = int(next(iter(train_loader))["graph_batch"].num_node_features)
    model_config = {
        "node_in": node_features_dim,
        "hidden": HIDDEN_CHANNELS,
        "time_period": float(args.time_period),
        "phase_fourier_modes": int(args.phase_fourier_modes),
        "param_fourier_modes": int(args.param_fourier_modes),
        "cross_attention_heads": int(args.cross_attention_heads),
        "pointwise_layers": int(args.pointwise_layers),
        "num_modes": int(args.num_modes),
        "pressure_scale": float(args.pressure_scale),
        "normalize_pressure": bool(args.normalize_pressure),
    }
    model = CGFENet(**model_config).to(DEVICE)
    logger.info(f"Parameters | {sum(p.numel() for p in model.parameters()):,}")
    logger.info(
        f"Schedule | lr={args.lr:g} -> min {args.min_lr:g} (factor {args.lr_factor:g}, "
        f"patience {args.lr_patience} checks) | val_every={args.val_every} | "
        f"val_smoothing=running-median-{args.val_smoothing_window} | "
        f"early_stop_patience={args.early_stop_patience} checks at "
        f"{args.early_stop_min_delta_rel * 100:g}% relative improvement | "
        f"best_metric={BEST_MODEL_METRIC}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold_rel,
        threshold_mode="rel",
        min_lr=args.min_lr,
    )

    best_path = args.output_dir / "best_model.pt"
    curve_path = args.output_dir / "loss_curve.png"
    # Only what the figure draws.  The validation losses live in raw_val_losses,
    # which is what the running median and every progress decision read.
    history: Dict[str, List[float]] = {"train_loss": [], "lr": []}
    raw_val_losses: List[float] = []
    early_stopping = EarlyStoppingState()
    best_metric = math.inf
    best_epoch = -1
    epoch = 0
    should_stop = False

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, args, logger)
        history["train_loss"].append(train_metrics["loss"])
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if epoch % args.val_every != 0 and epoch != args.epochs:
            logger.info(
                f"Epoch {epoch:03d} | train={train_metrics['loss']:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.3e} | "
                f"P_RMSE={train_metrics['pressure_rmse']:.4f} mmHg | "
                f"disp_rmse={train_metrics['disp_rmse']:.5f}"
            )
            continue

        val_metrics = evaluate(model, val_loader, args)
        raw_val_losses.append(val_metrics["loss"])
        smoothed_val = running_median(raw_val_losses, args.val_smoothing_window)

        previous_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(smoothed_val)
        current_lr = float(optimizer.param_groups[0]["lr"])
        lr_reduced = current_lr < previous_lr - max(abs(previous_lr) * 1e-12, 1e-15)
        history["lr"][-1] = current_lr

        meaningful_improvement, should_stop = early_stopping.update(
            smoothed_val,
            lr_reduced=lr_reduced,
            min_delta_rel=args.early_stop_min_delta_rel,
            patience=args.early_stop_patience,
            warmup_validations=args.early_stop_warmup_validations,
            lr_grace_validations=args.early_stop_lr_grace_validations,
        )
        logger.info(
            f"Epoch {epoch:03d} | train={train_metrics['loss']:.6f} | "
            f"val={val_metrics['loss']:.6f} | "
            f"val median-{args.val_smoothing_window}={smoothed_val:.6f} | "
            f"lr={current_lr:.3e} | P_RMSE={val_metrics['pressure_rmse']:.4f} mmHg | "
            f"disp_rmse={val_metrics['disp_rmse']:.5f} | "
            f"improved={'yes' if meaningful_improvement else 'no'} | "
            f"bad={early_stopping.bad_validations}/{args.early_stop_patience}"
            + (f" | lr reduced {previous_lr:.3e} -> {current_lr:.3e}" if lr_reduced else "")
        )

        if smoothed_val < best_metric:
            best_metric, best_epoch = smoothed_val, epoch
            save_checkpoint(
                best_path, model, model_config, stats, splits, args,
                epoch=epoch, best_metric=best_metric, validation_metrics=val_metrics,
            )
            logger.info(f"Saved {best_path.name} | {BEST_MODEL_METRIC}={best_metric:.6f}")

        if epoch % args.plot_every == 0 or should_stop or epoch == args.epochs:
            plot_loss_curve(history, curve_path, best_epoch)

        if should_stop:
            logger.info(
                f"Early stopping at epoch {epoch} after {early_stopping.bad_validations} "
                f"validation checks without a >={args.early_stop_min_delta_rel * 100:.3g}% "
                "relative improvement"
            )
            break

    plot_loss_curve(history, curve_path, best_epoch)
    if best_epoch < 0:
        raise RuntimeError("Training finished without a validation check; no model was selected")

    logger.info(f"Best epoch {best_epoch} | {BEST_MODEL_METRIC}={best_metric:.6f}")
    logger.info(f"Wrote {best_path} and {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
