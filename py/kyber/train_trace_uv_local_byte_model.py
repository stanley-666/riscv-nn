#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _configure_runtime_env() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        ld_parts = [part for part in current_ld.split(":") if part]
        if conda_lib not in ld_parts:
            os.environ["LD_LIBRARY_PATH"] = (
                f"{conda_lib}:{current_ld}" if current_ld else conda_lib
            )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


_configure_runtime_env()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from pipeline.poi_detection import DEFAULT_TRACE_START, segment_mean_trace

DATASETS = ROOT / "datasets"
CONFIGS = ROOT / "configs"
UV_DIR = ROOT / "final_cts_4regions"
REGIONS = ["zero_left", "one_left", "one_right", "zero_right"]
TRACE_STATS_CACHE = DATASETS / "all_40000_cts_trace_stats.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a local 3-byte-to-1-byte model from power trace + unpacked Kyber u/v."
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "trace_uv_local_byte_model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="none", help="optional learning-rate schedule")
    parser.add_argument("--min-lr", type=float, default=0.0, help="minimum lr used by cosine scheduler")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--margin-loss-weight", type=float, default=0.0, help="weight for additional logit margin loss; 0 disables it")
    parser.add_argument("--margin-loss-value", type=float, default=0.2, help="target margin between true-class logit and hardest negative logit")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-messages", type=int, default=32000)
    parser.add_argument("--val-messages", type=int, default=8000)
    parser.add_argument("--max-messages", type=int, default=40000)
    parser.add_argument(
        "--trace-file-prefix",
        default="",
        help="prefix for 4-region trace files, e.g. 'new_kyber_' loads new_kyber_zero_left_cts_trace_10000.npy",
    )
    parser.add_argument(
        "--trace-stats-cache",
        type=Path,
        default=None,
        help="optional cache path for per-trace mean/std; defaults depend on trace-file-prefix",
    )
    parser.add_argument("--segmentation-source-tag", default="01")
    parser.add_argument("--segmentation-config-path", type=Path, default=CONFIGS / "fix_trace_config.json")
    parser.add_argument("--trace-window-len", type=int, default=2400)
    parser.add_argument("--train-trace-shift-max", type=int, default=0, help="uniform random train-time shift sampled from [-k, +k] samples")
    parser.add_argument("--mask-mode", choices=("hard", "gaussian"), default="hard")
    parser.add_argument("--mask-sigma", type=float, default=10.0, help="sigma in samples for gaussian mask")
    parser.add_argument("--mask-expand-samples", type=int, default=0, help="expand hard-mask region by this many samples on both sides")
    parser.add_argument("--context-bytes", type=int, default=1, help="left/right byte context count for local u/v; 1 means 3-byte uv context")
    parser.add_argument("--trace-channels", type=int, default=32)
    parser.add_argument("--trace-extra-conv-layers", type=int, default=0, help="extra stride-1 conv blocks appended after the base trace encoder")
    parser.add_argument("--trace-extra-conv-kernel", type=int, default=5, help="kernel size for extra stride-1 trace conv blocks")
    parser.add_argument("--fusion-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--head-layernorm", type=int, default=1)
    parser.add_argument("--use-attention", type=int, default=1)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--use-trace", type=int, default=1)
    parser.add_argument("--use-uv", type=int, default=1)
    parser.add_argument("--use-peak-positions", type=int, default=0)
    parser.add_argument("--use-peak-window-features", type=int, default=0)
    parser.add_argument("--peak-window-radius", type=int, default=8, help="radius in samples around each bit POI for local mean-abs-amplitude features")
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_prefix_name(prefix: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in prefix)
    return safe.strip("_")


def load_trace_memmaps(trace_file_prefix: str = ""):
    memmaps = []
    for region in REGIONS:
        path = DATASETS / f"{trace_file_prefix}{region}_cts_trace_10000.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing trace file: {path}")
        memmaps.append(np.load(path, mmap_mode="r"))
    return memmaps


def load_muv_bundle(max_messages: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = np.load(UV_DIR / "all_40000_muv_bundle.npy", allow_pickle=True).item()
    m = np.asarray(bundle["m"][:max_messages], dtype=np.uint8)
    u = np.asarray(bundle["u"][:max_messages], dtype=np.float32) / 3329.0
    v = np.asarray(bundle["v"][:max_messages], dtype=np.float32) / 3329.0
    return m, u, v


def _trace_row_from_memmaps(trace_memmaps, global_idx: int) -> np.ndarray:
    region_idx = int(global_idx // 10000)
    local_idx = int(global_idx % 10000)
    return np.asarray(trace_memmaps[region_idx][local_idx], dtype=np.float32)


def _default_trace_stats_cache(trace_file_prefix: str) -> Path:
    prefix_name = _safe_prefix_name(trace_file_prefix)
    if not prefix_name:
        return TRACE_STATS_CACHE
    return DATASETS / f"{prefix_name}_all_40000_cts_trace_stats.npz"


def compute_or_load_trace_stats(
    trace_memmaps,
    max_messages: int,
    trace_file_prefix: str = "",
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_cache = Path(cache_path) if cache_path is not None else _default_trace_stats_cache(trace_file_prefix)
    if resolved_cache.exists():
        cached = np.load(resolved_cache)
        mean = np.asarray(cached["mean"][:max_messages], dtype=np.float32)
        std = np.asarray(cached["std"][:max_messages], dtype=np.float32)
        if len(mean) >= max_messages and len(std) >= max_messages:
            return mean, std

    mean_parts = []
    std_parts = []
    for arr in trace_memmaps:
        mean_parts.append(np.asarray(arr.mean(axis=1), dtype=np.float32))
        std_parts.append(np.asarray(arr.std(axis=1), dtype=np.float32))
    mean = np.concatenate(mean_parts, axis=0)
    std = np.concatenate(std_parts, axis=0)
    std = np.maximum(std, 1e-6)
    resolved_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(resolved_cache, mean=mean, std=std)
    return mean[:max_messages], std[:max_messages]


def load_fix_trace_config(segmentation_source_tag: str, config_path: Path | None = None) -> dict:
    resolved_path = Path(config_path) if config_path is not None else (CONFIGS / "fix_trace_config.json")
    with open(resolved_path, "r", encoding="ascii") as f:
        config_by_tag = json.load(f)
    return config_by_tag.get(segmentation_source_tag, {})


def build_segmentation_metadata(segmentation_source_tag: str, config_path: Path | None = None) -> dict:
    config = load_fix_trace_config(segmentation_source_tag, config_path=config_path)
    trace_source_tag = str(config.get("trace_source_tag", config.get("source_tag", segmentation_source_tag)))
    if "trace_path" in config:
        trace_path = Path(config["trace_path"])
        if not trace_path.is_absolute():
            trace_path = ROOT / trace_path
    else:
        trace_path = DATASETS / f"fix_trace_{trace_source_tag}.npy"
    if not trace_path.exists():
        raise FileNotFoundError(f"missing segmentation source trace: {trace_path}")
    fix_trace = np.load(trace_path)
    mean_trace = fix_trace.mean(axis=0)
    manual_main_boundaries = config.get("main_boundaries")
    segments = segment_mean_trace(
        mean_trace,
        main_count=32,
        sub_count=8,
        trace_start=int(config.get("trace_start", DEFAULT_TRACE_START)),
        manual_main_boundaries=manual_main_boundaries,
        enable_edge_boundary_refinement=True,
    )
    return {
        "main_regions": [(int(start), int(end)) for start, end in segments["main_regions"][:32]],
        "sub_peaks_per_region": [
            [int(x) for x in np.asarray(region_peaks, dtype=int).tolist()]
            for region_peaks in segments["sub_peaks_per_region"][:32]
        ],
    }


def build_main_regions(segmentation_source_tag: str, config_path: Path | None = None) -> list[tuple[int, int]]:
    return build_segmentation_metadata(segmentation_source_tag, config_path=config_path)["main_regions"]


def split_indices(total: int, train_messages: int, val_messages: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    train_n = min(train_messages, total)
    val_n = min(val_messages, max(0, total - train_n))
    train_idx = np.sort(perm[:train_n])
    if val_n > 0:
        val_idx = np.sort(perm[train_n : train_n + val_n])
    else:
        val_idx = train_idx.copy()
    return train_idx, val_idx


class LocalByteDataset(Dataset):
    def __init__(
        self,
        trace_memmaps,
        trace_mean: np.ndarray,
        trace_std: np.ndarray,
        m_bytes: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        message_indices: np.ndarray,
        main_regions: list[tuple[int, int]],
        trace_window_len: int,
        context_bytes: int,
        trace_shift_samples: int = 0,
        random_shift_max: int = 0,
        mask_mode: str = "hard",
        mask_sigma: float = 10.0,
        mask_expand_samples: int = 0,
        sub_peaks_per_region: list[list[int]] | None = None,
        peak_window_radius: int = 8,
    ):
        self.trace_memmaps = trace_memmaps
        self.trace_mean = trace_mean
        self.trace_std = trace_std
        self.m_bytes = m_bytes
        self.u = u
        self.v = v
        self.message_indices = np.asarray(message_indices, dtype=np.int64)
        self.main_regions = list(main_regions)
        self.trace_window_len = int(trace_window_len)
        self.context_bytes = int(context_bytes)
        self.trace_shift_samples = int(trace_shift_samples)
        self.random_shift_max = max(0, int(random_shift_max))
        self.mask_mode = str(mask_mode).strip().lower()
        self.mask_sigma = float(mask_sigma)
        self.mask_expand_samples = max(0, int(mask_expand_samples))
        self.peak_window_radius = max(0, int(peak_window_radius))
        self.sub_peaks_per_region = [
            [int(x) for x in region_peaks]
            for region_peaks in (sub_peaks_per_region or [[] for _ in self.main_regions])
        ]
        self.bytes_per_message = len(self.main_regions)
        self.num_slots = 2 * self.context_bytes + 1
        self.trace_input_len = self.trace_window_len
        self.coeffs_per_byte = 8
        self.uv_slot_coeffs = self.coeffs_per_byte * self.num_slots
        self.trace_len = int(self.trace_memmaps[0].shape[1])

        if self.trace_input_len <= 0:
            raise ValueError(f"trace_window_len must be positive, got {self.trace_input_len}")
        if len(self.sub_peaks_per_region) != self.bytes_per_message:
            raise ValueError("sub_peaks_per_region must match main_regions length")

        self.uv_neighbor_bytes = np.full((self.bytes_per_message, self.num_slots), -1, dtype=np.int32)
        self._precompute_byte_metadata()

    def _window_bounds_for_byte(
        self,
        raw_trace_len: int,
        byte_idx: int,
        shift_samples: int | None = None,
    ) -> tuple[int, int]:
        start, end = self.main_regions[byte_idx]
        center = int(round((start + end) / 2.0))
        half = self.trace_input_len // 2
        applied_shift = self.trace_shift_samples if shift_samples is None else int(shift_samples)
        window_start = center - half + applied_shift
        window_end = window_start + self.trace_input_len

        if window_start < 0:
            window_start = 0
            window_end = min(raw_trace_len, self.trace_input_len)
        if window_end > raw_trace_len:
            window_end = raw_trace_len
            window_start = max(0, window_end - self.trace_input_len)
        return int(window_start), int(window_end)

    def _target_mask_for_byte(self, byte_idx: int, window_start: int) -> np.ndarray:
        target_start, target_end = self.main_regions[byte_idx]
        expanded_start = int(target_start - self.mask_expand_samples)
        expanded_end = int(target_end + self.mask_expand_samples)
        local_start = max(0, int(expanded_start - window_start))
        local_end = min(self.trace_input_len, int(expanded_end - window_start))
        if self.mask_mode == "hard":
            target_mask = np.zeros((self.trace_input_len,), dtype=np.float32)
            if local_end > local_start:
                target_mask[local_start:local_end] = 1.0
            return target_mask
        if self.mask_mode == "gaussian":
            center = ((float(target_start) + float(target_end)) * 0.5) - float(window_start)
            sigma = max(1e-3, float(self.mask_sigma))
            xs = np.arange(self.trace_input_len, dtype=np.float32)
            target_mask = np.exp(-((xs - float(center)) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
            peak = float(target_mask.max())
            if peak > 0.0:
                target_mask /= peak
            return target_mask
        raise ValueError(f"unsupported mask_mode: {self.mask_mode}")

    def _peak_positions_for_byte(self, byte_idx: int, window_start: int) -> np.ndarray:
        peaks = self.sub_peaks_per_region[byte_idx]
        peak_pos = np.zeros((8,), dtype=np.float32)
        if not peaks:
            return peak_pos
        for idx, peak in enumerate(peaks[:8]):
            local_peak = float(peak - window_start)
            local_peak = min(max(local_peak, 0.0), float(max(self.trace_input_len - 1, 0)))
            peak_pos[idx] = local_peak / float(max(self.trace_input_len - 1, 1))
        return peak_pos

    def _peak_window_features_for_byte(self, byte_idx: int, trace_local: np.ndarray, window_start: int) -> np.ndarray:
        peaks = self.sub_peaks_per_region[byte_idx]
        feats = np.zeros((8,), dtype=np.float32)
        if not peaks:
            return feats
        radius = int(self.peak_window_radius)
        trace_len = int(trace_local.shape[0])
        for idx, peak in enumerate(peaks[:8]):
            local_peak = int(peak - window_start)
            left = max(0, local_peak - radius)
            right = min(trace_len, local_peak + radius + 1)
            if right <= left:
                continue
            window = trace_local[left:right]
            feats[idx] = float(np.mean(np.abs(window)))
        return feats

    def _precompute_byte_metadata(self) -> None:
        for byte_idx in range(self.bytes_per_message):
            for slot_idx, neighbor_byte_idx in enumerate(
                range(byte_idx - self.context_bytes, byte_idx + self.context_bytes + 1)
            ):
                if 0 <= neighbor_byte_idx < self.bytes_per_message:
                    self.uv_neighbor_bytes[byte_idx, slot_idx] = int(neighbor_byte_idx)

    def __len__(self) -> int:
        return int(len(self.message_indices) * self.bytes_per_message)

    def __getitem__(self, idx: int):
        msg_pos = int(idx // self.bytes_per_message)
        byte_idx = int(idx % self.bytes_per_message)
        global_msg_idx = int(self.message_indices[msg_pos])

        raw_trace = _trace_row_from_memmaps(self.trace_memmaps, global_msg_idx)
        trace_mean = float(self.trace_mean[global_msg_idx])
        trace_std = float(self.trace_std[global_msg_idx])

        effective_shift = int(self.trace_shift_samples)
        if self.random_shift_max > 0:
            effective_shift += int(torch.randint(-self.random_shift_max, self.random_shift_max + 1, (1,)).item())

        window_start, window_end = self._window_bounds_for_byte(
            self.trace_len,
            byte_idx,
            shift_samples=effective_shift,
        )
        trace_x = np.asarray(raw_trace[window_start:window_end], dtype=np.float32)
        trace_x = ((trace_x - trace_mean) / trace_std).astype(np.float32, copy=False)
        target_mask = self._target_mask_for_byte(byte_idx, window_start)
        peak_pos = self._peak_positions_for_byte(byte_idx, window_start)
        peak_window_feat = self._peak_window_features_for_byte(byte_idx, trace_x, window_start)
        trace_x = np.stack([trace_x, target_mask], axis=0)
        local_u = np.zeros((2, self.uv_slot_coeffs), dtype=np.float32)
        local_v = np.zeros((self.uv_slot_coeffs,), dtype=np.float32)

        for slot_idx, neighbor_byte_idx in enumerate(self.uv_neighbor_bytes[byte_idx]):
            if int(neighbor_byte_idx) < 0:
                continue

            coeff_start = int(neighbor_byte_idx) * self.coeffs_per_byte
            coeff_end = coeff_start + self.coeffs_per_byte
            uv_base = slot_idx * self.coeffs_per_byte
            local_u[:, uv_base : uv_base + self.coeffs_per_byte] = self.u[global_msg_idx, :, coeff_start:coeff_end]
            local_v[uv_base : uv_base + self.coeffs_per_byte] = self.v[global_msg_idx, coeff_start:coeff_end]

        target = np.int64(self.m_bytes[global_msg_idx, byte_idx])
        return (
            torch.from_numpy(trace_x),
            torch.from_numpy(local_u),
            torch.from_numpy(local_v).unsqueeze(0),
            torch.from_numpy(peak_pos),
            torch.from_numpy(peak_window_feat),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(byte_idx, dtype=torch.long),
            torch.tensor(global_msg_idx, dtype=torch.long),
        )


class TraceEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        use_attention: bool = True,
        attn_heads: int = 4,
        in_channels: int = 2,
        extra_conv_layers: int = 0,
        extra_conv_kernel: int = 5,
    ):
        super().__init__()
        feature_layers: list[nn.Module] = [
            nn.Conv1d(in_channels, channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(channels * 4),
            nn.GELU(),
            nn.Conv1d(channels * 4, channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(channels * 4),
            nn.GELU(),
        ]
        extra_kernel = max(1, int(extra_conv_kernel))
        extra_padding = extra_kernel // 2
        for _ in range(max(0, int(extra_conv_layers))):
            feature_layers.extend(
                [
                    nn.Conv1d(channels * 4, channels * 4, kernel_size=extra_kernel, stride=1, padding=extra_padding),
                    nn.BatchNorm1d(channels * 4),
                    nn.GELU(),
                ]
            )
        self.features = nn.Sequential(*feature_layers)
        self.use_attention = bool(use_attention)
        embed_dim = channels * 4
        if self.use_attention:
            if embed_dim % attn_heads != 0:
                raise ValueError(f"trace embed_dim={embed_dim} must be divisible by attn_heads={attn_heads}")
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=attn_heads, dropout=0.1, batch_first=True)
            self.attn_norm = nn.LayerNorm(embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.out_dim = channels * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.use_attention:
            seq = x.transpose(1, 2)
            attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
            seq = self.attn_norm(seq + attn_out)
            x = seq.transpose(1, 2)
        avg = self.avg_pool(x).flatten(1)
        mx = self.max_pool(x).flatten(1)
        return torch.cat([avg, mx], dim=1)


class UVEncoder(nn.Module):
    def __init__(self, num_slots: int, fusion_dim: int = 128):
        super().__init__()
        input_dim = num_slots * 8 * 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, fusion_dim),
            nn.GELU(),
        )
        self.out_dim = fusion_dim

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        uv = torch.cat([u.flatten(1), v.flatten(1)], dim=1)
        return self.net(uv)


class LocalByteModel(nn.Module):
    def __init__(
        self,
        num_slots: int,
        trace_channels: int = 32,
        fusion_dim: int = 128,
        head_hidden_dim: int = 0,
        head_dropout: float = 0.2,
        head_layernorm: bool = False,
        use_attention: bool = True,
        attn_heads: int = 4,
        use_trace: bool = True,
        use_uv: bool = True,
        use_byte_position: bool = True,
        use_peak_positions: bool = False,
        use_peak_window_features: bool = False,
        trace_extra_conv_layers: int = 0,
        trace_extra_conv_kernel: int = 5,
    ):
        super().__init__()
        self.use_trace = bool(use_trace)
        self.use_uv = bool(use_uv)
        self.use_byte_position = bool(use_byte_position)
        self.use_peak_positions = bool(use_peak_positions)
        self.use_peak_window_features = bool(use_peak_window_features)
        if not self.use_trace and not self.use_uv:
            raise ValueError("at least one of use_trace or use_uv must be enabled")

        feat_dim = 0
        if self.use_trace:
            self.trace_encoder = TraceEncoder(
                channels=trace_channels,
                use_attention=use_attention,
                attn_heads=attn_heads,
                in_channels=2,
                extra_conv_layers=trace_extra_conv_layers,
                extra_conv_kernel=trace_extra_conv_kernel,
            )
            feat_dim += self.trace_encoder.out_dim
        if self.use_uv:
            self.uv_encoder = UVEncoder(num_slots=num_slots, fusion_dim=fusion_dim)
            feat_dim += self.uv_encoder.out_dim

        if self.use_byte_position:
            feat_dim += 1
        if self.use_peak_positions:
            feat_dim += 8
        if self.use_peak_window_features:
            feat_dim += 8

        layers: list[nn.Module] = []
        if bool(head_layernorm):
            layers.append(nn.LayerNorm(feat_dim))
        if int(head_hidden_dim) > 0:
            layers.extend(
                [
                    nn.Linear(feat_dim, int(head_hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(head_dropout)),
                    nn.Linear(int(head_hidden_dim), 256),
                ]
            )
        else:
            # Legacy single-layer head for backward compatibility with old checkpoints.
            layers.append(nn.Linear(feat_dim, 256))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        trace_x: torch.Tensor | None,
        u: torch.Tensor | None,
        v: torch.Tensor | None,
        byte_pos: torch.Tensor | None = None,
        peak_pos: torch.Tensor | None = None,
        peak_window_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = []
        if self.use_trace:
            assert trace_x is not None
            feats.append(self.trace_encoder(trace_x))
        if self.use_uv:
            assert u is not None and v is not None
            feats.append(self.uv_encoder(u, v))
        if self.use_byte_position:
            assert byte_pos is not None
            feats.append(byte_pos)
        if self.use_peak_positions:
            assert peak_pos is not None
            feats.append(peak_pos)
        if self.use_peak_window_features:
            assert peak_window_feat is not None
            feats.append(peak_window_feat)
        fused = torch.cat(feats, dim=1)
        return self.head(fused)


class ByteClassificationLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.05, margin_loss_weight: float = 0.0, margin_loss_value: float = 0.2):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.margin_loss_weight = float(margin_loss_weight)
        self.margin_loss_value = float(margin_loss_value)
        self.ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def _margin_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_mask = torch.zeros_like(logits, dtype=torch.bool)
        target_mask.scatter_(1, targets.unsqueeze(1), True)
        hardest_negative = logits.masked_fill(target_mask, float("-inf")).max(dim=1).values
        return F.relu(self.margin_loss_value - (target_logits - hardest_negative)).mean()

    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ce_loss = self.ce(logits, targets)
        if self.margin_loss_weight > 0.0:
            margin_loss = self._margin_loss(logits, targets)
        else:
            margin_loss = logits.new_zeros(())
        total_loss = ce_loss + (self.margin_loss_weight * margin_loss)
        return total_loss, {
            "ce_loss": ce_loss.detach(),
            "margin_loss": margin_loss.detach(),
            "loss": total_loss.detach(),
        }

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss, _ = self.compute(logits, targets)
        return total_loss


def byte_logits_to_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    pred = logits.argmax(dim=-1)
    byte_acc = float((pred == targets).float().mean().item())
    top2_acc = float((logits.topk(k=min(2, logits.shape[-1]), dim=-1).indices == targets.unsqueeze(1)).any(dim=1).float().mean().item())
    top5_acc = float((logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices == targets.unsqueeze(1)).any(dim=1).float().mean().item())
    probs = torch.softmax(logits, dim=-1)
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)).mean().item())
    pred_bits = np.unpackbits(pred.detach().cpu().numpy().astype(np.uint8).reshape(-1, 1), axis=1, bitorder="little")
    gt_bits = np.unpackbits(targets.detach().cpu().numpy().astype(np.uint8).reshape(-1, 1), axis=1, bitorder="little")
    bit_acc = float((pred_bits == gt_bits).mean())
    return {
        "bit_acc": bit_acc,
        "byte_acc": byte_acc,
        "top2_acc": top2_acc,
        "top5_acc": top5_acc,
        "mean_entropy": entropy,
    }


def reconstruct_message_metrics(logits: torch.Tensor, targets: torch.Tensor, byte_positions: torch.Tensor, message_indices: torch.Tensor) -> dict:
    pred = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.uint8)
    gt = targets.detach().cpu().numpy().astype(np.uint8)
    byte_positions_np = byte_positions.detach().cpu().numpy().astype(np.int64)
    message_indices_np = message_indices.detach().cpu().numpy().astype(np.int64)
    probs = torch.softmax(logits, dim=-1)
    entropy_np = (-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)).detach().cpu().numpy().astype(np.float64)
    top2_hit_np = (
        (logits.topk(k=min(2, logits.shape[-1]), dim=-1).indices == targets.unsqueeze(1))
        .any(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    top5_hit_np = (
        (logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices == targets.unsqueeze(1))
        .any(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    unique_msgs = np.unique(message_indices_np)
    pred_msg = np.zeros((len(unique_msgs), 32), dtype=np.uint8)
    gt_msg = np.zeros((len(unique_msgs), 32), dtype=np.uint8)
    msg_lookup = {msg_idx: pos for pos, msg_idx in enumerate(unique_msgs.tolist())}

    for sample_idx in range(len(pred)):
        row = msg_lookup[int(message_indices_np[sample_idx])]
        col = int(byte_positions_np[sample_idx])
        pred_msg[row, col] = pred[sample_idx]
        gt_msg[row, col] = gt[sample_idx]

    byte_match = pred_msg == gt_msg
    bit_match = (
        np.unpackbits(pred_msg, axis=1, bitorder="little")
        == np.unpackbits(gt_msg, axis=1, bitorder="little")
    )
    per_byte_entropy = []
    per_byte_top2 = []
    per_byte_top5 = []
    for byte_idx in range(32):
        mask = byte_positions_np == byte_idx
        if not np.any(mask):
            per_byte_entropy.append(float("nan"))
            per_byte_top2.append(float("nan"))
            per_byte_top5.append(float("nan"))
            continue
        per_byte_entropy.append(float(entropy_np[mask].mean()))
        per_byte_top2.append(float(top2_hit_np[mask].mean()))
        per_byte_top5.append(float(top5_hit_np[mask].mean()))
    return {
        "message_byte_acc": float(byte_match.mean()),
        "message_bit_acc": float(bit_match.mean()),
        "message_exact_match": float(byte_match.all(axis=1).mean()),
        "per_byte_acc": byte_match.mean(axis=0).astype(np.float64).tolist(),
        "per_byte_top2_acc": per_byte_top2,
        "per_byte_top5_acc": per_byte_top5,
        "per_byte_entropy": per_byte_entropy,
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_margin_loss = 0.0
    total_samples = 0
    use_non_blocking = device.type == "cuda"
    for trace_x, u, v, peak_pos, peak_window_feat, y, byte_pos, _ in loader:
        trace_x = trace_x.to(device, non_blocking=use_non_blocking)
        u = u.to(device, non_blocking=use_non_blocking)
        v = v.to(device, non_blocking=use_non_blocking)
        peak_pos = peak_pos.to(device, non_blocking=use_non_blocking)
        peak_window_feat = peak_window_feat.to(device, non_blocking=use_non_blocking)
        y = y.to(device, non_blocking=use_non_blocking)
        byte_pos = byte_pos.to(device, non_blocking=use_non_blocking).float().unsqueeze(1) / 31.0
        optimizer.zero_grad(set_to_none=True)
        logits = model(trace_x, u, v, byte_pos=byte_pos, peak_pos=peak_pos, peak_window_feat=peak_window_feat)
        loss, loss_parts = criterion.compute(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * trace_x.size(0)
        total_ce_loss += float(loss_parts["ce_loss"].item()) * trace_x.size(0)
        total_margin_loss += float(loss_parts["margin_loss"].item()) * trace_x.size(0)
        total_samples += trace_x.size(0)
    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "ce_loss": total_ce_loss / denom,
        "margin_loss": total_margin_loss / denom,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_margin_loss = 0.0
    total_samples = 0
    use_non_blocking = device.type == "cuda"
    logits_all = []
    y_all = []
    byte_positions_all = []
    message_indices_all = []
    for trace_x, u, v, peak_pos, peak_window_feat, y, byte_pos, msg_idx in loader:
        trace_x = trace_x.to(device, non_blocking=use_non_blocking)
        u = u.to(device, non_blocking=use_non_blocking)
        v = v.to(device, non_blocking=use_non_blocking)
        peak_pos = peak_pos.to(device, non_blocking=use_non_blocking)
        peak_window_feat = peak_window_feat.to(device, non_blocking=use_non_blocking)
        y = y.to(device, non_blocking=use_non_blocking)
        byte_pos_model = byte_pos.to(device, non_blocking=use_non_blocking).float().unsqueeze(1) / 31.0
        logits = model(trace_x, u, v, byte_pos=byte_pos_model, peak_pos=peak_pos, peak_window_feat=peak_window_feat)
        loss, loss_parts = criterion.compute(logits, y)
        total_loss += float(loss.item()) * trace_x.size(0)
        total_ce_loss += float(loss_parts["ce_loss"].item()) * trace_x.size(0)
        total_margin_loss += float(loss_parts["margin_loss"].item()) * trace_x.size(0)
        total_samples += trace_x.size(0)
        logits_all.append(logits.cpu())
        y_all.append(y.cpu())
        byte_positions_all.append(byte_pos.cpu())
        message_indices_all.append(msg_idx.cpu())

    logits_all_t = torch.cat(logits_all, dim=0)
    y_all_t = torch.cat(y_all, dim=0)
    byte_positions_t = torch.cat(byte_positions_all, dim=0)
    message_indices_t = torch.cat(message_indices_all, dim=0)
    metrics = byte_logits_to_metrics(logits_all_t, y_all_t)
    metrics.update(reconstruct_message_metrics(logits_all_t, y_all_t, byte_positions_t, message_indices_t))
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["ce_loss"] = total_ce_loss / max(total_samples, 1)
    metrics["margin_loss"] = total_margin_loss / max(total_samples, 1)
    return metrics, logits_all_t, y_all_t, byte_positions_t, message_indices_t


def save_training_plot(out_path: Path, history: list[dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_byte = [row["train_byte_acc"] for row in history]
    val_byte = [row["val_byte_acc"] for row in history]
    train_bit = [row["train_bit_acc"] for row in history]
    val_bit = [row["val_bit_acc"] for row in history]
    val_msg = [row["val_message_exact_match"] for row in history]

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(epochs, train_loss, label="train_loss", color="#2563eb", linewidth=1.8)
    axes[0].plot(epochs, val_loss, label="val_loss", color="#dc2626", linewidth=1.8)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend()

    axes[1].plot(epochs, train_byte, label="train_byte_acc", color="#1d4ed8", linewidth=1.8)
    axes[1].plot(epochs, val_byte, label="val_byte_acc", color="#dc2626", linewidth=1.8)
    axes[1].set_ylabel("Byte Acc")
    axes[1].set_title("Byte Accuracy")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend()

    axes[2].plot(epochs, train_bit, label="train_bit_acc", color="#0f766e", linewidth=1.8)
    axes[2].plot(epochs, val_bit, label="val_bit_acc", color="#ea580c", linewidth=1.8)
    axes[2].set_ylabel("Bit Acc")
    axes[2].set_title("Bit Accuracy")
    axes[2].grid(True, alpha=0.22)
    axes[2].legend()

    axes[3].plot(epochs, val_msg, label="val_message_exact", color="#7c3aed", linewidth=1.8)
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Exact")
    axes[3].set_title("Validation Message Exact Match")
    axes[3].grid(True, alpha=0.22)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_per_byte_plot(out_path: Path, per_byte_acc: list[float]) -> None:
    import matplotlib.pyplot as plt

    xs = np.arange(len(per_byte_acc), dtype=int)
    ys = np.asarray(per_byte_acc, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(xs, ys, color="#2563eb", linewidth=1.8)
    ax.set_xlabel("Byte Index")
    ax.set_ylabel("Validation Byte Acc")
    ax.set_xticks(np.arange(0, len(per_byte_acc), 1))
    ax.set_ylim(max(0.0, ys.min() - 0.03), 1.01)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    trace_memmaps = load_trace_memmaps(trace_file_prefix=str(args.trace_file_prefix))
    trace_stats_cache = Path(args.trace_stats_cache) if args.trace_stats_cache is not None else _default_trace_stats_cache(str(args.trace_file_prefix))
    trace_mean, trace_std = compute_or_load_trace_stats(
        trace_memmaps,
        args.max_messages,
        trace_file_prefix=str(args.trace_file_prefix),
        cache_path=trace_stats_cache,
    )
    m_bytes, u, v = load_muv_bundle(args.max_messages)
    if len(m_bytes) != args.max_messages:
        args.max_messages = len(m_bytes)
        trace_mean = trace_mean[:args.max_messages]
        trace_std = trace_std[:args.max_messages]
        u = u[:args.max_messages]
        v = v[:args.max_messages]
        m_bytes = m_bytes[:args.max_messages]

    segmentation_meta = build_segmentation_metadata(args.segmentation_source_tag, config_path=args.segmentation_config_path)
    main_regions = segmentation_meta["main_regions"]
    sub_peaks_per_region = segmentation_meta["sub_peaks_per_region"]
    train_idx, val_idx = split_indices(args.max_messages, args.train_messages, args.val_messages, args.seed)

    train_dataset = LocalByteDataset(
        trace_memmaps,
        trace_mean,
        trace_std,
        m_bytes,
        u,
        v,
        train_idx,
        main_regions,
        trace_window_len=args.trace_window_len,
        context_bytes=args.context_bytes,
        random_shift_max=args.train_trace_shift_max,
        mask_mode=args.mask_mode,
        mask_sigma=args.mask_sigma,
        mask_expand_samples=args.mask_expand_samples,
        sub_peaks_per_region=sub_peaks_per_region,
        peak_window_radius=args.peak_window_radius,
    )
    val_dataset = LocalByteDataset(
        trace_memmaps,
        trace_mean,
        trace_std,
        m_bytes,
        u,
        v,
        val_idx,
        main_regions,
        trace_window_len=args.trace_window_len,
        context_bytes=args.context_bytes,
        mask_mode=args.mask_mode,
        mask_sigma=args.mask_sigma,
        mask_expand_samples=args.mask_expand_samples,
        sub_peaks_per_region=sub_peaks_per_region,
        peak_window_radius=args.peak_window_radius,
    )

    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": pin_memory,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = LocalByteModel(
        num_slots=2 * args.context_bytes + 1,
        trace_channels=args.trace_channels,
        fusion_dim=args.fusion_dim,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        head_layernorm=bool(args.head_layernorm),
        use_attention=bool(args.use_attention),
        attn_heads=args.attn_heads,
        use_trace=bool(args.use_trace),
        use_uv=bool(args.use_uv),
        use_peak_positions=bool(args.use_peak_positions),
        use_peak_window_features=bool(args.use_peak_window_features),
        trace_extra_conv_layers=int(args.trace_extra_conv_layers),
        trace_extra_conv_kernel=int(args.trace_extra_conv_kernel),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
            eta_min=float(args.min_lr),
        )
    criterion = ByteClassificationLoss(
        label_smoothing=float(args.label_smoothing),
        margin_loss_weight=float(args.margin_loss_weight),
        margin_loss_value=float(args.margin_loss_value),
    )

    best_val_message_exact = -1.0
    best_state = None
    history: list[dict[str, float]] = []
    epoch_iter = range(1, args.epochs + 1)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, total=args.epochs, desc="trace_uv_local_byte", leave=True)

    for epoch in epoch_iter:
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics, *_ = evaluate(model, train_loader, criterion, device)
        val_metrics, *_ = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "lr": current_lr,
            "train_loss": float(train_loss["loss"]),
            "train_ce_loss": float(train_loss["ce_loss"]),
            "train_margin_loss": float(train_loss["margin_loss"]),
            "val_loss": float(val_metrics["loss"]),
            "val_ce_loss": float(val_metrics["ce_loss"]),
            "val_margin_loss": float(val_metrics["margin_loss"]),
            "train_bit_acc": float(train_metrics["bit_acc"]),
            "val_bit_acc": float(val_metrics["bit_acc"]),
            "train_byte_acc": float(train_metrics["byte_acc"]),
            "val_byte_acc": float(val_metrics["byte_acc"]),
            "train_message_exact_match": float(train_metrics["message_exact_match"]),
            "val_message_exact_match": float(val_metrics["message_exact_match"]),
        }
        history.append(row)
        if tqdm is not None:
            epoch_iter.set_postfix(
                train_byte=f"{train_metrics['byte_acc']:.4f}",
                val_byte=f"{val_metrics['byte_acc']:.4f}",
                val_msg=f"{val_metrics['message_exact_match']:.4f}",
                margin=f"{val_metrics['margin_loss']:.4f}",
                lr=f"{current_lr:.2e}",
            )
        if val_metrics["message_exact_match"] >= best_val_message_exact:
            best_val_message_exact = float(val_metrics["message_exact_match"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics, *_ = evaluate(model, train_loader, criterion, device)
    val_metrics, *_ = evaluate(model, val_loader, criterion, device)

    ckpt_path = args.out_dir / "trace_uv_local_byte_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "segmentation_source_tag": args.segmentation_source_tag,
                "trace_file_prefix": str(args.trace_file_prefix),
                "trace_stats_cache": str(trace_stats_cache),
                "lr": float(args.lr),
                "lr_scheduler": str(args.lr_scheduler),
                "min_lr": float(args.min_lr),
                "trace_window_len": int(args.trace_window_len),
                "context_bytes": int(args.context_bytes),
                "trace_channels": int(args.trace_channels),
                "trace_extra_conv_layers": int(args.trace_extra_conv_layers),
                "trace_extra_conv_kernel": int(args.trace_extra_conv_kernel),
                "fusion_dim": int(args.fusion_dim),
                "head_hidden_dim": int(args.head_hidden_dim),
                "head_dropout": float(args.head_dropout),
                "head_layernorm": bool(args.head_layernorm),
                "use_attention": bool(args.use_attention),
                "attn_heads": int(args.attn_heads),
                "use_trace": bool(args.use_trace),
                "use_uv": bool(args.use_uv),
                "use_peak_positions": bool(args.use_peak_positions),
                "use_peak_window_features": bool(args.use_peak_window_features),
                "use_byte_position": True,
                "label_smoothing": float(args.label_smoothing),
                "margin_loss_weight": float(args.margin_loss_weight),
                "margin_loss_value": float(args.margin_loss_value),
                "train_trace_shift_max": int(args.train_trace_shift_max),
                "mask_mode": str(args.mask_mode),
                "mask_sigma": float(args.mask_sigma),
                "mask_expand_samples": int(args.mask_expand_samples),
                "peak_window_radius": int(args.peak_window_radius),
                "seed": int(args.seed),
                "max_messages": int(args.max_messages),
                "train_messages": int(len(train_idx)),
                "val_messages": int(len(val_idx)),
            },
        },
        ckpt_path,
    )

    plot_path = args.out_dir / "trace_uv_local_byte_training.png"
    plot_error = ""
    try:
        save_training_plot(plot_path, history)
    except Exception as exc:  # plotting should never invalidate a finished training run
        plot_error = f"{type(exc).__name__}: {exc}"
        print(f"Warning: failed to save training plot: {plot_error}")

    per_byte_plot_path = args.out_dir / "trace_uv_local_byte_per_byte_acc.png"
    per_byte_plot_error = ""
    try:
        save_per_byte_plot(per_byte_plot_path, val_metrics["per_byte_acc"])
    except Exception as exc:  # plotting should never invalidate a finished training run
        per_byte_plot_error = f"{type(exc).__name__}: {exc}"
        print(f"Warning: failed to save per-byte plot: {per_byte_plot_error}")

    summary = {
        "train_messages": int(len(train_idx)),
        "val_messages": int(len(val_idx)),
        "device": str(device),
        "segmentation_source_tag": args.segmentation_source_tag,
        "trace_file_prefix": str(args.trace_file_prefix),
        "trace_stats_cache": str(trace_stats_cache),
        "lr": float(args.lr),
        "lr_scheduler": str(args.lr_scheduler),
        "min_lr": float(args.min_lr),
        "trace_window_len": int(args.trace_window_len),
        "context_bytes": int(args.context_bytes),
        "trace_input_len": int(args.trace_window_len),
        "use_trace": bool(args.use_trace),
        "use_uv": bool(args.use_uv),
        "use_peak_positions": bool(args.use_peak_positions),
        "use_peak_window_features": bool(args.use_peak_window_features),
        "label_smoothing": float(args.label_smoothing),
        "margin_loss_weight": float(args.margin_loss_weight),
        "margin_loss_value": float(args.margin_loss_value),
        "train_trace_shift_max": int(args.train_trace_shift_max),
        "mask_mode": str(args.mask_mode),
        "mask_sigma": float(args.mask_sigma),
        "mask_expand_samples": int(args.mask_expand_samples),
        "peak_window_radius": int(args.peak_window_radius),
        "seed": int(args.seed),
        "max_messages": int(args.max_messages),
        "trace_channels": int(args.trace_channels),
        "trace_extra_conv_layers": int(args.trace_extra_conv_layers),
        "trace_extra_conv_kernel": int(args.trace_extra_conv_kernel),
        "fusion_dim": int(args.fusion_dim),
        "head_hidden_dim": int(args.head_hidden_dim),
        "head_dropout": float(args.head_dropout),
        "head_layernorm": bool(args.head_layernorm),
        "best_val_message_exact": float(best_val_message_exact),
        "final_train": train_metrics,
        "final_val": val_metrics,
        "checkpoint_path": str(ckpt_path),
        "plot_path": str(plot_path),
        "plot_error": plot_error,
        "per_byte_plot_path": str(per_byte_plot_path),
        "per_byte_plot_error": per_byte_plot_error,
        "history": history,
    }

    out_json = args.out_dir / "trace_uv_local_byte_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="ascii")

    out_md = args.out_dir / "trace_uv_local_byte_summary.md"
    out_md.write_text(
        "\n".join(
            [
                "# Trace+UV Local Byte Model Summary",
                "",
                f"- train_messages: `{len(train_idx)}`",
                f"- val_messages: `{len(val_idx)}`",
                f"- segmentation_source_tag: `{args.segmentation_source_tag}`",
                f"- lr: `{float(args.lr):.6g}`",
                f"- lr_scheduler: `{str(args.lr_scheduler)}`",
                f"- min_lr: `{float(args.min_lr):.6g}`",
                f"- trace_window_len: `{args.trace_window_len}`",
                f"- context_bytes: `{args.context_bytes}`",
                f"- trace_input_len: `{args.trace_window_len}`",
                f"- use_trace: `{int(bool(args.use_trace))}`",
                f"- use_uv: `{int(bool(args.use_uv))}`",
                f"- use_peak_positions: `{int(bool(args.use_peak_positions))}`",
                f"- use_peak_window_features: `{int(bool(args.use_peak_window_features))}`",
                f"- label_smoothing: `{float(args.label_smoothing):.4f}`",
                f"- margin_loss_weight: `{float(args.margin_loss_weight):.4f}`",
                f"- margin_loss_value: `{float(args.margin_loss_value):.4f}`",
                f"- train_trace_shift_max: `{int(args.train_trace_shift_max)}`",
                f"- mask_mode: `{str(args.mask_mode)}`",
                f"- mask_sigma: `{float(args.mask_sigma):.2f}`",
                f"- mask_expand_samples: `{int(args.mask_expand_samples)}`",
                f"- peak_window_radius: `{int(args.peak_window_radius)}`",
                f"- head_hidden_dim: `{int(args.head_hidden_dim)}`",
                f"- trace_extra_conv_layers: `{int(args.trace_extra_conv_layers)}`",
                f"- trace_extra_conv_kernel: `{int(args.trace_extra_conv_kernel)}`",
                f"- head_dropout: `{float(args.head_dropout):.2f}`",
                f"- head_layernorm: `{int(bool(args.head_layernorm))}`",
                f"- final_train_bit_acc: `{train_metrics['bit_acc']:.4f}`",
                f"- final_train_byte_acc: `{train_metrics['byte_acc']:.4f}`",
                f"- final_train_top2_acc: `{train_metrics['top2_acc']:.4f}`",
                f"- final_train_top5_acc: `{train_metrics['top5_acc']:.4f}`",
                f"- final_train_entropy: `{train_metrics['mean_entropy']:.4f}`",
                f"- final_train_ce_loss: `{train_metrics['ce_loss']:.4f}`",
                f"- final_train_margin_loss: `{train_metrics['margin_loss']:.4f}`",
                f"- final_train_message_exact: `{train_metrics['message_exact_match']:.4f}`",
                f"- final_val_bit_acc: `{val_metrics['bit_acc']:.4f}`",
                f"- final_val_byte_acc: `{val_metrics['byte_acc']:.4f}`",
                f"- final_val_top2_acc: `{val_metrics['top2_acc']:.4f}`",
                f"- final_val_top5_acc: `{val_metrics['top5_acc']:.4f}`",
                f"- final_val_entropy: `{val_metrics['mean_entropy']:.4f}`",
                f"- final_val_ce_loss: `{val_metrics['ce_loss']:.4f}`",
                f"- final_val_margin_loss: `{val_metrics['margin_loss']:.4f}`",
                f"- final_val_message_exact: `{val_metrics['message_exact_match']:.4f}`",
                "",
                f"- checkpoint: `{ckpt_path}`",
                f"- training_plot: `{plot_path}`",
                f"- per_byte_plot: `{per_byte_plot_path}`",
            ]
        ),
        encoding="ascii",
    )

    print(
        "trace_uv_local_byte "
        f"train_bit_acc={train_metrics['bit_acc']:.4f} "
        f"train_byte_acc={train_metrics['byte_acc']:.4f} "
        f"train_top2_acc={train_metrics['top2_acc']:.4f} "
        f"train_top5_acc={train_metrics['top5_acc']:.4f} "
        f"train_entropy={train_metrics['mean_entropy']:.4f} "
        f"train_msg_exact={train_metrics['message_exact_match']:.4f} "
        f"val_bit_acc={val_metrics['bit_acc']:.4f} "
        f"val_byte_acc={val_metrics['byte_acc']:.4f} "
        f"val_top2_acc={val_metrics['top2_acc']:.4f} "
        f"val_top5_acc={val_metrics['top5_acc']:.4f} "
        f"val_entropy={val_metrics['mean_entropy']:.4f} "
        f"val_msg_exact={val_metrics['message_exact_match']:.4f}"
    )
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
    if plot_error:
        print(f"Skipped {plot_path}: {plot_error}")
    else:
        print(f"Saved {plot_path}")
    if per_byte_plot_error:
        print(f"Skipped {per_byte_plot_path}: {per_byte_plot_error}")
    else:
        print(f"Saved {per_byte_plot_path}")
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
