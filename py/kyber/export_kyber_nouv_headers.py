#!/usr/bin/env python3
# SPDX-FileContributor: Person: Stanley Lee
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "py" / "kyber" / "trace_uv_local_byte_model.pt"
SAMPLE_JSON_PATH = ROOT / "py" / "kyber" / "c_infer_sample_msg0_byte0.json"
OUT_DIR = ROOT / "testbench" / "kyber"
WEIGHTS_HEADER = OUT_DIR / "kyber_nouv_weights.h"
SAMPLE_HEADER = OUT_DIR / "kyber_nouv_sample.h"


class TraceEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        use_attention: bool = True,
        attn_heads: int = 4,
        in_channels: int = 2,
        extra_conv_layers: int = 0,
        extra_conv_kernel: int = 5,
    ) -> None:
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
    def __init__(self, num_slots: int, fusion_dim: int = 128) -> None:
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
    ) -> None:
        super().__init__()
        self.use_trace = bool(use_trace)
        self.use_uv = bool(use_uv)
        self.use_byte_position = bool(use_byte_position)
        self.use_peak_positions = bool(use_peak_positions)
        self.use_peak_window_features = bool(use_peak_window_features)

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
        return self.head(torch.cat(feats, dim=1))


def fuse_conv_bn_eval(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Conv1d:
    return torch.nn.utils.fuse_conv_bn_eval(conv, bn)


def build_model(model_path: Path) -> tuple[LocalByteModel, dict]:
    checkpoint = torch.load(model_path, map_location="cpu")
    meta = checkpoint["meta"]
    model = LocalByteModel(
        num_slots=0,
        trace_channels=meta["trace_channels"],
        fusion_dim=meta["fusion_dim"],
        head_hidden_dim=meta["head_hidden_dim"],
        head_dropout=meta["head_dropout"],
        head_layernorm=meta["head_layernorm"],
        use_attention=meta["use_attention"],
        attn_heads=meta["attn_heads"],
        use_trace=meta["use_trace"],
        use_uv=meta["use_uv"],
        use_byte_position=meta["use_byte_position"],
        use_peak_positions=meta["use_peak_positions"],
        use_peak_window_features=meta["use_peak_window_features"],
        trace_extra_conv_layers=meta["trace_extra_conv_layers"],
        trace_extra_conv_kernel=meta["trace_extra_conv_kernel"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, meta


def format_value(v: float) -> str:
    s = f"{float(v):.9g}"
    if "e" not in s and "." not in s:
        s = f"{s}.0"
    return f"{s}f"


def write_flat_array(f, c_type: str, name: str, tensor: torch.Tensor, shape_comment: str) -> None:
    data = tensor.detach().cpu().contiguous().view(-1).numpy()
    f.write(f"// {shape_comment}: {tuple(tensor.shape)}\n")
    f.write(f"static const {c_type} {name}[{tensor.numel()}] = {{\n")
    for i, val in enumerate(data):
        if i % 8 == 0:
            f.write("  ")
        f.write(format_value(float(val)))
        if i != len(data) - 1:
            f.write(", ")
        if i % 8 == 7:
            f.write("\n")
    if len(data) % 8 != 0:
        f.write("\n")
    f.write("};\n\n")


def write_weights_header(model: LocalByteModel, meta: dict, header_path: Path) -> None:
    trace = model.trace_encoder
    features = trace.features
    fused_convs = [
        fuse_conv_bn_eval(features[0], features[1]),
        fuse_conv_bn_eval(features[3], features[4]),
        fuse_conv_bn_eval(features[6], features[7]),
        fuse_conv_bn_eval(features[9], features[10]),
        fuse_conv_bn_eval(features[12], features[13]),
    ]

    stage_lengths = [meta["trace_window_len"]]
    seq_len = meta["trace_window_len"]
    for conv in fused_convs:
        seq_len = (seq_len + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        stage_lengths.append(seq_len)

    head_ln = model.head[0]
    head_fc1 = model.head[1]
    head_fc2 = model.head[4]

    header_path.parent.mkdir(parents=True, exist_ok=True)
    with open(header_path, "w", encoding="ascii") as f:
        f.write("#pragma once\n")
        f.write("#include <stddef.h>\n\n")
        f.write("// Generated by py/kyber/export_kyber_nouv_headers.py\n")
        f.write(f"#define KYBER_NOUV_TRACE_LEN {meta['trace_window_len']}\n")
        f.write(f"#define KYBER_NOUV_STAGE1_LEN {stage_lengths[1]}\n")
        f.write(f"#define KYBER_NOUV_STAGE2_LEN {stage_lengths[2]}\n")
        f.write(f"#define KYBER_NOUV_STAGE3_LEN {stage_lengths[3]}\n")
        f.write(f"#define KYBER_NOUV_STAGE4_LEN {stage_lengths[4]}\n")
        f.write(f"#define KYBER_NOUV_STAGE5_LEN {stage_lengths[5]}\n")
        f.write(f"#define KYBER_NOUV_TRACE_IN_CHANNELS 2\n")
        f.write(f"#define KYBER_NOUV_TRACE_SEQ_LEN {stage_lengths[5]}\n")
        f.write(f"#define KYBER_NOUV_EMBED_DIM {trace.attn.embed_dim}\n")
        f.write(f"#define KYBER_NOUV_ATTN_HEADS {trace.attn.num_heads}\n")
        f.write(f"#define KYBER_NOUV_ATTN_HEAD_DIM {trace.attn.head_dim}\n")
        f.write(f"#define KYBER_NOUV_FUSED_DIM {head_ln.normalized_shape[0]}\n")
        f.write(f"#define KYBER_NOUV_HEAD_HIDDEN_DIM {head_fc1.out_features}\n")
        f.write("#define KYBER_NOUV_NUM_CLASSES 256\n\n")

        for idx, conv in enumerate(fused_convs, start=1):
            write_flat_array(f, "float", f"kyber_conv{idx}_weight", conv.weight, "Conv1d weight O,I,K")
            write_flat_array(f, "float", f"kyber_conv{idx}_bias", conv.bias, "Conv1d bias O")

        write_flat_array(f, "float", "kyber_attn_in_proj_weight", trace.attn.in_proj_weight, "Attention in_proj weight O,I")
        write_flat_array(f, "float", "kyber_attn_in_proj_bias", trace.attn.in_proj_bias, "Attention in_proj bias O")
        write_flat_array(f, "float", "kyber_attn_out_proj_weight", trace.attn.out_proj.weight, "Attention out_proj weight O,I")
        write_flat_array(f, "float", "kyber_attn_out_proj_bias", trace.attn.out_proj.bias, "Attention out_proj bias O")
        write_flat_array(f, "float", "kyber_attn_norm_weight", trace.attn_norm.weight, "Attention LayerNorm weight")
        write_flat_array(f, "float", "kyber_attn_norm_bias", trace.attn_norm.bias, "Attention LayerNorm bias")
        write_flat_array(f, "float", "kyber_head_ln_weight", head_ln.weight, "Head LayerNorm weight")
        write_flat_array(f, "float", "kyber_head_ln_bias", head_ln.bias, "Head LayerNorm bias")
        write_flat_array(f, "float", "kyber_head_fc1_weight", head_fc1.weight, "Head FC1 weight O,I")
        write_flat_array(f, "float", "kyber_head_fc1_bias", head_fc1.bias, "Head FC1 bias O")
        write_flat_array(f, "float", "kyber_head_fc2_weight", head_fc2.weight, "Head FC2 weight O,I")
        write_flat_array(f, "float", "kyber_head_fc2_bias", head_fc2.bias, "Head FC2 bias O")


def write_sample_header(sample_json_path: Path, header_path: Path) -> None:
    sample = json.loads(sample_json_path.read_text(encoding="utf-8"))
    trace_x = sample["trace_x"]
    top5 = sample["top5"]

    header_path.parent.mkdir(parents=True, exist_ok=True)
    with open(header_path, "w", encoding="ascii") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Generated by py/kyber/export_kyber_nouv_headers.py\n")
        f.write(f"#define KYBER_NOUV_SAMPLE_TRACE_LEN {len(trace_x[0])}\n")
        f.write("#define KYBER_NOUV_SAMPLE_NUM_CLASSES 256\n")
        f.write("#define KYBER_NOUV_SAMPLE_TOPK 5\n\n")
        f.write(f"static const int KYBER_NOUV_SAMPLE_MESSAGE_IDX = {int(sample['message_idx'])};\n")
        f.write(f"static const int KYBER_NOUV_SAMPLE_BYTE_IDX = {int(sample['byte_idx'])};\n")
        f.write(f"static const float KYBER_NOUV_SAMPLE_BYTE_POS = {format_value(sample['byte_pos_model'])};\n")
        f.write(f"static const uint8_t KYBER_NOUV_SAMPLE_TARGET = {int(sample['target_groundtruth'])};\n")
        f.write(f"static const uint8_t KYBER_NOUV_SAMPLE_PYTORCH_ARGMAX = {int(sample['prediction_argmax'])};\n\n")

        f.write("static const uint16_t KYBER_NOUV_SAMPLE_TOP5_CLASS[KYBER_NOUV_SAMPLE_TOPK] = {")
        f.write(", ".join(str(int(item["class"])) for item in top5))
        f.write("};\n")
        f.write("static const float KYBER_NOUV_SAMPLE_TOP5_LOGIT[KYBER_NOUV_SAMPLE_TOPK] = {")
        f.write(", ".join(format_value(item["logit"]) for item in top5))
        f.write("};\n\n")

        f.write("static const float KYBER_NOUV_SAMPLE_TRACE_X[2][KYBER_NOUV_SAMPLE_TRACE_LEN] = {\n")
        for channel_idx, channel in enumerate(trace_x):
            f.write("  {\n")
            for i, value in enumerate(channel):
                if i % 8 == 0:
                    f.write("    ")
                f.write(format_value(value))
                if i != len(channel) - 1:
                    f.write(", ")
                if i % 8 == 7:
                    f.write("\n")
            if len(channel) % 8 != 0:
                f.write("\n")
            trailer = "  },\n" if channel_idx == 0 else "  }\n"
            f.write(trailer)
        f.write("};\n")


def main() -> None:
    model, meta = build_model(MODEL_PATH)
    write_weights_header(model, meta, WEIGHTS_HEADER)
    write_sample_header(SAMPLE_JSON_PATH, SAMPLE_HEADER)
    print(f"wrote {WEIGHTS_HEADER}")
    print(f"wrote {SAMPLE_HEADER}")


if __name__ == "__main__":
    main()
