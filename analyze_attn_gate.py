#!/usr/bin/env python3
"""
analyze_attn_gate.py — Analyze learned per-head attention gates after training.

Reads the final attn_gate model checkpoint and computes:
  1. Average sigmoid activation per (layer, head) on a validation slice
  2. Min/max gate values to see if any heads are fully suppressed/amplified
  3. Per-token gate variance (does the model use the gate dynamically?)
  4. Correlation with train_loss patterns

Run AFTER train_gpt_mlx_attn_gate.py finishes:
  .venv/bin/python analyze_attn_gate.py logs/attn_gate_v1_mlx_model.int8.ptz
"""
from __future__ import annotations

import sys
import os
import pickle
import zlib
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("HOURGLASS_MLP_MULTS", "2,2,2,2,1,1,1,1,2,2,2,2")

from train_gpt_mlx_attn_gate import (
    GPT,
    parse_hourglass_mlp_mults,
    dequantize_state_dict_int8,
)


def load_model_from_checkpoint(checkpoint_path: str) -> GPT:
    """Load the trained model from the int8 zlib quantized checkpoint."""
    with open(checkpoint_path, "rb") as f:
        blob = f.read()
    quant_obj = pickle.loads(zlib.decompress(blob))
    flat_state = dequantize_state_dict_int8(quant_obj)

    mlp_mults = parse_hourglass_mlp_mults("2,2,2,2,1,1,1,1,2,2,2,2", 12, 2)
    model = GPT(
        vocab_size=1024, num_layers=12, dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=2, logit_chunk_tokens=0, logit_softcap=30.0, rope_base=10000.0,
        tied_embed_init_std=0.005, qk_gain_init=1.5, stp_beta=0.005, stp_layer="last",
        mlp_mults_per_layer=mlp_mults, disable_skip_connections=True,
    )
    # Load state from flat dict
    from mlx.utils import tree_unflatten
    nested = tree_unflatten(list(flat_state.items()))
    model.update(nested)
    mx.eval(model.state)
    return model


def analyze_gates_on_data(model: GPT, val_tokens: np.ndarray, num_seqs: int = 16) -> dict:
    """
    Forward pass on validation data, capturing the gate activations at each layer.
    Returns per-layer, per-head statistics.
    """
    seq_len = 1024
    # Hook each block's attention to record gates
    gate_records = {i: [] for i in range(len(model.blocks))}

    # Wrap each attention's __call__ to capture the gate
    original_calls = {}
    for layer_idx, block in enumerate(model.blocks):
        attn = block.attn
        original = attn.__call__

        def make_wrapper(layer_i, attn_module, orig):
            def wrapped(x):
                # Compute the gate exactly like the attention does
                bsz, seqlen, dim = x.shape
                gate_logits = attn_module.c_g(x)  # (bsz, seqlen, num_heads)
                gate_sigmoid = mx.sigmoid(gate_logits.astype(mx.float32))
                mx.eval(gate_sigmoid)
                gate_records[layer_i].append(np.array(gate_sigmoid))
                return orig(x)
            return wrapped

        attn.__call__ = make_wrapper(layer_idx, attn, original)
        original_calls[layer_idx] = original

    # Run forward passes on a few validation sequences
    for seq_i in range(num_seqs):
        start = seq_i * seq_len
        if start + seq_len + 1 > val_tokens.size:
            break
        x = mx.array(val_tokens[start : start + seq_len].reshape(1, seq_len), dtype=mx.int32)
        y = mx.array(val_tokens[start + 1 : start + 1 + seq_len].reshape(1, seq_len), dtype=mx.int32)
        _ = model.eval_loss(x, y)
        mx.eval(_)

    # Restore originals
    for layer_idx, block in enumerate(model.blocks):
        block.attn.__call__ = original_calls[layer_idx]

    # Aggregate stats per (layer, head)
    stats = {}
    for layer_idx, records in gate_records.items():
        if not records:
            continue
        all_gates = np.concatenate(records, axis=1)  # (1, total_seqs*seqlen, num_heads)
        all_gates = all_gates.reshape(-1, all_gates.shape[-1])  # (N, num_heads)
        per_head_mean = all_gates.mean(axis=0)
        per_head_std = all_gates.std(axis=0)
        per_head_min = all_gates.min(axis=0)
        per_head_max = all_gates.max(axis=0)
        stats[layer_idx] = {
            "mean": per_head_mean,
            "std": per_head_std,
            "min": per_head_min,
            "max": per_head_max,
        }
    return stats


def print_summary(stats: dict, model: GPT) -> None:
    print(f"\n{'Layer':<6} {'MLP':<6}  ", end="")
    for h in range(8):
        print(f"H{h:<5}", end=" ")
    print()
    print("-" * 80)
    for layer_idx in sorted(stats.keys()):
        s = stats[layer_idx]
        block = model.blocks[layer_idx]
        mlp_hidden = block.mlp.fc.weight.shape[0]
        mlp_mult = mlp_hidden // 512
        kind = "wide" if mlp_mult >= 2 else "narrow"
        print(f"L{layer_idx:<5} {kind:<6}  ", end="")
        for h in range(8):
            print(f"{s['mean'][h]:.3f}", end=" ")
        print(f"  std=[{s['std'].min():.3f}-{s['std'].max():.3f}]")

    # Most/least gated heads
    print("\nMost SUPPRESSED heads (lowest mean gate):")
    all_means = []
    for layer_idx, s in stats.items():
        for h in range(len(s["mean"])):
            all_means.append((s["mean"][h], layer_idx, h))
    all_means.sort()
    for v, l, h in all_means[:5]:
        print(f"  L{l} H{h}: {v:.3f}")
    print("\nMost AMPLIFIED heads (highest mean gate):")
    for v, l, h in all_means[-5:]:
        print(f"  L{l} H{h}: {v:.3f}")

    print("\nMost DYNAMIC heads (highest gate std — context-sensitive):")
    all_stds = []
    for layer_idx, s in stats.items():
        for h in range(len(s["std"])):
            all_stds.append((s["std"][h], layer_idx, h))
    all_stds.sort(reverse=True)
    for v, l, h in all_stds[:5]:
        print(f"  L{l} H{h}: std={v:.3f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: analyze_attn_gate.py <checkpoint.int8.ptz>")
        sys.exit(1)
    ckpt = sys.argv[1]
    print(f"Loading model from {ckpt}...")
    model = load_model_from_checkpoint(ckpt)
    print(f"Model loaded: {len(model.blocks)} blocks")

    # Load validation tokens
    val_files = sorted(Path("./data/datasets/fineweb10B_sp1024").glob("fineweb_val_*.bin"))
    if not val_files:
        print("No validation files found")
        sys.exit(1)
    val_tokens = np.fromfile(val_files[0], dtype=np.uint16).astype(np.int32)
    print(f"Loaded {val_tokens.size:,} validation tokens")

    print("\nAnalyzing gate activations on 16 validation sequences...")
    stats = analyze_gates_on_data(model, val_tokens, num_seqs=16)
    print_summary(stats, model)


if __name__ == "__main__":
    main()
