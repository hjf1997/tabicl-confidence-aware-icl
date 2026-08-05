"""Analytical FLOP model for TabICL v2 inference.

Counts matmul FLOPs (1 multiply-accumulate = 2 FLOPs), matching the convention
of torch.utils.flop_counter.FlopCounterMode. Elementwise ops (LayerNorm, GELU,
softmax, RoPE rotations, residual adds) are excluded — they are O(tokens * dim)
and contribute well under 1% next to the matmul terms counted here.

Architecture constants below are the TabICL v2 checkpoint configuration
(tabicl-classifier-v2-20260212.ckpt). They equal the TabICL() constructor
defaults in tabicl/_model/tabicl.py, verified by exact parameter count
(27,552,258 params) and by the SSMax query-MLP shapes in the state dict.

Scaling structure (n = support rows, q = query rows, H = features):

- Column embedding: 3 induced-attention blocks per feature. Attention is
  128 inducing points vs rows — LINEAR in rows, linear in H.
- Row interaction: 3 attention blocks over the feature axis (H + 4 CLS
  tokens) per row — QUADRATIC in H, linear in rows.
- ICL predictor: 12 layers at d=512 where all rows attend only to the n
  support rows (rectangular attention, tabicl/_model/layers.py:430-435).
  Per-query cost is LINEAR in n; the n x n support self-attention exists
  only when support rows are (re)encoded.

kv_cache=True (the pipeline default) splits the cost:
- fit(): encodes the support set once (includes the n x n ICL term) and
  caches per-layer K/V.
- predict_proba(): per query, only the Q-side runs — col stage 2 against
  cached inducing K/V, row interaction, ICL with cached K/V. No n^2 term.
kv_cache=False re-encodes the support set inside every predict call.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class TabICLDims:
    """TabICL v2 architecture constants."""

    embed_dim: int = 128        # E: col/row stages
    col_num_blocks: int = 3
    col_num_inds: int = 128     # m: inducing points per ISAB
    col_group_size: int = 3     # in_linear: Linear(3 -> 128); "same" mode => H groups
    row_num_blocks: int = 3
    row_num_cls: int = 4        # C: CLS tokens; row seq len = H + C
    icl_num_blocks: int = 12
    icl_dim: int = 512          # d = embed_dim * row_num_cls
    ff_factor: int = 2          # FFN hidden = dim * ff_factor
    ssmax_hidden: int = 64      # QASSMaxMLP hidden width (col attn1 + every ICL layer)
    max_classes: int = 10       # y one-hot width and decoder output
    decoder_hidden: int = 1024  # Linear(512->1024) -> GELU -> Linear(1024->10)


D = TabICLDims()


def _lin(tokens: int, d_in: int, d_out: int) -> int:
    """FLOPs of a linear layer over `tokens` tokens."""
    return 2 * tokens * d_in * d_out


def _attn(tgt: int, src: int, dim: int) -> int:
    """QK^T + AV for tgt queries vs src keys at total width `dim` (all heads)."""
    return 4 * tgt * src * dim


def _ssmax(tokens: int, dim: int, d: TabICLDims = D) -> int:
    """QASSMaxMLP-elementwise on query tokens: per head, head_dim->64->head_dim.

    Summed over heads this is 4 * tokens * ssmax_hidden * dim regardless of the
    head count. The scalar base_mlp (input width 1) is negligible and skipped.
    """
    return 4 * tokens * d.ssmax_hidden * dim


# ---------------------------------------------------------------------------
# Column embedding (per feature group; "same" mode => H groups, and the 4
# reserved CLS columns ride the skip path, so real work is x H).
# ---------------------------------------------------------------------------

def _col_isab_block(n_rows_q2: int, n_kv: int, d: TabICLDims, cached: bool) -> int:
    """One InducedSelfAttentionBlock for a single feature.

    attn1: m inducing queries vs the n_kv support rows (skipped when cached).
    attn2: the rows vs the m inducing outputs (K/V of `hidden` read from cache
    when cached; recomputed otherwise).
    """
    E, m, ff = d.embed_dim, d.col_num_inds, d.embed_dim * d.ff_factor
    total = 0
    if not cached:
        # attn1: Q proj (m) + K,V proj (n_kv) + attention + out proj + FFN + SSMax
        total += _lin(m, E, E) + 2 * _lin(n_kv, E, E) + _attn(m, n_kv, E)
        total += _lin(m, E, E) + 4 * m * E * ff + _ssmax(m, E, d)
        # attn2 K/V proj of the m hidden outputs
        total += 2 * _lin(m, E, E)
    # attn2: rows attend to the m inducing outputs
    total += _lin(n_rows_q2, E, E) + _attn(n_rows_q2, m, E)
    total += _lin(n_rows_q2, E, E) + 4 * n_rows_q2 * E * ff
    return total


def col_embedding_flops(n_rows: int, n_kv: int, n_features: int,
                        cached: bool, d: TabICLDims = D) -> int:
    # The 4 reserved CLS columns do full compute through this stage: the
    # SkippableLinear path masks their outputs but does not skip the matmuls
    # (verified against FlopCounterMode: tf_col cost is exactly (H+4) groups).
    G = n_features + d.row_num_cls
    E = d.embed_dim
    total = _lin(n_rows * G, d.col_group_size, E)  # in_linear per group
    if not cached:
        total += G * _lin(n_kv, d.max_classes, E)  # y encoder applied per group
    total += G * d.col_num_blocks * _col_isab_block(n_rows, n_kv, d, cached)
    return total


# ---------------------------------------------------------------------------
# Row interaction (per row; sequence axis = H + 4 CLS tokens). Never cached.
# ---------------------------------------------------------------------------

def row_interaction_flops(n_rows: int, n_features: int, d: TabICLDims = D) -> int:
    E, ff, C = d.embed_dim, d.embed_dim * d.ff_factor, d.row_num_cls
    L = n_features + C
    # First row_num_blocks - 1 blocks: full self-attention over L tokens.
    full = 3 * _lin(L, E, E) + _attn(L, L, E) + _lin(L, E, E) + 4 * L * E * ff
    # Last block: only the C CLS tokens query all L tokens.
    cls = (_lin(C, E, E) + 2 * _lin(L, E, E) + _attn(C, L, E)
           + _lin(C, E, E) + 4 * C * E * ff)
    return n_rows * ((d.row_num_blocks - 1) * full + cls)


# ---------------------------------------------------------------------------
# ICL predictor (12 layers, d=512; all rows attend to the n support rows only).
# ---------------------------------------------------------------------------

def icl_flops(n_rows: int, n_support: int, cached: bool,
              with_y_encoder: bool, d: TabICLDims = D) -> int:
    dm, ff = d.icl_dim, d.icl_dim * d.ff_factor
    per_block = _lin(n_rows, dm, dm) + _attn(n_rows, n_support, dm)
    per_block += _lin(n_rows, dm, dm) + 4 * n_rows * dm * ff + _ssmax(n_rows, dm, d)
    if not cached:
        per_block += 2 * _lin(n_support, dm, dm)  # K/V projections of support
    total = d.icl_num_blocks * per_block
    if with_y_encoder:
        total += _lin(n_support, d.max_classes, dm)
    return total


def decoder_flops(n_rows: int, d: TabICLDims = D) -> int:
    """Decoder MLP. Runs unconditionally over every row present in the ICL
    sequence (learning.py: `out = self.decoder(src)`), with test positions
    sliced afterwards — so pass the number of rows encoded, not n_query."""
    return _lin(n_rows, d.icl_dim, d.decoder_hidden) + _lin(n_rows, d.decoder_hidden, d.max_classes)


# ---------------------------------------------------------------------------
# End-to-end costs
# ---------------------------------------------------------------------------

def fit_flops(n_support: int, n_features: int, n_estimators: int = 8,
              kv_cache: bool = True, d: TabICLDims = D) -> Dict[str, int]:
    """One-time cost of TabICLClassifier.fit().

    With kv_cache=True this encodes the support set through all three stages
    (including the n x n ICL attention) and stores K/V. With kv_cache=False,
    fit() only preprocesses — encoding happens inside every predict call.
    """
    if not kv_cache:
        return {"col": 0, "row": 0, "icl": 0, "decoder": 0, "total": 0}
    n = n_support
    col = col_embedding_flops(n, n, n_features, cached=False, d=d)
    row = row_interaction_flops(n, n_features, d=d)
    icl = icl_flops(n, n, cached=False, with_y_encoder=True, d=d)
    dec = decoder_flops(n, d=d)  # decoder runs over support rows at fit too
    out = {"col": col * n_estimators, "row": row * n_estimators,
           "icl": icl * n_estimators, "decoder": dec * n_estimators}
    out["total"] = sum(out.values())
    return out


def predict_flops(n_query: int, n_support: int, n_features: int, n_estimators: int = 8,
                  kv_cache: bool = True, d: TabICLDims = D) -> Dict[str, int]:
    """Cost of one predict_proba() call on n_query samples.

    kv_cache=True: strictly linear in n_query, no n_support^2 term.
    kv_cache=False: the support set is re-encoded within this call (the full
    forward runs over n_support + n_query rows), so each call repays the
    support cost — total cost then depends on how queries are chunked.
    """
    n, q = n_support, n_query
    if kv_cache:
        col = col_embedding_flops(q, n, n_features, cached=True, d=d)
        row = row_interaction_flops(q, n_features, d=d)
        icl = icl_flops(q, n, cached=True, with_y_encoder=False, d=d)
        dec = decoder_flops(q, d=d)  # only the q rows are encoded
    else:
        T = n + q
        col = col_embedding_flops(T, n, n_features, cached=False, d=d)
        row = row_interaction_flops(T, n_features, d=d)
        icl = icl_flops(T, n, cached=False, with_y_encoder=True, d=d)
        dec = decoder_flops(T, d=d)  # decoder runs over all T rows
    out = {"col": col * n_estimators, "row": row * n_estimators,
           "icl": icl * n_estimators, "decoder": dec * n_estimators}
    out["total"] = sum(out.values())
    return out


def per_query_flops(n_support: int, n_features: int, n_estimators: int = 8,
                    d: TabICLDims = D) -> Dict[str, int]:
    """Marginal FLOPs per test sample with a fitted KV cache.

    Exact (not an average): with kv_cache=True every per-call term is
    proportional to n_query, so this equals predict_flops(q)/q for any q.
    """
    return predict_flops(1, n_support, n_features, n_estimators, kv_cache=True, d=d)
