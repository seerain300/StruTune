# mla_paged_decode_h16_ckv512_kpe64_ps1 — best candidate c002 (batch formal-kda-h100-20260920)
# feedback (w2/i10 coarse): 51.46x | final (100 iters, paired timing): 47.19x, valid
# source: kda-control/formal-kda-h100-20260920--flashinfer--mla_paged_decode_h16_ckv512_kpe64_ps1/candidates/c002/solution.py (sha256-locked snapshot)
"""
Solution for KDA task: mla_paged_decode_h16_ckv512_kpe64_ps1  (H100 / sm_90)

Candidate c002 — faithful two-stage flash-decoding (split-KV) baseline.
Identical to c001 except the base-2 LSE constant is passed as a tl.constexpr
kernel argument instead of a bare module-level global (c001 failed to compile
with `NameError: Cannot access global variable LOG2E from within @jit'ed
function`). No other change vs c001.

Multi-head Latent Attention paged decode:
  - H = 16 query heads share a single latent KV ("head").
  - ckv_cache [num_pages, 1, 512] is used as BOTH key (score) and value (output).
  - kpe_cache [num_pages, 1, 64] contributes the RoPE score term.
  - page_size == 1  =>  kv_indices are token rows directly (row = kv_indices[beg + n]).
  - Output bf16 [B, 16, 512]; lse fp32 [B, 16] in BASE-2 (natural / ln2).

Design (docs/plan.md §2):
  Stage 1: grid (B, NUM_KV_SPLITS). All 16 heads in one program (BLOCK_H=16) so the
           gathered KV tile is reused across heads. Online softmax over each split's
           token range. Writes partial normalized acc [16,512] and partial NATURAL
           lse (e_max + log(e_sum)) [16] to a scratch `mid` buffer.
  Stage 2: grid (B, 16). Merges partials across splits with a second online softmax,
           writes output (bf16) and lse (base-2 = *log2(e)).

Triton only; no Torch/CPU/NumPy computational fallback.
"""

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634  # 1 / ln(2); passed into stage-2 as a constexpr arg


@triton.jit
def _mla_decode_stage1(
    Q_nope,        # [B, H, Dckv] bf16
    Q_pe,          # [B, H, Dkpe] bf16
    CKV,           # [num_pages, 1, Dckv] bf16 (key AND value)
    KPE,           # [num_pages, 1, Dkpe] bf16
    KV_indptr,     # [B+1] int32
    KV_indices,    # [num_kv_indices] int32
    Mid,           # [B, H, NUM_KV_SPLITS, Dckv+1] fp32 scratch
    sm_scale,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_ckv_p,
    stride_kpe_p,
    stride_mid_b, stride_mid_h, stride_mid_s,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,   # Dckv (512)
    BLOCK_DPE: tl.constexpr,      # Dkpe (64)
    BLOCK_DV: tl.constexpr,       # Dckv (512)
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    split_kv_id = tl.program_id(1)

    base = tl.load(KV_indptr + cur_batch)
    seq_len = tl.load(KV_indptr + cur_batch + 1) - base

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dpe = tl.arange(0, BLOCK_DPE)

    # Query (all 16 heads); H == BLOCK_H and Dckv/Dkpe are powers of two -> no mask.
    q = tl.load(
        Q_nope + cur_batch * stride_qn_b + offs_h[:, None] * stride_qn_h + offs_d[None, :]
    )
    qpe = tl.load(
        Q_pe + cur_batch * stride_qp_b + offs_h[:, None] * stride_qp_h + offs_dpe[None, :]
    )

    kv_len_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = kv_len_per_split * split_kv_id
    split_end = tl.minimum(split_start + kv_len_per_split, seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_end > split_start:
        for start_n in range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_end

            kv_loc = tl.load(KV_indices + base + offs_n, mask=mask_n, other=0).to(tl.int64)

            # Key tile [Dckv, BLOCK_N]  ->  qk = q @ k  = [BLOCK_H, BLOCK_N]
            k = tl.load(
                CKV + kv_loc[None, :] * stride_ckv_p + offs_d[:, None],
                mask=mask_n[None, :],
                other=0.0,
            )
            qk = tl.dot(q, k)

            # RoPE score term
            kpe = tl.load(
                KPE + kv_loc[None, :] * stride_kpe_p + offs_dpe[:, None],
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe)

            qk *= sm_scale
            qk = tl.where(mask_n[None, :], qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]

            # Value tile [BLOCK_N, Dckv] (same ckv data)  ->  acc += p @ v
            v = tl.load(
                CKV + kv_loc[:, None] * stride_ckv_p + offs_dv[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        off_mid = (
            cur_batch * stride_mid_b
            + offs_h[:, None] * stride_mid_h
            + split_kv_id * stride_mid_s
            + offs_dv[None, :]
        )
        tl.store(Mid + off_mid, acc / e_sum[:, None])

        off_mid_lse = (
            cur_batch * stride_mid_b
            + offs_h * stride_mid_h
            + split_kv_id * stride_mid_s
            + BLOCK_DV
        )
        tl.store(Mid + off_mid_lse, e_max + tl.log(e_sum))


@triton.jit
def _mla_decode_stage2(
    Mid,           # [B, H, NUM_KV_SPLITS, Dckv+1] fp32
    O,             # [B, H, Dckv] bf16
    Lse,           # [B, H] fp32
    KV_indptr,     # [B+1] int32
    stride_mid_b, stride_mid_h, stride_mid_s,
    stride_o_b, stride_o_h,
    stride_lse_b,
    LOG2E: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    seq_len = tl.load(KV_indptr + cur_batch + 1) - tl.load(KV_indptr + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    off_base = cur_batch * stride_mid_b + cur_head * stride_mid_h

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = kv_len_per_split * split_kv_id
        split_end = tl.minimum(split_start + kv_len_per_split, seq_len)

        if split_end > split_start:
            tv = tl.load(Mid + off_base + split_kv_id * stride_mid_s + offs_d)
            tlogic = tl.load(Mid + off_base + split_kv_id * stride_mid_s + BLOCK_DV)

            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    out = tl.where(e_sum > 0, acc / e_sum, 0.0)
    tl.store(
        O + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d,
        out.to(O.dtype.element_ty),
    )
    # base-2 LSE; empty sequence (e_sum==0) -> e_max=-inf, log(0)=-inf -> -inf
    lse_val = (e_max + tl.log(e_sum)) * LOG2E
    tl.store(Lse + cur_batch * stride_lse_b + cur_head, lse_val)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    B, H, Dckv = q_nope.shape
    Dkpe = q_pe.shape[-1]
    device = q_nope.device

    # Fixed configuration for the baseline.
    NUM_KV_SPLITS = 4
    BLOCK_N = 32
    BLOCK_H = H  # 16

    sm_scale = float(sm_scale)

    output = torch.zeros((B, H, Dckv), dtype=torch.bfloat16, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)
    mid = torch.empty((B, H, NUM_KV_SPLITS, Dckv + 1), dtype=torch.float32, device=device)

    grid1 = (B, NUM_KV_SPLITS)
    _mla_decode_stage1[grid1](
        q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, mid,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        ckv_cache.stride(0),
        kpe_cache.stride(0),
        mid.stride(0), mid.stride(1), mid.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_DMODEL=Dckv,
        BLOCK_DPE=Dkpe,
        BLOCK_DV=Dckv,
        BLOCK_N=BLOCK_N,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        num_warps=4,
        num_stages=2,
    )

    grid2 = (B, H)
    _mla_decode_stage2[grid2](
        mid, output, lse, kv_indptr,
        mid.stride(0), mid.stride(1), mid.stride(2),
        output.stride(0), output.stride(1),
        lse.stride(0),
        LOG2E=_LOG2E,
        BLOCK_DV=Dckv,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        num_warps=4,
        num_stages=2,
    )

    return output, lse
