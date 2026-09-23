"""Triton MLA paged decode — candidate c002.

Task: mla_paged_decode_h16_ckv512_kpe64_ps1 (DeepSeek-V3 MLA decode, weight-absorbed).
Target: NVIDIA A800 (sm_80).

c002 = mechanical compile fix over c001 (identical algorithm/config):
  * single fused Triton kernel, grid = (batch_size,)
  * all 16 query heads processed together (KV read once, reused for score + value)
  * online softmax in fp32 (natural base internally), fp32 accumulator
  * bf16 tensor-core QK and PV dots
  * empty-sequence guard -> output 0, lse -inf
  * base-2 LSE emitted as (m + ln l) * log2(e), with log2(e) inlined as a literal
    inside the kernel (c001 failed to compile because it referenced a module-global
    python float `_LOG2E` from within the @jit kernel, which this Triton version
    forbids).

Primary implementation is Triton. PyTorch is used only for output allocation and
launch plumbing. No Torch/CPU/NumPy/CUDA-extension computational fallback.
"""

import torch
import triton
import triton.language as tl

# Constants for this definition (asserted at runtime to catch plumbing mistakes).
_H = 16       # num_qo_heads
_DCKV = 512   # head_dim_ckv (also the value dim, since V == compressed latent)
_DKPE = 64    # head_dim_kpe


@triton.jit
def _mla_decode_kernel(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr,
    kv_indptr_ptr, kv_indices_ptr,
    out_ptr, lse_ptr,
    sm_scale,
    stride_qn_b, stride_qn_h, stride_qn_d,
    stride_qp_b, stride_qp_h, stride_qp_d,
    stride_ckv_p, stride_ckv_d,
    stride_kpe_p, stride_kpe_d,
    stride_o_b, stride_o_h, stride_o_d,
    stride_lse_b, stride_lse_h,
    H: tl.constexpr, DCKV: tl.constexpr, DKPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)

    beg = tl.load(kv_indptr_ptr + b)
    end = tl.load(kv_indptr_ptr + b + 1)
    L = end - beg

    offs_h = tl.arange(0, H)
    offs_ckv = tl.arange(0, DCKV)
    offs_kpe = tl.arange(0, DKPE)

    out_row_ptr = (out_ptr + b * stride_o_b
                   + offs_h[:, None] * stride_o_h
                   + offs_ckv[None, :] * stride_o_d)
    lse_row_ptr = lse_ptr + b * stride_lse_b + offs_h * stride_lse_h

    # Empty sequence: reproduce reference (zeros + -inf).
    if L <= 0:
        tl.store(out_row_ptr, tl.zeros([H, DCKV], dtype=tl.bfloat16))
        tl.store(lse_row_ptr, tl.full([H], float("-inf"), dtype=tl.float32))
        return

    # Load queries for all heads (reused across the whole KV loop).
    qn = tl.load(q_nope_ptr + b * stride_qn_b
                 + offs_h[:, None] * stride_qn_h
                 + offs_ckv[None, :] * stride_qn_d)          # [H, DCKV] bf16
    qp = tl.load(q_pe_ptr + b * stride_qp_b
                 + offs_h[:, None] * stride_qp_h
                 + offs_kpe[None, :] * stride_qp_d)          # [H, DKPE] bf16

    m_i = tl.full([H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, DCKV], dtype=tl.float32)

    for n in range(0, L, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        col_mask = offs_n < L
        idx = tl.load(kv_indices_ptr + beg + offs_n, mask=col_mask, other=0).to(tl.int64)

        Kc = tl.load(ckv_ptr + idx[:, None] * stride_ckv_p
                     + offs_ckv[None, :] * stride_ckv_d,
                     mask=col_mask[:, None], other=0.0)       # [BLOCK_N, DCKV] bf16
        Kp = tl.load(kpe_ptr + idx[:, None] * stride_kpe_p
                     + offs_kpe[None, :] * stride_kpe_d,
                     mask=col_mask[:, None], other=0.0)       # [BLOCK_N, DKPE] bf16

        # Scores over the 576-dim key: qn @ Kc^T + qp @ Kp^T  -> [H, BLOCK_N] fp32
        qk = tl.dot(qn, tl.trans(Kc))
        qk += tl.dot(qp, tl.trans(Kp))
        qk = qk * sm_scale
        qk = tl.where(col_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])                       # [H, BLOCK_N] fp32
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # PV over the 512-dim latent (value == Kc). bf16 tensor-core dot, fp32 acc.
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), Kc)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(out_row_ptr, out.to(tl.bfloat16))
    # base-2 log-sum-exp: (m + ln l) / ln 2 = (m + ln l) * log2(e)
    # log2(e) inlined as a literal (module globals are not accessible from @jit).
    lse = (m_i + tl.log(l_i)) * 1.4426950408889634
    tl.store(lse_row_ptr, lse)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]

    assert num_qo_heads == _H
    assert head_dim_ckv == _DCKV
    assert head_dim_kpe == _DKPE
    assert page_size == 1

    device = q_nope.device
    kv_indptr = kv_indptr.to(device=device, dtype=torch.int32)
    kv_indices = kv_indices.to(device=device, dtype=torch.int32)

    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv),
                         dtype=torch.bfloat16, device=device)
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    # ckv_cache / kpe_cache: [num_pages, 1, D]; page stride is stride(0), feature stride is stride(2).
    grid = (batch_size,)
    _mla_decode_kernel[grid](
        q_nope, q_pe, ckv_cache, kpe_cache,
        kv_indptr, kv_indices,
        output, lse,
        float(sm_scale),
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        H=_H, DCKV=_DCKV, DKPE=_DKPE,
        BLOCK_N=64,
        num_warps=4,
        num_stages=2,
    )
    return output, lse
