import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, [D] (we pass q_nope[b, j, :])
    qp_ptr,           # *float32, [Dp] (we pass q_pe[b, j, :])
    Kc_ptr,           # *float32, [L, D], row-major
    Kp_ptr,           # *float32, [L, Dp], row-major
    v_ptr,            # *float32, [L]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    Dp: tl.int32,     # head_dim_kpe
    BLOCK_K: tl.constexpr,
):
    # One program per token index i
    i = tl.program_id(0)
    acc = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        knc = tl.load(Kc_ptr + i * D + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        qn = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)
        acc += tl.sum(qn * knc, axis=0)
    # Reduce over Kp dimension (Dp)
    acc2 = 0.0
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        kmp = tl.load(Kp_ptr + i * Dp + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        qp = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)
        acc2 += tl.sum(qp * kmp, axis=0)
    v = acc + acc2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, scalar
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # Two-pass reduction: max, then sum(exp(v - max) / ln(2))
    max_v = -float('inf')
    for i in range(0, L, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-float('inf'))
        m = tl.max(v, axis=0)
        max_v = tl.maximum(max_v, m)
    sum_exp = 0.0
    for i in range(0, L, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-float('inf'))
        e = tl.exp(v - max_v) * inv_ln2
        sum_exp += tl.sum(e, axis=0)
    lse = max_v + tl.log(sum_exp)
    tl.store(lse_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    attn_ptr,         # *float32, [L]
    lse_ptr,          # *float32, scalar lse
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # Compute attn[i] = exp(v[i] - lse * ln(2))
    lse = tl.load(lse_ptr)
    for i in range(0, L, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        attn = tl.exp(v - lse * inv_ln2)
        tl.store(attn_ptr + offs, attn, mask=mask)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D], row-major
    out_ptr,          # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # One program per output dimension h
    h = tl.program_id(0)
    acc = 0.0
    for i in range(0, L, BLOCK_K):
        offs = i + tl.arange(0, BLOCK_K)
        mask_i = offs < L
        attn = tl.load(attn_ptr + offs, mask=mask_i, other=0.0)
        kc = tl.load(Kc_ptr + offs * D + h, mask=mask_i, other=0.0)
        acc += tl.sum(attn * kc, axis=0)
    tl.store(out_ptr + h, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    # Ensure inputs are float32 for computation
    q_nope = q_nope.to(torch.float32)
    q_pe = q_pe.to(torch.float32)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64

    # Prepare output and lse
    output = torch.empty(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
    )
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    for b in range(batch_size):
        # token range
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = max(page_end - page_beg, 0)
        if L_tokens == 0:
            # No tokens in this batch element -> zero output, no lse update
            output[b].zero_()
            continue

        # Gather token indices
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32)
        # Kc and Kp for these tokens
        Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]

        # Precompute inv ln(2)
        inv_ln2 = 1.0 / math.log(2.0)

        # For each head j
        for j in range(num_qo_heads):
            # Per-head q vectors (float32)
            qn = q_nope[b, j, :].to(torch.float32).contiguous()  # [D]
            qp = q_pe[b, j, :].to(torch.float32).contiguous()   # [Dp]

            # 1) Compute v[j, :] = sum over tokens of (qn · Kc[i, :]) + (qp · Kp[i, :])
            v = torch.empty(L_tokens, dtype=torch.float32, device=device)
            grid_v = (L_tokens,)
            matvec_add_kernel[grid_v](
                qn, qp, Kc, Kp, v, L_tokens, head_dim_ckv, head_dim_kpe,
                BLOCK_K=128, num_warps=4
            )

            # 2) Compute lse_j = logsumexp_base2(v)
            lse_j = torch.empty((), dtype=torch.float32, device=device)
            grid_lse = (1,)
            lse_kernel[grid_lse](
                v, lse_j, L_tokens, inv_ln2,
                BLOCK=128, num_warps=4
            )
            lse[b, j] = lse_j

            # 3) Compute attn[j, :] = softmax_base2(v)
            attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
            grid_softmax = (L_tokens,)
            softmax_base2_kernel[grid_softmax](
                v, attn, lse[b, j], L_tokens, inv_ln2,
                BLOCK=128, num_warps=4
            )

            # 4) Final matvec: out[b, j, :] = attn @ Kc[:, :]
            out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
            grid_y = (head_dim_ckv,)
            matvec_write_y_kernel[grid_y](
                attn, Kc, out_row, L_tokens, head_dim_ckv,
                BLOCK=128, num_warps=4
            )
            output[b, j, :] = out_row

    # Cast output to bfloat16 to match original return type
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


def get_inputs():
    # Helper: do not call run here to avoid recursion.
    # The harness will provide inputs to ModelNew.forward.
    pass


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and dtype handling
        for t in (q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices):
            if isinstance(t, torch.Tensor) and t.device.type != 'cuda':
                t = t.to('cuda')
        # Run Triton-orchestrated computation
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
