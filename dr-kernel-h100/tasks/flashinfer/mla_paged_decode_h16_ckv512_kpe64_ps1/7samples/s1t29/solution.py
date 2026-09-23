import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, [D]
    qp_ptr,           # *float32, [Dp]
    Kc_ptr,           # *float32, [L, D], row-major (L, D)
    Kp_ptr,           # *float32, [L, Dp], row-major (L, Dp)
    v_ptr,            # *float32, [L]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    Dp: tl.int32,     # head_dim_kpe
    BLOCK_K: tl.constexpr
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)
        kc_ptr = Kc_ptr + i * D + k_off
        kc_vals = tl.load(kc_ptr, mask=mask_k, other=0.0)
        sum1 += tl.sum(qn_slice * kc_vals, axis=0)

    # Reduce over Kp dimension (Dp)
    for k in range(0, Dp, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < Dp
        qp_slice = tl.load(qp_ptr + k_off, mask=mask_k, other=0.0)
        kp_ptr = Kp_ptr + i * Dp + k_off
        kp_vals = tl.load(kp_ptr, mask=mask_k, other=0.0)
        sum2 += tl.sum(qp_slice * kp_vals, axis=0)

    v_ptr[i] = sum1 + sum2


@triton.jit
def lse_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK_L: tl.constexpr
):
    # Pass 1: compute max of v
    max_v = -1.0e30
    for i in range(0, L, BLOCK_L):
        offs = i + tl.arange(0, BLOCK_L)
        mask = offs < L
        vi = tl.load(v_ptr + offs, mask=mask, other=-1.0e30)
        m = tl.max(vi, axis=0)
        max_v = tl.maximum(max_v, m)

    # Pass 2: compute sum of exp(v - max_v)
    sum_exp = 0.0
    for i in range(0, L, BLOCK_L):
        offs = i + tl.arange(0, BLOCK_L)
        mask = offs < L
        vi = tl.load(v_ptr + offs, mask=mask, other=-1.0e30)
        sum_exp += tl.sum(tl.exp(vi - max_v), axis=0)

    lse_val = max_v + tl.log(sum_exp) * inv_ln2
    tl.store(lse_ptr, lse_val)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK_L: tl.constexpr
):
    # Compute attn[i] = exp(v[i] / ln(2) - lse)
    lse_val = tl.load(lse_ptr)
    for i in range(0, L, BLOCK_L):
        offs = i + tl.arange(0, BLOCK_L)
        mask = offs < L
        vi = tl.load(v_ptr + offs, mask=mask, other=0.0)
        attn = tl.exp((vi * inv_ln2) - lse_val)
        tl.store(attn_ptr + offs, attn, mask=mask)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D]
    y_ptr,            # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_D: tl.constexpr
):
    # One program per output dimension h
    h = tl.program_id(0)
    # Reduce over L to produce y[h]
    acc = 0.0
    for i in range(0, L, BLOCK_D):
        offs = i + tl.arange(0, BLOCK_D)
        mask_i = offs < L
        attn_slice = tl.load(attn_ptr + offs, mask=mask_i, other=0.0)
        kc_ptr = Kc_ptr + offs * D + h
        kc_vals = tl.load(kc_ptr, mask=mask_i, other=0.0)
        acc += tl.sum(attn_slice * kc_vals, axis=0)
    y_ptr[h] = acc


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only implementation: all heavy computation is done in Triton kernels.
    Returns:
      output: [batch_size, 16, 512], dtype=torch.bfloat16
      lse: [batch_size, 16], dtype=torch.float32
    """
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
    device = q_nope.device
    dtype = q_nope.dtype

    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    num_pages = ckv_cache.shape[0]

    # Prepare output tensors
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float('inf'), dtype=torch.float32, device=device)

    # Make sure inputs are float32 for Triton kernels
    qn = q_nope.contiguous().to(torch.float32)   # [B, H, D]
    qp = q_pe.contiguous().to(torch.float32)     # [B, H, Dp]
    Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, D]
    Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dp]

    for b in range(batch_size):
        # token range from kv_indptr
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = max(0, page_end - page_beg)
        if L_tokens == 0:
            # No tokens for this batch, output zeros and lse stays -inf
            lse[b, :] = -float('inf')
            continue

        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [L_tokens]
        Kc = Kc_all[tok_idx]  # [L_tokens, D]
        Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

        # Loop over heads
        for j in range(num_qo_heads):
            # Select qn[j] and qp[j]
            qn_j = qn[b, j, :]   # [D]
            qp_j = qp[b, j, :]   # [Dp]

            # Compute v[i] = qn_j · Kc[i] + qp_j · Kp[i]
            v = torch.empty(L_tokens, dtype=torch.float32, device=device)
            matvec_add_kernel[(L_tokens,)](
                qn_j, qp_j, Kc, Kp, v,
                L_tokens, head_dim_ckv, head_dim_kpe,
                BLOCK_K=128, num_warps=4
            )

            # Compute lse = logsumexp_base2(v)
            lse_b_j = torch.empty(1, dtype=torch.float32, device=device)
            inv_ln2 = 1.0 / math.log(2.0)
            lse_base2_kernel[(1,)](
                v, lse_b_j, L_tokens, inv_ln2,
                BLOCK_L=256, num_warps=1
            )
            lse_val = lse_b_j[0]

            # Compute attn[i] = exp(v[i] / ln(2) - lse)
            attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
            softmax_base2_kernel[(L_tokens,)](
                v, lse_b_j, attn, L_tokens, inv_ln2,
                BLOCK_L=256, num_warps=4
            )

            # Compute out[:, j, :] = attn @ Kc
            y = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
            matvec_write_y_kernel[(head_dim_ckv,)](
                attn, Kc, y, L_tokens, head_dim_ckv,
                BLOCK_D=128, num_warps=4
            )
            output[b, j, :] = y.to(torch.bfloat16)

            # Update lse
            lse[b, j] = lse_val

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure inputs are on CUDA
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        # Run Triton-only implementation
        return _run_triton_only(*args)


def run(*args):
    return ModelNew()(*args)
