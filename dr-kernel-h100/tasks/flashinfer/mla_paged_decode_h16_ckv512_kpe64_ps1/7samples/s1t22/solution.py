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
    BLOCK_K: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    acc = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptrs = Kc_ptr + i * D + k_off
        kc = tl.load(kc_ptrs, mask=mask_k, other=0.0)              # [BLOCK_K]
        acc += tl.sum(qn_slice * kc, axis=0)
    # Reduce over Kp dimension (Dp)
    for kp in range(0, Dp, BLOCK_K):
        kp_off = kp + tl.arange(0, BLOCK_K)
        mask_kp = kp_off < Dp
        qp_slice = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)  # [BLOCK_K]
        kp_ptrs = Kp_ptr + i * Dp + kp_off
        kp_vec = tl.load(kp_ptrs, mask=mask_kp, other=0.0)            # [BLOCK_K]
        acc += tl.sum(qp_slice * kp_vec, axis=0)
    tl.store(v_ptr + i, acc)


@triton.jit
def lse_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    L: tl.int32,
    ln2: tl.float32,  # natural log of 2
    BLOCK_L: tl.constexpr,
):
    # Compute stable logsumexp in base 2: lse = (max(v) + log(sum(exp(v - max)))) * ln2
    m = -float("inf")
    sum_exp = 0.0
    for i in range(0, L, BLOCK_L):
        offs = i + tl.arange(0, BLOCK_L)
        mask = offs < L
        vi = tl.load(v_ptr + offs, mask=mask, other=-float("inf"))
        local_max = tl.max(vi, axis=0)
        m = tl.maximum(m, local_max)
        exp_val = tl.exp(vi - m)  # sum already in local block
        sum_exp += tl.sum(exp_val, axis=0)
    lse = (m + tl.log(sum_exp)) * ln2
    tl.store(lse_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1] scalar lse for this head
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    ln2: tl.float32,
    BLOCK_L: tl.constexpr,
):
    # Compute attn[i] = exp((v[i] - lse) / ln2)
    lse_val = tl.load(lse_ptr)
    for i in range(0, L, BLOCK_L):
        offs = i + tl.arange(0, BLOCK_L)
        mask = offs < L
        vi = tl.load(v_ptr + offs, mask=mask, other=0.0)
        attn = tl.exp((vi - lse_val) / ln2)
        tl.store(attn_ptr + offs, attn, mask=mask)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D]
    y_ptr,            # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # One program per output dimension h in [0, D)
    h = tl.program_id(0)
    acc = 0.0
    for i in range(0, L, BLOCK_K):
        k_off = i + tl.arange(0, BLOCK_K)
        mask = k_off < L
        ai = tl.load(attn_ptr + k_off, mask=mask, other=0.0)   # [BLOCK_K]
        kc_ptrs = Kc_ptr + k_off * D + h
        kc = tl.load(kc_ptrs, mask=mask, other=0.0)            # [BLOCK_K]
        acc += tl.sum(ai * kc, axis=0)
    tl.store(y_ptr + h, acc)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Shapes and constants
    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    num_pages = ckv_cache.shape[0]
    # Cast inputs to float32 for computation
    device = q_nope.device
    q_nope_f32 = q_nope.to(torch.float32)
    q_pe_f32 = q_pe.to(torch.float32)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    # Natural log of 2 for base-2 normalization
    ln2 = math.log(2.0)

    for b in range(batch_size):
        # Determine token range for this batch element
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L = page_end - page_beg
        if L <= 0:
            # No tokens for this batch element
            lse[b] = -float("inf")
            continue

        # Gather token indices
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [L]
        # Gather Kc and Kp for these tokens
        Kc = Kc_all[tok_idx]  # [L, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L, head_dim_kpe]

        # Compute per-head logits vector v using Triton
        for j in range(num_qo_heads):
            # qn[j, :] and qp[j, :]
            qn = q_nope_f32[b, j, :]             # [D]
            qp = q_pe_f32[b, j, :]               # [Dp]

            # Allocate v for this head
            v = torch.empty(L, dtype=torch.float32, device=device)

            grid_v = (L,)
            matvec_add_kernel[grid_v](
                qn, qp, Kc, Kp, v,
                L=L, D=head_dim_ckv, Dp=head_dim_kpe, BLOCK_K=128,
            )

            # Compute lse (base-2 logsumexp) for this head using Triton
            lse_scalar = torch.empty(1, dtype=torch.float32, device=device)
            lse_base2_kernel[(1,)](
                v, lse_scalar,
                L=L, ln2=ln2, BLOCK_L=1024,
            )
            lse[b, j] = lse_scalar[0]

            # Compute attn (softmax with base-2 normalization) using Triton
            attn = torch.empty(L, dtype=torch.float32, device=device)
            softmax_base2_kernel[(1,)](
                v, lse[b, j], attn,
                L=L, ln2=ln2, BLOCK_L=1024,
            )

            # Compute final output vector y = attn @ Kc using Triton
            y = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
            grid_y = (head_dim_ckv,)
            matvec_write_y_kernel[grid_y](
                attn, Kc, y,
                L=L, D=head_dim_ckv, BLOCK_K=128,
            )
            output[b, j, :] = y

    # Cast output to bfloat16 to match original signature
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse  # lse kept in float32 as in original


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure inputs are on CUDA
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)


def run(*args):
    return ModelNew()(*args)
