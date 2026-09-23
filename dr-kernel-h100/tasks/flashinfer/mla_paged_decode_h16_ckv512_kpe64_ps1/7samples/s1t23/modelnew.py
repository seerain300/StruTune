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
    D: tl.int32,      # head_dim_ckv (512)
    Dp: tl.int32,     # head_dim_kpe (64)
    ln2: tl.float32,  # natural log of 2
    BLOCK_K: tl.constexpr,
):
    # One program per output index i
    i = tl.program_id(0)
    acc1 = 0.0
    acc2 = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        # Load qn slice [BLOCK_K]
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)
        # Load Kc row i for these K offsets
        kc = tl.load(Kc_ptr + i * D + k_off, mask=mask_k, other=0.0)
        acc1 += tl.sum(qn_slice * kc, axis=0)
    # Reduce over Kp dimension (Dp)
    for kp in range(0, Dp, BLOCK_K):
        kp_off = kp + tl.arange(0, BLOCK_K)
        mask_kp = kp_off < Dp
        qp_slice = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)
        kp = tl.load(Kp_ptr + i * Dp + kp_off, mask=mask_kp, other=0.0)
        acc2 += tl.sum(qp_slice * kp, axis=0)
    v = acc1 + acc2  # contribution of token i to logits for this head
    # Store v[i]
    tl.store(v_ptr + i, v)


@triton.jit
def lse_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    L: tl.int32,      # number of tokens
    ln2: tl.float32,  # natural log of 2
    BLOCK_L: tl.constexpr,
):
    # Stable logsumexp base-2: lse = (max(v) + log(sum(exp(v - max)))) * (1/ln(2))
    # Pass scalar lse back via lse_ptr[0]
    # Compute max(v)
    max_v = -float("inf")
    for l in range(0, L, BLOCK_L):
        offs = l + tl.arange(0, BLOCK_L)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-float("inf"))
        # reduce to scalar
        block_max = tl.max(v, axis=0)
        max_v = tl.maximum(max_v, block_max)
    sum_exp = 0.0
    for l in range(0, L, BLOCK_L):
        offs = l + tl.arange(0, BLOCK_L)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(v * (1.0 / ln2) - max_v * (1.0 / ln2)), axis=0)
    lse_val = (max_v + tl.log(sum_exp)) * (1.0 / ln2)
    # store scalar
    tl.store(lse_ptr, lse_val)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_val,          # scalar float32 (already normalized by ln2)
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    ln2: tl.float32,  # we'll use lse_val directly; ln2 only for clarity
    BLOCK_L: tl.constexpr,
):
    # Compute attn[i] = exp((v[i] - lse_val) * (1/ln2))
    for l in range(0, L, BLOCK_L):
        offs = l + tl.arange(0, BLOCK_L)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        attn = tl.exp((v - lse_val) * (1.0 / ln2))
        tl.store(attn_ptr + offs, attn, mask=mask)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D], row-major
    y_ptr,            # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # One program per output dimension h
    h = tl.program_id(0)
    acc = 0.0
    for l in range(0, L, BLOCK_K):
        k_off = l + tl.arange(0, BLOCK_K)
        mask_l = k_off < L
        attn = tl.load(attn_ptr + k_off, mask=mask_l, other=0.0)
        kc = tl.load(Kc_ptr + k_off * D + h, mask=mask_l, other=0.0)
        acc += tl.sum(attn * kc, axis=0)
    tl.store(y_ptr + h, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Assertions and constants (as in original)
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    device = q_nope.device

    # Extract L per batch element from kv_indptr (batch_size + 1 entries)
    # L[b] = kv_indptr[b+1] - kv_indptr[b]
    L_per_b = kv_indptr[1:].to(torch.int32) - kv_indptr.to(torch.int32)  # [batch_size]
    # Build Kc_all and Kp_all (as float32)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

    # Allocate outputs
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # we'll return bfloat16
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    ln2 = math.log(2.0)

    # Loop over batch and heads; perform all heavy compute in Triton
    for b in range(batch_size):
        L = int(L_per_b[b].item())
        if L <= 0:
            # No tokens for this batch element; output zeros
            for j in range(num_qo_heads):
                output[b, j, :] = 0.0
            lse[b, :] = -float("inf")
            continue

        # Determine token range and gather indices
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        # Gather tokens for this batch element
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [L]
        # Gather Kc and Kp for this batch element
        Kc = Kc_all[tok_idx]  # [L, 512], row-major contiguous
        Kp = Kp_all[tok_idx]  # [L, 64]

        # Prepare qn and qp (float32) for this head j
        for j in range(num_qo_heads):
            # Load qn[j, :] and qp[j, :]
            qn = q_nope[:, j, :].reshape(-1).to(torch.float32)  # [512]
            qp = q_pe[:, j, :].reshape(-1).to(torch.float32)    # [64]

            # Allocate v, attn, y
            v = torch.empty(L, dtype=torch.float32, device=device)
            # Triton: compute v[j, :]
            grid_v = (L,)
            matvec_add_kernel[grid_v](
                qn, qp, Kc, Kp, v,
                L=L, D=head_dim_ckv, Dp=head_dim_kpe, ln2=ln2, BLOCK_K=128,
            )

            # Triton: compute lse (base-2) for this head
            lse_scalar = torch.empty(1, dtype=torch.float32, device=device)
            lse_base2_kernel[(1,)](
                v, lse_scalar,
                L=L, ln2=ln2, BLOCK_L=1024,
            )
            lse[b, j] = lse_scalar[0]

            # Triton: compute attn[j, :]
            attn = torch.empty(L, dtype=torch.float32, device=device)
            softmax_base2_kernel[(1,)](
                v, lse[b, j], attn,
                L=L, ln2=ln2, BLOCK_L=1024,
            )

            # Triton: compute y[j, :] = attn @ Kc
            y = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
            grid_y = (head_dim_ckv,)
            matvec_write_y_kernel[grid_y](
                attn, Kc, y,
                L=L, D=head_dim_ckv, BLOCK_K=128,
            )
            output[b, j, :] = y

    # Cast output to bfloat16 to match original signature
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure inputs are on CUDA
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)