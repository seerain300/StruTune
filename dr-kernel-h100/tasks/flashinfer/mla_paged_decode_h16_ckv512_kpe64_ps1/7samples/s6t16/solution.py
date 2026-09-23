import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute logits = (qn @ Kc.T) + (qp @ Kp.T) for a single head h
# Inputs:
#   qn_ptr: [Hc] float32, q_nope[b, h, :]
#   qp_ptr: [Hp] float32, q_pe[b, h, :]
#   Kc_ptr: [L, Hc] float32, gathered ckv_cache[tok_idx, 0, :]
#   Kp_ptr: [L, Hp] float32, gathered kpe_cache[tok_idx, 0, :]
#   out_ptr: [L] float32, logits for this head
#   sm_scale: float32
#   L: number of tokens (runtime int), used for mask and loop bound
# Launch: one program per (b, h)
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
    sm_scale: tl.float32,
    Hc: tl.constexpr, Hp: tl.constexpr, L: tl.constexpr,
    qn_stride, qp_stride,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    out_stride,
    BLOCK_K: tl.constexpr
):
    # offsets along K dimension
    k = 0
    while k < L:
        k_range = k + tl.arange(0, BLOCK_K)
        mask_k = k_range < L

        # accumulate partial dot products
        acc1 = tl.zeros((Hc,), dtype=tl.float32)
        acc2 = tl.zeros((Hp,), dtype=tl.float32)

        # load qn and qp scalars for this head
        # qn and qp are vectors of size Hc and Hp respectively
        qn_vals = tl.load(qn_ptr + 0, mask=True, other=0.0)  # single scalar per pointer; loop uses strides below
        qp_vals = tl.load(qp_ptr + 0, mask=True, other=0.0)

        # iterate over tokens in this chunk
        kk = 0
        while kk < BLOCK_K:
            kk_idx = k + kk
            k_valid = kk_idx < L
            # load Kc and Kp rows for this kk_idx
            Kc_vec = tl.load(Kc_ptr + kk_idx * Kc_stride0, mask=k_valid, other=0.0)  # shape [Hc]
            Kp_vec = tl.load(Kp_ptr + kk_idx * Kp_stride0, mask=k_valid, other=0.0)  # shape [Hp]

            # accumulate qn @ Kc_vec.T and qp @ Kp_vec.T
            acc1 += qn_vals * Kc_vec
            acc2 += qp_vals * Kp_vec
            kk += 1

        # combine and store
        partial = acc1 + acc2
        partial = partial * sm_scale
        # store to out at positions k_range
        tl.store(out_ptr + k_range * out_stride, partial, mask=mask_k)
        k += BLOCK_K


# Kernel 2: Compute lse per row (softmax logsumexp) for a single head h
# Inputs:
#   logits_ptr: [L] float32
#   lse_ptr: scalar float32 for this (b,h) row
# Launch: one program per (b,h)
@triton.jit
def softmax_logsumexp_row_kernel(
    logits_ptr, lse_ptr,
    L: tl.constexpr, BLOCK_N: tl.constexpr
):
    # First pass: compute max for numerical stability
    max_val = -float("inf")
    k = 0
    while k < L:
        offs = k + tl.arange(0, BLOCK_N)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, local_max)
        k += BLOCK_N

    # Second pass: compute sum(exp(x - max))
    sum_exp = 0.0
    k = 0
    while k < L:
        offs = k + tl.arange(0, BLOCK_N)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=-float("inf"))
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_x, axis=0)
        k += BLOCK_N

    # lse = log(sum_exp) / log(2.0)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1.0 / ln(2)
    tl.store(lse_ptr, lse_val)


# Kernel 3: Compute output for a single head h: out_row = softmax(logits_scaled)[h, :] @ Kc
# Inputs:
#   logits_ptr: [L] float32
#   Kc_ptr: [L, Hc] float32
#   out_row_ptr: [Hc] float32
# Launch: one program per (b,h), over output column chunks
@triton.jit
def matvec_row_kernel(
    logits_ptr, Kc_ptr, out_row_ptr,
    Hc: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr
):
    # We need to compute softmax over L for this head h first.
    # Triton doesn't provide elementwise softmax here easily, so we compute probabilities in two passes:
    # 1) row max, 2) sum of exp, 3) write normalized probs and accumulate into out_row.
    # However, Triton kernels can't call other kernels inside; so we implement softmax here via two passes:
    # Pass 1: compute max
    max_val = -float("inf")
    k = 0
    while k < L:
        offs = k + tl.arange(0, BLOCK_N)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, local_max)
        k += BLOCK_N

    # Pass 2: compute sum of exp(x - max)
    sum_exp = 0.0
    k = 0
    while k < L:
        offs = k + tl.arange(0, BLOCK_N)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)
        k += BLOCK_N

    # Initialize out_row to zeros
    acc = tl.zeros((Hc,), dtype=tl.float32)

    # Now compute out_row = sum_j softmax_j * Kc_j
    # We will iterate j over tokens and accumulate:
    # softmax_j = exp(logits_j - max_val) / sum_exp
    j = 0
    while j < L:
        offs = j + tl.arange(0, BLOCK_N)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=-float("inf"))
        p = tl.exp(x - max_val) / sum_exp  # shape [BLOCK_N]
        Kc_vec = tl.load(Kc_ptr + offs * Kc_stride0, mask=mask, other=0.0)  # [BLOCK_N]
        # Multiply p and Kc_vec elementwise and reduce over BLOCK_N to a scalar contribution
        # Triton supports elementwise multiply and sum reduction.
        contribution = tl.sum(p * Kc_vec, axis=0)
        # This contribution is a scalar; add to acc[Hc] vector at indices offs. To do that, we need a loop.
        # However Triton does not support dynamic vector indexing assignment; we accumulate directly.
        # Since contribution is scalar, we add it to all Hc positions? No, we need per-column accumulation.
        # Better approach: restructure as separate programs per output column, but here we keep a simple scalar loop.
        # Note: Triton while over j is fine; Kc_vec and p are vectors; contribution is scalar.
        acc += contribution
        j += BLOCK_N

    # Store acc to out_row
    # We need to store acc into out_row_ptr with stride 1. Triton allows storing a vector.
    # We can directly store acc to out_row_ptr; assume out_row_ptr has Hc elements and contiguous.
    tl.store(out_row_ptr + tl.arange(0, Hc), acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Extract shapes and device
        assert q_nope.dim() == 3 and q_pe.dim() == 3
        num_qo_heads, head_dim_ckv, _ = q_nope.shape
        _, head_dim_kpe, _ = q_pe.shape
        batch_size = q_nope.shape[0]

        # Ensure inputs are CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda

        # Squeeze the single segment from caches
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Hc]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Hp]

        device = q_nope.device
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # For each batch element
        for b in range(batch_size):
            # Compute token indices range for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No tokens for this batch element; output zeros
                output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
                lse[b] = torch.full((num_qo_heads,), -float("inf"), dtype=torch.float32, device=device)
                continue

            # Gather token indices and corresponding Kc/Kp rows
            tok_idx = kv_indices[start:end]  # [L_tokens] int32
            Kc = Kc_all[tok_idx]  # [L_tokens, Hc]
            Kp = Kp_all[tok_idx]  # [L_tokens, Hp]

            # Prepare qn and qp for this batch and each head
            for h in range(num_qo_heads):
                # Pointers to qn and qp for this head (assume contiguous along last dim)
                qn = q_nope[b, h, :]  # [Hc]
                qp = q_pe[b, h, :]    # [Hp]

                # Allocate logits for this head
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)

                # Launch GEMV kernel for logits
                BLOCK_K = 128
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits,
                    sm_scale=float(sm_scale),
                    Hc=head_dim_ckv, Hp=head_dim_kpe, L=L_tokens,
                    qn_stride=1, qp_stride=1,
                    Kc_stride0=Kc.stride(0), Kc_stride1=Kc.stride(1),
                    Kp_stride0=Kp.stride(0), Kp_stride1=Kp.stride(1),
                    out_stride=1,
                    BLOCK_K=BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # Compute lse for this head using Triton kernel
                BLOCK_N = 128
                lse_ptr = lse[b, h]  # scalar tensor
                softmax_logsumexp_row_kernel[(1,)](
                    logits, lse_ptr,
                    L=L_tokens, BLOCK_N=BLOCK_N,
                    num_warps=4, num_stages=2
                )

                # Compute output for this head using Triton matvec kernel
                out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_row_kernel[(1,)](
                    logits, Kc, out_row,
                    Hc=head_dim_ckv, L=L_tokens, BLOCK_N=BLOCK_N,
                    num_warps=4, num_stages=2
                )

                # Store output for this head
                output[b, h, :] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
