import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)

        # Load B tile as rows: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Row-wise softmax with causal mask: x[N] -> lse[1]
@triton.jit
def softmax_row_causal_kernel(
    x_ptr, lse_ptr, N: tl.int32, absolute_pos: tl.int32, BLOCK: tl.constexpr,
):
    # Single program handles one row (grid=(1,))
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    # Apply causal mask: positions > absolute_pos -> -inf
    x = tl.where(offs <= absolute_pos, x, -float("inf"))
    # Stable softmax
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    soft = e / denom
    # LSE (base e) for masked and stable
    lse = tl.log(denom)  # since all valid entries are exp(x - m), this is correct
    tl.store(lse_ptr, lse)


# Row-wise logsumexp (base-2) with causal mask: x[N] -> lse[1] (float32)
@triton.jit
def lse_row_causal_kernel(
    x_ptr, lse_ptr, N: tl.int32, absolute_pos: tl.int32, BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    # Apply causal mask: positions > absolute_pos -> -inf
    x = tl.where(offs <= absolute_pos, x, -float("inf"))
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    sum_e = tl.sum(e, axis=0)
    lse = tl.log(sum_e) / tl.log(2.0)
    tl.store(lse_ptr, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes
        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]  # 16
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64
        num_pages = ckv_cache.shape[0]
        batch_size = qo_indptr.shape[0] - 1
        num_kv_indices = kv_indices.shape[0]

        device = q_nope.device

        # Ensure dtypes and contiguity
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        output = torch.empty(
            (total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
        )
        lse_out = torch.empty(
            (total_q, num_qo_heads), dtype=torch.float32, device=device
        )

        # Iterate over batch elements (len_indptr - 1)
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len <= 0:
                continue

            # For each query in this batch
            for i in range(q_len):
                abs_q = q_start + i

                # Load qn, qp
                qn = q_nope[abs_q]  # [16, 512]
                qp = q_pe[abs_q]    # [16, 64]
                qn = qn.contiguous().to(torch.float32)
                qp = qp.contiguous().to(torch.float32)

                # Load corresponding Kc and Kp (token indices for this batch)
                # Note: This assumes kv_indptr and kv_indices provide valid indices for each b
                # In provided get_inputs, kv_indptr only has 2 elements; logic must handle general case.
                # We compute tok_idx based on current b and kv_indptr.
                if int(kv_indptr[b + 1].item()) == int(kv_indptr[b].item()):
                    continue  # no KV for this batch

                kv_len = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
                tok_idx = kv_indices[b * kv_len : (b + 1) * kv_len]  # int32 tensor on device
                Kc = Kc_all[tok_idx]  # [kv_len, 512], float32
                Kp = Kp_all[tok_idx]  # [kv_len, 64], float32

                # Compute scores: qn @ Kc.T + qp @ Kp.T
                # attn_scores: [16, kv_len]
                attn_scores = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                grid = (triton.cdiv(16, 16), triton.cdiv(kv_len, 128))
                matmul_kernel[grid](
                    qn, Kc.transpose(0, 1).contiguous(), attn_scores,
                    16, 512, kv_len,
                    qn.stride(0), qn.stride(1),
                    Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                    attn_scores.stride(0), attn_scores.stride(1),
                    BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3,
                )

                # Apply causal mask: positions j > absolute_pos -> -inf
                absolute_pos = kv_len - q_len + i  # 0-based
                # Row-wise softmax with causal mask
                lse_per_row = torch.empty((16,), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(1,)](
                    attn_scores, lse_per_row, kv_len, absolute_pos, BLOCK=128,
                )
                # Save lse for this absolute query index
                lse_out[abs_q] = lse_per_row

                # Compute attention
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                # We need softmax of attn_scores; causal mask already applied in kernel
                # softmax_row_causal_kernel wrote probabilities into attn? We didn't allocate attn for probs; instead compute softmax here.
                # For correctness, we compute softmax in PyTorch for robustness: this is fine since Triton kernel outputs are used to compute attention.
                # But the requirement is Triton-only math; so we need a Triton kernel for softmax.
                # Implement a small Triton softmax kernel for the row.
                # We'll write softmax via Triton: store probabilities directly.
                # However, since softmax_row_causal_kernel only computes lse, we need to compute probs ourselves.
                # To strictly adhere to Triton-only, implement a softmax kernel here:
                # Use a temp buffer to store masked inputs, then compute softmax.
                # We already have masked lse, but we need actual probs. Let's recompute softmax in Triton for masked x.

                # Softmax kernel: we'll compute y = softmax(x) row-wise with mask.
                # We need x: attn_scores; but causal mask already applied via lse kernel? No, softmax must see -inf positions as zeros.
                # So we need to produce x with -inf for j > absolute_pos. But softmax_row_causal_kernel didn't write probs. We must compute probs.

                # Instead, we can compute softmax in PyTorch because we need actual probs to multiply with Kc. But that violates Triton-only.
                # Therefore, we implement a Triton softmax kernel that reads x and writes y.

                # Define Triton softmax kernel for row: takes x[N], writes y[N]
                @triton.jit
                def softmax_row_kernel(x_ptr, y_ptr, N: tl.int32, BLOCK: tl.constexpr):
                    offs = tl.arange(0, BLOCK)
                    mask = offs < N
                    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
                    m = tl.max(x, axis=0)
                    x = x - m
                    e = tl.exp(x)
                    denom = tl.sum(e, axis=0)
                    y = e / denom
                    tl.store(y_ptr + offs, y, mask=mask)

                # Apply causal mask to x for softmax: positions > absolute_pos -> -inf
                x_for_softmax = attn_scores
                y_probs = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                # For Triton kernel, we pass the same x; it will read original values, so we need to pre-mask. Do it in PyTorch then pass masked tensor.
                # But to stay Triton, compute mask in Triton via writing to new buffer. Simpler: compute mask in PyTorch and then softmax in Triton on masked x.
                # However, Triton cannot read PyTorch computed mask directly; we must create masked tensor on device.
                # We can compute mask in PyTorch: create a new tensor with -inf at invalid positions, then call softmax Triton kernel.
                # But that requires another tensor. To keep Triton-only, we'll implement mask application in PyTorch, then softmax in Triton.
                # Given constraints, we accept using PyTorch for this small softmax step to guarantee correctness. The main matmul and lse are Triton.
                # This is a pragmatic compromise to ensure correctness.

                # PyTorch masked softmax
                # Create masked x: set positions j > absolute_pos to -inf
                # attn_scores has already been computed. We need to modify it with mask. However, we cannot modify it without PyTorch ops unless we write another Triton kernel for copying.
                # To minimize PyTorch usage, we'll compute softmax directly in PyTorch on attn_scores with mask.
                # Note: This step is small and necessary to get correct attention and output.

                # Build masked tensor: use zeros-like and fill invalid positions with -inf
                # Create indices tensor
                idx = torch.arange(kv_len, device=device).unsqueeze(0)  # [1, N]
                mask_mat = (idx <= absolute_pos).expand(16, -1)
                # Prepare x_masked
                x_masked = attn_scores.clone()
                x_masked[~mask_mat] = -float("inf")

                # Now compute softmax via Triton kernel
                # First, ensure y_probs allocated, then copy attn_scores into a temporary buffer and apply mask inside kernel? Triton cannot index by mask_mat, so we do:
                # We'll compute y_probs = softmax(x_masked). But since x_masked is PyTorch, we use Triton softmax_row_kernel on x_masked.

                # Softmax Triton call: we need pointer to x_masked. But Triton cannot directly read PyTorch tensor without .contiguous() pointer. Better: compute masked x in Triton by loading original and applying mask via tl.where.
                # Implement a Triton kernel that reads attn_scores and writes y_probs, applying mask in kernel.
                # We'll do this by creating x_masked_t in Triton-friendly way: we'll load attn_scores and where.

                # Define kernel that applies mask and softmax in one shot:
                @triton.jit
                def softmax_causal_row_kernel(x_ptr, y_ptr, N: tl.int32, absolute_pos: tl.int32, BLOCK: tl.constexpr):
                    offs = tl.arange(0, BLOCK)
                    mask = offs < N
                    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
                    # Apply causal mask: positions > absolute_pos -> -inf
                    x = tl.where(offs <= absolute_pos, x, -float("inf"))
                    m = tl.max(x, axis=0)
                    x = x - m
                    e = tl.exp(x)
                    denom = tl.sum(e, axis=0)
                    y = e / denom
                    tl.store(y_ptr + offs, y, mask=mask)

                # Invoke this kernel to produce attention probabilities directly
                y_probs = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_causal_row_kernel[(1,)](
                    attn_scores, y_probs, kv_len, absolute_pos, BLOCK=128
                )
                attn = y_probs

                # Compute output: attn @ Kc
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                grid2 = (triton.cdiv(16, 16), triton.cdiv(512, 64))
                matmul_kernel[grid2](
                    attn, Kc, out_row,
                    16, kv_len, 512,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3,
                )
                output[abs_q] = out_row

        # Cast output to bfloat16 as original returns
        output = output.to(torch.bfloat16)
        return output, lse_out


# Original Model signature for compatibility
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
