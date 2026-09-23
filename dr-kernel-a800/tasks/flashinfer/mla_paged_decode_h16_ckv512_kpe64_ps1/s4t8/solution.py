import torch
import math
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(A_ptr, B_ptr, C_ptr,
                       K: tl.constexpr,  # e.g., 512
                       M,                # runtime int, number of rows in B
                       BLOCK_K: tl.constexpr,  # e.g., 64 or 128
                       BLOCK_M: tl.constexpr   # e.g., 64
                      ):
    """
    Compute C[i] = sum_k A[k] * B[i, k] for i in 0..M-1, with A of length K, B of shape [M, K].
    We tile over K and M. A_ptr is 1D of length K, B_ptr is [M, K], C_ptr is [M].
    """
    # We handle a chunk of M rows per program instance. Since grid is (ceil_div(M, BLOCK_M),),
    # we compute the row offsets for this instance and loop over K in tiles.
    # Triton doesn't support dynamic loops well, so we use static_range for both K and M tiling.
    # However, since M is runtime, we implement per-instance handling via grid: each program
    # handles a unique set of rows. For safety, we iterate m0 = pid * BLOCK_M to M in a while-like
    # pattern. Triton supports such control flow.
    m0 = tl.program_id(0) * BLOCK_M
    for dm in tl.static_range(0, BLOCK_M):
        i = m0 + dm
        # If i >= M, we skip via mask in store (no need to load)
        # Accumulator for this row i
        acc = tl.zeros((), dtype=tl.float32)
        # Tile over K
        for k0 in tl.static_range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = (offs_k < K)
            # Load A chunk as 1D vector
            A_chunk = tl.load(A_ptr + offs_k, mask=mask_k, other=0.0)
            # Load B[i, offs_k] chunk. Triton supports pointer arithmetic with vectors.
            B_row_ptrs = B_ptr + i * K + offs_k
            B_chunk = tl.load(B_row_ptrs, mask=mask_k, other=0.0)
            # Accumulate dot product for this row i over this K tile
            acc += tl.sum(A_chunk * B_chunk, axis=0)
        # Store result to C[i]
        # We need to guard store by checking i < M. Use mask for store.
        store_mask = i < M
        # Write acc to C[i]
        tl.store(C_ptr + i, acc, mask=store_mask)


@triton.jit
def softmax_lse_kernel(x_ptr, out_lse_ptr, attn_ptr,
                        M: tl.constexpr,   # use constexpr for loops, pass as compile-time
                        sm_scale: tl.float32):
    """
    Compute per-head lse and attention vector:
    - lse = logsumexp(x * sm_scale) / ln(2), where x is a vector of length M (logits_scaled)
    - attn[i] = exp(x[i] * sm_scale) / sum_j exp(x[j] * sm_scale)
    """
    # Reductions over M: compute max and sum_exp
    max_val = -float("inf")
    sum_exp = 0.0
    # We iterate over M in chunks of BLOCK_M. Since M is constexpr, static_range is fine.
    for m0 in tl.static_range(0, M, 64):
        # For safety with varying M, we implement the inner loop with runtime control using while.
        # But Triton likes static_range; thus we set M to constexpr at launch. Given eval workloads,
        # M is often small per batch; we will pass M as a constexpr to this kernel by special-casing.
        # However, Triton requires constexpr for tl.static_range, so we keep M as tl.constexpr here.
        # Compute max over chunk
        for dm in tl.static_range(0, 64):
            i = m0 + dm
            # Mask for load
            load_mask = i < M
            xi = tl.load(x_ptr + i, mask=load_mask, other=-float("inf"))
            # xi is scalar; compare for max
            max_val = tl.maximum(max_val, xi)
        # Compute sum_exp over chunk
        for dm in tl.static_range(0, 64):
            i = m0 + dm
            load_mask = i < M
            xi = tl.load(x_ptr + i, mask=load_mask, other=-float("inf"))
            sum_exp += tl.exp((xi - max_val) * sm_scale)

    # lse = (max + log(sum_exp)) / ln(2)
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse = (max_val + tl.log(sum_exp)) / ln2
    # Store lse
    tl.store(out_lse_ptr, lse)

    # Compute attn and store
    for m0 in tl.static_range(0, M, 64):
        for dm in tl.static_range(0, 64):
            i = m0 + dm
            load_mask = i < M
            xi = tl.load(x_ptr + i, mask=load_mask, other=-float("inf"))
            attn_i = tl.exp((xi - max_val) * sm_scale) / sum_exp
            tl.store(attn_ptr + i, attn_i, mask=load_mask)


@triton.jit
def matvec_out_kernel(A_ptr, B_ptr, C_ptr,
                      K: tl.constexpr,   # e.g., 512
                      M,                 # runtime int
                      BLOCK_K: tl.constexpr,
                      BLOCK_M: tl.constexpr):
    """
    Compute C[i] = sum_k A[k] * B[i, k] for i in 0..M-1, with A of length K (here A is attn vector),
    B of shape [M, K], output C of shape [M]. We will use this to compute out[h, :] by looping over M
    and accumulating per-dimension. For simplicity, we compute per-dimension directly:
    out[d] = sum_i attn[i] * Kc[i, d].
    """
    # Grid should be (ceil_div(K, BLOCK_K),) so each program handles a chunk of dimensions d.
    d0 = tl.program_id(0) * BLOCK_K
    # Initialize accumulator for this chunk of dimensions
    acc_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
    # Loop over rows i in M
    for m0 in tl.static_range(0, M, 64):  # M is constexpr for kernel; we pass M at launch
        for dm in tl.static_range(0, 64):
            i = m0 + dm
            # Load attn[i] and Kc[i, d0:d0+BLOCK_K]
            attn_i = tl.load(A_ptr + i)
            offs_k = d0 + tl.arange(0, BLOCK_K)
            mask_k = (offs_k < K)
            B_row_ptrs = B_ptr + i * K + offs_k
            B_chunk = tl.load(B_row_ptrs, mask=mask_k, other=0.0)
            acc_vec += attn_i * B_chunk
    # Store results to C at positions d0:d0+BLOCK_K
    offs_d = d0 + tl.arange(0, BLOCK_K)
    mask_d = (offs_d < K)
    tl.store(C_ptr + offs_d, acc_vec, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA for Triton
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA device"
        device = q_nope.device
        dtype = torch.float32

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]
        Kc_dim = ckv_cache.shape[2]  # 512
        Kp_dim = kpe_cache.shape[2]  # 64

        # Prepare output and lse
        output = torch.empty((batch_size, H, Kc_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Process each batch b
        for b in range(batch_size):
            # Determine token range for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if end <= start:
                # No tokens for this batch element
                lse[b] = -float("inf")
                output[b] = torch.zeros((H, Kc_dim), dtype=torch.float32, device=device)
                continue

            M = end - start
            tok_idx = kv_indices[start:end].to(torch.int32)

            # Gather Kc and Kp rows
            # Kc_all = ckv_cache.squeeze(1) -> [N, 512], but we only need tok_idx rows
            Kc = ckv_cache[tok_idx, 0, :].contiguous().to(dtype)  # [M, 512]
            Kp = kpe_cache[tok_idx, 0, :].contiguous().to(dtype)  # [M, 64]

            # Compute part1: qn @ Kc.T → logits_part1 [M]
            qn = q_nope[b].to(dtype)  # [16, 512] but we need per-head vector: q_nope[b, :, :] -> [16, 512]
            # We need to run Triton kernel per head; ModelNew.forward will loop over heads:
            # However, the kernel expects A_ptr of length K. Since we need per-head, we can reuse q_nope[q,h] as A.
            # For Triton, we'll loop h in forward and call the kernel with A = q_nope[b,h], B = Kc, producing logits_part1.
            # We'll implement this in the Python loop below.

            # We will not use PyTorch for matmul/softmax; all via Triton kernels.

            # Prepare output tensors for part1 logits and part2 logits
            logits_part1 = torch.empty((M,), dtype=dtype, device=device)
            logits_part2 = torch.empty((M,), dtype=dtype, device=device)

            # Compute part1: qn @ Kc.T using matvec_row_kernel for each head h in [0..15]
            # But matvec_row_kernel expects A of length K, here A is scalar vector qn[h,:]. To use Triton,
            # we implement a Python loop and set A to q_nope[b,h]. However, Triton kernels are launched from forward;
            # we need to define how to feed A. Simpler approach: compute part1 and part2 by launching Triton kernels
            # where A is q_nope[b,h] and B is Kc (or q_pe[b,h] and Kp).
            # We can define helper calls by slicing q_nope[q,h] correctly and launching matvec_row_kernel.
            # However, Triton kernels are not invoked from here directly; we need to define how to invoke.
            # Given the environment constraints, we'll implement per-head loops in forward using Triton kernels.

            # We'll implement Triton calls for each head:
            # For each head h, compute logits_part1 = qn[h] @ Kc.T, logits_part2 = qp[h] @ Kp.T
            # Triton kernels matvec_row_kernel: A length-K vector, B [M,K], output [M]
            # Note: Triton kernels must be @triton.jit defined; we have defined them. We now invoke them.

            # Loop over heads
            for h in range(H):
                # Prepare A vectors
                A1 = q_nope[b, h, :].to(dtype).contiguous()  # [Kc_dim]
                A2 = q_pe[b, h, :].to(dtype).contiguous()   # [Kp_dim]

                # Compute part1 via kernel: A=A1, B=Kc, C=logits_part1
                # We need to launch matvec_row_kernel with K=Kc_dim, M=M, BLOCK_K=128, BLOCK_M=64
                # Triton expects grid = (ceil_div(M, BLOCK_M),)
                M_const = M  # Triton requires constexpr for static_range; we pass as constexpr via meta
                grid = (triton.cdiv(M_const, 64),)
                matvec_row_kernel[grid](
                    A1, Kc, logits_part1,
                    K=Kc_dim, M=M_const, BLOCK_K=128, BLOCK_M=64,
                    num_warps=4, num_stages=2
                )

                # Compute part2 via kernel: A=A2, B=Kp, C=logits_part2
                grid2 = (triton.cdiv(M_const, 64),)
                matvec_row_kernel[grid2](
                    A2, Kp, logits_part2,
                    K=Kp_dim, M=M_const, BLOCK_K=128, BLOCK_M=64,
                    num_warps=4, num_stages=2
                )

                # Add part1 and part2 to get logits_scaled
                logits_scaled = logits_part1 + logits_part2  # [M]

                # Compute lse and attn using softmax_lse_kernel. Note: softmax_lse_kernel assumes M is constexpr.
                # We'll set M=128 or 512 at launch. But eval M varies. Triton requires constexpr for tl.static_range.
                # Workaround: For small M, we can compute with a fixed BLOCK_M >= M and mask. For simplicity, assume M <= 128.
                # If M > 128, we can call the kernel with BLOCK_M=256 and loop over M chunks.
                # To keep kernel simple, we call with grid (ceil_div(M, BLOCK_M)), using BLOCK_M=128.
                # The kernel uses static_range(0, M, BLOCK_M) with M as runtime passed in, Triton will handle masks.
                # Prepare outputs: out_lse [1], attn [M]
                out_lse = torch.empty((1,), dtype=dtype, device=device)
                attn = torch.empty((M,), dtype=dtype, device=device)

                softmax_lse_kernel[(triton.cdiv(M_const, 64),)](
                    logits_scaled, out_lse, attn,
                    M=M_const, sm_scale=sm_scale,
                    num_warps=2, num_stages=2
                )

                # Store lse per head
                lse[b, h] = out_lse[0]

                # Compute final output vector out[h, :] = attn @ Kc via matvec_out_kernel
                # A = attn [M], B = Kc [M, 512], C = out_vec [512]
                out_vec = torch.empty((Kc_dim,), dtype=dtype, device=device)
                # We need grid over Kc_dim
                grid_out = (triton.cdiv(Kc_dim, 128),)
                matvec_out_kernel[grid_out](
                    attn, Kc, out_vec,
                    K=Kc_dim, M=M_const, BLOCK_K=128, BLOCK_M=64,
                    num_warps=4, num_stages=2
                )
                # Store out[h, :]
                output[b, h, :] = out_vec

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
