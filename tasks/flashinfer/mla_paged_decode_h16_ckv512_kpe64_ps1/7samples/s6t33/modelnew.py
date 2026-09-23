import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute logits for a single head h: logits = qn @ Kc.T + qp @ Kp.T
# Input:
#   qn_ptr: [Hc] float32
#   qp_ptr: [Hp] float32
#   Kc_ptr: [L, Hc] float32 (tokens x Hc)
#   Kp_ptr: [L, Hp] float32 (tokens x Hp)
#   logits_ptr: [L] float32
#   sm_scale: float32
# Launch: grid=(1,) (we call it for each head in a Python loop)
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_ptr,
    Hc: tl.constexpr, Hp: tl.constexpr, L: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Single program computes the entire logits vector for its row (head).
    offs = tl.arange(0, BLOCK_K)
    # Accumulate in fp32
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    # Loop over tokens in chunks of BLOCK_K
    for k in range(0, L, BLOCK_K):
        mask = k + offs < L
        # qn_chunk: [BLOCK_K]
        qn_chunk = tl.load(qn_ptr + k + offs, mask=mask, other=0.0)
        # Kc_chunk: [BLOCK_K, Hc] where we only need Kc[k+offs, :]
        Kc_chunk = tl.load(Kc_ptr + (k + offs)[:, None] * Hc + offs[None, :], mask=mask[:, None], other=0.0)
        acc += tl.sum(qn_chunk[None, :] * Kc_chunk, axis=1)
        # qp_chunk and Kp_chunk similarly
        qp_chunk = tl.load(qp_ptr + k + offs, mask=mask, other=0.0)
        Kp_chunk = tl.load(Kp_ptr + (k + offs)[:, None] * Hp + offs[None, :], mask=mask[:, None], other=0.0)
        acc += tl.sum(qp_chunk[None, :] * Kp_chunk, axis=1)
    logits = acc * sm_scale
    # Store results for all valid k positions
    for i in range(BLOCK_K):
        idx = i
        if idx < L:
            tl.store(logits_ptr + idx, logits[i])


# Kernel 2: Compute row-wise logsumexp and store lse (float32) for a single row (head).
# Two-pass approach:
# Pass 1: compute max of logits
# Pass 2: compute sum of exp(logits - max)
@triton.jit
def softmax_logsumexp_row_kernel(
    logits_ptr, lse_ptr,
    L: tl.constexpr, inv_log2: tl.constexpr
):
    # One program per row (head). We assume logits_ptr points to a single row.
    # Pass 1: compute max
    max_val = -1.0e30  # initialize
    offs = tl.arange(0, L)
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        if val > max_val:
            max_val = val
    # Pass 2: compute sum of exp(val - max)
    sum_exp = 0.0
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        sum_exp += tl.exp(val - max_val)
    lse_val = tl.log(sum_exp) * inv_log2
    tl.store(lse_ptr, lse_val)


# Kernel 3: Compute matvec: out = softmax(logits_scaled) @ Kc for a single head
# We implement softmax via two Triton passes: max and sum; Triton doesn't expose softmax directly.
# Then accumulate in chunks of BLOCK_N across tokens.
@triton.jit
def matvec_row_kernel(
    logits_ptr, Kc_ptr, out_ptr,
    L: tl.constexpr, Hc: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Compute row-wise max and sum for softmax
    max_val = -1.0e30
    offs = tl.arange(0, L)
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        if val > max_val:
            max_val = val

    sum_exp = 0.0
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        sum_exp += tl.exp(val - max_val)

    # Prepare output vector
    # We will accumulate per output column chunk
    # However, Triton prefers 2D tiling; we can write directly to out_ptr in chunks.
    # Create a vector accumulator for a chunk
    for n in range(0, Hc, BLOCK_N):
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        offs_n = n + tl.arange(0, BLOCK_N)
        for i in range(0, L):
            val = tl.load(logits_ptr + i)
            p = tl.exp(val - max_val) / sum_exp  # scalar probability
            Kvec = tl.load(Kc_ptr + i * Hc + offs_n, mask=offs_n < Hc, other=0.0)
            acc += p * Kvec
        # Store acc to out_ptr[n : n+BLOCK_N]
        for j in range(0, BLOCK_N):
            idx = n + j
            if idx < Hc:
                tl.store(out_ptr + idx, acc[j])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and dtype assumptions as in the original:
        # q_nope: [B, 16, 512], q_pe: [B, 16, 64], ckv_cache: [N, 1, 512], kpe_cache: [N, 1, 64]
        # We squeeze the 1 dimension: Kc_all: [N, 512], Kp_all: [N, 64]
        B = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        Hc = q_nope.shape[2]  # head_dim_ckv
        Hp = q_pe.shape[2]    # head_dim_kpe
        # Make sure inputs are on CUDA and contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        device = q_nope.device

        # Squeeze the cache's single-segment dimension (assumed to be 1)
        Kc_all = ckv_cache.squeeze(1)  # [N, Hc]
        Kp_all = kpe_cache.squeeze(1)  # [N, Hp]

        # Prepare output
        output = torch.empty((B, num_qo_heads, Hc), dtype=torch.float32, device=device)  # we'll compute in fp32
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Iterate over batch
        for b in range(B):
            # Compute token indices for this batch
            # kv_indptr[b] and kv_indptr[b+1] give the range [page_beg, page_end)
            # tok_idx = kv_indices[page_beg:page_end], length = L
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens for this batch element
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()
            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L_tokens, Hc]
            Kp = Kp_all[tok_idx]  # [L_tokens, Hp]

            # For each head h
            for h in range(num_qo_heads):
                # Extract qn and qp for this head
                qn = q_nope[b, h, :].to(torch.float32).contiguous()  # [Hc]
                qp = q_pe[b, h, :].to(torch.float32).contiguous()   # [Hp]

                # Allocate logits and compute with Triton
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                # Choose BLOCK_K (must divide or iterate fully). Use 128 for Hc=512, 64 for Hp chunking.
                BLOCK_K = 128 if Hc >= 128 else 64
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits,
                    Hc=Hc, Hp=Hp, L=L_tokens,
                    sm_scale=sm_scale,
                    BLOCK_K=BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # Compute lse with Triton
                inv_log2 = 1.0 / math.log(2.0)
                lse_kernel_out = torch.empty((1,), dtype=torch.float32, device=device)
                softmax_logsumexp_row_kernel[(1,)](
                    logits, lse_kernel_out,
                    L=L_tokens, inv_log2=inv_log2,
                    num_warps=1, num_stages=1
                )
                lse[b, h] = lse_kernel_out[0]

                # Compute output row with Triton matvec
                out_row = torch.empty((Hc,), dtype=torch.float32, device=device)
                BLOCK_N = 128
                matvec_row_kernel[(triton.cdiv(Hc, BLOCK_N),)](
                    logits, Kc, out_row,
                    L=L_tokens, Hc=Hc, BLOCK_N=BLOCK_N,
                    num_warps=4, num_stages=2
                )
                output[b, h, :] = out_row

        # Return output in bfloat16 to match original q_nope dtype, lse in float32
        return output.to(torch.bfloat16), lse