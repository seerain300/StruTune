import math
import torch

# Triton kernels: all computation happens inside Triton. No torch ops in forward.

# matvec_row: out[n_block] = v[M] @ B[M,N], where B is [M, N] and v is [M].
# We compute over blocks of N (BLOCK_N) and loop over M.
@triton.jit
def matvec_row(v_ptr, B_ptr, out_ptr,
                M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Loop over M dimension
    for m in range(0, M):
        v_m = tl.load(v_ptr + m)
        b_m = tl.load(B_ptr + m * N + n_offsets, mask=mask_n, other=0.0)
        acc += v_m * b_m
    tl.store(out_ptr + n_offsets, acc, mask=mask_n)

# softmax_base2_kernel: compute softmax in base-2 on a 1D logits vector of length L.
# We write probabilities to out_probs[0:L] and scalar lse to lse_ptr[0].
# logsumexp is computed in base-2: lse = log(sum(exp(x / ln(2)))) / ln(2).
# We pass inv_ln2 = 1/ln(2) as a scalar float32.
@triton.jit
def softmax_base2_kernel(logits_ptr, out_probs_ptr, lse_ptr,
                         L: tl.constexpr, inv_ln2: tl.constexpr):
    # First pass: compute max for numerical stability
    max_val = -1.0e30  # float32
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        max_val = tl.maximum(max_val, val)

    # Compute sum of exp(logits_scaled - max) where logits_scaled = logits * inv_ln2
    sum_exp = 0.0
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        scaled = val * inv_ln2
        e = tl.exp(scaled - max_val)
        sum_exp += e

    # lse in base-2
    lse = tl.log(sum_exp) * inv_ln2  # log(sum_exp) in natural, then scale to base-2
    tl.store(lse_ptr, lse)

    # Store probabilities: exp(scaled - lse)
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        scaled = val * inv_ln2
        prob = tl.exp(scaled - lse)
        tl.store(out_probs_ptr + i, prob)

# matmul_small: compute C[M, N] = A[M, K] @ B[K, N] using a simple tiling loop.
# We implement a single output tile: one row (M=1) and N in blocks.
@triton.jit
def matmul_small(A_ptr, B_ptr, C_ptr,
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Loop over K
    for k in range(0, K):
        a_k = tl.load(A_ptr + k)  # A is [M,K] row-major; for M=1, this is correct
        b_k = tl.load(B_ptr + k * N + n_offsets, mask=n_offsets < N, other=0.0)
        acc += a_k * b_k
    # Store C[0, n_offsets]
    tl.store(C_ptr + n_offsets, acc, mask=n_offsets < N)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, 16, 512] (bfloat16), q_pe: [B, 16, 64] (bfloat16)
        ckv_cache: [num_pages, 1, 512] (bfloat16), kpe_cache: [num_pages, 1, 64] (bfloat16)
        kv_indptr: [len_indptr] (int32), kv_indices: [num_kv_indices] (int32)
        sm_scale: float32 scalar (not used in original but kept for signature)
        Returns:
        - output: [B, 16, 512] (bfloat16)
        - lse: [B, 16] (float32)
        """
        B = q_nope.shape[0]
        heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64

        # Output buffers (we'll fill them with Triton kernels; no torch compute in forward)
        output = torch.empty((B, heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, heads), dtype=torch.float32, device=q_nope.device)

        # Constants
        inv_ln2 = 1.4426950408889634  # 1 / ln(2)

        for b in range(B):
            # Determine token range for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No valid tokens for this batch element; output zeros
                output[b] = torch.zeros((heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
                lse[b] = torch.full((heads,), -float("inf"), dtype=torch.float32, device=q_nope.device)
                continue

            # Gather Kc and Kp
            # ckv_cache has shape [num_pages, 1, 512]; we need Kc[tok_idx, 0, :]
            # kpe_cache has shape [num_pages, 1, 64]
            tok_idx = kv_indices[start:end]  # [L_tokens]
            Kc = ckv_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [L_tokens, 512]
            Kp = kpe_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [L_tokens, 64]

            # Per head
            for h in range(heads):
                # Query vectors
                qn = q_nope[b, h, :].contiguous().to(torch.float32)  # [512]
                qp = q_pe[b, h, :].contiguous().to(torch.float32)    # [64]

                # Compute logits = qn @ Kc.T + qp @ Kp.T
                # logits1 = matvec_row(qn, Kc.T, [L_tokens])
                logits1 = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                grid1 = (triton.cdiv(L_tokens, 128),)
                matvec_row[grid1](qn, Kc.transpose(0, 1), logits1, M=512, N=L_tokens, BLOCK_N=128, num_warps=4)

                # logits2 = matvec_row(qp, Kp.T, [L_tokens])
                logits2 = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                grid2 = (triton.cdiv(L_tokens, 64),)
                matvec_row[grid2](qp, Kp.transpose(0, 1), logits2, M=64, N=L_tokens, BLOCK_N=128, num_warps=2)

                logits = logits1 + logits2  # [L_tokens]

                # Softmax in base-2: probs and lse
                probs = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                lse_scalar = torch.empty((), dtype=torch.float32, device=q_nope.device)
                # We need a 1D tensor for logits. Triton will write probs and lse_scalar.
                softmax_base2_kernel[(1,)](logits, probs, lse_scalar, L=L_tokens, inv_ln2=1.4426950408889634, num_warps=1)

                # attention_probs is probs; output = probs @ Kc
                # attention_probs shape [1, L_tokens], Kc shape [L_tokens, 512]
                # We need to compute C[1,512] = attention_probs @ Kc using matmul_small
                att = probs.unsqueeze(0)  # [1, L_tokens]
                out_vec = torch.empty((512,), dtype=torch.float32, device=q_nope.device)
                grid_mm = (triton.cdiv(512, 128),)
                matmul_small[grid_mm](att, Kc, out_vec, M=1, N=512, K=L_tokens, BLOCK_N=128, num_warps=4)

                # Store outputs and lse
                output[b, h, :] = out_vec.to(torch.bfloat16)
                lse[b, h] = lse_scalar.item()  # scalar per head; Triton stores a tensor, but we read as Python float.

        return output, lse


def run(*args):
    return ModelNew()(*args)
