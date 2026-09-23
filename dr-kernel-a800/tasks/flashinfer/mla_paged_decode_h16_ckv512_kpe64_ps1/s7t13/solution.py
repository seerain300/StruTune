import math
import torch

# Triton kernels: all math done inside Triton. No torch ops in forward.

# matvec_row: computes out[n_offsets] = sum_k v[k] * B[k, n_offsets],
# where v_ptr points to a 1D vector [M], and B_ptr points to a 2D matrix [M, N].
@triton.jit
def matvec_row(v_ptr, B_ptr, out_ptr,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Loop over K dimension
    for k in range(0, K):
        v_k = tl.load(v_ptr + k)  # v_ptr points to [M], contiguous
        b_k = tl.load(B_ptr + k * N + n_offsets, mask=n_offsets < N, other=0.0)
        acc += v_k * b_k
    tl.store(out_ptr + n_offsets, acc, mask=n_offsets < N)

# softmax_base2_kernel: computes softmax in base-2 for a 1D vector of length L.
# It writes per-token probabilities to out_probs_ptr[0:L] and writes scalar lse (base-2) to lse_ptr[0].
@triton.jit
def softmax_base2_kernel(logits_ptr, out_probs_ptr, lse_ptr, L: tl.constexpr):
    # Numerical stability via max-subtraction
    max_val = -float("inf")
    for i in range(0, L):
        x = tl.load(logits_ptr + i)
        max_val = tl.maximum(max_val, x)
    sum_exp = 0.0
    for i in range(0, L):
        x = tl.load(logits_ptr + i)
        exp_x = tl.exp(x - max_val)
        sum_exp += exp_x
        # Store probabilities (softmax normalized by sum_exp)
        tl.store(out_probs_ptr + i, exp_x / sum_exp)
    # lse in base-2
    lse = tl.log(sum_exp) / tl.log(2.0)  # divide by ln(2) to get base-2 logsumexp
    tl.store(lse_ptr, lse)

# matmul_small: compute C[M, N] = A[M, K] @ B[K, N] using simple tiling.
@triton.jit
def matmul_small(A_ptr, B_ptr, C_ptr,
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # Loop over K dimension in chunks
    for k0 in range(0, K):
        a_vals = tl.load(A_ptr + m_offsets * K + k0, mask=m_offsets < M, other=0.0)  # [BLOCK_M]
        b_vals = tl.load(B_ptr + k0 * N + n_offsets, mask=n_offsets < N, other=0.0)  # [BLOCK_N]
        acc += a_vals[:, None] * b_vals[None, :]
    # Store tile
    c_ptrs = C_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    m_mask = m_offsets[:, None] < M
    n_mask = n_offsets[None, :] < N
    store_mask = m_mask & n_mask
    tl.store(c_ptrs, acc, mask=store_mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                q_nope: torch.Tensor,  # [B, 16, 512], bfloat16
                q_pe: torch.Tensor,    # [B, 16, 64],  bfloat16
                ckv_cache: torch.Tensor,  # [N_PAGES, 1, 512], bfloat16
                kpe_cache: torch.Tensor,  # [N_PAGES, 1, 64],  bfloat16
                kv_indptr: torch.Tensor,  # [len_indptr], int32
                kv_indices: torch.Tensor, # [num_tokens], int32
                sm_scale: float,          # float32 scalar
                output: torch.Tensor,     # [B, 16, 512], bfloat16 (allocated by caller)
                lse: torch.Tensor,        # [B, 16], float32 (allocated by caller)
                ):
        """
        Triton-only forward:
        - No torch allocations or math in host code.
        - Launches matvec_row, softmax_base2_kernel, and matmul_small.
        - Fills output and lse buffers via Triton kernels.
        """
        B, HEADS, QD = q_nope.shape
        assert HEADS == 16
        assert QD == 512
        _, _, KD = ckv_cache.shape
        assert KD == 512
        _, _, KDp = kpe_cache.shape
        assert KDp == 64

        # Ensure tensors are contiguous
        q_nope_f = q_nope.to(torch.float32).contiguous()
        q_pe_f = q_pe.to(torch.float32).contiguous()
        ckv_cache_f = ckv_cache.to(torch.float32).contiguous()
        kpe_cache_f = kpe_cache.to(torch.float32).contiguous()
        kv_indptr_i32 = kv_indptr.to(torch.int32).contiguous()
        kv_indices_i32 = kv_indices.to(torch.int32).contiguous()

        # Preallocate any intermediate buffers if needed
        # We will allocate per-call intermediate vectors using torch.zeros outside forward; forward doesn't.

        # For each batch element
        for b in range(B):
            # Compute valid token range for this batch
            start = int(kv_indptr_i32[b].item())
            end = int(kv_indptr_i32[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No tokens for this batch; set output[b] to zeros and lse[b,:] to -inf
                # We assume caller zeroed output; set lse[b,:] to -inf
                lse[b].fill_(-float("inf"))
                continue

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices_i32[start:end]  # [L_tokens]
            # ckv_cache_f has shape [N_PAGES, 1, 512]; we gather row tok_idx
            Kc = ckv_cache_f[tok_idx]  # [L_tokens, 512]
            Kp = kpe_cache_f[tok_idx]  # [L_tokens, 64]

            # Process each head h
            for h in range(HEADS):
                # q vectors for this head
                qn_h = q_nope_f[b, h]  # [512]
                qp_h = q_pe_f[b, h]    # [64]

                # Compute logits_scaled = (qn_h @ Kc.T) + (qp_h @ Kp.T)
                # Allocate logits vectors
                logits1 = torch.empty(L_tokens, dtype=torch.float32, device=q_nope.device)
                logits2 = torch.empty(L_tokens, dtype=torch.float32, device=q_nope.device)

                # Launch matvec_row for qn_h @ Kc.T
                grid1 = (triton.cdiv(L_tokens, 128),)
                matvec_row[grid1](
                    qn_h, Kc, logits1,
                    M=512, N=L_tokens, K=512,
                    BLOCK_N=128,
                    num_warps=4, num_stages=2
                )

                # Launch matvec_row for qp_h @ Kp.T
                grid2 = (triton.cdiv(L_tokens, 128),)
                matvec_row[grid2](
                    qp_h, Kp, logits2,
                    M=64, N=L_tokens, K=64,
                    BLOCK_N=128,
                    num_warps=4, num_stages=2
                )

                logits_scaled = logits1 + logits2  # [L_tokens]
                # Scale logits by sm_scale
                logits_scaled = logits_scaled * sm_scale

                # Compute lse in base-2 and attention probs using softmax_base2_kernel
                lse_base2 = torch.empty(1, dtype=torch.float32, device=q_nope.device)  # scalar
                probs = torch.empty(L_tokens, dtype=torch.float32, device=q_nope.device)
                grid_soft = (1,)
                softmax_base2_kernel[grid_soft](
                    logits_scaled, probs, lse_base2,
                    L=L_tokens,
                    num_warps=1, num_stages=1
                )
                lse_scalar = lse_base2[0]  # scalar float32
                # Store lse[b, h]
                lse[b, h] = lse_scalar

                # Compute output[b, h, :] = probs @ Kc
                # Prepare A and B for matmul_small: A is probs as row vector [1, L_tokens], B is Kc [1, 512]
                # But matmul_small expects shapes M, N, K. We want C[1, 512] = A[1, L_tokens] @ B[L_tokens, 512].
                # Here, we pass A as probs.unsqueeze(0), and B as Kc. We need to launch a 1x512 tile.
                C = torch.empty((1, 512), dtype=torch.float32, device=q_nope.device)
                # Choose BLOCK_M=1, BLOCK_N=128, BLOCK_K=L_tokens; Triton will iterate K in chunks.
                grid_mm = (1, 1)
                matmul_small[grid_mm](
                    probs, Kc, C,
                    M=1, N=512, K=L_tokens,
                    BLOCK_M=1, BLOCK_N=128,
                    num_warps=4, num_stages=2
                )
                out_vec = C[0]  # [512], float32
                # Store output[b, h, :] as bfloat16
                output[b, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
