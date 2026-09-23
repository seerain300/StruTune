import torch
import triton
import triton.language as tl

# Triton kernels: all math inside kernels, host only does data movement.
# - matvec_row: out = v @ B, v is 1xM, B is MxN (we pass B transposed for dot-products).
# - softmax_base2_kernel: softmax in base-2 and returns logsumexp (base-2) as a scalar.
# - matmul_small: C[M,N] = A[M,K] @ B[K,N] for small sizes.

@triton.jit
def matvec_row(B, v, out, L, N, stride_b_m, stride_b_n, stride_v, stride_out,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # out = v @ B where v is [M] (M=L tokens), B is [M, N]
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over tokens dimension (M=L) in tiles
    for k0 in range(0, L, BLOCK_M):
        k_offsets = k0 + tl.arange(0, BLOCK_M)
        mask_k = k_offsets < L

        # Load v[k]
        v_tile = tl.load(v + k_offsets * stride_v, mask=mask_k, other=0.0)  # [BLOCK_M]

        # Load B[k, n] tile: [BLOCK_M, BLOCK_N]
        B_tile = tl.load(B + k_offsets[:, None] * stride_b_m + n_offsets[None, :] * stride_b_n,
                         mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate: acc[n] += sum_k v[k] * B[k, n]
        acc += tl.sum(B_tile * v_tile[:, None], axis=0)

    # Store results to out
    tl.store(out + n_offsets * stride_out, acc, mask=mask_n)


@triton.jit
def softmax_base2_kernel(logits, out_ptr, L, sm_scale, BLOCK: tl.constexpr):
    # Compute softmax in base 2 for a 1D logits vector of length L
    # Writes: out_ptr[0:L] = probabilities, out_ptr[L] = base-2 logsumexp scalar
    offs = tl.arange(0, BLOCK)
    mask = offs < L
    x = tl.load(logits + offs, mask=mask, other=-float("inf"))
    x = x * sm_scale

    # Max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # Exponentiate
    exp_x = tl.exp(x)

    # Sum in base 2: sum_exp_base2 = sum(exp(x)) / ln(2)
    sum_exp = tl.sum(exp_x, axis=0)
    ln2 = 0.6931471805599453  # 1 / log(2)
    sum_exp_base2 = sum_exp / ln2

    # Store logsumexp (base 2) to out_ptr[L]
    tl.store(out_ptr + L, sum_exp_base2)

    # Compute probabilities
    probs = exp_x / sum_exp_base2  # already scaled to base-2 sense

    # Store probabilities to out_ptr[0:L]
    tl.store(out_ptr + offs, probs, mask=mask)


@triton.jit
def matmul_small(A, B, C, M, K, N, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute C[M,N] = A[M,K] @ B[K,N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        a = tl.load(A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
                    mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
                    other=0.0)  # [BLOCK_M, BLOCK_K]
        b = tl.load(B + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn,
                    mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
                    other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(a, b)

    # Write back
    tl.store(C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
             acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Entry point model
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda

        batch_size = q_nope.shape[0]
        heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        assert heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Prepare output buffers (torch allocations only for storage; math in Triton)
        output = torch.empty((batch_size, heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((batch_size, heads), dtype=torch.float32, device=q_nope.device)

        for b in range(batch_size):
            # Determine valid tokens for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg

            # If no tokens, zero output for this batch and skip
            if L_tokens <= 0:
                output[b].zero_()
                continue

            # Gather Kc and Kp for these tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L_tokens]
            Kc_f = ckv_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [L_tokens, 512]
            Kp_f = kpe_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [L_tokens, 64]

            for h in range(heads):
                # Per-head queries
                qn_h = q_nope[b, h, :].contiguous().to(torch.float32)  # [512]
                qp_h = q_pe[b, h, :].contiguous().to(torch.float32)   # [64]

                # Compute logits_scaled = (qn_h @ Kc.T) + (qp_h @ Kp.T) -> [L_tokens]
                logits1 = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                Kc_T = Kc_f.T  # [512, L_tokens]
                stride_kc_m = Kc_T.stride(0)  # 512
                stride_kc_n = Kc_T.stride(1)  # L_tokens
                stride_qm = qn_h.stride(0)    # 1
                stride_out = logits1.stride(0)  # 1

                grid1 = (triton.cdiv(L_tokens, 128),)
                matvec_row[grid1](Kc_T, qn_h, logits1, L_tokens, L_tokens,
                                  stride_kc_m, stride_kc_n, stride_qm, stride_out,
                                  BLOCK_M=128, BLOCK_N=128, num_warps=4, num_stages=2)

                logits2 = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                Kp_T = Kp_f.T  # [64, L_tokens]
                stride_kp_m = Kp_T.stride(0)  # 64
                stride_kp_n = Kp_T.stride(1)  # L_tokens

                grid2 = (triton.cdiv(L_tokens, 128),)
                matvec_row[grid2](Kp_T, qp_h, logits2, L_tokens, L_tokens,
                                  stride_kp_m, stride_kp_n, 1, 1,
                                  BLOCK_M=128, BLOCK_N=128, num_warps=4, num_stages=2)

                logits_scaled = logits1 + logits2  # [L_tokens], float32

                # Compute softmax (base 2) and logsumexp (base 2)
                probs_and_lse = torch.empty((L_tokens + 1,), dtype=torch.float32, device=q_nope.device)
                grid_soft = (1,)
                softmax_base2_kernel[grid_soft](logits_scaled, probs_and_lse, L_tokens, sm_scale, 1024,
                                                BLOCK=1024, num_warps=4, num_stages=2)

                # lse per head (base-2)
                lse[b, h] = probs_and_lse[L_tokens]  # scalar tensor

                # Compute output vector: out[b, h, :] = attention_probs @ Kc -> [512]
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=q_nope.device)

                # A: attention_probs as [1, L_tokens], B: Kc_f as [L_tokens, 512], C: out_vec as [1, 512]
                probs_row = probs_and_lse[:L_tokens].unsqueeze(0)  # [1, L_tokens]
                stride_am = probs_row.stride(0)  # 1
                stride_ak = probs_row.stride(1)  # L_tokens

                stride_bk = Kc_f.stride(0)       # L_tokens
                stride_bn = Kc_f.stride(1)       # 512

                C = out_vec  # [512] acts as [1, 512] with strides (1,1)
                stride_cm = 1
                stride_cn = 1

                grid_mm = (triton.cdiv(1, 64), triton.cdiv(head_dim_ckv, 128))
                matmul_small[grid_mm](probs_row, Kc_f, C, 1, L_tokens, head_dim_ckv,
                                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                                      BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2)

                # Store to output[b, h, :] as bfloat16
                output[b, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
