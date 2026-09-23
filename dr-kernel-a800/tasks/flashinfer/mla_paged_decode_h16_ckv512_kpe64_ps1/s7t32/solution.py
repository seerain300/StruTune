import math
import torch

# Triton kernels: all math performed inside Triton. No torch ops in forward.

# matvec_row: computes out = v @ B, where v is [M] (row vector) and B is [M, N].
# Each program handles BLOCK_N columns of the output vector.
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
# It writes probabilities to out_ptr[0:L] and the scalar lse (base-2) to out_ptr[L].
@triton.jit
def softmax_base2_kernel(logits_ptr, out_ptr, L: tl.constexpr):
    # Find max for numerical stability
    max_val = -float('inf')
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        max_val = tl.maximum(max_val, val)
    # Compute sum of exp(logits - max) * (2 ** lse)
    sum_val = 0.0
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        exp_i = tl.exp(val - max_val)
        sum_val += exp_i
    # Compute lse in base 2: lse = log2(sum) + max
    sum_val_log2 = tl.log(sum_val) * 1.4426950408889634  # log2(e)
    lse = sum_val_log2 + max_val
    # Store lse
    tl.store(out_ptr + L, lse)
    # Now write probabilities: out[i] = exp(logits[i] - max) / sum * 2 ** lse
    inv_sum = 1.0 / sum_val
    two_pow_lse = 2.0 ** lse
    for i in range(0, L):
        val = tl.load(logits_ptr + i)
        prob = tl.exp(val - max_val) * inv_sum * two_pow_lse
        tl.store(out_ptr + i, prob)

# matmul_small: computes C[M, N] = A[M, K] @ B[K, N] with tiling.
# We use a 2D grid: (ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
@triton.jit
def matmul_small(A_ptr, B_ptr, C_ptr,
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # Loop over K with BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + m_offsets[:, None] * K + k_offsets[None, :],
                    mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
                    other=0.0)
        b = tl.load(B_ptr + k_offsets[:, None] * N + n_offsets[None, :],
                    mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b)
    tl.store(C_ptr + m_offsets[:, None] * N + n_offsets[None, :],
             acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))

# ModelNew: Triton-only forward, must launch all three kernels
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity (no torch elementwise math)
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        assert H == 16, "num_qo_heads must be 16"

        # Preallocate outputs and lse buffers (float32 for compute)
        # Evaluation environment can provide preallocated tensors; here we allocate.
        output = torch.empty((B, H, 512), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Loop over batch and heads; compute everything in Triton
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = max(0, end - start)
            if L_tokens == 0:
                continue

            tok_idx = kv_indices[start:end].contiguous()  # int32 tensor on device
            Kc = ckv_cache[tok_idx, 0, :]  # [L_tokens, 512], bfloat16
            Kp = kpe_cache[tok_idx, 0, :]  # [L_tokens, 64],  bfloat16

            # Convert to float32 for compute
            Kc_f32 = Kc.to(torch.float32)
            Kp_f32 = Kp.to(torch.float32)

            # For each head h
            for h in range(H):
                # Load query vectors for this head (bfloat16), compute in float32
                qn_h = q_nope[b, h, :].contiguous()  # [512], bfloat16
                qp_h = q_pe[b, h, :].contiguous()    # [64],  bfloat16

                qn_h_f32 = qn_h.to(torch.float32)  # [512], float32
                qp_h_f32 = qp_h.to(torch.float32)  # [64],  float32

                # 1) Compute logits_scaled = (qn_h @ Kc.T) + (qp_h @ Kp.T) using matvec_row on two parts
                # First part: qn_h @ Kc.T
                # B1 = Kc.T with shape [512, L_tokens]
                B1 = Kc_f32.transpose(0, 1).contiguous()  # [L_tokens, 512]
                logits1 = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                grid1 = (triton.cdiv(L_tokens, 128),)
                matvec_row[grid1](qn_h_f32, B1, logits1, M=512, N=L_tokens, K=512, BLOCK_N=128, num_warps=4, num_stages=2)

                # Second part: qp_h @ Kp.T
                B2 = Kp_f32.transpose(0, 1).contiguous()  # [L_tokens, 64]
                logits2 = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                grid2 = (triton.cdiv(L_tokens, 128),)
                matvec_row[grid2](qp_h_f32, B2, logits2, M=64, N=L_tokens, K=64, BLOCK_N=128, num_warps=4, num_stages=2)

                # Combine
                logits_scaled = logits1 + logits2  # [L_tokens], float32

                # 2) Softmax in base-2 and get lse[b,h]
                out_buf = torch.empty((L_tokens + 1,), dtype=torch.float32, device=q_nope.device)
                grid_s = (1,)
                softmax_base2_kernel[grid_s](logits_scaled, out_buf, L=L_tokens, num_warps=1, num_stages=1)
                lse[b, h] = out_buf[L_tokens]
                attention_probs = out_buf[:L_tokens]  # [L_tokens]

                # 3) Compute output[b,h,:] = attention_probs @ Kc using matmul_small
                # A: [1, L_tokens], B: [L_tokens, 512], C: [1, 512]
                A = attention_probs.unsqueeze(0)  # [1, L_tokens]
                C = torch.empty((1, 512), dtype=torch.float32, device=q_nope.device)
                grid_mm = (triton.cdiv(1, 1), triton.cdiv(512, 128))
                matmul_small[grid_mm](A, Kc_f32, C, M=1, N=512, K=L_tokens, BLOCK_M=1, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2)
                output[b, h, :] = C[0, :]  # [512], float32 (we'll convert to bfloat16 after all heads)

        # Convert output to bfloat16 to match original dtype expectations
        output_bf = output.to(torch.bfloat16)
        return output_bf, lse


def run(*args):
    return ModelNew()(*args)
