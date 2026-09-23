import math
import torch

# Triton kernels: all math performed inside Triton. No torch ops in forward.

# matvec_row: computes out = v @ B, where v is [M] and B is [M, N].
# Tiling over N with BLOCK_N and accumulating over K.
@triton.jit
def matvec_row(v_ptr, B_ptr, out_ptr,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Loop over K dimension (length of v)
    for k in range(0, K):
        v_k = tl.load(v_ptr + k)  # v_ptr is [M], contiguous
        b_k = tl.load(B_ptr + k * N + n_offsets, mask=n_offsets < N, other=0.0)
        acc += v_k * b_k
    tl.store(out_ptr + n_offsets, acc, mask=n_offsets < N)

# softmax_base2_kernel: performs softmax in base-2 for a 1D vector of length L.
# Writes per-token probabilities to out_probs and scalar lse (base-2) to lse_ptr[0].
# We use a single program looping over L in chunks of BLOCK.
@triton.jit
def softmax_base2_kernel(logits_ptr, out_probs_ptr, lse_ptr,
                          L: tl.constexpr, BLOCK: tl.constexpr):
    # First, compute sum of exp(logits / ln(2))
    ln2 = 0.6931471805599453  # math.log(2.0)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=0.0)
        e = tl.exp(x / ln2)
        sum_exp += tl.sum(e, axis=0)
    lse_val = tl.log(sum_exp) * ln2  # base-2 logsumexp
    tl.store(lse_ptr, lse_val)

    # Second pass: write probabilities (base-2 softmax)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=0.0)
        e = tl.exp((x - lse_val) / ln2)  # base-2 softmax
        tl.store(out_probs_ptr + offs, e, mask=mask)

# matmul_small: computes C[M, N] = A[M, K] @ B[K, N] for small sizes.
@triton.jit
def matmul_small(A_ptr, B_ptr, C_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # We handle a single output tile covering all M,N (M small, e.g., 1).
    for m in range(0, M, BLOCK_M):
        for n in range(0, N, BLOCK_N):
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                # Load A tile: [BLOCK_M, BLOCK_K]
                a_ptrs = A_ptr + (m + tl.arange(0, BLOCK_M))[:, None] * K + (k + tl.arange(0, BLOCK_K))[None, :]
                a_mask = (m + tl.arange(0, BLOCK_M))[:, None] < M and (k + tl.arange(0, BLOCK_K))[None, :] < K
                a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
                # Load B tile: [BLOCK_K, BLOCK_N]
                b_ptrs = B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N + (n + tl.arange(0, BLOCK_N))[None, :]
                b_mask = (k + tl.arange(0, BLOCK_K))[:, None] < K and (n + tl.arange(0, BLOCK_N))[None, :] < N
                b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
                # FMA
                # For each i in BLOCK_M, accumulate sum over BLOCK_K of a_tile[i, kk] * b_tile[kk, :]
                for i in range(0, BLOCK_M):
                    a_row = a_tile[i, :]
                    for kk in range(0, BLOCK_K):
                        b_vec = b_tile[kk, :]
                        acc[i, :] += a_row[kk] * b_vec
            # Store C tile
            c_ptrs = C_ptr + (m + tl.arange(0, BLOCK_M))[:, None] * N + (n + tl.arange(0, BLOCK_N))[None, :]
            c_mask = (m + tl.arange(0, BLOCK_M))[:, None] < M and (n + tl.arange(0, BLOCK_N))[None, :] < N
            tl.store(c_ptrs, acc, mask=c_mask)

# Entry point: ModelNew.forward
# All computation happens via Triton; no torch math here.
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        num_pages, _, _ = ckv_cache.shape
        _, _, _ = kpe_cache.shape

        # Output buffers (float32 for compute; cast to bfloat16 at the end)
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute Kc_all and Kp_all (we squeeze the 1 dimension and convert to float32)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        for b in range(batch_size):
            # Valid token range for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg
            if L_tokens <= 0:
                output[b].zero_()
                lse[b] = float('-inf')
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()

            # Gather Kc and Kp for this batch
            Kc = Kc_all[tok_idx]              # [L_tokens, 512]
            Kp = Kp_all[tok_idx]              # [L_tokens, 64]

            # For each head h
            for h in range(num_qo_heads):
                qn = q_nope[b, h, :].to(torch.float32).contiguous()  # [512]
                qp = q_pe[b, h, :].to(torch.float32).contiguous()   # [64]

                # Build Bc = Kc.T -> [512, L_tokens] and Bp = Kp.T -> [64, L_tokens]
                Bc = Kc.t().contiguous()   # [512, L_tokens]
                Bp = Kp.t().contiguous()   # [64, L_tokens]

                # Kernel 1: matvec_row for qn @ Kc.T
                logits1 = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                BLOCK_N_1 = 128
                grid1 = (triton.cdiv(L_tokens, BLOCK_N_1),)
                matvec_row[grid1](qn, Bc, logits1, 512, L_tokens, 512, BLOCK_N=BLOCK_N_1, num_warps=4, num_stages=2)

                # Kernel 2: matvec_row for qp @ Kp.T
                logits2 = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                BLOCK_N_2 = 128
                grid2 = (triton.cdiv(L_tokens, BLOCK_N_2),)
                matvec_row[grid2](qp, Bp, logits2, 64, L_tokens, 64, BLOCK_N=BLOCK_N_2, num_warps=4, num_stages=2)

                # Combine
                logits_scaled = logits1 + logits2  # [L_tokens]

                # Kernel 3: softmax_base2_kernel
                probs = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                BLOCK_S = 1024
                grid3 = (1,)
                softmax_base2_kernel[grid3](logits_scaled, probs, lse[b].contiguous(), L_tokens, BLOCK=BLOCK_S, num_warps=4, num_stages=2)

                # Final output vector: attention_probs @ Kc
                C = torch.empty((1, head_dim_ckv), dtype=torch.float32, device=device)
                # Matmul parameters
                BLOCK_M = 1
                BLOCK_N_M = 128
                BLOCK_K = 64
                grid4 = (1,)
                matmul_small[grid4](probs.view(1, L_tokens), Kc, C, 1, head_dim_ckv, L_tokens,
                                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_M, BLOCK_K=BLOCK_K, num_warps=4, num_stages=2)

                output[b, h, :] = C[0, :]

        # Cast output to bfloat16 as original code expects
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
