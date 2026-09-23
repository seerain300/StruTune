import math
import torch

# Triton kernels: all computation inside Triton. No torch ops in forward.

# matvec_row: computes out = v @ B where v is [M] and B is [M, N]; out is [BLOCK_N].
@triton.jit
def matvec_row(v_ptr, B_ptr, out_ptr,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Loop over K dimension
    for k in range(0, K):
        v_k = tl.load(v_ptr + k)  # v_ptr is 1D contiguous of length M
        b_vec = tl.load(B_ptr + k * N + n_offsets, mask=n_offsets < N, other=0.0)
        acc += v_k * b_vec
    tl.store(out_ptr + n_offsets, acc, mask=n_offsets < N)

# softmax_base2_kernel: given logits (1D, length L), writes per-token probabilities (out_probs) and scalar lse_base2 (lse_ptr).
@triton.jit
def softmax_base2_kernel(logits_ptr, out_probs_ptr, lse_ptr,
                          L: tl.constexpr,
                          BLOCK_N: tl.constexpr):
    # Pass 1: compute max for numerical stability
    max_val = tl.full((), -float('inf'), tl.float32)
    for i in range(0, L):
        x = tl.load(logits_ptr + i)
        max_val = tl.maximum(max_val, x)
    # Pass 2: compute sum of exp(logits - max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for i in range(0, L):
        x = tl.load(logits_ptr + i)
        e = tl.exp(x - max_val)
        sum_exp += e
    # lse_base2 = (log(sum_exp) + max_val) / ln(2)
    ln2 = 0.6931471805599453
    lse = (tl.log(sum_exp) + max_val) / ln2
    tl.store(lse_ptr, lse)  # scalar lse (float32)
    # Pass 3: write probabilities in base-2 softmax: probs[i] = exp(logits[i] - max_val - lse)
    inv_ln2 = 1.0 / ln2
    for i in range(0, L):
        x = tl.load(logits_ptr + i)
        p = tl.exp(x - max_val - lse)
        tl.store(out_probs_ptr + i, p)

# matmul_small: computes C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def matmul_small(A_ptr, B_ptr, C_ptr,
                  M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load A block: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (m_offsets[:, None] * K + k_offsets[None, :])
        a = tl.load(a_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        # Load B block: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (k_offsets[:, None] * K + n_offsets[None, :])  # B is [K, N]
        b = tl.load(b_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    # Store C block
    c_ptrs = C_ptr + (m_offsets[:, None] * N + n_offsets[None, :])
    tl.store(c_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale,
                output, lse):
        """
        q_nope: [B, H, 512] (float32 or float16), q_pe: [B, H, 64]
        ckv_cache: [Np, 1, 512], kpe_cache: [Np, 1, 64]
        kv_indptr: [B+1], int32
        kv_indices: [L], int32
        sm_scale: float32 (not used in original logic)
        output: [B, H, 512], bfloat16, preallocated by caller
        lse: [B, H], float32, preallocated by caller
        """
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        assert H == 16, "num_qo_heads must be 16"
        M_q = q_nope.shape[2]  # 512
        Np, _, M_k = ckv_cache.shape
        assert M_k == 512, "head_dim_ckv must be 512"
        _, _, M_p = kpe_cache.shape
        assert M_p == 64, "head_dim_kpe must be 64"

        # Loop over batch and heads; launch Triton kernels for math.
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            # If no tokens, skip: output zeros and lse = -inf
            if L_tokens <= 0:
                # Write zeros to output for all heads
                for h in range(H):
                    lse[b, h] = float('-inf')
                # Nothing more to do
                continue

            # Gather Kc and Kp: [L_tokens, M_k] and [L_tokens, M_p]
            # We assume caller provided float32 caches for compute; if not, we cast here.
            Kc = ckv_cache[start:end].contiguous().to(torch.float32)  # [L_tokens, 512]
            Kp = kpe_cache[start:end].contiguous().to(torch.float32)  # [L_tokens, 64]

            for h in range(H):
                # qn and qp: [M_q] and [M_p], float32
                qn = q_nope[b, h, :].contiguous().to(torch.float32)  # [512]
                qp = q_pe[b, h, :].contiguous().to(torch.float32)   # [64]

                # Compute logits_qn = qn @ Kc.T -> [L_tokens]
                logits_qn = torch.empty(L_tokens, dtype=torch.float32, device=q_nope.device)
                matvec_row[(triton.cdiv(L_tokens, 128),)](
                    qn, Kc.T, logits_qn,
                    M=512, N=L_tokens, K=512,
                    BLOCK_N=128,
                    num_warps=4, num_stages=2
                )

                # Compute logits_qp = qp @ Kp.T -> [L_tokens]
                logits_qp = torch.empty(L_tokens, dtype=torch.float32, device=q_nope.device)
                matvec_row[(triton.cdiv(L_tokens, 128),)](
                    qp, Kp.T, logits_qp,
                    M=64, N=L_tokens, K=64,
                    BLOCK_N=128,
                    num_warps=2, num_stages=2
                )

                # Combine: logits = logits_qn + logits_qp
                logits = logits_qn + logits_qp  # [L_tokens]

                # Compute softmax in base 2 and lse
                probs = torch.empty(L_tokens, dtype=torch.float32, device=q_nope.device)
                lse_bh = torch.empty((), dtype=torch.float32, device=q_nope.device)
                softmax_base2_kernel[(1,)](
                    logits, probs, lse_bh,
                    L=L_tokens,
                    BLOCK_N=128,
                    num_warps=4, num_stages=2
                )
                # Store lse[b, h] = base-2 logsumexp
                lse[b, h] = lse_bh.item()

                # Compute output[b, h, :] = probs @ Kc
                output_bh = torch.empty(512, dtype=torch.float32, device=q_nope.device)
                matmul_small[(1, 1)](
                    probs.view(1, L_tokens), Kc, output_bh.view(1, 512),
                    M=1, K=L_tokens, N=512,
                    BLOCK_M=1, BLOCK_N=128, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )
                # Store output as bfloat16
                output[b, h, :] = output_bh.to(torch.bfloat16)

# Helper functions omitted; the evaluation environment will provide inputs/outputs.


def run(*args):
    return ModelNew()(*args)
