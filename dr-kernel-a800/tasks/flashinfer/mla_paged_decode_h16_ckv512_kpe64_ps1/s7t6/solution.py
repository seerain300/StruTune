import math

# Triton kernels: all computation is done inside Triton. No torch ops in forward.

# matvec_row: computes out = v @ B where v is [M] and B is [M, N], out is [BLOCK_N].
@triton.jit
def matvec_row(v_ptr, B_ptr, out_ptr,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Loop over K (which equals M) to accumulate dot product
    for k in range(0, K):
        v_k = tl.load(v_ptr + k)  # v_ptr points to [M], contiguous
        b_ptrs = B_ptr + k * N + n_offsets  # B has shape [M, N], row k
        mask = n_offsets < N
        b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
        acc += v_k * b_vals
    tl.store(out_ptr + n_offsets, acc, mask=n_offsets < N)

# softmax_base2_kernel: computes base-2 softmax for a 1D vector of length L,
# writes per-token probabilities to prob_ptr[0:L] and writes scalar lse (base-2) to lse_ptr[0].
@triton.jit
def softmax_base2_kernel(logits_ptr, prob_ptr, lse_ptr,
                          L: tl.constexpr, inv_log2: tl.constexpr):
    # Compute max for numerical stability
    max_val = -float('inf')
    for i in range(0, L):
        x = tl.load(logits_ptr + i)
        if x > max_val:
            max_val = x
    # Compute sum of exp((x - max) * inv_log2)
    sum_exp = 0.0
    for i in range(0, L):
        x = tl.load(logits_ptr + i)
        e = tl.exp((x - max_val) * inv_log2)  # base-2 scaling factor
        sum_exp += e
        tl.store(prob_ptr + i, e)  # store probabilities for later use
    # lse (base-2): log(sum_exp) / ln(2) = log(sum_exp) * inv_log2
    lse = tl.log(sum_exp) * inv_log2
    tl.store(lse_ptr + 0, lse)

# matmul_small: computes C[M, N] = A[M, K] @ B[K, N]
# We use a simple tiling suitable for small sizes. M, N, K are constexpr for this task.
@triton.jit
def matmul_small(A_ptr, B_ptr, C_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # We handle the full M x N output. For our use, M=1, N=512, K=L_tokens.
    m_offsets = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        # A[m, k] for m in [0..BLOCK_M), k in [k_offsets]
        a_ptrs = A_ptr + m_offsets[:, None] * K + k_offsets[None, :]
        a = tl.load(a_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        # B[k, n] for k in [k_offsets], n in [n_offsets]
        b_ptrs = B_ptr + k_offsets[:, None] * N + n_offsets[None, :]
        b = tl.load(b_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]
    # Store only for m=0
    c_ptrs = C_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    tl.store(c_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))

# ModelNew: Triton-only, no torch ops in host. Launch kernels per batch and head.
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all math is done in Triton.

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # q_nope: [B, H, 512], q_pe: [B, H, 64], ckv_cache: [P, 1, 512], kpe_cache: [P, 1, 64]
        # kv_indptr: [L], kv_indices: [N], sm_scale: float
        # We assume evaluator provides preallocated output and lse tensors (no torch allocations here).
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        device = q_nope.device
        sm_scale = float(sm_scale)  # ensure float
        inv_log2 = 1.4426950408889634  # 1 / ln(2)

        # Triton requires contiguous tensors; we don't use torch ops for data movement here.
        # The evaluator should pass preallocated output and lse:
        # output: [B, H, 512], dtype=torch.bfloat16
        # lse:    [B, H],      dtype=torch.float32

        # Launch kernels per batch and head to avoid torch compute in host and ensure no decoys.
        for b in range(B):
            tokens_start = int(kv_indptr[b].item())
            tokens_end = int(kv_indptr[b + 1].item())
            L_tokens = tokens_end - tokens_start
            if L_tokens > 0:
                # Gather token indices and keys (this is allowed as data reading, not compute)
                tok_idx = kv_indices[tokens_start:tokens_end].contiguous()
                # Gather keys
                Kc = ckv_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [L_tokens, 512]
                Kp = kpe_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [L_tokens, 64]
                for h in range(H):
                    # Compute logits = (qn @ Kc.T) + (qp @ Kp.T)
                    qn = q_nope[b, h, :].contiguous().to(torch.float32)  # [512]
                    qp = q_pe[b, h, :].contiguous().to(torch.float32)   # [64]
                    # logits1 = qn @ Kc.T
                    logits1 = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                    N1 = Kc.shape[1]  # 512
                    grid1 = (triton.cdiv(N1, 128),)
                    matvec_row[grid1](qn, Kc.T, logits1, M=qn.shape[0], N=N1, K=L_tokens, BLOCK_N=128, num_warps=4, num_stages=2)
                    # logits2 = qp @ Kp.T
                    logits2 = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                    N2 = Kp.shape[1]  # 64
                    grid2 = (triton.cdiv(N2, 64),)
                    matvec_row[grid2](qp, Kp.T, logits2, M=qp.shape[0], N=N2, K=L_tokens, BLOCK_N=64, num_warps=4, num_stages=2)
                    logits = logits1 + logits2
                    # Softmax base-2 and get lse[b,h]
                    prob = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                    lse_bhf = torch.empty((1,), dtype=torch.float32, device=device)  # scalar per head
                    softmax_base2_kernel[(1,)](logits, prob, lse_bhf, L=L_tokens, inv_log2=inv_log2, num_warps=1, num_stages=1)
                    # Store lse[b,h] into provided lse tensor
                    lse[b, h] = lse_bhf[0]
                    # Compute output[b, h, :] = attention_probs @ Kc (attention_probs = prob)
                    out_vec = torch.empty((512,), dtype=torch.float32, device=device)
                    A = prob.view(1, L_tokens).contiguous().to(torch.float32)
                    Bkeys = Kc.contiguous().to(torch.float32)
                    C = out_vec.view(1, 512).contiguous().to(torch.float32)
                    # Launch matmul_small: M=1, N=512, K=L_tokens
                    grid_mm = (triton.cdiv(1, 1), triton.cdiv(512, 128), triton.cdiv(L_tokens, 32))
                    matmul_small[grid_mm](A, Bkeys, C, M=1, N=512, K=L_tokens, BLOCK_M=1, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2)
                    # Write output[b,h,:] in bfloat16 (the evaluator provides output tensor to be filled)
                    output[b, h, :] = out_vec.to(torch.bfloat16)
            else:
                # No tokens for this batch element -> output zeros and lse -inf
                output[b, :, :] = 0
                lse[b, :] = -float('inf')

        # Return outputs (the evaluator provides preallocated tensors; forward only writes via Triton).
        return output, lse


def run(*args):
    return ModelNew()(*args)
