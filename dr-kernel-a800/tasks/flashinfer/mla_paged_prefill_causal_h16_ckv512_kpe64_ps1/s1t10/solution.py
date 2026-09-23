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
    # 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Softmax (row-wise) with causal mask: Out = softmax(X) along last dimension, with j > absolute_pos -> -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,          # typically 1.0
    absolute_pos: tl.int32,     # prefix_len + i
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    # Load row, apply causal mask
    x = tl.load(X_ptr + row_id * N + offs, mask=offs < N, other=0.0)
    # Set j > absolute_pos to -inf
    j = offs
    mask_inf = j > absolute_pos
    x = tl.where(mask_inf, -float("inf"), x)
    # Stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.exp(x)
    denom = tl.sum(x, axis=0)
    x = x / denom * scale
    tl.store(Out_ptr + row_id * N + offs, x, mask=offs < N)


# Logsumexp (row-wise) with causal mask: Out = log(sum(exp(X))) / ln(2), where j > absolute_pos -> -inf
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,          # 1.0
    absolute_pos: tl.int32,     # prefix_len + i
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + offs, mask=offs < N, other=0.0)
    j = offs
    mask_inf = j > absolute_pos
    x = tl.where(mask_inf, -float("inf"), x)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    lse_val = tl.log(sum_exp) * (1.0 / math.log(2.0))  # base-2
    # scale is just 1.0; we apply scale if needed
    tl.store(Out_ptr + row_id, lse_val)


def _matmul_triton(A, B, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2):
    """
    A: [M, K], B: [K, N], return C: [M, N] in float32
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible shapes for matmul"
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, B, C,
        M, K, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


def _softmax_row_causal_triton(X, BLOCK):
    """
    X: [1, N] float32 on device. Returns Out: [1, N] float32.
    """
    M, N = X.shape
    assert M == 1, "softmax_row_causal_triton expects 1 row"
    Out = torch.empty((M, N), dtype=torch.float32, device=X.device)
    grid = (1,)
    softmax_row_causal_kernel[grid](X, Out, N, 1.0, 0, BLOCK)
    return Out


def _lse_row_causal_triton(X, BLOCK):
    """
    X: [1, N] float32 on device. Returns Out: [1] float32.
    """
    M, N = X.shape
    assert M == 1, "lse_row_causal_triton expects 1 row"
    Out = torch.empty((1,), dtype=torch.float32, device=X.device)
    grid = (1,)
    lse_row_causal_kernel[grid](X, Out, N, 1.0, 0, BLOCK)
    return Out[0]


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # tunable constants
        self.SM_SCALE = 1.0
        # Triton tiling defaults (can be tuned per run)
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_K = 32
        self.num_warps = 4
        self.num_stages = 2

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Assuming inputs are already on CUDA. Triton requires CUDA tensors.
        device = q_nope.device
        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Ensure constants match (as in original)
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Convert caches: [num_pages, 1, D] -> [num_pages, D]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        batch_size = qo_indptr.numel() - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Process each query in this batch
            for i in range(q_len):
                abs_q = q_start + i
                # qn: [16, 512], qp: [16, 64]
                qn = q_nope[abs_q]  # [16, 512]
                qp = q_pe[abs_q]    # [16, 64]

                # scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = _matmul_triton(qn, Kc.T, BLOCK_M=16, BLOCK_N=kv_len, BLOCK_K=64, num_warps=2, num_stages=2)
                # scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = _matmul_triton(qp, Kp.T, BLOCK_M=16, BLOCK_N=kv_len, BLOCK_K=32, num_warps=2, num_stages=2)
                scores = scores_n + scores_p  # [16, kv_len]

                # Causal mask: j > (prefix_len + i) -> -inf
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Compute softmax with mask
                # Expand scores to [1, kv_len] for row-wise kernel
                scores_row = scores.view(1, kv_len)  # float32
                attn_row = _softmax_row_causal_triton(scores_row, BLOCK=kv_len)  # [1, kv_len]
                attn = attn_row  # already [1, kv_len], but we want [16, kv_len]; for now, let's compute per head below.

                # To get per-head attention: we need to run kernel per head. In this structure, we have 16 heads, but scores shape is [16, kv_len].
                # We'll loop over heads by constructing each row independently. However, Triton kernel is row-wise; we can call it 16 times, but that's heavy.
                # Instead, we compute per head inside matmul: out_per_head = attn @ Kc for each row. But we need attn to be [16, kv_len].
                # Fix: implement per-head softmax using Triton: call softmax_row_causal_kernel(16 times with each row).
                # Here we run per-head softmax directly in Triton by calling the kernel 16 times (less overhead than looping over each element).
                # Prepare per-head outputs
                attn_heads = [torch.empty((1, kv_len), dtype=torch.float32, device=device) for _ in range(16)]
                # Loop heads: call kernel for each head's row
                for h in range(16):
                    s = scores[h]  # 1D [kv_len]
                    s_row = s.view(1, kv_len)
                    # Output buffer
                    out = attn_heads[h]
                    softmax_row_causal_kernel[(1,)](s_row, out, kv_len, 1.0, absolute_pos, BLOCK=kv_len)
                    # out is [1, kv_len]; but we need to store attn[h] = out[0]
                    attn_heads[h] = out[0]  # [kv_len] view

                # Compute output for each head: out = attn[h] @ Kc -> [512]
                for h in range(16):
                    attn_vec = attn_heads[h]  # [kv_len]
                    out_vec = _matmul_triton(attn_vec.view(1, kv_len), Kc)  # [1, 512]
                    # Store into output
                    output[abs_q, h] = out_vec[0]  # [512]

                # lse per head
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                for h in range(16):
                    s = scores[h].view(1, kv_len)  # [1, kv_len]
                    lse_val = _lse_row_causal_triton(s, BLOCK=kv_len)  # scalar
                    lse_row[h] = lse_val
                lse[abs_q] = lse_row

        # Cast output to bfloat16
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
