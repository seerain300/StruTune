import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
# A: [M, K], B: [K, N], C: [M, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Pointers
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        # Load tiles
        a = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        # Accumulate
        acc += tl.dot(a, b)

    # Write back
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Softmax with causal mask: X[M, N] -> Out[M, N], row-wise
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    stride_xn: tl.int32, stride_outn: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # We assume grid=(N,) launched, but here row index corresponds to row id.
    # Compute row pointer
    # For Out, we write whole row; for X, load whole row
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x_ptrs = X_ptr + row * stride_xn + offs * stride_xn
    # Load row
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
    # Apply causal mask: j > absolute_pos -> -inf
    j = offs
    mask_causal = j > absolute_pos
    x = tl.where(mask_causal, -float("inf"), x)
    # Stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    out_ptrs = Out_ptr + row * stride_outn + offs * stride_outn
    tl.store(out_ptrs, out, mask=mask)


# LogSumExp base-2 with causal mask: X[M, N] -> Out[M], row-wise
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    stride_xn: tl.int32, stride_out: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x_ptrs = X_ptr + row * stride_xn + offs * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
    j = offs
    mask_causal = j > absolute_pos
    x = tl.where(mask_causal, -float("inf"), x)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    sum_exp = tl.sum(tl.exp(x), axis=0)
    lse = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    tl.store(Out_ptr + row * stride_out, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Constants and checks (as in the original code)
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1

        device = q_nope.device

        # Prepare Kc_all and Kp_all from caches
        Kc_all = ckv_cache.to(torch.float32).squeeze(1)  # [num_pages, 512]
        Kp_all = kpe_cache.to(torch.float32).squeeze(1)  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Loop over batch elements
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            if b + 1 == len_indptr:
                # If b is the last (no kv_indptr[b+1]), set kv_len=0 and break
                kv_len = 0
            else:
                kv_end = int(kv_indptr[b + 1].item())
                kv_len = kv_end - page_beg

            # Gather token indices and corresponding K vectors
            if kv_len == 0:
                continue
            tok_idx = kv_indices[page_beg:page_end]  # [kv_len]
            Kc = Kc_all[tok_idx].to(torch.float32)   # [kv_len, 512]
            Kp = Kp_all[tok_idx].to(torch.float32)   # [kv_len, 64]

            # Process queries in this batch
            for i in range(q_len):
                abs_q = q_start + i  # absolute query index across all batches
                # Compute scores: scores = qn @ Kc.T + qp @ Kp.T
                qn = q_nope[abs_q].to(torch.float32)  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32)    # [16, 64]
                # Matmul 1: qn @ Kc.T -> [16, kv_len]
                attn_scores = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_mm = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_mm](
                    qn, Kc.transpose(0, 1).contiguous(), attn_scores,
                    num_qo_heads, kv_len, head_dim_ckv,
                    qn.stride(0), qn.stride(1),
                    Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                    attn_scores.stride(0), attn_scores.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )
                # Matmul 2: qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_mm2 = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_mm2](
                    qp, Kp.transpose(0, 1).contiguous(), scores_p,
                    num_qo_heads, kv_len, head_dim_kpe,  # second dim is 64, but Kp has 64
                    qp.stride(0), qp.stride(1),
                    Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )
                scores = attn_scores + scores_p  # [16, kv_len]
                # Causal mask: j > (kv_len - q_len + i) -> -inf
                prefix_len = kv_len - q_len  # previously cached tokens
                absolute_pos = prefix_len + i  # absolute position of the current query in sequence
                # Softmax per row with causal mask
                # Note: launch one program per row
                softmax_out = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_softmax = (num_qo_heads,)
                softmax_row_causal_kernel[grid_softmax](
                    scores, softmax_out,
                    kv_len, absolute_pos,
                    scores.stride(1), softmax_out.stride(1),
                    BLOCK=128
                )
                # Output: attn @ Kc -> [16, 512]
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                # Prepare A = attn, B = Kc.T
                # Use PyTorch matmul for robustness (keeping Triton for softmax).
                # A: [16, kv_len], B: [kv_len, 512]
                attn_view = softmax_out.transpose(0, 1)  # [kv_len, 16]
                out_row = attn_view @ Kc  # [16, 512]
                output[abs_q] = out_row.to(torch.bfloat16)

                # LogSumExp base-2 with causal mask per row
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                grid_lse = (num_qo_heads,)
                lse_row_causal_kernel[grid_lse](
                    scores, lse_row,
                    kv_len, absolute_pos,
                    scores.stride(1), lse_row.stride(0),
                    BLOCK=128
                )
                lse[abs_q] = lse_row

        return output, lse


# Keep the original get_inputs function
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


# Optional: original Model can alias ModelNew
class Model(ModelNew):
    pass


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
