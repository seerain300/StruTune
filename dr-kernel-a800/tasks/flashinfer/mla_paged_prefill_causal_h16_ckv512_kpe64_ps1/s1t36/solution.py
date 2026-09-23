import math
import torch
import triton
import triton.language as tl


# Simple matmul kernel: C[M, N] = A[M, K] @ B[K, N]
# We implement a straightforward tiled matmul using tl.dot.
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        # B tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Masks
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        a = tl.load(A_ptrs, mask=A_mask, other=0.0)
        b = tl.load(B_ptrs, mask=B_mask, other=0.0)

        acc += tl.dot(a, b)

    # Store C tile
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


# Softmax with causal mask: X [1, N] -> Out [1, N], row-wise
# Mask: j > absolute_pos => -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    stride_xn: tl.constexpr, stride_outn: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + offs * stride_xn, mask=offs < N, other=0.0)
    mask = offs > absolute_pos
    x = tl.where(mask, -float("inf"), x)
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    y = e / denom
    tl.store(Out_ptr + offs * stride_outn, y, mask=offs < N)


# LogSumExp base-2 with causal mask: X [1, N] -> Out [1], row-wise
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, absolute_pos: tl.int32,
    stride_xn: tl.constexpr, stride_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + offs * stride_xn, mask=offs < N, other=0.0)
    mask = offs > absolute_pos
    x = tl.where(mask, -float("inf"), x)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    sum_e = tl.sum(e, axis=0)
    lse = tl.log(sum_e) / 0.6931471805599453  # 1 / ln(2)
    tl.store(Out_ptr, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Cast inputs to float32 for compute
        device = q_nope.device
        q_nope = q_nope.contiguous().to(torch.float32)
        q_pe = q_pe.contiguous().to(torch.float32)

        # Prepare key caches (fp32)
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue
            q_len = q_end - q_start

            # KV for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg
            if kv_len == 0:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512], fp32
            Kp = Kp_all[tok_idx]  # [kv_len, 64],  fp32

            # Loop queries in this batch
            for i in range(q_len):
                # query vectors
                qn = q_nope[q_start + i]  # [16, 512] fp32
                qp = q_pe[q_start + i]    # [16, 64]  fp32

                # scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_n = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_n](
                    qn, Kc.transpose(0, 1), scores_n,
                    num_qo_heads, head_dim_ckv, kv_len,
                    qn.stride(0), qn.stride(1),
                    Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_p = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_p](
                    qp, Kp.transpose(0, 1), scores_p,
                    num_qo_heads, head_dim_kpe, kv_len,
                    qp.stride(0), qp.stride(1),
                    Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # Combine
                scores = scores_n + scores_p  # [16, kv_len]

                # Causal mask: j > (prefix_len + i) -> -inf, prefix_len = kv_len - q_len
                absolute_pos = kv_len - q_len + i

                # softmax scores -> attn
                attn = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(num_qo_heads,)](
                    scores, attn,
                    kv_len, absolute_pos,
                    scores.stride(1), attn.stride(1),
                    BLOCK=128
                )

                # out = attn @ Kc -> [16, 512]
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                grid_mm = (triton.cdiv(num_qo_heads, 16), triton.cdiv(head_dim_ckv, 64))
                matmul_kernel[grid_mm](
                    attn, Kc, out_row,
                    num_qo_heads, kv_len, head_dim_ckv,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # store output
                output[q_start + i] = out_row  # fp32, will cast later

                # lse base-2
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(num_qo_heads,)](
                    scores, lse_row,
                    kv_len, absolute_pos,
                    scores.stride(1), lse_row.stride(0),
                    BLOCK=128
                )
                lse[q_start + i] = lse_row

        # Cast output to bfloat16 as original
        output = output.to(torch.bfloat16)
        return output, lse


# Optional: reuse the original get_inputs
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


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
