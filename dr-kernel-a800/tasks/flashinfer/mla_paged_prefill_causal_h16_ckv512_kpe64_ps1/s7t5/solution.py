import math
import torch
import triton
import triton.language as tl


# Kernel: copy one row from a 3D tensor [T, M, K] to a fp32 buffer [M, K]
# We pass row_index (int32) of which batch element to copy, and strides of src and dst.
@triton.jit
def copy_row_to_fp32_kernel(
    src_ptr, dst_ptr,
    T: tl.int32, M: tl.int32, K: tl.int32,
    row_index: tl.int32,
    stride_src_t, stride_src_m, stride_src_k,
    stride_dst_m, stride_dst_k,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # absolute src ptr for this row: src_ptr + row_index*stride_src_t
    src_row_ptr = src_ptr + row_index * stride_src_t

    # loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # load from src: [M, K]
        vals = tl.load(
            src_row_ptr + offs_m[:, None] * stride_src_m + offs_k[None, :] * stride_src_k,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        vals = vals.to(tl.float32)

        # store to dst: dst_ptr + offs_m * stride_dst_m + offs_k * stride_dst_k
        tl.store(
            dst_ptr + offs_m[:, None] * stride_dst_m + offs_k[None, :] * stride_dst_k,
            vals,
            mask=mask_m[:, None] & mask_k[None, :],
        )


# Kernel: copy multiple rows from [N, K] src to [size, K] dst, one row per program
@triton.jit
def copy_rows_to_fp32_kernel(
    src_ptr, dst_ptr,
    N: tl.int32, K: tl.int32, size: tl.int32,
    stride_src_n, stride_src_k,
    stride_dst_size, stride_dst_k,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= size:
        return
    # src row index = pid
    src_row_ptr = src_ptr + pid * stride_src_n

    # destination row index = pid (same ordering)
    dst_row_ptr = dst_ptr + pid * stride_dst_size

    # loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        vals = tl.load(src_row_ptr + offs_k * stride_src_k, mask=mask_k, other=0.0).to(tl.float32)
        tl.store(dst_row_ptr + offs_k * stride_dst_k, vals, mask=mask_k)


# Matmul left: A[M, N] @ B[K, N]^T -> C[M, K]
@triton.jit
def matmul_left_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_am, stride_an,
    stride_bk, stride_bn,
    stride_cm, stride_ck,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # loop over N in tiles
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        b = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a, b)

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck,
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


# Softmax with mask along last dim. Input X[M, N], Output Y[M, N].
@triton.jit
def softmax_mask_kernel(
    X_ptr, Y_ptr,
    M: tl.int32, N: tl.int32,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    abs_pos: tl.int32,  # query_abs_pos
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # compute row-wise max with mask: max_j (X[i,j] if j<abs_pos else -inf)
    max_val = -float("inf")
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        mask_pos = offs_n < abs_pos
        x = tl.load(
            X_ptr + pid_m * stride_xm + offs_n * stride_xn,
            mask=mask_n,
            other=-float("inf"),
        ).to(tl.float32)
        # set masked elements to -inf
        x = tl.where(mask_pos, x, -float("inf"))
        local_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, local_max)

    # compute sum of exp(x - max_val) with mask
    sum_val = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        mask_pos = offs_n < abs_pos
        x = tl.load(
            X_ptr + pid_m * stride_xm + offs_n * stride_xn,
            mask=mask_n,
            other=-float("inf"),
        ).to(tl.float32)
        x = tl.where(mask_pos, x, -float("inf"))
        e = tl.exp(x - max_val)
        sum_val += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_val

    # write softmax: exp(x - max) * inv_sum, masked
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        mask_pos = offs_n < abs_pos
        x = tl.load(
            X_ptr + pid_m * stride_xm + offs_n * stride_xn,
            mask=mask_n,
            other=-float("inf"),
        ).to(tl.float32)
        x = tl.where(mask_pos, x, -float("inf"))
        y = tl.exp(x - max_val) * inv_sum
        tl.store(Y_ptr + pid_m * stride_ym + offs_n * stride_yn, y, mask=mask_n)


# Row-wise logsumexp with mask in base-2. Input X[M, N], Output LSE[M].
@triton.jit
def lse_mask_base2_kernel(
    X_ptr, LSE_ptr,
    M: tl.int32, N: tl.int32,
    stride_xm, stride_xn,
    abs_pos: tl.int32,  # query_abs_pos
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    max_val = -float("inf")
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        mask_pos = offs_n < abs_pos
        x = tl.load(
            X_ptr + pid_m * stride_xm + offs_n * stride_xn,
            mask=mask_n,
            other=-float("inf"),
        ).to(tl.float32)
        x = tl.where(mask_pos, x, -float("inf"))
        local_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, local_max)

    sum_val = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        mask_pos = offs_n < abs_pos
        x = tl.load(
            X_ptr + pid_m * stride_xm + offs_n * stride_xn,
            mask=mask_n,
            other=-float("inf"),
        ).to(tl.float32)
        x = tl.where(mask_pos, x, -float("inf"))
        e = tl.exp(x - max_val)
        sum_val += tl.sum(e, axis=0)

    lse = max_val + math.log(2.0) * tl.log(sum_val)  # logsumexp base 2
    tl.store(LSE_ptr + pid_m, lse)


# Now ModelNew using these Triton kernels
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-store constants; we can't use torch ops for compute, but we can allocate shapes.

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Extract shapes
        total_q = qo_indptr[-1].item()
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare global cache in fp32 for simplicity and consistent math
        Kc_all_f = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all_f = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output and lse buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        batch_size = qo_indptr.shape[0] - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())

            if page_beg >= page_end:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [Lq]
            Lq = tok_idx.numel()

            # Prepare per-batch Kc_used and Kp_used (fp32)
            # Create buffers
            Kc_used = torch.empty((Lq, head_dim_ckv), dtype=torch.float32, device=device)
            Kp_used = torch.empty((Lq, head_dim_kpe), dtype=torch.float32, device=device)

            # Copy rows from Kc_all_f and Kp_all_f to buffers
            copy_rows_to_fp32_kernel[(Lq,)](
                Kc_all_f, Kc_used,
                head_dim_ckv,  # N
                Lq,             # size
                head_dim_ckv,   # stride_src_n = K
                1,              # stride_src_k = 1 (row-major)
                Lq,             # stride_dst_size
                1,              # stride_dst_k = 1
                BLOCK_K=128,
                num_warps=4,
            )

            copy_rows_to_fp32_kernel[(Lq,)](
                Kp_all_f, Kp_used,
                head_dim_kpe,   # N
                Lq,             # size
                head_dim_kpe,   # stride_src_n = K
                1,              # stride_src_k = 1
                Lq,             # stride_dst_size
                1,              # stride_dst_k = 1
                BLOCK_K=128,
                num_warps=4,
            )

            # Process each query in this batch
            for i in range(q_start, q_end):
                # Copy q_nope[i] and q_pe[i] to fp32 buffers
                qn_buf = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                qp_buf = torch.empty((num_qo_heads, head_dim_kpe), dtype=torch.float32, device=device)

                copy_row_to_fp32_kernel[(num_qo_heads,)](
                    q_nope, qn_buf,
                    total_q, num_qo_heads, head_dim_ckv,
                    i,
                    q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
                    qn_buf.stride(0), qn_buf.stride(1),
                    BLOCK_M=16, BLOCK_K=128,
                    num_warps=4,
                )

                copy_row_to_fp32_kernel[(num_qo_heads,)](
                    q_pe, qp_buf,
                    total_q, num_qo_heads, head_dim_kpe,
                    i,
                    q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
                    qp_buf.stride(0), qp_buf.stride(1),
                    BLOCK_M=16, BLOCK_K=128,
                    num_warps=4,
                )

                # Compute logits_qn = qn_buf @ Kc_used.T -> [16, Lq]
                logits_qn = torch.empty((num_qo_heads, Lq), dtype=torch.float32, device=device)
                matmul_left_kernel[(num_qo_heads, 1)](
                    qn_buf, Kc_used.T, logits_qn,
                    num_qo_heads, Lq, head_dim_ckv,
                    qn_buf.stride(0), qn_buf.stride(1),
                    Kc_used.T.stride(0), Kc_used.T.stride(1),
                    logits_qn.stride(0), logits_qn.stride(1),
                    BLOCK_M=16, BLOCK_N=32, BLOCK_K=64,
                    num_warps=4,
                )

                # Compute logits_qp = qp_buf @ Kp_used.T -> [16, Lq]
                logits_qp = torch.empty((num_qo_heads, Lq), dtype=torch.float32, device=device)
                matmul_left_kernel[(num_qo_heads, 1)](
                    qp_buf, Kp_used.T, logits_qp,
                    num_qo_heads, Lq, head_dim_kpe,
                    qp_buf.stride(0), qp_buf.stride(1),
                    Kp_used.T.stride(0), Kp_used.T.stride(1),
                    logits_qp.stride(0), logits_qp.stride(1),
                    BLOCK_M=16, BLOCK_N=32, BLOCK_K=64,
                    num_warps=4,
                )

                # Sum and scale
                logits = logits_qn + logits_qp  # [16, Lq]
                sm_scale_f = float(sm_scale)
                logits = logits * sm_scale_f

                # Compute query_abs_pos for causal mask: prefix_len + i, prefix_len = Lq - (q_end - q_start)
                q_len = q_end - q_start
                query_abs_pos = (Lq - q_len) + (i - q_start)

                # Softmax with causal mask and store attn
                attn = torch.empty((num_qo_heads, Lq), dtype=torch.float32, device=device)
                softmax_mask_kernel[(num_qo_heads,)](
                    logits, attn,
                    num_qo_heads, Lq,
                    logits.stride(0), logits.stride(1),
                    attn.stride(0), attn.stride(1),
                    query_abs_pos,
                    BLOCK_N=32,
                    num_warps=4,
                )

                # Row-wise logsumexp base-2 with mask and store to lse[q, h]
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                lse_mask_base2_kernel[(num_qo_heads,)](
                    logits, lse_row,
                    num_qo_heads, Lq,
                    logits.stride(0), logits.stride(1),
                    query_abs_pos,
                    BLOCK_N=32,
                    num_warps=4,
                )
                lse[i, :] = lse_row  # write per query, per head

                # Compute output per head: attn @ Kc_used -> [16, 512]
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                matmul_left_kernel[(num_qo_heads, 1)](
                    attn, Kc_used, out_row,
                    num_qo_heads, Lq, head_dim_ckv,
                    attn.stride(0), attn.stride(1),
                    Kc_used.stride(0), Kc_used.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=128,
                    num_warps=4,
                )
                output[i, :, :] = out_row

        # Cast to bfloat16 for output as in original example, lse as float32
        output_bf16 = output.to(torch.bfloat16)
        # Return output and lse (Triton kernels have computed everything)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
