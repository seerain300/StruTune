import math
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch over M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop across K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    # Store C tile
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,            # typically 1.0
    absolute_pos: tl.int32,       # positions j > absolute_pos will be masked
    BLOCK: tl.constexpr,          # BLOCK >= N
):
    row_id = tl.program_id(0)
    # Load row
    x = tl.load(X_ptr + row_id * N + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < N, other=-float("inf"))
    # Apply causal mask
    j = tl.arange(0, BLOCK)
    mask_causal = j <= absolute_pos
    x = tl.where(mask_causal, x, -float("inf"))
    # Stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(Out_ptr + row_id * N + tl.arange(0, BLOCK), out, mask=tl.arange(0, BLOCK) < N)


@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,            # typically 1.0
    absolute_pos: tl.int32,       # positions j > absolute_pos will be masked
    BLOCK: tl.constexpr,          # BLOCK >= N
):
    row_id = tl.program_id(0)
    x = tl.load(X_ptr + row_id * N + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < N, other=-float("inf"))
    j = tl.arange(0, BLOCK)
    mask_causal = j <= absolute_pos
    x = tl.where(mask_causal, x, -float("inf"))
    # Stable logsumexp
    x_max = tl.max(x, axis=0)
    x = x - x_max
    e = tl.exp(x)
    sum_e = tl.sum(e, axis=0)
    l = tl.log(sum_e) / 0.6931471805599453  # ln(2)
    tl.store(Out_ptr + row_id, l)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # All tensors must be on CUDA for Triton
        if not (q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda):
            raise RuntimeError("ModelNew.forward expects CUDA tensors")

        device = q_nope.device
        total_q = int(q_nope.shape[0])
        batch_size = int(qo_indptr.shape[0] - 1)

        output = torch.empty((total_q, 16, 512), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)
        scale = float(sm_scale)

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

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)
            Kc = ckv_cache.squeeze(1)[tok_idx].to(torch.float32)  # [kv_len, 512]
            Kp = kpe_cache.squeeze(1)[tok_idx].to(torch.float32)  # [kv_len, 64]

            for i in range(q_len):
                abs_q = q_start + i
                # qn and qp for current query
                qn = q_nope[abs_q]  # [16, 512]
                qp = q_pe[abs_q]    # [16, 64]

                # scores_n = qn @ Kc.T -> [


def run(*args):
    return ModelNew()(*args)
