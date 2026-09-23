import torch
import triton
import triton.language as tl


# Triton kernel: copy a row from a 3D tensor A[T, M, K] into a 2D fp32 buffer B[T, M, K] using strides.
# We treat A and B as 3D tensors with strides (stride_at, stride_am, stride_ak). This avoids mixing 2D/3D issues.
@triton.jit
def copy_row_3d_to_fp32_kernel(A_ptr, B_ptr,
                               T, M, K,
                               stride_at, stride_am, stride_ak,
                               stride_bt, stride_bm, stride_bk,
                               BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_t = tl.program_id(0)  # row index in T
    for m_start in range(0, M, BLOCK_M):
        for k_start in range(0, K, BLOCK_K):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_m = offs_m < M
            mask_k = offs_k < K
            # Pointer for A row: [M_tile, K_tile]
            A_block_ptr = A_ptr + pid_t * stride_at + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a = tl.load(A_block_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            # Store into B row at time index pid_t
            B_block_ptr = B_ptr + pid_t * stride_bt + offs_m[:, None] * stride_bm + offs_k[None, :] * stride_bk
            tl.store(B_block_ptr, a, mask=mask_m[:, None] & mask_k[None, :])


# Triton kernel: transpose a single row of a 2D tensor src[N, K] into dst[K, N].
# Each program instance handles one row, writing transposed values.
@triton.jit
def transpose_single_row_kernel(src_ptr, dst_ptr,
                                row_idx, N, K,
                                stride_sn, stride_sk,
                                stride_dk, stride_dn):
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    # Load row from src: shape [N, K]
    src_row_ptr = src_ptr + row_idx * stride_sn + n * stride_sk
    vals = tl.load(src_row_ptr)
    # Store to dst: dst[k, n] = src[row_idx, n]
    dst_row_ptr = dst_ptr + k * stride_dk + n * stride_dn
    tl.store(dst_row_ptr, vals)


# Triton kernel: left-multiply matmul. Computes C[M, K] = A[M, N] @ B[N, K], where A is provided as [M, N],
# and B is provided as [N, K] (B^T). We assume fp32 tensors and 2D shapes. Strides provided for A and B^T.
@triton.jit
def matmul_left_kernel(A_ptr, BT_ptr, C_ptr,
                       M, N, K,
                       stride_am, stride_an,
                       stride_bk, stride_bn,  # BT has shape (N, K): stride_bk = stride along K, stride_bn = stride along N
                       stride_cm, stride_ck,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)  # tile along M
    pid_k = tl.program_id(1)  # tile along K

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_idx = n_start + offs_n
        # Load A tile: [BLOCK_M, BLOCK_N]
        A_tile_ptr = A_ptr + offs_m[:, None] * stride_am + n_idx[None, :] * stride_an
        A_mask = (offs_m[:, None] < M) & (n_idx[None, :] < N)
        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0)

        # Load BT tile: [BLOCK_N, BLOCK_K]
        BT_tile_ptr = BT_ptr + n_idx[:, None] * stride_bk + offs_k[None, :] * stride_bn
        BT_mask = (n_idx[:, None] < N) & (offs_k[None, :] < K)
        BT_tile = tl.load(BT_tile_ptr, mask=BT_mask, other=0.0)

        acc += tl.dot(A_tile, BT_tile)

    # Write back to C
    C_tile_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    C_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(C_tile_ptr, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
        device = q_nope.device

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[1]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        num_kv_indices = kv_indices.shape[0]

        # Original constraints
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64).to(device)

            # Gather cached keys; since cache has one "page", we can gather rows directly
            # Kc_all: [num_pages, 1, 512] -> squeeze(


def run(*args):
    return ModelNew()(*args)
