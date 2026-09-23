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
    # 2D launch grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        A_tile = tl.load(
            A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0
        )
        B_tile = tl.load(
            B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0
        )
        acc += tl.dot(A_tile, B_tile)

    tl.store(
        C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
        acc,
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    )


# Row-wise softmax with causal mask
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,             # usually 1.0
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,          # typically N
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + row_id * N + offsets, mask=mask, other=-float("inf"))

    causal_mask = offsets > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)

    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    out = exp_x / denom

    tl.store(Out_ptr + row_id * N + offsets, out, mask=mask)


# Row-wise logsumexp with causal mask (base-2)
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,             # 1.0 (no scaling)
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,          # typically N
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + row_id * N + offsets, mask=mask, other=-float("inf"))

    causal_mask = offsets > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)

    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    lse = tl.log(sum_exp) / tl.log(2.0)  # base-2
    tl.store(Out_ptr + row_id, lse)  # per-row scalar


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        assert device.type == "cuda", "This Triton implementation requires CUDA tensors."

        total_q, num_q


def run(*args):
    return ModelNew()(*args)
