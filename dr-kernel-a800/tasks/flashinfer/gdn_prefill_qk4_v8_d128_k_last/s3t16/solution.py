import torch
import math
import triton
import triton.language as tl


# Triton kernels: elementwise ops
@triton.jit
def _softplus_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes softplus(x) = log(1 + exp(x)) elementwise, N is head dimension
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y)


@triton.jit
def _sigmoid_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes sigmoid(x) = 1 / (1 + exp(-x)) elementwise, N is head dimension
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y)


# GEMV: 1xK x KxV -> 1xV, single output vector out_vec[i] for a fixed i
@triton.jit
def _gemv_1xKxKxV_row(q_ptr, A_ptr, out_ptr, i, K: tl.constexpr, V: tl.constexpr):
    # q_ptr: [K], A_ptr: [K, V] row-major, out_ptr: [V]
    # Compute out[i] = sum_j q[j] * A[j, i]
    acc = tl.zeros((1,), dtype=tl.float32)
    for j in range(0, K):
        qj = tl.load(q_ptr + j)
        Aj = tl.load(A_ptr + j * V + i)
        acc += qj * Aj
    tl.store(out_ptr + i, acc)


# GEMV: 1xV x VxK -> 1xK (q @ A), single output vector out[k] for a fixed k
@triton.jit
def _gemv_1xVxK_col(A_ptr, q_ptr, out_ptr, k, V: tl.constexpr, K: tl.constexpr):
    # A_ptr: [V, K] row-major, q_ptr: [K], out_ptr: [K]
    # Compute out[k] = sum_j A[k, j] * q[j]
    acc = tl.zeros((1,), dtype=tl.float32)
    for j in range(0, K):
        qj = tl.load(q_ptr + j)
        Akj = tl.load(A_ptr + k * K + j)
        acc += qj * Akj
    tl.store(out_ptr + k, acc)


# Elementwise vector op: out = alpha * v + beta * x
@triton.jit
def _elementwise_mul_add(alpha, beta, v_ptr, x_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(v_ptr + i)
    x = tl.load(x_ptr + i)
    y = alpha * v + beta * x
    tl.store(out_ptr + i, y)


# Dot product of two vectors: dot(k, x) where k, x are [K]
@triton.jit
def _dot_scalar_row(k_ptr, x_ptr, out_ptr, i, K: tl.constexpr):
    acc = tl.zeros((1,), dtype=tl.float32)
    for j in range(0, K):
        kj = tl.load(k_ptr + j)
        xj = tl.load(x_ptr + j)
        acc += kj * xj
    tl.store(out_ptr + i, acc)


# Add scalar alpha to a matrix [M, N] (generic for updating state)
@triton.jit
def _add_scalar_to_matrix(A_ptr, alpha, M: tl.constexpr, N: tl.constexpr):
    # A_ptr is row-major [M, N], alpha is scalar
    for i in range(0, M):
        for j in range(0, N):
            val = tl.load(A_ptr + i * N + j)
            val = val + alpha
            tl.store(A_ptr + i * N + j, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized version of the original run function.
        All numerical computation is done by Triton kernels. Host code orchestrates launches.
        """
        device = q.device
        dtype = torch.float32  # compute in fp32; output stored as bfloat16

        # Shapes as per the original code assertions
        total_seq_len = q.shape[0]
        H = q.shape[1]
        V = q.shape[2]  # head_size
        assert V == 128, "head_size must be 128"

        # Repeat q and k for v heads
        num_q_heads = 4
        num_k_heads = 4
        num_v_heads = v.shape[1]  # 8
        q_exp = torch.repeat_interleave(q, num_v_heads // num_q_heads, dim=1).contiguous()
        k_exp = torch.repeat_interleave(k, num_v_heads // num_k_heads, dim=1).contiguous()

        # Output buffer [total_seq_len, H, V], bfloat16
        output = torch.empty((total_seq_len, H, V), dtype=torch.bfloat16, device=device)

        # Number of segments
        num_seqs = cu_seqlens.shape[0] - 1

        # Scale: if None or 0.0, use 1/sqrt(V)
        scale_val = float(1.0 / math.sqrt(V)) if (scale is None or scale == 0.0) else float(scale)

        # Process per sequence
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Move A_log, a, dt_bias, b to device and float32
            A_log_dev = A_log.to(device).to(torch.float32).contiguous()  # [H]
            a_dev = a[seq_start:seq_end].to(device).to(torch.float32).contiguous


def run(*args):
    return ModelNew()(*args)
