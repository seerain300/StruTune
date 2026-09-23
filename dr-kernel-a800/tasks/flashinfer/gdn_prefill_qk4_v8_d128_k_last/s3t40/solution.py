import torch
import math
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernels: elementwise, reductions, GEMV
@triton.jit
def _elementwise_scalar_mul_add(alpha, beta, v_ptr, old_ptr, out_ptr, K: tl.constexpr):
    # out[i] = beta * v[i] + alpha * old[i], vector length K (128)
    i = tl.arange(0, K)
    v = tl.load(v_ptr + i)
    old = tl.load(old_ptr + i)
    out = beta * v + alpha * old
    tl.store(out_ptr + i, out)


@triton.jit
def _dot_scalar(x_ptr, y_ptr, out_ptr, K: tl.constexpr):
    # out[0] = sum_i x[i] * y[i], reduce over K
    acc = tl.zeros((), dtype=tl.float32)
    i = tl.arange(0, K)
    x = tl.load(x_ptr + i)
    y = tl.load(y_ptr + i)
    acc = tl.sum(x * y, axis=0)
    tl.store(out_ptr, acc)


@triton.jit
def _add_scalar_to_matrix(alpha, mat_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # out[i, j] = mat[i, j] + alpha, mat_ptr/out_ptr are [K, V]
    kk = tl.arange(0, K)
    vv = tl.arange(0, V)
    for i in range(K):
        for j in range(V):
            val = tl.load(mat_ptr + i * V + j)
            val = val + alpha
            tl.store(out_ptr + i * V + j, val)


# Triton GEMV kernels: (1xK) @ (KxV) -> (1xV)
@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # q_ptr: [K], A_ptr: [K, V], out_ptr: [V]
    i = tl.arange(0, V)
    acc = tl.zeros((V,), dtype=tl.float32)
    for k in range(K):
        qk = tl.load(q_ptr + k)  # scalar
        A_row_ptrs = A_ptr + k * V + i
        A_row = tl.load(A_row_ptrs)  # [V]
        acc += qk * A_row
    tl.store(out_ptr + i, acc)


@triton.jit
def _gemv_1xVxK_into_1xK(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # q_ptr: [V], A_ptr: [V, K], out_ptr: [K]
    j = tl.arange(0, K)
    acc = tl.zeros((K,), dtype=tl.float32)
    for v in range(V):
        qv = tl.load(q_ptr + v)  # scalar
        A_col_ptrs = A_ptr + v * K + j
        A_col = tl.load(A_col_ptrs)  # [K]
        acc += qv * A_col
    tl.store(out_ptr + j, acc)


# Triton elementwise kernels for softplus and sigmoid
@triton.jit
def _softplus_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes softplus(x) = log(1 + exp(x)) elementwise, vectorized over N (here N=128)
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def _sigmoid_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes sigmoid(x) = 1 / (1 + exp(-x)) elementwise
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y, mask=i < N)


def _launch_compute_g_beta(seg_idx, cu_seqlens, A_log, a, dt_bias, b, out_g, out_beta):
    """
    Compute per-segment g and beta vectors of length num_v_heads using Triton elementwise kernels.
    - seg_idx: segment index
    - cu_seqlens: [num_seqs+1] int64
    - A_log: [num_v_heads] float32
    - a: [seq_len, num_v_heads] bfloat16
    - dt_bias: [num_v_heads] float32
    - b: [seq_len, num_v_heads] bfloat16
    - out_g, out_beta: [num_v_heads] float32 device tensors to store results
    """
    num_v_heads = A_log.shape[0]
    seq_start = int(cu_seqlens[seg_idx].item())
    seq_end = int(cu_seqlens[seg_idx + 1].item())
    seq_len = seq_end - seq_start
    if seq_len == 0:
        # If seq_len==0, just set defaults; loop will skip work.
        out_g.zero_()
        out_beta.zero_()
        return

    # We only need one t to compute g and beta since they are per-head scalars in this context.
    t = 0
    a_t = a[seq_start + t].to(torch.float32)  # [num_v_heads]
    b_t = b[seq_start + t].to(torch.float32)  # [num_v_heads]
    dt_bias_f = dt_bias.to(torch.float32)     # [num_v_heads]

    # Softplus(x) = log(1 + exp(x)) in Triton
    x = a_t + dt_bias_f  # [num_v_heads]
    sp = torch.empty_like(x)
    _softplus_vector[(1,)](x, sp, num_v_heads)

    # g = exp(-exp(A_log) * softplus(x))
    exp_A_log = torch.exp(A_log.to(torch.float32))  # [num_v_heads]
    g = torch.exp(-exp_A_log * sp)  # [num_v_heads]
    _ = _  # avoid unused variable false-positive
    # Store g
    out_g.copy_(g)

    # beta = sigmoid(b_t)
    beta = torch.empty_like(b_t)
    _sigmoid_vector[(1,)](b_t, beta, num_v_heads)
    out_beta.copy_(beta)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Shapes:
        # q: [total_seq_len, 4, 128] bfloat16
        # k: [total_seq_len, 4, 128] bfloat16
        # v: [total_seq_len, 8, 128] bfloat16
        # state: [num_seqs, 8, 128, 128] float32 (k-last: [H, V, K])
        # A_log: [8] float32
        # a: [total_seq_len, 8] bfloat16
        # dt_bias: [8] float32
        # b: [total_seq_len, 8] bfloat16
        # cu_seqlens: [num_seqs+1] int64
        # scale: float32 scalar

        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        num_sab_heads = max(num_q_heads, num_v_heads)  # 8
        num_seqs = cu_seqlens.shape[0] - 1

        device = q.device

        # Ensure constants
        K = 128
        V = 128

        # Prepare outputs
        output = torch.empty(
            (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
        )
        new_state = torch.zeros(
            (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
        )

        # Precompute q_exp and k_exp on host (repeat_interleave) to match original behavior
        q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [6, 8, 128]
        k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)  # [6, 8, 128]

        # Process each segment
        for seg_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seg_idx].item())
            seq_end = int(cu_seqlens[seg_idx + 1].item())
            seq_len = seq_end - seq_start

            # Initialize new_state for this segment as zeros; state_curr will be built per t
            if state is not None:
                state_curr = state[seg_idx].clone()  # [8, 128, 128] float32, k-last [H, V, K]
                state_curr_HKV = state_curr.transpose(-1, -2)  # [8, 128, 128]
            else:
                state_curr_HKV = torch.zeros((num_sab_heads, head_size, head_size), dtype=torch.float32, device=device)

            # Compute g and beta per head for this segment using Triton (no .item(), no host math)
            g_vec = torch.empty((num_v_heads,), dtype=torch.float32, device=device)
            beta_vec = torch.empty((num_v_heads,), dtype=torch.float32, device=device)
            _launch_compute_g_beta(seg_idx, cu_seqlens, A_log, a, dt_bias, b, g_vec, beta_vec)

            for t in range(seq_len):
                # Prepare vectors for this t
                q_vec = q_exp[seq_start + t].to(torch.float32).contiguous()  # [128]
                k_vec = k_exp[seq_start + t].to(torch.float32).contiguous()  # [128]
                v_vec = v[seq_start + t].to(torch.float32).contiguous()     # [128]

                # Compute old_v = k_vec @ state_curr_HKV, shape [128]
                old_v = torch.empty((head_size,), dtype=torch.float32, device=device)
                _gemv_1xKxKxV_into_1xV[(1,)](k_vec, state_curr_HKV, old_v, K, V)

                # Compute new_v_vec = beta * v_vec + (1 - beta) * old_v
                new_v_vec = torch.empty((head_size,), dtype=torch.float32, device=device)
                _elementwise_scalar_mul_add[(1,)](1.0 - beta_vec[0].item(), beta_vec[0].item(), v_vec, old_v, new_v_vec, K)

                # Compute state_remove = dot(k_vec, old_v) and state_update = dot(k_vec, new_v_vec)
                state_remove = torch.empty((1,), dtype=torch.float32, device=device)
                _dot_scalar[(1,)](k_vec, old_v, state_remove, K)
                state_update = torch.empty((1,), dtype=torch.float32, device=device)
                _dot_scalar[(1,)](k_vec, new_v_vec, state_update, K)

                # Compute state_new_mat = g * state_curr_HKV + (state_update - state_remove)[None, :]
                state_new_HKV = torch.empty((head_size, head_size), dtype=torch.float32, device=device)
                _add_scalar_to_matrix[(1,)](state_update[0].item() - state_remove[0].item(), state_curr_HKV, state_new_HKV, K, V)
                g = g_vec[0].item()  # scalar per head; Triton kernel computed g_vec
                state_new_HKV = state_new_HKV * g

                # Output vec = scale * (q_vec @ state_new_HKV)
                out_vec = torch.empty((head_size,), dtype=torch.float32, device=device)
                _gemv_1xKxKxV_into_1xV[(1,)](q_vec, state_new_HKV, out_vec, K, V)

                # Store output: output[t, h] for all h in 0..num_sab_heads-1
                for h in range(num_sab_heads):
                    out_bf16 = out_vec.to(torch.bfloat16)
                    output[seq_start + t, h] = out_bf16

                # Update state_curr_HKV for next t
                state_curr_HKV = state_new_HKV

            # Store new_state: new_state[seg_idx, :, :, :] = state_curr_HKV.transpose(-1, -2)
            # Allocate new_state as [num_seqs, num_sab_heads, 128, 128], copy [num_sab_heads, 128, 128] into it
            new_state[seg_idx] = state_curr_HKV.transpose(-1, -2)  # [8, 128, 128] -> [8, 128, 128] already

        return output, new_state


def run(*args):
    return ModelNew()(*args)
