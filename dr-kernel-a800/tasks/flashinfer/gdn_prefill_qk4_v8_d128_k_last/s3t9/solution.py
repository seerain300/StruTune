import torch
import math
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernels: elementwise ops (to be launched from host)
@triton.jit
def _softplus_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes softplus(x) = log(1 + exp(x)) elementwise, vectorized over N (here N=128)
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def _sigmoid_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Sigmoid(x) = 1 / (1 + exp(-x))
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # Compute out_vec[i] = sum_{k=0..K-1} q_vec[k] * A_mat[k, i], where A_mat is [K, V], q_vec is [K].
    # A_ptr is laid out as [K, V] contiguous. out_ptr is [V].
    i = tl.arange(0, V)
    acc = tl.zeros([V], dtype=tl.float32)
    # Iterate over K dimension (compile-time unrolled since K is constexpr)
    for k in range(0, K):
        a_row = tl.load(A_ptr + k * V + i, mask=i < V, other=0.0)
        qk = tl.load(q_ptr + k, mask=True, other=0.0)  # scalar q[k]
        acc += qk * a_row
    tl.store(out_ptr + i, acc, mask=i < V)


@triton.jit
def _elementwise_vector_mul_add(alpha, beta, v_ptr, old_ptr, out_ptr, N: tl.constexpr):
    # out[i] = alpha * v[i] + beta * old[i]
    i = tl.arange(0, N)
    v = tl.load(v_ptr + i, mask=i < N, other=0.0)
    old = tl.load(old_ptr + i, mask=i < N, other=0.0)
    out = alpha * v + beta * old
    tl.store(out_ptr + i, out, mask=i < N)


@triton.jit
def _dot_scalar(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    # Computes scalar = sum_{i=0..N-1} x[i] * y[i]
    acc = 0.0
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.load(y_ptr + i, mask=i < N, other=0.0)
    # Reduce elementwise
    for j in range(0, N):
        acc += x[j] * y[j]
    tl.store(out_ptr, acc)


# Host-side ModelNew.forward orchestrates computation with Triton kernels
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # q: [T, H_q, V], k: [T, H_k, V], v: [T, H_v, V], state: [S, H_v, K, V]
        # A_log: [H_v], a: [T, H_v], dt_bias: [H_v], b: [T, H_v], cu_seqlens: [S+1], scale: float
        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        assert num_q_heads == 4, "num_q_heads must be 4"
        assert num_k_heads == 4, "num_k_heads must be 4"
        assert num_v_heads == 8, "num_v_heads must be 8"
        assert head_size == 128, "head_size must be 128"
        assert cu_seqlens.dtype in (torch.int32, torch.int64), "cu_seqlens must be int"
        num_seqs = cu_seqlens.shape[0] - 1

        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(head_size)

        # Repeat q and k along heads for v heads (4 -> 8)
        q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1).contiguous()  # [T, H_v, V]
        k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1).contiguous()  # [T, H_v, V]

        # Allocate outputs
        output = torch.empty((total_seq_len, num_v_heads, head_size), dtype=torch.float32, device=q.device)
        new_state = torch.empty((num_seqs, num_v_heads, head_size, head_size), dtype=torch.float32, device=q.device)

        # Orchestrate per segment and per t and per h
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # state_curr: [H_v, K, V] contiguous
            state_curr = state[seq_idx].contiguous()  # [H_v, 128, 128]

            # Prepare dt_bias and A_log for Triton (scalar per head)
            dt_bias_vec = dt_bias.to(torch.float32).contiguous()       # [H_v]
            A_log_vec = A_log.to(torch.float32).contiguous()          # [H_v]

            for t in range(seq_len):
                t_abs = seq_start + t

                # Compute g and beta for each head h via Triton (elementwise vectors)
                # g = exp(-exp(A_log[h]) * softplus(a[t, h] + dt_bias[h]))
                # beta = sigmoid(b[t, h])

                # Prepare a_t and b_t for Triton
                a_t = a[t].to(torch.float32).contiguous()             # [H_v]
                b_t = b[t].to(torch.float32).contiguous()             # [H_v]

                # 1) softplus(a_t + dt_bias_vec)
                softplus_vals = torch.empty_like(dt_bias_vec)         # [H_v]
                _softplus_vector[(1,)](a_t, softplus_vals, N=dt_bias_vec.shape[0])

                # 2) g = exp(-exp(A_log_vec) * softplus)
                g_vals = torch.empty_like(dt_bias_vec)                # [H_v]
                _compute_g_scalar_vec[(1,)](A_log_vec, softplus_vals, g_vals, N=dt_bias_vec.shape[0])

                # 3) beta = sigmoid(b_t)
                beta_vals = torch.empty_like(b_t)                     # [H_v]
                _sigmoid_vector[(1,)](b_t, beta_vals, N=b_t.shape[0])

                for h in range(num_v_heads):
                    # Extract vectors
                    k_vec = k_exp[t_abs, h].to(torch.float32).contiguous()   # [128]
                    q_vec = q_exp[t_abs, h].to(torch.float32).contiguous()   # [128]
                    v_vec = v[t_abs, h].to(torch.float32).contiguous()       # [128]

                    # state_old_T: [K, V] = [128, 128], contiguous
                    state_old_T = state_curr[h].contiguous()                 # [128, 128]

                    # 4) old_v = k_vec @ state_old_T (GEMV)
                    old_v = torch.empty(128, dtype=torch.float32, device=q.device)
                    _gemv_1xKxKxV_into_1xV[(1,)](k_vec, state_old_T, old_v, K=128, V=128)

                    # 5) new_v_vec = beta[h] * v_vec + (1 - beta[h]) * old_v
                    new_v_vec = torch.empty(128, dtype=torch.float32, device=q.device)
                    _elementwise_vector_mul_add[(1,)](beta_vals[h], 1.0 - beta_vals[h], v_vec, old_v, new_v_vec, N=128)

                    # 6) Compute state_remove = dot(k_vec, old_v) (scalar)
                    state_remove = torch.empty((), dtype=torch.float32, device=q.device)
                    _dot_scalar[(1,)](k_vec, old_v, state_remove, N=128)

                    # 7) state_update = dot(k_vec, new_v_vec) (scalar)
                    state_update = torch.empty((), dtype=torch.float32, device=q.device)
                    _dot_scalar[(1,)](k_vec, new_v_vec, state_update, N=128)

                    # 8) state_new_mat = g[h] * state_old_T + (state_update - state_remove)[None, :]
                    #    First, scale state_old_T by g[h]
                    g_scalar = g_vals[h]
                    # Triton kernel to add scalar to each element: out[j] = state_old_T[j] + alpha
                    # We can implement: out = state_old_T + (state_update - state_remove) broadcast across rows
                    alpha = (state_update - state_remove).item()
                    # Launch add_scalar_to_matrix_elements: A_ptr -> state_old_T, out_ptr -> state_new_mat, alpha
                    # We need a separate tensor for state_new_mat to avoid modifying state_old_T.
                    state_new_mat = torch.empty_like(state_old_T, dtype=torch.float32, device=q.device)
                    _add_scalar_to_matrix_elements[(1,)](state_old_T, state_new_mat, alpha, K=128, V=128)
                    # Multiply by g_scalar
                    state_new_mat = state_new_mat * g_scalar

                    # 9) output_vec = scale * (q_vec @ state_new_mat)
                    output_vec = torch.empty(128, dtype=torch.float32, device=q.device)
                    _gemv_1xKxKxV_into_1xV[(1,)](q_vec, state_new_mat, output_vec, K=128, V=128)
                    output[t_abs, h, :] = (output_vec * scale).to(torch.bfloat16)

                    # 10) Update new_state[seq_idx, h, :, :]
                    #     state_new_mat is [K, V]; new_state expects [V, K], so transpose.
                    new_state[seq_idx, h, :, :] = state_new_mat.transpose(0, 1).contiguous()

        return output, new_state


# Define Triton kernel: add scalar alpha to each element of a KxV matrix (row-major [K, V])
@triton.jit
def _add_scalar_to_matrix_elements(A_ptr, out_ptr, alpha, K: tl.constexpr, V: tl.constexpr):
    i = tl.arange(0, V)
    j = tl.arange(0, K)
    # Loop over rows and columns to update each element
    for r in range(0, K):
        for c in range(0, V):
            val = tl.load(A_ptr + r * V + c)
            val = val + alpha
            tl.store(out_ptr + r * V + c, val)

# Note: The original code computes scale = 1.0 / sqrt(head_size) and uses it to scale the final output.
# We keep that logic in host code (no Triton needed there), but avoid using torch.sqrt/exp on host.
# We also avoid .item() for scalars on host; we pass them as constexpr or keep them in tensors and use Triton where possible.
# However, Triton currently does not support storing to non-pointer tensors without explicit grid; hence we keep simple single-item outputs and use .item() to retrieve scalars for arithmetic, which is allowed here as Triton doesn’t compute those host values, and we only use Triton for numerical ops.


def run(*args):
    return ModelNew()(*args)
