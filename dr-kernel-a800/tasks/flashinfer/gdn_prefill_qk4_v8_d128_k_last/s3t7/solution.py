import torch
import math
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernels for elementwise ops (to be launched from host)
@triton.jit
def _softplus_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes softplus(x) = log(1 + exp(x)) elementwise, vectorized
    i = tl.arange(0, 128)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def _sigmoid_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Sigmoid(x) = 1 / (1 + exp(-x))
    i = tl.arange(0, 128)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y, mask=i < N)


# Triton kernel: 1xK x KxV -> 1xV
# out_vec[i] = sum_{k=0..K-1} q_vec[k] * A_mat[k, i]
@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    i = tl.arange(0, V)
    acc = tl.zeros([V], dtype=tl.float32)
    # Loop over K dimension; K=128
    for k in range(0, K):
        a_col = tl.load(A_ptr + k * V + i, mask=i < V, other=0.0)
        qk = tl.load(q_ptr + k)
        acc += qk * a_col
    tl.store(out_ptr + i, acc, mask=i < V)


# Triton kernel: elementwise vector operation
# out_vec = alpha * x_vec + beta * y_vec
@triton.jit
def _elementwise_mul_add(x_ptr, y_ptr, out_ptr, alpha, beta, N: tl.constexpr):
    i = tl.arange(0, N)
    xv = tl.load(x_ptr + i, mask=i < N, other=0.0)
    yv = tl.load(y_ptr + i, mask=i < N, other=0.0)
    out = alpha * xv + beta * yv
    tl.store(out_ptr + i, out, mask=i < N)


# Triton kernel: dot product of two vectors (1xK -> scalar)
# out[0] = sum_{k=0..K-1} x_vec[k] * y_vec[k]
@triton.jit
def _dot_scalar(x_ptr, y_ptr, out_ptr, K: tl.constexpr):
    acc = tl.zeros([1], dtype=tl.float32)
    for k in range(0, K):
        xv = tl.load(x_ptr + k)
        yv = tl.load(y_ptr + k)
        acc += xv * yv
    tl.store(out_ptr, acc)


# Triton kernel: add scalar to all elements of a KxV matrix
# out_ptr[k*V + j] = A_ptr[k*V + j] + alpha
@triton.jit
def _add_scalar_to_matrix(A_ptr, out_ptr, alpha, K: tl.constexpr, V: tl.constexpr):
    for k in range(0, K):
        for j in range(0, V):
            val = tl.load(A_ptr + k * V + j)
            val = val + alpha
            tl.store(out_ptr + k * V + j, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure tensors on same device and dtype conversions
        device = q.device
        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        assert num_q_heads == 4
        assert num_k_heads == 4
        assert num_v_heads == 8
        assert head_size == 128

        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(head_size)

        # Repeat q/k along heads (data movement, not computation)
        q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

        # Allocate outputs (float32, cast to bfloat16 before return to match original behavior)
        output = torch.empty(
            (total_seq_len, num_v_heads, head_size), dtype=torch.float32, device=device
        )
        new_state = torch.empty(
            (cu_seqlens.shape[0] - 1, num_v_heads, head_size, head_size), dtype=torch.float32, device=device
        )

        H = num_v_heads  # heads = 8

        # Compute num_seqs from cu_seqlens (lens may be provided as in original)
        # The original cu_seqlens is computed from _lens in the get_inputs helper; here we rely on input cu_seqlens.
        num_seqs = cu_seqlens.shape[0] - 1

        # Orchestrate per segment and per t and per head
        # Note: Triton kernels will do all heavy math; host uses only minimal torch ops to form params.
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            # loop over positions t within this segment
            for t in range(seq_start, seq_end):
                # Gather q_exp, k_exp, v for this t
                # q_exp shape: [total_seq_len, num_v_heads, head_size]
                q_vec = q_exp[t].contiguous()          # [128]
                k_vec = k_exp[t].contiguous()          # [128]
                v_vec = v[t].contiguous()              # [128]

                # Compute g and beta scalars using Triton kernels
                # g = exp(-exp(A_log[h]) * softplus(a[t,h] + dt_bias[h]))
                A_log_h = float(A_log[h].item()) if isinstance(A_log, torch.Tensor) else A_log  # h in 0..H-1
                a_th = float(a[t, h].item()) if isinstance(a, torch.Tensor) else a
                dt_bh = float(dt_bias[h].item()) if isinstance(dt_bias, torch.Tensor) else dt_bias
                softplus_input = a_th + dt_bh
                # softplus vector kernel expects length N=128; we pass scalars as 1-element tensors
                softplus_out = torch.empty(1, dtype=torch.float32, device=device)
                _softplus_vector[(1,)](torch.tensor([softplus_input], device=device, dtype=torch.float32),
                                       softplus_out, N=128)
                softplus_val = softplus_out[0]
                g_val = torch.exp(-torch.exp(torch.tensor([A_log_h], device=device, dtype=torch.float32)) * softplus_val)
                g_val = float(g_val.item()) if isinstance(g_val, torch.Tensor) else g_val

                # beta = sigmoid(b[t,h])
                b_th = float(b[t, h].item()) if isinstance(b, torch.Tensor) else b
                beta_out = torch.empty(1, dtype=torch.float32, device=device)
                _sigmoid_vector[(1,)](torch.tensor([b_th], device=device, dtype=torch.float32),
                                      beta_out, N=128)
                beta_val = float(beta_out[0].item()) if isinstance(beta_out[0], torch.Tensor) else beta_out[0]

                # Compute old_v = k_vec @ state_old_T via GEMV
                # state_old_T is the h-th matrix from state_curr transposed to [K,V]
                # state_curr shape: [num_seqs, num_v_heads, head_size, head_size]; we need the matrix at (seq_idx, h)
                # For correctness, we can assume the given state is the current state for this segment; otherwise,
                # we would need to pass an explicit state_curr tensor. Here we use 'state' as the current state.
                # Extract state_old as [head_size, head_size], then transpose to [K, V] for kernel.
                state_old = state[seq_idx, h].contiguous()  # [128, 128]
                state_old_T = state_old.transpose(0, 1).contiguous()  # [128, 128]
                old_v = torch.empty(128, dtype=torch.float32, device=device)
                _gemv_1xKxKxV_into_1xV[(1,)](k_vec, state_old_T, old_v, K=128, V=128)

                # Compute new_v_vec = beta * v_vec + (1 - beta) * old_v
                new_v_vec = torch.empty(128, dtype=torch.float32, device=device)
                _elementwise_mul_add[(1,)](v_vec, old_v, new_v_vec, beta_val, (1 - beta_val), N=128)

                # Compute state_remove = dot(k_vec, old_v) and state_update = dot(k_vec, new_v_vec)
                state_remove = torch.empty((), dtype=torch.float32, device=device)
                _dot_scalar[(1,)](k_vec, old_v, state_remove, K=128)
                state_update = torch.empty((), dtype=torch.float32, device=device)
                _dot_scalar[(1,)](k_vec, new_v_vec, state_update, K=128)
                alpha = (state_update - state_remove).item()  # scalar

                # Create state_new_mat as g * state_old_T + alpha
                state_new_mat = torch.empty((128, 128), dtype=torch.float32, device=device)
                # Use Triton to add scalar alpha to state_old_T
                _add_scalar_to_matrix[(1,)](state_old_T, state_new_mat, alpha, K=128, V=128)

                # Compute output_vec = scale * (q_vec @ state_new_mat) via GEMV
                output_vec = torch.empty(128, dtype=torch.float32, device=device)
                _gemv_1xKxKxV_into_1xV[(1,)](q_vec, state_new_mat, output_vec, K=128, V=128)
                output_vec_scaled = output_vec * float(scale)

                # Store output[t, h, :]
                output[t, h, :] = output_vec_scaled

                # Update new_state[seq_idx, h, :, :]
                new_state[seq_idx, h, :, :] = state_new_mat

        # Cast outputs to bfloat16 to match original behavior
        return output.to(torch.bfloat16), new_state


def run(*args):
    return ModelNew()(*args)
