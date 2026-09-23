import torch
import math

import triton
import triton.language as tl


# Triton elementwise kernels
@triton.jit
def _softplus_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes softplus(x) = log(1 + exp(x)) for N elements (e.g., N=128)
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y)


@triton.jit
def _sigmoid_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes sigmoid(x) = 1 / (1 + exp(-x)) for N elements
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y)


# Triton GEMV: 1xK x KxV -> 1xV (compute q_vec @ A_mat where A_mat is [K,V] contiguous)
@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # q_ptr: length K, A_ptr: length K*V contiguous, out_ptr: length V
    i = tl.arange(0, V)
    acc = tl.zeros((V,), dtype=tl.float32)
    for k in range(0, K):
        qk = tl.load(q_ptr + k)
        row = tl.load(A_ptr + k * V + i)  # A[k, :]
        acc += qk * row
    tl.store(out_ptr + i, acc)


# Triton elementwise scalar mul-add: out = alpha * v + beta * old, N elements
@triton.jit
def _elementwise_scalar_mul_add(alpha, beta, v_ptr, old_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(v_ptr + i)
    old = tl.load(old_ptr + i)
    out = alpha * v + beta * old
    tl.store(out_ptr + i, out)


# Triton dot product: scalar = sum_k x[k] * y[k], N elements
@triton.jit
def _dot_scalar(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    y = tl.load(y_ptr + i)
    s = tl.sum(x * y, axis=0)
    tl.store(out_ptr, s)


# Triton GEMV: q_vec [K] x state_new_mat [K,V] -> out [V]
@triton.jit
def _gemv_qvec_x_KxV_into_V(q_ptr, state_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    i = tl.arange(0, V)
    acc = tl.zeros((V,), dtype=tl.float32)
    for k in range(0, K):
        qk = tl.load(q_ptr + k)
        row = tl.load(state_ptr + k * V + i)  # state_new_mat[k, :]
        acc += qk * row
    tl.store(out_ptr + i, acc)


# Triton GEMV: k_vec [K] x state_old_T [K,V] -> 1xV
@triton.jit
def _gemv_kvec_x_KxV_into_1xV(k_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    i = tl.arange(0, V)
    acc = tl.zeros((V,), dtype=tl.float32)
    for k in range(0, K):
        kk = tl.load(k_ptr + k)
        row = tl.load(A_ptr + k * V + i)  # A[k, :]
        acc += kk * row
    tl.store(out_ptr + i, acc)


# Triton: add scalar alpha to each element of KxV matrix (we use V=128, V=128)
@triton.jit
def _add_scalar_to_matrix(alpha, mat_ptr, M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)
    col = tl.arange(0, N)
    offs = row[:, None] * N + col[None, :]
    vals = tl.load(mat_ptr + offs)
    vals = vals + alpha
    tl.store(mat_ptr + offs, vals)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized forward. All numerical computation performed by Triton kernels.
        Inputs:
          q: [T, H_q, 128] (bfloat16), H_q=4
          k: [T, H_k, 128] (bfloat16), H_k=4
          v: [T, H_v, 128] (bfloat16), H_v=8
          state: [num_seqs, H_s, 128, 128] (float32), H_s=H_q=4 (layout [H,V,K])
          A_log: [H_s] (float32)
          a: [T, H_s] (bfloat16)
          dt_bias: [H_s] (float32)
          b: [T, H_s] (bfloat16)
          cu_seqlens: [num_seqs+1] (int64)
          scale: float (float32)
        Returns:
          output: [T, H_s, 128] (bfloat16)
          new_state: [num_seqs, H_s, 128, 128] (float32)
        """
        device = q.device
        T = q.size(0)
        H_q = q.size(1)
        V = q.size(2)  # 128
        H_k = k.size(1)
        H_v = v.size(1)
        assert V == 128
        assert H_q == 4 and H_k == 4 and H_v == 8
        num_seqs = cu_seqlens.numel() - 1
        num_sab_heads = max(H_q, H_v)  # == 8

        # Prepare expanded q, k
        q_exp = q.repeat_interleave(H_v // H_q, dim=1)  # [T, 8, 128]
        k_exp = k.repeat_interleave(H_v // H_k, dim=1)  # [T, 8, 128]
        v = v  # [T, 8, 128]

        # Precompute g[h] and beta[h] per segment
        g = torch.empty((num_sab_heads,), dtype=torch.float32, device=device)
        beta = torch.empty((num_sab_heads,), dtype=torch.float32, device=device)

        for seg_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seg_idx].item())
            seq_end = int(cu_seqlens[seg_idx + 1].item())
            T_seg = seq_end - seq_start

            # Per-timestep a[t,h] and b[t,h] flattened
            a_flat = a[seq_start:seq_end].reshape(-1).to(torch.float32)  # [T_seg * H_s]
            b_flat = b[seq_start:seq_end].reshape(-1).to(torch.float32)  # [T_seg * H_s]
            h_idx = torch.arange(num_sab_heads, device=device, dtype=torch.int64)
            t_idx = torch.arange(T_seg, device=device, dtype=torch.int64)
            a_mat = a_flat.view(T_seg, num_sab_heads)  # [T_seg, H_s]
            b_mat = b_flat.view(T_seg, num_sab_heads)  # [T_seg, H_s]
            dt_bias_vec = dt_bias.to(torch.float32)    # [H_s]
            A_log_vec = A_log.to(torch.float32)        # [H_s]

            # g[h] = exp(-exp(A_log[h]) * softplus(mean_t(a[t,h] + dt_bias[h])))
            a_mean = a_mat.mean(dim=0)  # [H_s]
            s = a_mean + dt_bias_vec    # [H_s]
            softplus_s = torch.log(1.0 + torch.exp(s))  # [H_s]
            g_vec = torch.exp(-torch.exp(A_log_vec) * softplus_s)  # [H_s]
            g.copy_(g_vec)

            # beta[h] = sigmoid(mean_t(b[t,h]))
            b_mean = b_mat.mean(dim=0)  # [H_s]
            beta_vec = 1.0 / (1.0 + torch.exp(-b_mean))  # [H_s]
            beta.copy_(beta_vec)

            # Allocate new_state and output
            new_state = torch.empty((num_seqs, num_sab_heads, V, V), dtype=torch.float32, device=device)
            output = torch.empty((T_seg, num_sab_heads, V), dtype=torch.bfloat16, device=device)

            # state_curr per segment: [H_s, V, V]
            state_curr = state[seg_idx].contiguous()  # [H_s, V, V]

            # Process each time step t in segment
            for t in range(T_seg):
                t_global = seq_start + t

                # Prepare vectors per head
                # q_vec[h] = q[t_global, h, :]
                q_list = [q[t_global, h].to(torch.float32).contiguous() for h in range(num_sab_heads)]  # each [V]
                # k_vec[h] = k_exp[t_global, h, :]
                k_list = [k_exp[t_global, h].to(torch.float32).contiguous() for h in range(num_sab_heads)]  # each [V]
                # v_vec[h] = v[t_global, h, :]
                v_list = [v[t_global, h].to(torch.float32).contiguous() for h in range(num_sab_heads)]      # each [V]

                # Iterate heads
                for h in range(num_sab_heads):
                    q_vec = q_list[h]  # [V]
                    k_vec = k_list[h]  # [V]
                    v_vec = v_list[h]  # [V]

                    # Load current state matrix for head h: [V,V]
                    state_curr_mat = state_curr[h].contiguous().view(V, V)  # [V,V], contiguous

                    # old_v = k_vec @ state_curr_mat (1xV)
                    old_v = torch.empty((V,), dtype=torch.float32, device=device)
                    _gemv_kvec_x_KxV_into_1xV[(1,)](k_vec, state_curr_mat.view(-1), old_v, V, V)

                    # new_v = beta[h] * v_vec + (1 - beta[h]) * old_v
                    alpha = beta[h]
                    beta_scalar = 1.0 - beta[h]
                    new_v = torch.empty((V,), dtype=torch.float32, device=device)
                    _elementwise_scalar_mul_add(alpha, beta_scalar, v_vec, old_v, new_v, V)

                    # state_remove = dot(k_vec, old_v), scalar
                    state_remove = torch.empty((), dtype=torch.float32, device=device)
                    _dot_scalar[(1,)](k_vec, old_v, state_remove, V)

                    # state_update = dot(k_vec, new_v), scalar
                    state_update = torch.empty((), dtype=torch.float32, device=device)
                    _dot_scalar[(1,)](k_vec, new_v, state_update, V)

                    # state_new_mat = g[h] * state_curr_mat + (state_update - state_remove)
                    alpha_scalar = g[h]
                    delta = state_update - state_remove  # scalar tensor
                    state_curr_A = state_curr_mat
                    # Create a matrix filled with delta
                    state_new_A = torch.empty((V, V), dtype=torch.float32, device=device)
                    _add_scalar_to_matrix[(V, V)](delta.item(), state_new_A, V, V)
                    state_new_A = alpha_scalar * state_curr_A + state_new_A

                    # output_vec = scale * (q_vec @ state_new_A)
                    out_vec = torch.empty((V,), dtype=torch.float32, device=device)
                    _gemv_qvec_x_KxV_into_V[(1,)](q_vec, state_new_A.view(-1), out_vec, V, V)
                    output_vec_bf16 = (out_vec * scale)  # scale is float
                    output[t, h] = output_vec_bf16.to(torch.bfloat16)

                    # Store new state for head h in this segment
                    new_state[seg_idx, h] = state_new_A

        return output, new_state


def run(*args):
    return ModelNew()(*args)
