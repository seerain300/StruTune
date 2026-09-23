import torch
import math
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton elementwise and GEMV kernels (scalar-loop based to avoid shape mismatches)
@triton.jit
def softplus_scalar(x_ptr, out_ptr, N: tl.constexpr):
    # Computes softplus(x) = log(1 + exp(x)) elementwise for N elements.
    for i in range(N):
        x = tl.load(x_ptr + i)
        y = tl.log(1.0 + tl.exp(x))
        tl.store(out_ptr + i, y)


@triton.jit
def sigmoid_scalar(x_ptr, out_ptr, N: tl.constexpr):
    # Computes sigmoid(x) = 1 / (1 + exp(-x)) elementwise for N elements.
    for i in range(N):
        x = tl.load(x_ptr + i)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + i, y)


@triton.jit
def scale_vector(x_ptr, out_ptr, scale, N: tl.constexpr):
    # Elementwise multiply by scalar: out[i] = scale * x[i]
    for i in range(N):
        x = tl.load(x_ptr + i)
        tl.store(out_ptr + i, x * scale)


@triton.jit
def gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # Computes out[i] = sum_{k=0..K-1} q[k] * A[k, i], where:
    # q_ptr: [K] contiguous
    # A_ptr: [K, V] contiguous (row-major: offset = k*V + i)
    # out_ptr: [V]
    for i in range(V):
        acc = 0.0
        for k in range(K):
            qk = tl.load(q_ptr + k)
            Aki = tl.load(A_ptr + k * V + i)
            acc += qk * Aki
        tl.store(out_ptr + i, acc)


@triton.jit
def elementwise_mul_add_scalar(alpha, beta, v_ptr, old_ptr, out_ptr, N: tl.constexpr):
    # alpha and beta are scalars. v_ptr and old_ptr are [N]. out_ptr is [N].
    for i in range(N):
        vi = tl.load(v_ptr + i)
        oldi = tl.load(old_ptr + i)
        outi = alpha * vi + beta * oldi
        tl.store(out_ptr + i, outi)


@triton.jit
def dot_scalar(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    # Computes scalar = sum_{i=0..N-1} x[i] * y[i]
    acc = 0.0
    for i in range(N):
        xi = tl.load(x_ptr + i)
        yi = tl.load(y_ptr + i)
        acc += xi * yi
    tl.store(out_ptr, acc)


@triton.jit
def add_scalar_to_matrix_scalar_alpha(mat_ptr, alpha, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # Adds scalar alpha to all elements of [K, V] matrix mat_ptr, writes to out_ptr
    for i in range(K):
        for j in range(V):
            val = tl.load(mat_ptr + i * V + j)
            tl.store(out_ptr + i * V + j, val + alpha)


# Forward function that orchestrates Triton kernels
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_size = 128
        self.K = 128
        self.V = 128

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        device = q.device
        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        num_sab_heads = max(num_q_heads, num_v_heads)

        # Assertions
        assert head_size == self.head_size, "head_size must be 128"
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8, "Head counts must match the original code"
        num_seqs = cu_seqlens.numel() - 1

        # Compute scale if not provided
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(self.head_size)
        else:
            scale = float(scale)

        # Precompute q_exp and k_exp: q_exp repeats q by (num_v_heads // num_q_heads), k similarly
        q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1).contiguous()
        k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1).contiguous()

        # Prepare output and new_state
        output = torch.empty((total_seq_len, num_sab_heads, self.head_size), dtype=torch.bfloat16, device=device)
        new_state = torch.zeros((num_seqs, num_sab_heads, self.head_size, self.head_size), dtype=torch.float32, device=device)

        # Loop over segments defined by cu_seqlens
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # If state is provided, use it; original state is [1, 8, 128, 128], so select seq_idx 0 and head h
            if state is None:
                state_curr = None
            else:
                # state is [1, 8, 128, 128]; select seq_idx 0
                state_curr = state[0].contiguous()  # [8, 128, 128]

            # Loop over time steps
            for t in range(seq_len):
                q_vec = q_exp[t].contiguous()  # [128]
                k_vec = k_exp[t].contiguous()  # [128]
                v_vec = v[t].contiguous()      # [128]

                # Compute per-head g and beta for h in 0..num_sab_heads-1
                for h in range(num_sab_heads):
                    # Extract scalars
                    a_val = float(a[t, h].item())          # a[t, h]
                    dt_bias_val = float(dt_bias[h].item()) # dt_bias[h]
                    A_log_val = float(A_log[h].item())     # A_log[h]
                    b_val = float(b[t, h].item())          # b[t, h]

                    # Compute softplus(a[t,h] + dt_bias[h]) and g
                    a_plus_dt = a_val + dt_bias_val
                    softplus_a_dt = math.log(1.0 + math.exp(a_plus_dt))
                    g_scalar = math.exp(-math.exp(A_log_val) * softplus_a_dt)

                    # Compute beta = sigmoid(b[t,h])
                    beta_scalar = 1.0 / (1.0 + math.exp(-b_val))
                    alpha_scalar = 1.0 - beta_scalar

                    # Compute state_old_T: [K, V] = [128, 128]
                    if state_curr is None:
                        state_old_T = torch.zeros((self.K, self.V), dtype=torch.float32, device=device)
                    else:
                        # state_curr: [8, 128, 128]; select head h
                        state_old_T = state_curr[h].contiguous()  # [128, 128]

                    # old_v = k_vec @ state_old_T
                    old_v = torch.empty((self.V,), dtype=torch.float32, device=device)
                    q_vec_k = k_vec.contiguous()  # [128]
                    A_mat = state_old_T.contiguous()  # [128, 128]
                    out_old_v = torch.empty((self.V,), dtype=torch.float32, device=device)
                    # Launch Triton GEMV
                    gemv_1xKxKxV_into_1xV[q_vec_k, A_mat, out_old_v, self.K, self.V]  # meta dict not needed when using positional args

                    # new_v = beta * v_vec + (1 - beta) * old_v
                    new_v = torch.empty((self.V,), dtype=torch.float32, device=device)
                    v_in = v_vec.contiguous()   # [128]
                    old_in = old_v               # [128]
                    elementwise_mul_add_scalar[alpha_scalar, beta_scalar, v_in, old_in, new_v, self.V]

                    # Compute state_remove and state_update: dot(k_vec, old_v) and dot(k_vec, new_v)
                    dot_k_old = torch.empty((), dtype=torch.float32, device=device)
                    dot_k_new = torch.empty((), dtype=torch.float32, device=device)
                    dot_scalar[k_vec, old_in, dot_k_old, self.K]
                    dot_scalar[k_vec, new_v, dot_k_new, self.K]

                    # state_new_mat = g * state_old_T + (dot_k_new - dot_k_old)[None, :]
                    state_new_mat = torch.empty((self.K, self.V), dtype=torch.float32, device=device)
                    alpha_add = (dot_k_new.item() - dot_k_old.item())  # scalar
                    add_scalar_to_matrix_scalar_alpha[state_old_T, alpha_add, state_new_mat, self.K, self.V]
                    state_new_mat = state_new_mat * g_scalar  # apply gate g

                    # output_vec = scale * (q_vec @ state_new_mat)
                    out_vec = torch.empty((self.V,), dtype=torch.float32, device=device)
                    gemv_1xKxKxV_into_1xV[q_vec, state_new_mat, out_vec, self.K, self.V]
                    out_vec_scaled = torch.empty((self.V,), dtype=torch.bfloat16, device=device)
                    scale_vector[out_vec, out_vec_scaled, scale, self.V]
                    # Store output[t, h, :]
                    output[t, h, :] = out_vec_scaled

                    # Update new_state[seq_idx, h, :, :] = state_new_mat.transpose(0, 1) -> [128, 128]
                    new_state[seq_idx, h, :, :] = state_new_mat.transpose(0, 1).contiguous()

        return output, new_state


def run(*args):
    return ModelNew()(*args)
