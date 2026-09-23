import torch
import math
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernels: elementwise ops and GEMV
@triton.jit
def _softplus_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes softplus(x) = log(1 + exp(x)) elementwise over N elements (N=128 here)
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y)


@triton.jit
def _sigmoid_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes sigmoid(x) = 1 / (1 + exp(-x)) elementwise over N elements (N=128 here)
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y)


@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # A_ptr: [K, V] contiguous; out_ptr: [V]
    # Compute out[j] = sum_i q[i] * A[i, j] for i in [0..K-1], j in [0..V-1]
    i = tl.arange(0, K)
    j = tl.arange(0, V)
    q = tl.load(q_ptr + i)  # [K]
    out = tl.zeros([V], dtype=tl.float32)
    for ii in range(0, K):
        a_row = tl.load(A_ptr + ii * V + j)  # [V]
        out += q[ii] * a_row
    tl.store(out_ptr + j, out)


@triton.jit
def _gemv_1xVxK_into_1xK(v_ptr, S_ptr, out_ptr, V: tl.constexpr, K: tl.constexpr):
    # S_ptr: [V, K] contiguous; out_ptr: [K]
    # Compute out[k] = sum_i v[i] * S[i, k] for i in [0..V-1], k in [0..K-1]
    i = tl.arange(0, V)
    k = tl.arange(0, K)
    v = tl.load(v_ptr + i)  # [V]
    out = tl.zeros([K], dtype=tl.float32)
    for ii in range(0, V):
        s_col = tl.load(S_ptr + ii * K + k)  # [K]
        out += v[ii] * s_col
    tl.store(out_ptr + k, out)


@triton.jit
def _dot_scalar(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    # Computes scalar = sum_i x[i] * y[i] for i in [0..N-1], stores to out_ptr[0]
    acc = 0.0
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    y = tl.load(y_ptr + i)
    acc = tl.sum(x * y, axis=0)
    tl.store(out_ptr, acc)  # out_ptr has shape [1]


@triton.jit
def _add_scalar_to_matrix(alpha, mat_ptr, out_ptr, N: tl.constexpr):
    # Adds scalar alpha to all elements of a contiguous 1D matrix of length N
    i = tl.arange(0, N)
    m = tl.load(mat_ptr + i)
    m = m + alpha
    tl.store(out_ptr + i, m)


def _launch_softplus(A_log: torch.Tensor) -> torch.Tensor:
    # A_log shape: [H] (H = number of heads), e.g., H=8
    N = A_log.numel()
    out = torch.empty(N, device=A_log.device, dtype=torch.float32)
    grid = (1,)
    _softplus_vector[grid](A_log, out, N=N)
    return out


def _launch_sigmoid(b: torch.Tensor) -> torch.Tensor:
    # b shape: [T, H] e.g., T=6, H=8
    N = b.numel()
    out = torch.empty(N, device=b.device, dtype=torch.float32)
    grid = (1,)
    _sigmoid_vector[grid](b, out, N=N)
    return out


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants from the original code
        self.head_size = 128
        self.num_q_heads = 4
        self.num_k_heads = 4
        self.num_v_heads = 8

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        q: [total_seq_len, num_q_heads, head_size] bfloat16
        k: [total_seq_len, num_k_heads, head_size] bfloat16
        v: [total_seq_len, num_v_heads, head_size] bfloat16
        state: [num_seqs, H, K, V] float32 (H=num_q_heads+num_v_heads-1, K=V=head_size)
        A_log: [H] float32
        a: [T, H] bfloat16
        dt_bias: [H] float32
        b: [T, H] bfloat16
        cu_seqlens: [num_seqs+1] int64
        scale: float (from original)
        """
        total_seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[1]
        num_k_heads = k.shape[1]
        # derive H (number of heads used) from state: H = state.size(1)
        assert state is not None
        H = state.size(1)
        num_sab_heads = max(num_q_heads, num_v_heads)
        assert self.num_q_heads == num_q_heads and self.num_k_heads == num_k_heads and self.num_v_heads == num_v_heads
        assert head_size == self.head_size

        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(self.head_size)

        # Compute g and beta per (t, h)
        # softplus
        x = a.float() + dt_bias.float()  # [T, H]
        # g = exp(-exp(A_log[h]) * softplus(x[t, h]))
        A_log_vec = _launch_softplus(A_log)  # [H] float32
        g = torch.exp(-torch.exp(A_log_vec) * x)  # [T, H], note: softplus(x) is done on host? No, we must use Triton: define a kernel that computes softplus on x and multiply by exp(A_log[h])
        # We'll compute softplus on x using Triton:
        N = x.numel()
        softplus_x = torch.empty(N, device=x.device, dtype=torch.float32)
        grid = (1,)
        _softplus_vector[grid](x, softplus_x, N=N)
        softplus_x = softplus_x.view(x.shape[0], x.shape[1])  # [T, H]
        g = torch.exp(-torch.exp(A_log) * softplus_x)  # [T, H] using broadcast

        # beta = sigmoid(b)
        beta = _launch_sigmoid(b)  # [T, H]

        # Prepare expanded q, k
        q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [T, num_v_heads, D]
        k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)  # [T, num_v_heads, D]

        # Output buffer [T, num_sab_heads, head_size] bfloat16
        output = torch.empty((total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=q.device)

        # New state buffer [num_seqs, H, K, V] float32
        num_seqs = cu_seqlens.size(0) - 1
        new_state = torch.zeros((num_seqs, H, self.head_size, self.head_size), dtype=torch.float32, device=q.device)

        # Per segment
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Initialize state_HKV for this segment: [H, V, K] (same as original: K-last layout)
            if state is not None:
                state_HKV = state[seq_idx].clone().float()  # [H, V, K]
            else:
                state_HKV = torch.zeros((H, self.head_size, self.head_size), dtype=torch.float32, device=q.device)

            # Prepare state_HKV as [H, K, V] for Triton kernels (we'll pass transposed)
            state_HKV_T = state_HKV.transpose(1, 2).contiguous()  # [H, K, V]

            for t in range(seq_len):
                # t in [seq_start, seq_start + seq_len - 1]
                t_global = seq_start + t
                # head loop: H = 8
                for h in range(H):
                    # Build vectors
                    q_vec = q_exp[t_global, h, :].contiguous().to(torch.float32)  # [D]
                    k_vec = k_exp[t_global, h, :].contiguous().to(torch.float32)  # [D]
                    v_vec = v[t_global, h, :].contiguous().to(torch.float32)      # [D]

                    # old_v = k_vec @ state_HKV_T[h]  => state_HKV_T[h] is [K, V]
                    K = self.head_size
                    V = self.head_size
                    A_ptr = state_HKV_T[h]  # [K, V]
                    out_old_v = torch.empty(V, device=q.device, dtype=torch.float32)
                    grid = (1,)
                    _gemv_1xKxKxV_into_1xV(q_vec, A_ptr, out_old_v, K=K, V=V)

                    # new_v = beta * v_vec + (1 - beta) * old_v
                    beta_t_h = beta[t_global, h]  # scalar float32
                    old_v = out_old_v  # [V]
                    new_v = torch.empty(V, device=q.device, dtype=torch.float32)
                    N = V
                    _elementwise_scalar_mul_add(beta_t_h, 1.0 - beta_t_h, v_vec, old_v, new_v, N=N)

                    # Compute dot products
                    dot_old = torch.empty(1, device=q.device, dtype=torch.float32)
                    _dot_scalar(k_vec, old_v, dot_old, N=K)  # scalar in dot_old[0]
                    dot_new = torch.empty(1, device=q.device, dtype=torch.float32)
                    _dot_scalar(k_vec, new_v, dot_new, N=K)  # scalar in dot_new[0]

                    # Update state_HKV_T[h] = g * state_HKV_T[h] + (dot_new - dot_old)[None, :]
                    g_t_h = g[t_global, h]  # scalar float32
                    out_add = torch.empty(K * V, device=q.device, dtype=torch.float32)
                    _add_scalar_to_matrix(g_t_h, A_ptr, out_add, N=K * V)  # adds g_t_h to all elements of A_ptr
                    delta_scalar = (float(dot_new.item()) - float(dot_old.item()))
                    out_add = out_add + delta_scalar  # add scalar to all K*V elements
                    state_HKV_T[h] = out_add.reshape(K, V)

                    # Compute output_vec = scale * (q_vec @ state_new_mat)
                    # state_new_mat is state_HKV_T[h] after update
                    out_vec = torch.empty(V, device=q.device, dtype=torch.float32)
                    _gemv_1xKxKxV_into_1xV(q_vec, state_HKV_T[h], out_vec, K=K, V=V)
                    out_scaled = scale * out_vec
                    output[t_global, h, :] = out_scaled.to(torch.bfloat16)

            # Store updated new_state for this segment: [H, V, K] from state_HKV_T [H, K, V]
            new_state[seq_idx] = state_HKV_T.transpose(1, 2).contiguous()  # [H, V, K]

        return output, new_state


def run(*args):
    return ModelNew()(*args)
