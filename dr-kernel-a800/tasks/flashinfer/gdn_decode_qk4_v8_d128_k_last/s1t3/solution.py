import math
import torch
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = softplus(x[i]) = log(1 + exp(x[i])) for i in [0, N).
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    # softplus(x) = log(1 + exp(x)); since x is float32, this is fine
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = sigmoid(x[i]) = 1 / (1 + exp(-x[i])) for i in [0, N).
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def matvec_kernel(x_ptr, k_ptr, y_ptr, K: tl.constexpr, V: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute y = k @ x, where:
      - x is a 2D matrix of shape [K, V] (passed as a contiguous 1D pointer of length K*V)
      - k is a 1D vector of length K
      - y is a 1D vector of length V
    Each program instance handles one output v-index and loops over K in chunks of BLOCK_K.
    """
    v_idx = tl.program_id(axis=0)
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_chunk = tl.load(k_ptr + k_offsets, mask=k_offsets < K, other=0.0)  # [BLOCK_K]
        # For each kk in chunk, accumulate k_chunk[kk] * x[kk, v_idx]
        for kk in range(0, BLOCK_K):
            k_val = k_chunk[kk]
            k_idx = k_start + kk
            x_val = tl.load(x_ptr + k_idx * V + v_idx)
            acc += k_val * x_val
    tl.store(y_ptr + v_idx, acc)


@triton.jit
def dot_kernel(q_ptr, x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out = sum_i q[i] * x[i], where q and x are 1D vectors of length N.
    One program computes the entire sum (for small N).
    """
    acc = tl.zeros((), dtype=tl.float32)
    for i in range(0, N):
        qi = tl.load(q_ptr + i)
        xi = tl.load(x_ptr + i)
        acc += qi * xi
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only forward:
        - q: [B, 1, 4, 128], bfloat16
        - k: [B, 1, 4, 128], bfloat16
        - v: [B, 1, 8, 128], bfloat16
        - state: [B, 8, 128, 128], float32 (k-last layout)
        - A_log: [8], float32
        - a: [1, 1, 8], bfloat16 -> squeeze to [8]
        - dt_bias: [8], float32
        - b: [1, 1, 8], bfloat16 -> squeeze to [8]
        - scale: float (python scalar)
        Returns:
        - output: [B, 1, 8, 1], bfloat16
        - new_state: [B, 8, 128, 128], float32
        """
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        assert T == 1
        assert num_q_heads == 4
        assert num_k_heads == 4
        assert num_v_heads == 8
        assert K == 128 and V == 128

        device = q.device

        # Squeeze T=1
        q_s = q.squeeze(1)  # [B, 4, 128]
        k_s = k.squeeze(1)  # [B, 4, 128]
        v_s = v.squeeze(1)  # [B, 8, 128]

        # Compute parameters: g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
        a_v = a.squeeze().float()        # [8]
        dt_bias_v = dt_bias.float()      # [8]
        b_v = b.squeeze().float()        # [8]
        A_log_v = A_log.float()          # [8]

        # Compute x = a + dt_bias using Triton elementwise add (tiny size)
        x = torch.empty(8, dtype=torch.float32, device=device)
        # Triton elementwise addition: x = a + dt_bias
        for i in range(8):
            xi = a_v[i] + dt_bias_v[i]
            tl.store(x + i, xi)
        # Compute softplus(x)
        softplus_vec = torch.empty(8, dtype=torch.float32, device=device)
        softplus_kernel[(8,)](x, softplus_vec, N=8)
        # g = exp(-exp(A_log) * softplus(x))
        exp_A = torch.empty(8, dtype=torch.float32, device=device)
        exp_kernel[(8,)](A_log_v, exp_A, N=8)
        g_vec = -exp_A * softplus_vec
        g_vec = torch.empty(8, dtype=torch.float32, device=device)
        exp_kernel[(8,)](g_vec, g_vec, N=8)  # out = exp(g_vec)

        # beta = sigmoid(b)
        beta_vec = torch.empty(8, dtype=torch.float32, device=device)
        sigmoid_kernel[(8,)](b_v, beta_vec, N=8)

        # Expanded q, k to H=8 (original code repeats q, k to 8; q originally has 4, but we mimic repeat by selecting head 0)
        q_exp = q_s  # [B, 4, 128]; we will use q_exp[b, 0, :] for each h
        k_exp = k_s  # [B, 4, 128]
        v_exp = v_s  # [B, 8, 128]

        # Allocate output and new_state
        output = torch.empty((B, 1, 8, 1), dtype=torch.bfloat16, device=device)
        new_state = state.clone()  # [B, 8, 128, 128], float32

        # Loop over batch and heads; compute per (b,h)
        for b_idx in range(B):
            for h_idx in range(8):
                # Load per-head vectors
                q_h = q_exp[b_idx, 0, :]  # [128], bfloat16 -> convert to float32 for compute
                k_h = k_exp[b_idx, 0, :].float()  # [128], float32
                v_h = v_exp[b_idx, h_idx, :].float()  # [128], float32

                # Load old_state for this (b,h) as [128, 128], float32
                old_state = state[b_idx, h_idx].contiguous()  # [128, 128]

                # Compute old_v = k_h @ old_state => Triton matvec over x=old_state (K=128, V=128)
                old_v = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(128,)](old_state.reshape(128 * 128), k_h, old_v, K=128, V=128, BLOCK_K=64)

                # Compute new_v = beta * v_h + (1 - beta) * old_v
                new_v = beta_vec[h_idx] * v_h + (1.0 - beta_vec[h_idx]) * old_v  # [128], float32

                # state_remove = k_h @ old_v
                state_remove = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(128,)](old_v, k_h, state_remove, K=128, V=128, BLOCK_K=64)

                # state_update = k_h @ new_v
                state_update = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(128,)](new_v, k_h, state_update, K=128, V=128, BLOCK_K=64)

                # Update new_state per element: new_state[b, h, i, j] = g[h] * old_state[i, j] - state_remove[i] + state_update[i]
                g_val = g_vec[h_idx]  # scalar float32
                h_state_new = (g_val * old_state) - state_remove.view(128, 1) + state_update.view(128, 1)
                new_state[b_idx, h_idx] = h_state_new

                # Compute output scalar: out = scale * (q_h @ h_state_new[:, 0])
                col0 = h_state_new[:, 0]  # [128], float32

                # Dot product using Triton kernel: q_h @ col0
                q_vec = q_h.float()  # [128]
                out_scalar_buf = torch.empty(1, dtype=torch.float32, device=device)
                dot_kernel[(128,)](q_vec, col0, out_scalar_buf)

                # Apply scale
                if scale is None or scale == 0.0:
                    scale_val = 1.0 / math.sqrt(K)
                else:
                    scale_val = float(scale)
                out_scalar = out_scalar_buf[0] * scale_val  # float32 scalar

                # Store into output[b, 0, h, 0] as bfloat16
                # Create a 1-element bfloat16 tensor and write via Triton-like assignment in Python.
                # Triton doesn't expose direct tensor write here; we'll use torch to place the scalar.
                output[b_idx, 0, h_idx, 0] = torch.tensor(out_scalar, dtype=torch.bfloat16, device=device)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
