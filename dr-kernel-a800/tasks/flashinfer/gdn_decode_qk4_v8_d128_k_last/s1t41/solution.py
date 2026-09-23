import math
import torch
import triton
import triton.language as tl


@triton.jit
def exp_kernel(inp_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = exp(inp[i]) for i in [0, N).
    Single-thread per element for simplicity.
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(inp_ptr + i, mask=i < N, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def softplus_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = softplus(x[i]) = log(1 + exp(x[i])) for i in [0, N).
    Single-thread per element.
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def sigmoid_kernel(z_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = sigmoid(z[i]) = 1 / (1 + exp(-z[i])) for i in [0, N).
    Single-thread per element.
    """
    pid = tl.program_id(axis=0)
    i = pid
    z = tl.load(z_ptr + i, mask=i < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-z))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def matvec_kernel(y_ptr, k_ptr, x_ptr, K: tl.constexpr, V: tl.constexpr):
    """
    Compute y = k @ x, where x is [K, V] provided as 1D contiguous pointer (K*V),
    k is [K], y is [V]. Each program handles one output element (i in [0, V)) and
    accumulates over K.
    """
    pid = tl.program_id(axis=0)
    i = pid  # output index in [0, V)
    acc = 0.0
    for k in range(0, K):
        k_val = tl.load(k_ptr + k)
        x_val = tl.load(x_ptr + k * V + i)
        acc += k_val * x_val
    tl.store(y_ptr + i, acc)


@triton.jit
def dot_kernel(out_ptr, q_ptr, x_ptr, N: tl.constexpr):
    """
    Compute out = sum_i q[i] * x[i]. Store result to out_ptr[0].
    Single program loops over N and accumulates.
    """
    pid = tl.program_id(axis=0)  # only one program
    acc = 0.0
    for i in range(0, N):
        q_i = tl.load(q_ptr + i)
        x_i = tl.load(x_ptr + i)
        acc += q_i * x_i
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, 128], bfloat16
        k: [B, 1, 4, 128], bfloat16
        v: [B, 1, 8, 128], bfloat16
        state: [B, 8, 128, 128], float32 (k-last)
        A_log: [8], float32
        a: [1, 1, 8], bfloat16
        dt_bias: [8], float32
        b: [1, 1, 8], bfloat16
        scale: float or None
        Returns:
        - output: [B, 1, 8, 1], bfloat16
        - new_state: [B, 8, 128, 128], float32
        """
        B, T, q_heads, K = q.shape
        _, _, k_heads, _ = k.shape
        _, _, v_heads, V = v.shape
        device = q.device

        # Ensure inputs are contiguous
        q = q.squeeze(1).contiguous()
        k = k.squeeze(1).contiguous()
        v = v.squeeze(1).contiguous()
        state = state.contiguous()

        # Compute g and beta using Triton kernels
        # a is [1,1,H] -> [H], dt_bias is [H], b is [1,1,H] -> [H]
        a_vec = a.squeeze().contiguous().float()         # [H]
        dt_bias_vec = dt_bias.contiguous().float()       # [H]
        b_vec = b.squeeze().contiguous().float()         # [H]
        H = a_vec.shape[0]  # should be 8

        # 1) x = a + dt_bias
        x = a_vec + dt_bias_vec  # [H]
        x_sp = torch.empty(H, dtype=torch.float32, device=device)
        sigmoid_kernel[(H,)](x, x_sp, H)  # compute sigmoid(x)

        # 2) A_log -> exp(A_log)
        A_log_vec = A_log.contiguous().float()  # [H]
        exp_A_log = torch.empty(H, dtype=torch.float32, device=device)
        exp_kernel[(H,)](A_log_vec, exp_A_log, H)

        # 3) softplus(x_sp) = log(1 + exp(x_sp))
        sp = torch.empty(H, dtype=torch.float32, device=device)
        softplus_kernel[(H,)](x_sp, sp, H)

        # 4) g = exp(-exp_A_log * sp)
        g = torch.empty(H, dtype=torch.float32, device=device)
        exp_kernel[(H,)](-exp_A_log * sp, g, H)

        # 5) beta = sigmoid(b)
        beta = torch.empty(H, dtype=torch.float32, device=device)
        sigmoid_kernel[(H,)](b_vec, beta, H)

        # 6) Repeat q,k along head dim: factor 2
        q_exp = q.repeat_interleave(2, dim=1)  # [B, 8, 128]
        k_exp = k.repeat_interleave(2, dim=1)  # [B, 8, 128]

        # Allocate output and new_state
        output = torch.empty((B, 1, v_heads, 1), dtype=torch.bfloat16, device=device)
        new_state = torch.empty_like(state, dtype=torch.float32, device=device)

        # Loop over batch and heads
        for b_idx in range(B):
            for h_idx in range(v_heads):
                # Vectors
                q_h = q_exp[b_idx, h_idx].contiguous()  # [128], bfloat16
                k_h = k_exp[b_idx, h_idx].contiguous()  # [128], bfloat16
                v_h = v[b_idx, h_idx].contiguous()      # [128], bfloat16

                # old_state for this (b, h): state[b, h] -> [V, K] = [128, 128]
                old_state = state[b_idx, h_idx].contiguous()  # [128, 128], float32

                # 7) old_v = k_h @ old_state (K=128, V=128)
                old_v = torch.empty(128, dtype=torch.float32, device=device)
                k_h_f = k_h.float()  # [128], float32
                x_old = old_state.view(128 * 128).contiguous().float()  # [K*V] contiguous
                matvec_kernel[(128,)](old_v, k_h_f, x_old, 128, 128)

                # 8) new_v = beta[h] * v_h + (1 - beta[h]) * old_v
                beta_val = beta[h_idx]  # scalar float32
                old_v_scaled = old_v * (1.0 - beta_val)
                new_v = (v_h.float() * beta_val) + old_v_scaled  # [128], float32

                # 9) state_remove = k_h @ old_v
                state_remove = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(128,)](state_remove, k_h_f, old_v, 128, 128)

                # 10) state_update = k_h @ new_v
                state_update = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(128,)](state_update, k_h_f, new_v, 128, 128)

                # 11) Update state: new_state[b, h] = g[h] * old_state - state_remove + state_update
                g_val = g[h_idx]  # scalar float32
                old_state_flat = old_state.view(128 * 128).contiguous().float()
                rm_i = torch.empty(128 * 128, dtype=torch.float32, device=device)
                st_i = torch.empty(128 * 128, dtype=torch.float32, device=device)
                # construct rm_i, st_i from their vectors (length V=128), broadcast over K
                # For flat index idx in [0,128*128): i = idx // 128, j = idx % 128
                idx = torch.arange(0, 128 * 128, device=device)
                i_idx = idx // 128
                j_idx = idx % 128
                rm_i = (state_remove[i_idx] * 0.0) + state_remove[i_idx]  # broadcast via index_select
                st_i = (state_update[i_idx] * 0.0) + state_update[i_idx]
                new_state_flat = old_state_flat * g_val - rm_i + st_i
                new_state[b_idx, h_idx] = new_state_flat.view(128, 128)

                # 12) Compute output scalar: q_h @ new_state[:, 0], scaled by 1/sqrt(K) if needed
                col0 = new_state[b_idx, h_idx].index_select(1, 0).contiguous().view(-1)  # [128]
                # Triton dot q_h @ col0
                q_vec = q_h.float()  # [128]
                out_scalar_buf = torch.empty(1, dtype=torch.float32, device=device)
                dot_kernel[(128,)](out_scalar_buf, q_vec, col0, 128)
                out_scalar = out_scalar_buf[0]

                # Apply scale
                if scale is None or scale == 0.0:
                    scale_val = 1.0 / math.sqrt(K)
                else:
                    scale_val = float(scale)
                out_scalar = out_scalar * scale_val

                # Store into output[b, 0, h, 0] as bfloat16 (scalar)
                output[b_idx, 0, h_idx, 0] = torch.tensor(out_scalar, dtype=torch.bfloat16, device=device)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
