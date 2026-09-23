import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(a_ptr, dt_bias_ptr, A_log_ptr, b_ptr,
                            g_ptr, beta_ptr,
                            B: tl.constexpr, H: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b * H + h)       # a[B, H]
    dt_val = tl.load(dt_bias_ptr + h)        # dt_bias[H]
    A_val = tl.load(A_log_ptr + h)           # A_log[H]

    # softplus(x) = log(1 + exp(x))
    softplus = tl.log(1.0 + tl.exp(a_val + dt_val))
    # g = exp(-exp(A_log[h]) * softplus)
    g_val = tl.exp(-tl.exp(A_val) * softplus)
    # beta = sigmoid(b[b,h]) = 1 / (1 + exp(-b[b,h]))
    b_val = tl.load(b_ptr + b * H + h)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to g_ptr[b,h] and beta_ptr[b,h]
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, state_ptr,
                          out_ptr,
                          K: tl.constexpr, V: tl.constexpr):
    # out_ptr is length K
    # state_ptr is [V, K] flattened row-major (offset = v*K + k)
    # out[k] = sum_v k[v] * state[v, k]
    for k in tl.static_range(K):
        sum_val = 0.0
        for v in tl.static_range(V):
            s = tl.load(state_ptr + v * K + k)
            kv = tl.load(k_ptr + v)
            sum_val += kv * s
        tl.store(out_ptr + k, sum_val)


@triton.jit
def _vec_matmul_scalar(k_ptr, vec_ptr,
                        out_ptr,
                        K: tl.constexpr):
    # out_ptr is scalar (single element write)
    sum_val = 0.0
    for k in tl.static_range(K):
        kv = tl.load(k_ptr + k)
        vv = tl.load(vec_ptr + k)
        sum_val += kv * vv
    tl.store(out_ptr, sum_val)


@triton.jit
def _q_dot_kernel(q_ptr, state_ptr, out_ptr,
                   V: tl.constexpr, K: tl.constexpr):
    # Compute sum over v of (q @ state_row_v) where state_ptr points to row-major [V,K]
    # For each v in [0..V-1], sum_k q[k] * state[v,k]
    for v in tl.static_range(V):
        row_sum = 0.0
        for k in tl.static_range(K):
            qk = tl.load(q_ptr + k)
            svk = tl.load(state_ptr + v * K + k)
            row_sum += qk * svk
        # Accumulate row_sum into out_ptr[0]
        tl.atomic_add(out_ptr, row_sum)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the provided logic.
        Inputs:
          q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
          A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float (not used here)
        Returns:
          output: [B, 8] in bfloat16
          new_state: [B, 8, 128, 128] in float32
        """

        # Cast to float32 and make contiguous
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)
        state = state.contiguous().to(torch.float32)

        B, T_q, Hq, K = q.shape
        _, T_k, Hk, _ = k.shape
        _, T_v, Hv, V = v.shape
        assert T_q == 1 and T_k == 1
        assert Hq == 4 and Hk == 4 and Hv == 8 and V == 128 and K == 128
        H = Hv  # number of heads (8)

        device = q.device

        # Allocate outputs
        g_out = torch.empty(B * H, device=device, dtype=torch.float32)
        beta_out = torch.empty(B * H, device=device, dtype=torch.float32)
        output_f = torch.empty(B * H, device=device, dtype=torch.float32)

        # Launch compute g and beta kernel
        _compute_g_beta_kernel[(B * H,), 128, 128, B, H](a, dt_bias, A_log, b,
                                                        g_out, beta_out)

        # new_state tensor [B, H, V, K] as float32
        new_state_f = torch.empty((B, H, V, K), device=device, dtype=torch.float32)

        # Compute new_state and output per (b,h)
        for b_idx in range(B):
            for h_idx in range(H):
                # Prepare vectors
                q_vec = q[b_idx, 0, :, :].contiguous().to(torch.float32)  # [K]
                k_vec = k[b_idx, 0, :, :].contiguous().to(torch.float32)  # [K]
                v_vec = v[b_idx, 0, :, :].contiguous().to(torch.float32)  # [V]

                # Load g and beta for this (b,h)
                g_val = g_out[b_idx * H + h_idx]
                beta_val = beta_out[b_idx * H + h_idx]

                # state_old = g * state[b,h,:,:]
                state_old = state[b_idx, h_idx, :, :].contiguous() * g_val  # [V,K]

                # old_v = k @ state_old = sum_v k[v] * state


def run(*args):
    return ModelNew()(*args)
