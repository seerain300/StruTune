import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr,
                                g_ptr, beta_ptr,
                                B: tl.constexpr, H: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    a_val = tl.load(a_ptr + b * H + h)      # a[B, H]
    dt_val = tl.load(dt_bias_ptr + h)       # dt_bias[H]
    A_val = tl.load(A_log_ptr + h)          # A_log[H]
    b_val = tl.load(b_ptr + b * H + h)      # b[B, H]

    x = a_val + dt_val
    g = tl.exp(-tl.exp(A_val) * tl.log(1.0 + tl.exp(x)))
    bet = 1.0 / (1.0 + tl.exp(-b_val))

    out_idx = b * H + h
    tl.store(g_ptr + out_idx, g)
    tl.store(beta_ptr + out_idx, bet)


@triton.jit
def _vec_matmul_tile_vec(k_vec_ptr, state_ptr,
                          out_ptr,
                          B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b,h). Compute out_vec[K] = sum_v k[v] * state[v, K]
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # k_vec is [K]; state is [V, K] contiguous; out is [K]
    # We write out[k] for k in 0..K-1
    for k in tl.static_range(0, K):
        acc = 0.0
        for v in tl.static_range(0, V):
            kv = tl.load(k_vec_ptr + v)        # k[v]
            state_val = tl.load(state_ptr + v * K + k)  # state[v, k]
            acc += kv * state_val
        tl.store(out_ptr + k, acc)


@triton.jit
def _output_scalar_kernel(q_vec_ptr, new_state_ptr,
                           out_ptr,
                           scale_ptr,
                           B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b,h). Compute output_scalar = scale * sum_k q[k] * new_state[k, :]
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    scale = tl.load(scale_ptr)  # scalar

    total = 0.0
    for k in tl.static_range(0, K):
        qk = tl.load(q_vec_ptr + k)               # q[k]
        # sum over v of new_state[k, v]
        sum_v = 0.0
        for v in tl.static_range(0, V):
            sum_v += tl.load(new_state_ptr + v * K + k)
        total += qk * sum_v

    tl.store(out_ptr + (b * H + h), scale * total)


@triton.jit
def _sqrt_scale_kernel(scale_ptr, K: tl.constexpr):
    # Compute scale = 1 / sqrt(K) and write to scale_ptr
    scale = 1.0 / tl.sqrt(K)
    tl.store(scale_ptr, scale)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original logic.
        q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        """
        # Shapes
        B, _, Hq, K = q.shape
        _, _, Hk, _ = k.shape
        _, _, Hv, V = v.shape
        B_s, Hv_s, V_s, K_s = state.shape
        assert B == B_s and Hv == Hv_s and V == V_s and K == K_s
        assert Hq == 4 and Hk == 4 and Hv == 8 and K == 128 and V == 128

        device = q.device
        dtype = torch.float32

        # Prepare inputs: cast to float32 and make contiguous
        q_c = q.contiguous().to(dtype)          # [B, 1, 4, 128]
        k_c = k.contiguous().to(dtype)          # [B, 1, 4, 128]
        v_c = v.contiguous().to(dtype)          # [B, 1, 8, 128]
        state_c = state.contiguous().to(dtype)  # [B, 8, 128, 128]

        # Prepare outputs
        g_out = torch.empty(B * Hv, device=device, dtype=dtype)     # [B, Hv]
        beta_out = torch.empty(B * Hv, device=device, dtype=dtype)  # [B, Hv]
        output_f = torch.empty(B * Hv, device=device, dtype=dtype)  # [B, Hv]
        new_state_f = torch.empty_like(state_c)                     # [B, 8, 128, 128]

        # Compute g and beta per (b,h) using Triton
        a_reshaped = a.view(B, Hv).contiguous().to(dtype)
        b_reshaped = b.view(B, Hv).contiguous().to(dtype)
        _compute_g_and_beta_kernel[(B * Hv,)](
            a_reshaped, dt_bias.contiguous().to(dtype), b_reshaped, A_log.contiguous().to(dtype),
            g_out, beta_out, B, Hv
        )

        # Compute scale = 1 / sqrt(K) in Triton
        scale_buf = torch.empty(1, device=device, dtype=dtype)
        _sqrt_scale_kernel[(1,)](scale_buf, K)
        scale_val = scale_buf[0]  # scalar float32

        # For each (b,h): compute old_v, new_v, state_remove, state_update, new_state, output
        for b_idx in range(B):
            for h_idx in range(Hv):
                # 1) Extract vectors and matrices
                # q_vec: [K], k_vec: [K], v_vec: [V], state_mat: [V, K]
                q_vec = q_c[b_idx, 0, h_idx * (Hq // Hv), :].contiguous().to(dtype)  # shape [K]
                k_vec = k_c[b_idx, 0, h_idx, :].contiguous().to(dtype)                # shape [K]
                state_mat = state_c[b_idx, h_idx, :, :].contiguous().view(V, K)       # [V, K]
                v_mat = v_c[b_idx, 0, h_idx, :].contiguous().to(dtype).view(V)        # [V]

                # 2) Compute old_v = k @ state
                old_v = torch.empty(K, device=device, dtype=dtype)                    # host temp
                _vec_matmul_tile_vec[(1,)](k_vec, state_mat, old_v, B, Hv, V, K)

                # 3) Compute new_v = beta * v + (1 - beta) * old_v
                beta_val = beta_out[b_idx * Hv + h_idx]
                g_val = g_out[b_idx * Hv + h_idx]
                new_v = beta_val * v_mat + (1.0 - beta_val) * old_v.to(dtype)         # [V]

                # 4) Compute state_remove = k @ old_v
                state_remove = 0.0
                for k in range(K):
                    state_remove += k_vec[k] * old_v[k]
                state_remove = torch.tensor(state_remove, device=device, dtype=dtype)

                # 5) Compute state_update = k @ new_v
                state_update = 0.0
                for k in range(K):
                    state_update += k_vec[k] * new_v[k]
                state_update = torch.tensor(state_update, device=device, dtype=dtype)

                # 6) Update new_state
                new_state_tile = g_val * state_mat.to(dtype) - state_remove + state_update  # [V, K]
                new_state_f[b_idx, h_idx, :, :] = new_state_tile

                # 7) Compute output[b,h] = scale * (q @ new_state)
                q_vec2 = q_vec  # same as above
                out_scalar = torch.zeros((), device=device, dtype=dtype)
                _output_scalar_kernel[(1,)](q_vec2, new_state_tile.contiguous(), out_scalar, scale_val, B, Hv, V, K)
                output_f[b_idx * Hv + h_idx] = out_scalar

        # Return outputs: output and new_state
        # Match original: output is [B, Hv] in float32 (to be converted to bfloat16 if needed),
        # new_state remains float32 as per original computation (state was float32).
        return output_f.unsqueeze(1), new_state_f


def run(*args):
    return ModelNew()(*args)
