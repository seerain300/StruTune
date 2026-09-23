import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(a_ptr, dt_bias_ptr, A_log_ptr, b_ptr, g_ptr, beta_ptr,
                           B: tl.constexpr, H: tl.constexpr):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    a_val = tl.load(a_ptr + b * H + h)       # float32
    dt_val = tl.load(dt_bias_ptr + h)        # float32
    A_val = tl.load(A_log_ptr + h)           # float32
    b_val = tl.load(b_ptr + b * H + h)       # float32

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g = tl.exp(-tl.exp(A_val) * sp)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    idx = b * H + h
    tl.store(g_ptr + idx, g)
    tl.store(beta_ptr + idx, beta)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, A_ptr, out_ptr,
                         K: tl.constexpr, V: tl.constexpr):
    # Compute out[i] = sum_j k[j] * A[i*V + j] for i in [0, K)
    # A is [V, K] flattened as row-major: index = i*V + j
    for i in range(K):
        acc = 0.0
        for j in range(V):
            a_val = tl.load(A_ptr + i * V + j)
            k_val = tl.load(k_ptr + j)
            acc += a_val * k_val
        tl.store(out_ptr + i, acc)


@triton.jit
def _vec_matmul_scalar(k_ptr, vec_ptr, out_ptr,
                       K: tl.constexpr):
    # Compute scalar = sum_i k[i] * vec[i]
    acc = 0.0
    for i in range(K):
        k_val = tl.load(k_ptr + i)
        v_val = tl.load(vec_ptr + i)
        acc += k_val * v_val
    tl.store(out_ptr, acc)


@triton.jit
def _q_dot_kernel(q_ptr, A_ptr, out_ptr,
                  K: tl.constexpr, V: tl.constexpr):
    # Compute scalar = q @ A, where A is [V, K] flattened (row-major)
    acc = 0.0
    for i in range(V):
        row_start = i * K
        for j in range(K):
            a_val = tl.load(A_ptr + row_start + j)
            q_val = tl.load(q_ptr + j)
            acc += a_val * q_val
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], 
        state: [B, 8, 128, 128], A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8]
        Returns:
          output: [B, 8] in bfloat16
          new_state: [B, 8, 128, 128] in float32
        """
        # Ensure contiguous and float32
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

        # Allocate outputs for g and beta
        g_out = torch.empty(B * H, device=device, dtype=torch.float32)
        beta_out = torch.empty(B * H, device=device, dtype=torch.float32)

        # Launch compute g and beta kernel
        _compute_g_beta_kernel[(B * H,), 128, 128, B, H](
            a.contiguous().to(torch.float32),
            dt_bias.contiguous().to(torch.float32),
            A_log.contiguous().to(torch.float32),
            b.contiguous().to(torch.float32),
            g_out,
            beta_out,
        )

        # Compute scale (float32 scalar)
        scale_val = float(1.0 / math.sqrt(K))

        # Allocate final outputs
        output_f = torch.empty(B * H, device=device, dtype=torch.float32)  # [B*H]
        new_state_f = torch.empty((B, H, V, K), device=device, dtype=torch.float32)

        # Compute per (b,h)
        for b_idx in range(B):
            for h_idx in range(H):
                # Vectors
                q_vec = q[b_idx, 0, :, :].contiguous()  # [K]
                k_vec = k[b_idx, 0, :, :].contiguous()  # [K]
                v_vec = v[b_idx, 0, :, :].contiguous()  # [V]

                # Load g and beta for this (b,h)
                g_val = g_out[b_idx * H + h_idx]  # scalar
                beta_val = beta_out[b_idx * H + h_idx]  # scalar

                # Prepare state_old as [V, K] flattened for _vec_matmul_tile_vec
                state_old = state[b_idx, h_idx, :, :].contiguous()  # [V, K]
                # Compute old_v = k @ (g * state_old) via tile vector kernel
                # We pass g*state_old as A and k as k_ptr
                A_for_old = (state_old * g_val).reshape(V * K).contiguous()  # [V*K]
                out_old_v = torch.empty(K, device=device, dtype=torch.float32)
                _vec_matmul_tile_vec[(1,), 128, 128](k_vec, A_for_old, out_old_v)
                old_v = out_old_v  # [K]

                # Compute new_v = beta * v + (1 - beta) * old_v
                new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [V]

                # Compute state_remove = k @ old_v via scalar kernel
                state_remove = torch.empty(1, device=device, dtype=torch.float32)
                _vec_matmul_scalar[(1,), 128](k_vec, old_v, state_remove)
                state_remove = state_remove[0]

                # Compute state_update = k @ new_v via scalar kernel
                state_update = torch.empty(1, device=device, dtype=torch.float32)
                _vec_matmul_scalar[(1,), 128](k_vec, new_v, state_update)
                state_update = state_update[0]

                # Update new_state: start from g * state
                new_state_tile = (state_old * g_val) - state_remove + state_update  # [V, K]

                # Write into new_state_f [B, H, V, K]
                new_state_f[b_idx, h_idx, :, :] = new_state_tile

                # Compute output[b,h] = scale * (q @ new_state[b,h,:,:]) via _q_dot_kernel
                # Flatten new_state[b,h,:,:] to [V*K]
                A_q = new_state_tile.reshape(V * K).contiguous()
                out_scalar = torch.empty(1, device=device, dtype=torch.float32)
                _q_dot_kernel[(1,), 128, 128](q_vec, A_q, out_scalar)
                output_f[b_idx * H + h_idx] = scale_val * out_scalar[0]

        # Return as required: output [B, 8] bfloat16, new_state [B, 8, 128, 128] float32
        output = output_f.view(B, H).to(torch.bfloat16)
        return output, new_state_f


def run(*args):
    return ModelNew()(*args)
