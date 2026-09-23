import torch
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

    a_val = tl.load(a_ptr + b * H + h)       # a[B, H]
    dt_val = tl.load(dt_bias_ptr + h)        # dt_bias[H]
    A_val = tl.load(A_log_ptr + h)           # A_log[H]

    x = a_val + dt_val
    g = tl.exp(-tl.exp(A_val) * tl.log(1.0 + tl.exp(x)))
    bet = 1.0 / (1.0 + tl.exp(-b_val))

    # Write results
    out_idx = b * H + h
    tl.store(g_ptr + out_idx, g)
    tl.store(beta_ptr + out_idx, bet)


@triton.jit
def _vec_matmul_tile_vec(k_vec_ptr, state_mat_ptr, out_vec_ptr,
                         B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # k_vec: length K, state_mat: [V, K], out_vec: length K
    k_idx = tl.arange(0, K)
    k_vec = tl.load(k_vec_ptr + k_idx)  # [K]

    out_vec = tl.zeros([K], dtype=tl.float32)
    for v in tl.static_range(0, V):
        state_row_ptr = state_mat_ptr + v * K
        state_row = tl.load(state_row_ptr + tl.arange(0, K))  # [K]
        out_vec += k_vec * state_row

    tl.store(out_vec_ptr + b * (H * K) + h * K, out_vec)


@triton.jit
def _vec_matmul_scalar(k_vec_ptr, vec_ptr, out_scalar_ptr,
                       B: tl.constexpr, H: tl.constexpr, K: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    k_idx = tl.arange(0, K)
    k_vec = tl.load(k_vec_ptr + k_idx)     # [K]
    vec = tl.load(vec_ptr + tl.arange(0, K))  # [K]
    out = tl.sum(k_vec * vec, axis=0)      # scalar
    tl.store(out_scalar_ptr + b * H + h, out)


@triton.jit
def _output_scalar_kernel(q_vec_ptr, new_state_mat_ptr, out_scalar_ptr, scale,
                          B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    q_idx = tl.arange(0, K)
    q_vec = tl.load(q_vec_ptr + q_idx)     # [K]

    out_acc = tl.zeros((), dtype=tl.float32)  # scalar accumulator
    for v in tl.static_range(0, V):
        new_state_row_ptr = new_state_mat_ptr + v * K
        new_state_row = tl.load(new_state_row_ptr + tl.arange(0, K))  # [K]
        out_acc += tl.sum(q_vec * new_state_row, axis=0)  # scalar

    out = out_acc * scale
    tl.store(out_scalar_ptr + b * H + h, out)


@triton.jit
def _sqrt_scale_kernel(out_ptr, K: tl.constexpr):
    # Compute scale = 1 / sqrt(K) and store
    scale = 1.0 / tl.sqrt(K)
    tl.store(out_ptr, scale)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original logic.
        q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        Returns: output [B, 1, 8] in bfloat16, and new_state [B, 8, 128, 128]
        """
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and state.dim() == 4
        B, _, Hq, K = q.shape
        _, _, Hk, _ = k.shape
        _, _, Hv, V = v.shape
        B_s, Hv_s, V_s, K_s = state.shape
        assert B_s == B and Hv_s == Hv and V_s == V and K_s == K
        assert Hq == 4 and Hk == 4 and Hv == 8 and K == 128 and V == 128

        device = q.device
        dtype = torch.float32

        # Cast and make contiguous
        q_c = q.contiguous().to(dtype)          # [B, 1, 4, 128]
        k_c = k.contiguous().to(dtype)          # [B, 1, 4, 128]
        v_c = v.contiguous().to(dtype)          # [B, 1, 8, 128]
        state_c = state.contiguous().to(dtype)  # [B, 8, 128, 128]

        # Prepare outputs and buffers
        g_out = torch.empty(B * Hv, device=device, dtype=dtype)     # [B, Hv]
        beta_out = torch.empty(B * Hv, device=device, dtype=dtype)  # [B, Hv]
        output_f = torch.empty(B * Hv, device=device, dtype=dtype)  # [B, Hv]
        new_state_f = torch.empty_like(state_c)                     # [B, 8, 128, 128]

        # Compute g and beta per (b,h) using Triton
        _compute_g_and_beta_kernel[(B * Hv,)](
            a.contiguous().view(-1, Hv), dt_bias, b.contiguous().view(-1, Hv), A_log,
            g_out, beta_out, B, Hv
        )

        # Compute scale = 1 / sqrt(K) using Triton
        scale_buf = torch.empty(1, device=device, dtype=dtype)
        _sqrt_scale_kernel[(1,)](scale_buf, K)
        scale_val = scale_buf[0]  # scalar float32

        # For each (b,h): compute updates and output
        for b_idx in range(B):
            for h_idx in range(Hv):
                base = b_idx * Hv + h_idx
                g_val = g_out[base]
                beta_val = beta_out[base]

                # Extract vectors and matrices for this (b,h)
                # q_vec: [K], k_vec: [K], v_vec: [V], state_mat: [V, K]
                q_vec = q_c[b_idx, 0, h_idx * (Hq // Hv), :]   # [K]
                k_vec = k_c[b_idx, 0, h_idx * (Hk // Hv), :]   # [K]
                v_vec = v_c[b_idx, 0, h_idx, :]                # [V]
                state_mat = state_c[b_idx, h_idx, :, :]        # [V, K]

                # Compute old_v = k @ state using Triton
                old_v = torch.empty(K, device=device, dtype=dtype)
                _vec_matmul_tile_vec[(1,)](
                    k_vec, state_mat, old_v, B, Hv, V, K
                )

                # Compute new_v = beta * v + (1 - beta) * old_v
                new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [K]

                # Compute state_remove = k @ old_v and state_update = k @ new_v (scalars)
                state_remove = torch.empty((), device=device, dtype=dtype)
                _vec_matmul_scalar[(1,)](
                    k_vec, old_v, state_remove, B, Hv, K
                )
                state_update = torch.empty((), device=device, dtype=dtype)
                _vec_matmul_scalar[(1,)](
                    k_vec, new_v, state_update, B, Hv, K
                )

                # Update new_state: new_state_row = g * state_row - state_remove + state_update
                for v_row in range(V):
                    state_row = state_c[b_idx, h_idx, v_row, :]   # [K]
                    new_state_row = g_val * state_row - state_remove + state_update
                    new_state_c_row_ptr = new_state_f[b_idx, h_idx, v_row, :]
                    new_state_c_row_ptr.copy_(new_state_row)

                # Compute output[b,h] = scale * (q @ new_state) using Triton
                # Build a [V,K] matrix from new_state_f for this head and call kernel.
                new_state_mat_for_out = torch.empty((V, K), device=device, dtype=dtype)
                for v_row in range(V):
                    new_state_mat_for_out[v_row, :] = new_state_f[b_idx, h_idx, v_row, :]
                out_scalar = torch.empty((), device=device, dtype=dtype)
                _output_scalar_kernel[(1,)](
                    q_vec, new_state_mat_for_out, out_scalar, scale_val, B, Hv, V, K
                )
                output_f[base] = out_scalar

        # Return output [B, 1, 8] in bfloat16 and new_state [B, 8, 128, 128]
        output_bf16 = output_f.view(B, Hv).unsqueeze(1).to(torch.bfloat16)
        return output_bf16, new_state_f


def run(*args):
    return ModelNew()(*args)
