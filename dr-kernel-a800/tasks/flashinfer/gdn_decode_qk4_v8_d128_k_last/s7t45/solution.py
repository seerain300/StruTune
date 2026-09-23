import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr,
                                g_ptr, beta_ptr,
                                B: tl.constexpr, H: tl.constexpr):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b * H + h)        # a[b, h]
    dt_val = tl.load(dt_bias_ptr + h)         # dt_bias[h]
    A_val = tl.load(A_log_ptr + h)            # A_log[h]
    b_val = tl.load(b_ptr + b * H + h)        # b[b, h]

    # softplus(x) = log(1 + exp(x))
    x = a_val + dt_val
    softplus_x = tl.log(1.0 + tl.exp(x))

    # g = exp(-exp(A_log[h]) * softplus(a + dt_bias))
    g_val = tl.exp(-tl.exp(A_val) * softplus_x)

    # beta = sigmoid(b[b,h]) = 1 / (1 + exp(-b[b,h]))
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, state_ptr, out_ptr,
                         K: tl.constexpr, V: tl.constexpr):
    # out[k] = sum_v k[v] * state[v, k], v in [0, V), k in [0, K)
    pid = tl.program_id(0)
    for k in tl.static_range(K):
        acc = 0.0
        for v in tl.static_range(V):
            state_index = v * K + k
            sv = tl.load(state_ptr + state_index)
            kv = tl.load(k_ptr + v)
            acc += sv * kv
        tl.store(out_ptr + k, acc)


@triton.jit
def _vec_matmul_scalar(k_ptr, vec_ptr, scalar_ptr,
                       K: tl.constexpr):
    # scalar = sum_k k[k] * vec[k], k in [0, K)
    acc = 0.0
    for k in tl.static_range(K):
        kv = tl.load(k_ptr + k)
        vv = tl.load(vec_ptr + k)
        acc += kv * vv
    tl.store(scalar_ptr, acc)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, output_ptr,
                          scale_ptr,
                          B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    scale = tl.load(scale_ptr)  # 1-element tensor

    acc = 0.0

    # For each vector v in [0..V-1], compute q @ new_state[:, v] and accumulate
    for v in tl.static_range(V):
        # q[b,h,:] -> [K]
        q_row = tl.zeros((K,), dtype=tl.float32)
        for k in tl.static_range(K):
            q_row[k] = tl.load(q_ptr + b * H * K + h * K + k)

        # new_state[b,h,v,:] -> [K], laid out as [V*K]
        vec_row = tl.zeros((K,), dtype=tl.float32)
        base = b * H * V * K + h * V * K + v * K
        for k in tl.static_range(K):
            vec_row[k] = tl.load(new_state_ptr + base + k)

        # Dot product q_row · vec_row
        dot = 0.0
        for k in tl.static_range(K):
            dot += q_row[k] * vec_row[k]

        acc += dot

    out_val = scale * acc
    tl.store(output_ptr + b * H + h, out_val)


@triton.jit
def _sqrt_scale_kernel(scale_ptr, K: tl.constexpr):
    inv_sqrt = 1.0 / tl.sqrt(K)
    tl.store(scale_ptr, inv_sqrt)


class ModelNew(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: torch.Tensor,
                A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor, scale):
        """
        Triton-only implementation. All computations happen inside Triton kernels.
        Shapes (matching get_inputs):
          - q: [B, 1, 4, 128] (bfloat16)
          - k: [B, 1, 4, 128] (bfloat16)
          - v: [B, 1, 8, 128] (bfloat16)
          - state: [B, 8, 128, 128] (float32)
          - A_log: [8] (float32)
          - a: [B, 1, 8] (bfloat16)
          - dt_bias: [8] (float32)
          - b: [B, 1, 8] (bfloat16)
          - scale: float32 scalar or None
        Returns:
          - output: [B, 1, 8] (bfloat16)
          - new_state: [B, 8, 128, 128] (float32)
        """
        # Constants (per task)
        K = 128
        V = 128
        H = 8
        B = q.shape[0]

        # Prepare inputs: float32 and contiguous
        device = q.device
        q_f32 = q.float().contiguous()      # [B, 1, 4, 128]
        k_f32 = k.float().contiguous()      # [B, 1, 4, 128]
        v_f32 = v.float().contiguous()      # [B, 1, 8, 128]
        state_f32 = state.float().contiguous()  # [B, 8, 128, 128] (float32)

        # Output buffers
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)
        output_f = torch.empty((B, H), dtype=torch.float32, device=device)  # we'll return bfloat16

        # Compute scale = 1/sqrt(K) via Triton (ensure kernel invocation)
        scale_buf = torch.empty((1,), dtype=torch.float32, device=device)
        _sqrt_scale_kernel[(1,)](scale_buf, K)

        # Compute g and beta per (b,h) using Triton
        a_f32 = a.float().contiguous()      # [B, 1, 8]
        b_f32 = b.float().contiguous()      # [B, 1, 8]
        A_log_f32 = A_log.float().contiguous()  # [8]
        _compute_g_and_beta_kernel[(B * H,)](a_f32.view(-1), dt_bias.float().contiguous(),
                                             b_f32.view(-1), A_log_f32,
                                             g_out, beta_out, B, H)

        # Now per-(b,h) updates
        new_state_f32 = torch.empty_like(state_f32)

        for b_idx in range(B):
            for h_idx in range(H):
                # Vectors and matrix for this head
                q_vec = q_f32[b_idx, 0, h_idx, :].contiguous()   # [K]
                k_vec = k_f32[b_idx, 0, h_idx, :].contiguous()   # [K]
                v_vec = v_f32[b_idx, 0, h_idx, :].contiguous()   # [V]

                # state_old[b,h,:,:] as [V,K]
                state_old = state_f32[b_idx, h_idx].contiguous()  # [V, K]
                state_old_flat = state_old.view(-1).contiguous()  # [V*K]

                # old_v = k @ state_old (vector [K])
                old_v = torch.empty((K,), dtype=torch.float32, device=device)
                _vec_matmul_tile_vec[(1,)](k_vec, state_old_flat, old_v, K, V)

                # g and beta for this (b,h)
                g_val = g_out[b_idx, h_idx]
                beta_val = beta_out[b_idx, h_idx]

                # new_v = beta * v + (1 - beta) * old_v
                new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [V]

                # state_remove = k @ old_v (scalar)
                state_remove = torch.empty((1,), dtype=torch.float32, device=device)
                _vec_matmul_scalar[(1,)](k_vec, old_v, state_remove, K)

                # state_update = k @ new_v (scalar)
                state_update = torch.empty((1,), dtype=torch.float32, device=device)
                _vec_matmul_scalar[(1,)](k_vec, new_v, state_update, K)

                # Update new_state[b,h,:,:] = g * state_old - state_remove + state_update
                g_scaled = g_val * state_old
                new_state_block = g_scaled - state_remove[0] + state_update[0]  # [V, K]
                new_state_f32[b_idx, h_idx] = new_state_block

                # Compute output scalar = scale * (q @ new_state[b,h,:,:])
                new_state_block_flat = new_state_block.view(-1).contiguous()    # [V*K]
                out_val = torch.empty((1,), dtype=torch.float32, device=device)
                _output_scalar_kernel[(1,)](q_vec, new_state_block_flat, out_val, scale_buf, B, H, V, K)
                output_f[b_idx, h_idx] = out_val[0]

        # Return output (bfloat16) and new_state (float32)
        return output_f.unsqueeze(1).to(torch.bfloat16), new_state_f32


def run(*args):
    return ModelNew()(*args)
