import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(a_ptr, dt_bias_ptr, A_log_ptr, b_ptr, g_ptr, beta_ptr,
                           B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b * H + h)         # a[b, h]
    dt_val = tl.load(dt_bias_ptr + h)          # dt_bias[h]
    A_val = tl.load(A_log_ptr + h)             # A_log[h]

    x = a_val + dt_val                         # a + dt
    soft = tl.log(1.0 + tl.exp(x))            # softplus(x) = log(1 + exp(x))
    g = tl.exp(-tl.exp(A_val) * soft)         # g = exp(-exp(A) * softplus(a + dt))
    beta = 1.0 / (1.0 + tl.exp(-b_val))       # beta = sigmoid(b)

    # Store to output
    idx = b * H + h
    tl.store(g_ptr + idx, g)
    tl.store(beta_ptr + idx, beta)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, M_flat_ptr, out_ptr,
                         K: tl.constexpr, V: tl.constexpr):
    # out_vec[k] = sum_v k[v] * M[v, k]
    pid = tl.program_id(0)  # we'll use a 1D grid over K
    # For robustness, loop over v and compute out for this pid (k-index)
    # We'll set grid size to K and let each program compute one k
    k_idx = pid
    # Accumulator
    acc = 0.0
    # Sum over V using static_range
    for v in tl.static_range(0, V):
        m_ptr = M_flat_ptr + v * K + k_idx
        m_val = tl.load(m_ptr)
        k_ptr_v = k_ptr + v
        k_val = tl.load(k_ptr_v)
        acc += k_val * m_val
    tl.store(out_ptr + k_idx, acc)


@triton.jit
def _vec_matmul_scalar(k_ptr, vec_ptr, out_ptr,
                       K: tl.constexpr):
    # scalar = sum_k k[k] * vec[k]
    acc = 0.0
    for k in tl.static_range(0, K):
        k_val = tl.load(k_ptr + k)
        v_val = tl.load(vec_ptr + k)
        acc += k_val * v_val
    tl.store(out_ptr, acc)


@triton.jit
def _q_dot_kernel(q_ptr, M_flat_ptr, out_ptr,
                  K: tl.constexpr, V: tl.constexpr):
    # Compute scalar = sum_k q[k] * sum_v M[v, k]
    acc_q = 0.0
    for k in tl.static_range(0, K):
        q_k = tl.load(q_ptr + k)
        sum_v = 0.0
        for v in tl.static_range(0, V):
            m_ptr = M_flat_ptr + v * K + k
            m_val = tl.load(m_ptr)
            sum_v += m_val
        acc_q += q_k * sum_v
    tl.store(out_ptr, acc_q)


class ModelNew(torch.nn.Module):
    def forward(q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float or None
        Returns: output [B, 8] in bfloat16, new_state [B, 8, 128, 128] in float32
        """
        # Ensure device consistency
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All tensors must be CUDA"
        device = q.device

        # Cast to float32 and make contiguous
        q = q.contiguous().to(torch.float32)  # [B,1,4,128]
        k = k.contiguous().to(torch.float32)  # [B,1,4,128]
        v = v.contiguous().to(torch.float32)  # [B,1,8,128]
        state = state.contiguous().to(torch.float32)  # [B,8,128,128]

        B, T_q, Hq, K = q.shape
        _, T_k, Hk, _ = k.shape
        _, T_v, Hv, V = v.shape
        assert T_q == 1 and T_k == 1
        assert Hq == 4 and Hk == 4 and Hv == 8 and V == 128 and K == 128
        H = Hv  # number of heads is 8

        # Allocate buffers for g and beta
        g_out = torch.empty(B * H, device=device, dtype=torch.float32)
        beta_out = torch.empty(B * H, device=device, dtype=torch.float32)

        # Launch compute g and beta kernel
        _compute_g_beta_kernel[(B * H,), 128, 128, B, H](a.contiguous().to(torch.float32),
                                                        dt_bias.contiguous().to(torch.float32),
                                                        A_log.contiguous().to(torch.float32),
                                                        b.contiguous().to(torch.float32),
                                                        g_out,
                                                        beta_out)

        # Output buffer [B,H] in float32
        output_f = torch.empty(B * H, device=device, dtype=torch.float32)

        # New state buffer [B,H,V,K] in float32
        new_state_f = torch.empty((B, H, V, K), device=device, dtype=torch.float32)

        # Compute per (b,h)
        for b_idx in range(B):
            for h_idx in range(H):
                # Vectors
                q_vec = q[b_idx, 0, :, :].contiguous().to(torch.float32)  # [K]
                k_vec = k[b_idx, 0, :, :].contiguous().to(torch.float32)  # [K]
                v_vec = v[b_idx, 0, :, :].contiguous().to(torch.float32)  # [V]

                # Load g and beta for this (b,h)
                g_val = g_out[b_idx * H + h_idx]
                beta_val = beta_out[b_idx * H + h_idx]

                # state_old = g * state[b,h,:,:]
                state_old = state[b_idx, h_idx, :, :].contiguous() * g_val  # [V,K]

                # old_v = k @ state_old (sum over V), compute via Triton tile vector
                old_v = torch.empty(K, device=device, dtype=torch.float32)
                _vec_matmul_tile_vec[(K,), 128, 128, B, H](k_vec, state_old.reshape(V * K), old_v)

                # new_v = beta * v + (1 - beta) * old_v
                new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [V]

                # state_remove = k @ old_v
                state_rm = torch.empty(1, device=device, dtype=torch.float32)
                _vec_matmul_scalar[(1,), 128, 128, B, H](k_vec, old_v, state_rm)
                state_rm = state_rm[0]

                # state_update = k @ new_v
                state_up = torch.empty(1, device=device, dtype=torch.float32)
                _vec_matmul_scalar[(1,), 128, 128, B, H](k_vec, new_v, state_up)
                state_up = state_up[0]

                # Update new state
                # new_state[b,h,:,:] = g * state[b,h,:,:] - state_rm + state_up
                new_state_tile = (state[b_idx, h_idx, :, :] * g_val) - state_rm + state_up
                new_state_f[b_idx, h_idx, :, :] = new_state_tile

                # Output: output[b,h] = scale * (q @ new_state[b,h,:,:])
                # Flatten new_state to [V*K]
                new_state_flat = new_state_tile.reshape(V * K)
                out_scalar = torch.empty(1, device=device, dtype=torch.float32)
                _q_dot_kernel[(1,), 128, 128, B, H](q_vec, new_state_flat, out_scalar)
                out_scalar = out_scalar[0]

                # scale handling: default scale = 1/sqrt(K) if not provided
                if scale is None:
                    scale = 1.0 / math.sqrt(K)
                # Write output
                output_f[b_idx * H + h_idx] = out_scalar * scale

        # Return output [B,8] in bfloat16 and new_state [B,8,128,128] in float32
        output = output_f.view(B, H).to(torch.bfloat16)
        return output, new_state_f


def run(*args):
    return ModelNew()(*args)
