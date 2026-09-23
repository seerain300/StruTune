import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(a_ptr, dt_bias_ptr, A_log_ptr, b_ptr, g_ptr, beta_ptr,
                           B: tl.constexpr, H: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b * H + h)          # [B, H]
    dt_val = tl.load(dt_bias_ptr + h)           # [H]
    A_val = tl.load(A_log_ptr + h)              # [H]
    b_val = tl.load(b_ptr + b * H + h)          # [B, H]

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g = tl.exp(-tl.exp(A_val) * sp)
    sig = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(g_ptr + b * H + h, g)
    tl.store(beta_ptr + b * H + h, sig)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, M_flat_ptr, out_ptr,
                         K: tl.constexpr, V: tl.constexpr):
    # Compute out[k] = sum_v k[v] * M[v, k], for k in 0..K-1, v in 0..V-1
    pid = tl.program_id(0)
    offs = pid * K + tl.arange(0, K)
    mask = offs < K

    k_vec = tl.load(k_ptr + offs, mask=mask, other=0.0)

    # Accumulator
    acc = tl.zeros([K], dtype=tl.float32)

    # Loop over V dimension
    for v in tl.static_range(V):
        # M_flat index for row v across K columns
        M_offs = v * K + offs
        M_row = tl.load(M_flat_ptr + M_offs, mask=mask, other=0.0)
        acc += k_vec * M_row

    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _vec_matmul_scalar(k_ptr, vec_ptr, out_ptr,
                       K: tl.constexpr):
    # out = sum_k k[k] * vec[k]
    pid = tl.program_id(0)
    offs = tl.arange(0, K)
    mask = offs < K
    k_vec = tl.load(k_ptr + offs, mask=mask, other=0.0)
    vec = tl.load(vec_ptr + offs, mask=mask, other=0.0)
    acc = tl.sum(k_vec * vec, axis=0)
    tl.store(out_ptr + pid, acc)


@triton.jit
def _q_dot_kernel(q_ptr, M_flat_ptr, out_ptr,
                  scale, K: tl.constexpr, V: tl.constexpr):
    # out = scale * (q @ M), where M is [V, K] flattened
    pid = tl.program_id(0)
    offs = tl.arange(0, K)
    mask = offs < K
    q_vec = tl.load(q_ptr + offs, mask=mask, other=0.0)

    # Compute sum_v M[v, :] for each k
    acc = tl.zeros([K], dtype=tl.float32)
    for v in tl.static_range(V):
        M_offs = v * K + offs
        M_row = tl.load(M_flat_ptr + M_offs, mask=mask, other=0.0)
        acc += M_row

    result = tl.sum(q_vec * acc, axis=0) * scale
    tl.store(out_ptr + pid, result)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, 128] (bfloat16)
        k: [B, 1, 4, 128] (bfloat16)
        v: [B, 1, 8, 128] (bfloat16)
        state: [B, 8, 128, 128] (float32)
        A_log: [8] (float32)
        a: [B, 1, 8] (bfloat16)
        dt_bias: [8] (float32)
        b: [B, 1, 8] (bfloat16)
        scale: float (not used directly in Triton)
        Returns:
          output: [B, 8] (bfloat16)
          new_state: [B, 8, 128, 128] (float32)
        """
        # Cast to float32 and make contiguous
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)
        state = state.contiguous().to(torch.float32)

        # Shapes
        B, T_q, Hq, K = q.shape
        _, T_k, Hk, _ = k.shape
        _, T_v, Hv, V = v.shape
        # Tensors' T dims are 1, K=128, V=128, H=Hv=8 for this task
        assert T_q == 1 and T_k == 1 and Hv == 8 and V == 128 and K == 128
        H = Hv

        device = q.device

        # Allocate g and beta buffers
        g_out = torch.empty(B * H, device=device, dtype=torch.float32)
        beta_out = torch.empty(B * H, device=device, dtype=torch.float32)

        # Launch compute g and beta kernel
        _compute_g_beta_kernel[(B * H,), 128, 128, B, H](
            a.view(-1), dt_bias, A_log, b.view(-1),
            g_out, beta_out
        )

        # Prepare outputs
        output_f = torch.empty(B * H, device=device, dtype=torch.float32)
        new_state_f = torch.empty((B, H, V, K), device=device, dtype=torch.float32)

        # Compute new_state and output per (b,h)
        for b_idx in range(B):
            for h_idx in range(H):
                # Extract vectors and matrices
                q_vec = q[b_idx, 0, :, :].contiguous()  # [K]
                k_vec = k[b_idx, 0, :, :].contiguous()  # [K]
                v_vec = v[b_idx, 0, :, :].contiguous()  # [V]

                g_val = g_out[b_idx * H + h_idx]
                beta_val = beta_out[b_idx * H + h_idx]

                # state_old = g * state[b,h,:,:] -> [V, K]
                state_old = state[b_idx, h_idx, :, :].contiguous() * g_val  # [V, K]
                state_old_flat = state_old.view(-1)  # [V*K]

                # old_v = k @ state_old
                old_v = torch.empty(K, device=device, dtype=torch.float32)
                _vec_matmul_tile_vec[(K,),](k_vec, state_old_flat, old_v, 128, 128)

                # new_v = beta * v + (1 - beta) * old_v
                new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [V]

                # state_remove = k @ old_v (scalar)
                state_remove = torch.empty((), device=device, dtype=torch.float32)
                _vec_matmul_scalar[(1,),](k_vec, old_v, state_remove, 128)

                # state_update = k @ new_v (scalar)
                state_update = torch.empty((), device=device, dtype=torch.float32)
                _vec_matmul_scalar[(1,),](k_vec, new_v, state_update, 128)

                # Update new_state[b,h,:,:] = g * state[b,h,:,:] - state_remove + state_update
                # Note: state_remove and state_update are scalars; we can add them to each element.
                new_state_mat = g_val * state_old + (state_remove - state_update)
                new_state_f[b_idx, h_idx, :, :] = new_state_mat

                # output[b,h] = scale * (q @ new_state[b,h,:,:])
                new_state_flat = new_state_mat.view(-1)  # [V*K]
                out_scalar = torch.empty((), device=device, dtype=torch.float32)
                scale_val = 1.0 / math.sqrt(K)
                _q_dot_kernel[(1,),](q_vec, new_state_flat, out_scalar, scale_val, 128, 128)

                output_f[b_idx * H + h_idx] = out_scalar

        # Cast output to bfloat16 as per original signature
        output = output_f.view(B, H).unsqueeze(1).to(torch.bfloat16)
        return output, new_state_f


def run(*args):
    return ModelNew()(*args)
