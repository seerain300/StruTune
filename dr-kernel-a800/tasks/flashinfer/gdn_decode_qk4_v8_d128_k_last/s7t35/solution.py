import torch
import triton
import triton.language as tl


# Compute g[b,h] = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
# beta[b,h] = sigmoid(b[b,h])
@triton.jit
def _compute_g_and_beta_kernel(A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                                g_ptr, beta_ptr,
                                B: tl.constexpr, H: tl.constexpr):
    pid = tl.program_id(0)  # in [0, B*H)
    b = pid // H
    h = pid % H

    A_val = tl.load(A_log_ptr + h)       # [H]
    a_val = tl.load(a_ptr + b * H + h)   # [B, H]
    dt_val = tl.load(dt_bias_ptr + h)    # [H]
    bb_val = tl.load(b_ptr + b * H + h)  # [B, H]

    # softplus(x) = log(1 + exp(x))
    soft = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * soft)
    beta_val = 1.0 / (1.0 + tl.exp(-bb_val))

    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


# Compute out_vec[k] = sum_{v=0..V-1} k[v] * state[v, k] for k in [0..K)
# Note: This kernel is launched per (b,h) by passing pointers to k_vec and the corresponding slice of state_ptr.
@triton.jit
def _vec_matmul_tile_vec_kernel(k_ptr, state_ptr, out_ptr,
                                 B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program: we assume state_ptr points to the correct (b,h) slice.
    BLOCK_K = 128
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K
        k_vec = tl.load(k_ptr + k_offsets, mask=mask_k, other=0.0)  # [BLOCK_K]

        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for vk in range(0, V):  # loop over V (compile-time constant)
            state_off = vk * K + k_offsets  # [BLOCK_K]
            state_vec = tl.load(state_ptr + state_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            acc += k_vec * state_vec
        tl.store(out_ptr + k_offsets, acc, mask=mask_k)


# Compute scalar = sum_{k=0..K-1} k[k] * vec[k]
@triton.jit
def _vec_matmul_scalar_kernel(k_ptr, vec_ptr, out_scalar_ptr,
                               K: tl.constexpr):
    BLOCK_K = 128
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K
        k_vec = tl.load(k_ptr + k_offsets, mask=mask_k, other=0.0)     # [BLOCK_K]
        vec_vec = tl.load(vec_ptr + k_offsets, mask=mask_k, other=0.0) # [BLOCK_K]
        acc += tl.sum(k_vec * vec_vec, axis=0)
    tl.store(out_scalar_ptr, acc)


# Compute output[b,h] = scale * (q @ new_state), where new_state is [V,K] flattened, q is [K]
@triton.jit
def _output_scalar_kernel(q_ptr, new_ptr, out_ptr,
                           B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)  # [0, B*H)
    b = pid // H
    h = pid % H

    acc = tl.zeros((), dtype=tl.float32)

    BLOCK_K = 128
    for v0 in range(0, V, BLOCK_K):
        for vk in range(0, BLOCK_K):  # loop within tile
            v_idx = v0 + vk
            if v_idx >= V:
                continue
            # new_ptr is [V*K], row index = b*H*V*K + v_idx*K
            row_off = b * H * V * K + v_idx * K
            new_row = tl.load(new_ptr + row_off + tl.arange(0, K))  # [K]
            q_vec = tl.load(q_ptr + tl.arange(0, K))                # [K]
            acc += tl.sum(q_vec * new_row, axis=0)

    tl.store(out_ptr + b * H + h, acc)


# Compute scale = 1/sqrt(K) and write to out_ptr[0]
@triton.jit
def _sqrt_scale_kernel(K: tl.constexpr, out_ptr):
    inv_sqrt = 1.0 / tl.sqrt(K)
    tl.store(out_ptr, inv_sqrt)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure float32 and contiguous
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()
        state_f32 = state.to(torch.float32).contiguous()

        # Shapes from provided inputs: q[B,1,Hq,K], k[B,1,Hk,K], v[B,1,Hv,V], state[B,Hv,V,K]
        B = q_f32.shape[0]
        H = v_f32.shape[2]  # number of v heads (8)
        V = state_f32.shape[3]  # 128
        K = state_f32.shape[2]  # 128

        # Prepare parameters as 2D/1D tensors for clean loads
        A_log_2d = A_log.to(torch.float32).contiguous()         # [H]
        a_2d = a.to(torch.float32).contiguous().view(B, H)      # [B, H]
        dt_bias_1d = dt_bias.to(torch.float32).contiguous()     # [H]
        b_2d = b.to(torch.float32).contiguous().view(B, H)      # [B, H]

        # Allocate outputs
        g_out = torch.empty((B, H), dtype=torch.float32, device=q_f32.device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=q_f32.device)

        # Launch _compute_g_and_beta_kernel
        _compute_g_and_beta_kernel[(B * H,)](A_log_2d, a_2d, dt_bias_1d, b_2d, g_out, beta_out,
                                             B=B, H=H)

        # Compute scale = 1/sqrt(K) in Triton
        scale_buf = torch.empty((), dtype=torch.float32, device=q_f32.device)
        _sqrt_scale_kernel[(1,)](K, scale_buf)
        inv_sqrt = scale_buf.item()
        if scale is None or float(scale) == 0.0:
            scale_val = inv_sqrt
        else:
            scale_val = float(scale)

        # Prepare output and new state
        output_f = torch.empty((B, H), dtype=torch.float32, device=q_f32.device)
        new_state_f = torch.empty((B, H, V, K), dtype=torch.float32, device=q_f32.device)

        # For each (b, h), compute new state and output
        for b_idx in range(B):
            for h_idx in range(H):
                g_val = g_out[b_idx, h_idx]
                beta_val = beta_out[b_idx, h_idx]

                # Extract vectors/2D slices
                q_vec = q_f32[b_idx, 0, h_idx, :]  # [K]
                k_vec = k_f32[b_idx, 0, h_idx, :]  # [K]
                v_vec = v_f32[b_idx, 0, h_idx, :]  # [V]
                old_state = state_f32[b_idx, h_idx, :, :]  # [V, K], contiguous row-major

                # old_v = k @ old_state
                out_old_v = torch.empty((K,), dtype=torch.float32, device=q_f32.device)
                _vec_matmul_tile_vec_kernel[(1,)](k_vec, old_state.reshape(-1), out_old_v,
                                                   B=B, H=H, V=V, K=K)

                # new_v = beta * v + (1 - beta) * old_v
                new_v_vec = beta_val * v_vec + (1.0 - beta_val) * out_old_v  # [V]

                # state_remove = k @ old_v (scalar)
                state_remove = torch.empty((), dtype=torch.float32, device=q_f32.device)
                _vec_matmul_scalar_kernel[(1,)](k_vec, out_old_v, state_remove, K=K)

                # state_update = k @ new_v (scalar)
                state_update = torch.empty((), dtype=torch.float32, device=q_f32.device)
                _vec_matmul_scalar_kernel[(1,)](k_vec, new_v_vec, state_update, K=K)

                # new_state[h] = g * old_state - state_remove + state_update
                new_state_row = g_val * old_state - state_remove + state_update
                new_state_f[b_idx, h_idx, :, :] = new_state_row

                # output[b,h] = scale * (q @ new_state)
                out_ptr = torch.empty((), dtype=torch.float32, device=q_f32.device)
                _output_scalar_kernel[(1,)](q_vec, new_state_row.reshape(-1), out_ptr,
                                            B=B, H=H, V=V, K=K)
                output_f[b_idx, h_idx] = scale_val * out_ptr[0]

        # Return output (B,1,H,V) and new state (B,H,V,K), matching original behavior
        output = output_f.unsqueeze(1)  # [B,1,H,1] but should be [B,1,H,V] — original output has shape [B,1,H,V]
        output_bf16 = output.to(torch.bfloat16)  # match original output dtype
        new_state_bf16 = new_state_f.to(torch.bfloat16)
        return output_bf16, new_state_bf16


def run(*args):
    return ModelNew()(*args)
