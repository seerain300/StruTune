import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    B, H,
    stride_A, stride_a, stride_dt, stride_b,
    stride_g, stride_beta,
):
    # Each program handles one (b, h) pair
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load a[b,h] and dt_bias[h]
    a_val = tl.load(a_ptr + b_idx * stride_a + h_idx * stride_a)
    dt_val = tl.load(dt_bias_ptr + h_idx * stride_dt)
    # Load A_log[h]
    A_val = tl.load(A_log_ptr + h_idx * stride_A)
    # x = a + dt_bias
    x = a_val + dt_val
    # softplus(x) = log(1 + exp(x))
    soft = tl.log(1.0 + tl.exp(x))
    # g = exp(-exp(A_log) * softplus(x))
    g_val = tl.exp(-tl.exp(A_val) * soft)
    # beta = sigmoid(b)
    b_val = tl.load(b_ptr + b_idx * stride_b + h_idx * stride_b)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    # Store results
    tl.store(g_ptr + b_idx * stride_g + h_idx * stride_g, g_val)
    tl.store(beta_ptr + b_idx * stride_beta + h_idx * stride_beta, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    B, H, V, K,
    stride_k_b_h, stride_state_bh_v, stride_state_bh_k,
    stride_tmp_bh,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load k[b,h] as vector [K]
    k_off = h_idx * stride_k_b_h
    k_vec = tl.load(k_ptr + b_idx * stride_k_b_h + k_off, mask=tl.arange(0, K) < K, other=0.0)
    # Compute tmp_old_v[b,h] = sum_k k[k] * state[b,h,k]
    tmp_old = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        for kk in range(0, 128):
            k_idx = k_start + kk
            k_valid = k_idx < K
            if k_valid:
                # Load state[b,h,k_idx,:] as vector [V]
                for v in range(0, 128):
                    val = tl.load(state_ptr + b_idx * stride_state_bh_v + h_idx * stride_state_bh_k + k_idx * stride_state_bh_k + v * stride_state_bh_v)
                    tmp_old += k_vec[k_idx] * val
    # Store tmp_old_v[b,h]
    tl.store(tmp_ptr + b_idx * stride_tmp_bh + h_idx * stride_tmp_bh, tmp_old)


@triton.jit
def kernel_update_state_and_output(
    k_ptr, tmp_old_ptr, v_ptr, beta_ptr, state_in_ptr, state_out_ptr, q_ptr, output_ptr,
    B, H, V, K,
    stride_k_b_h, stride_tmp_bh,
    stride_v_b_h, stride_v_v,
    stride_beta_b_h,
    stride_state_in_bh_v, stride_state_in_bh_k,
    stride_state_out_bh_v, stride_state_out_bh_k,
    stride_q_b_h, stride_q_k,
    stride_output_b_h,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load k[b,h] as vector [K]
    k_off = h_idx * stride_k_b_h
    k_vec = tl.load(k_ptr + b_idx * stride_k_b_h + k_off, mask=tl.arange(0, K) < K, other=0.0)
    # Load tmp_old_v[b,h] scalar
    tmp_old = tl.load(tmp_old_ptr + b_idx * stride_tmp_bh + h_idx * stride_tmp_bh)
    # Load beta[b,h] scalar
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b_h + h_idx * stride_beta_b_h)
    # Load v[b,h] as [V]
    v_vals = tl.load(v_ptr + b_idx * stride_v_b_h + h_idx * stride_v_v, mask=tl.arange(0, V) < V, other=0.0)
    # new_v = beta * v + (1 - beta) * tmp_old
    new_v = beta_val * v_vals + (1.0 - beta_val) * tmp_old
    # Compute state_remove = dot(k, tmp_old) scalar
    state_remove = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        for kk in range(0, 128):
            k_idx = k_start + kk
            k_valid = k_idx < K
            if k_valid:
                state_remove += k_vec[k_idx] * tmp_old
    # Compute state_update = dot(k, new_v) vector [V]
    state_update = tl.zeros([V], dtype=tl.float32)
    for k_start in range(0, K, 128):
        k_sub = k_start + tl.arange(0, 128)
        k_mask = k_sub < K
        for kk in range(0, 128):
            k_idx = k_start + kk
            k_valid = k_idx < K
            if k_valid:
                state_update += k_vec[k_idx] * new_v
    # Load q[b,h] as [K]
    q_off = h_idx * stride_q_b_h
    q_vec = tl.load(q_ptr + b_idx * stride_q_b_h + q_off, mask=tl.arange(0, K) < K, other=0.0)
    # Initialize output_scalar
    output_scalar = tl.zeros([1], dtype=tl.float32)
    # Compute output_scalar = scale * (q @ new_state), where new_state[b,h] is [V,K]
    # We need to compute new_state for the whole [V,K] before computing q @ new_state.
    # We can allocate a temporary [V,K] in state_out_ptr for this program, but Triton doesn't support 2D stores with variable strides well here; instead we compute new_state and output by iterating over V and K.
    # Since Triton kernel doesn't have a way to return multiple outputs cleanly, we compute new_state on the fly and accumulate q @ new_state here.
    for v_start in range(0, V, 128):
        v_idx = v_start + tl.arange(0, 128)
        v_mask = v_idx < V
        for k_start in range(0, K, 128):
            k_sub = k_start + tl.arange(0, 128)
            k_mask = k_sub < K
            # Compute new_state for this tile: old_state - state_remove + state_update
            old_state_ptrs = state_in_ptr + b_idx * stride_state_in_bh_v + h_idx * stride_state_in_bh_k + k_sub[None, :] * stride_state_in_bh_k + v_idx[:, None] * stride_state_in_bh_v
            old_state_vals = tl.load(old_state_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0)
            new_state_vals = old_state_vals - state_remove + state_update[None, :]
            # Store new_state temporarily in state_out
            new_state_ptrs = state_out_ptr + b_idx * stride_state_out_bh_v + h_idx * stride_state_out_bh_k + k_sub[None, :] * stride_state_out_bh_k + v_idx[:, None] * stride_state_out_bh_v
            tl.store(new_state_ptrs, new_state_vals, mask=v_mask[:, None] & k_mask[None, :])
            # Accumulate q @ new_state over this tile
            # Multiply q_vec[:, None] and new_state_vals and sum over K and V
            # q_vec has shape [K], new_state_vals has shape [128, 128]
            # For each k in tile, sum across V
            for kk in range(0, 128):
                k_idx = k_start + kk
                k_valid = k_idx < K
                if k_valid:
                    # Sum over V of q[kk] * new_state_vals[kk, :]
                    sum_v = tl.zeros([1], dtype=tl.float32)
                    for vv in range(0, 128):
                        sum_v += new_state_vals[kk, vv] * q_vec[kk]
                    output_scalar += sum_v
    # Compute scale * output_scalar
    # original code uses scale or 1/sqrt(K) if None
    scale_val = 1.0 / math.sqrt(K)
    output_scalar = output_scalar * scale_val
    # Store output[b, h, V] in output_ptr[b*stride_output_b_h + h*stride_output_b_h + V*stride_output_b_h]
    out_ptr = output_ptr + b_idx * stride_output_b_h + h_idx * stride_output_b_h + V * stride_output_b_h
    tl.store(out_ptr, output_scalar)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward. Computes:
          - output: [B, H, V] in bfloat16
          - new_state: [B, H, V, K] in float32
        """
        device = q.device
        # Shapes
        B = q.shape[0]  # batch
        H = v.shape[1]  # num heads
        V = v.shape[3]  # embedding dim V
        K = q.shape[3]  # embedding dim K

        # Prepare inputs: convert to FP32 for kernels
        A_log = A_log.to(device=device, dtype=torch.float32)
        a = a.to(device=device, dtype=torch.float32)
        dt_bias = dt_bias.to(device=device, dtype=torch.float32)
        b = b.to(device=device, dtype=torch.float32)
        q_s = q.squeeze(1).to(device=device, dtype=torch.float32)  # [B, QH, K] -> [B, QH, K]
        k_s = k.squeeze(1).to(device=device, dtype=torch.float32)  # [B, KH, K]
        v_s = v.squeeze(1).to(device=device, dtype=torch.float32)  # [B, VH, V]

        # Input state [B, H, V, K]
        if state is None:
            state_in = torch.zeros(B, H, V, K, dtype=torch.float32, device=device)
        else:
            state_in = state.to(device=device, dtype=torch.float32)

        # Allocate intermediates
        g = torch.empty(B, H, dtype=torch.float32, device=device)
        beta = torch.empty(B, H, dtype=torch.float32, device=device)
        tmp_old_v = torch.empty(B, H, dtype=torch.float32, device=device)
        new_state_out = torch.empty(B, H, V, K, dtype=torch.float32, device=device)
        output = torch.empty(B, H, V, dtype=torch.float32, device=device)

        # Launch Triton kernels
        grid = (B, H)
        # 1) compute g and beta
        kernel_g_beta[grid](
            A_log, a, dt_bias, b,
            g, beta,
            B, H,
            A_log.stride(0), a.stride(0), dt_bias.stride(0), b.stride(0),
            g.stride(0), beta.stride(0),
        )
        # 2) compute tmp_old_v[b,h] = dot(k[b,h], state[b,h])
        kernel_tmp_old_v[grid](
            k_s, state_in, tmp_old_v,
            B, H, V, K,
            k_s.stride(0), state_in.stride(1), state_in.stride(3),
            tmp_old_v.stride(0),
        )
        # 3) update state and compute output[b,h] = scale * (q[b,h] @ new_state[b,h])
        #    We need beta, k, tmp_old_v, v, state_in, new_state_out, q, output
        kernel_update_state_and_output[grid](
            k_s, tmp_old_v, v_s, beta, state_in, new_state_out, q_s, output,
            B, H, V, K,
            k_s.stride(0), tmp_old_v.stride(0),
            v_s.stride(0), v_s.stride(2),
            beta.stride(0),
            state_in.stride(1), state_in.stride(3),
            new_state_out.stride(1), new_state_out.stride(3),
            q_s.stride(0), q_s.stride(2),
            output.stride(0),
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, new_state_out


def run(*args):
    return ModelNew()(*args)
