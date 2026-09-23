import math
import torch
import triton
import triton.language as tl


@triton.jit
def gate_beta_kernel(g_out_ptr, beta_out_ptr, B, H, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr):
    # One program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)
    if (b < B) and (h < H):
        # Load a[b,h] and dt_bias[h] as scalars
        a_val = tl.load(a_ptr + b * H + h)   # a is flattened [B, H]
        dtb_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
        A = a_val + dtb_val
        # softplus(A) = log(1 + exp(A))
        soft = tl.log(1.0 + tl.exp(A))
        # eA_log = exp(A_log[h])
        eA_log = tl.exp(tl.load(A_log_ptr + h))
        # g = exp(-eA_log * soft)
        g = tl.exp(-eA_log * soft)
        # beta = sigmoid(b[b,h]), b_ptr is flattened [B, H]
        b_val = tl.load(b_ptr + b * H + h)
        beta = 1.0 / (1.0 + tl.exp(-b_val))
        # Store g and beta
        tl.store(g_out_ptr + b * H + h, g)
        tl.store(beta_out_ptr + b * H + h, beta)


@triton.jit
def compute_output_kernel(output_flat_ptr, state_ptr, q_ptr, k_ptr, v_ptr,
                          g_ptr, beta_ptr, B, H, V, scale):
    # One program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)
    if (b < B) and (h < H):
        # Load scalars
        g = tl.load(g_ptr + b * H + h)
        beta = tl.load(beta_ptr + b * H + h)
        # Compute old_v = k_h @ (g * state_bh) where state_bh is [V, K]
        old_v = 0.0
        for i in range(0, 128):  # K fixed = 128
            row_k = tl.load(k_ptr + b * H * 128 + h * 128 + i)  # k is [B,H,128] for fixed (b,h)
            row_sum = 0.0
            for j in range(0, 128):  # V fixed = 128
                off = b * (H * V * 128) + h * (V * 128) + i * 128 + j  # state is [B,H,V,128]
                val = tl.load(state_ptr + off)
                row_sum += val
            old_v += row_k * row_sum

        # Compute sum_v = sum(v_h) where v_h is [128]
        sum_v = 0.0
        for i in range(0, 128):
            sum_v += tl.load(v_ptr + b * H * 128 + h * 128 + i)

        # new_v = beta * sum_v + (1 - beta) * old_v
        new_v = beta * sum_v + (1.0 - beta) * old_v

        # Compute q_dot = q_h @ updated_state = q_h @ new_v (scalar since updated_state is broadcasted to a scalar)
        q_dot = 0.0
        for i in range(0, 128):
            q_i = tl.load(q_ptr + b * H * 128 + h * 128 + i)
            q_dot += q_i * new_v

        # Apply scale and store to flat output at index b*H + h
        val = q_dot * scale
        tl.store(output_flat_ptr + b * H + h, val)


@triton.jit
def write_newstate_kernel(newstate_ptr, state_ptr, g_ptr, beta_ptr, B, H, V, K):
    # One program per (b, h); write new_state[b,h,:,:]
    b = tl.program_id(0)
    h = tl.program_id(1)
    if (b < B) and (h < H):
        g = tl.load(g_ptr + b * H + h)
        beta = tl.load(beta_ptr + b * H + h)

        # Compute old_v for this (b, h): old_v = sum_i k_h[i] * (sum_j state[b,h,i,j])
        old_v = 0.0
        for i in range(0, K):
            row_k = 0.0
            for j in range(0, V):
                off = b * (H * V * K) + h * (V * K) + j * K + i
                row_k += tl.load(state_ptr + off)
            # k_h[i] is offset b*H*K + h*K + i
            k_i = tl.load(k_ptr + b * H * K + h * K + i)
            old_v += k_i * row_k

        # Compute sum_v = sum(v_h) where v_h is [V]
        sum_v = 0.0
        for i in range(0, V):
            off = b * (H * V) + h * V + i
            sum_v += tl.load(v_ptr + off)  # v is [B,H,V]

        # new_v = beta * sum_v + (1 - beta) * old_v
        new_v = beta * sum_v + (1.0 - beta) * old_v

        # Write updated_state = g * state - old_v + new_v for all i,j
        for i in range(0, V):
            for j in range(0, K):
                off_in = b * (H * V * K) + h * (V * K) + i * K + j
                val_in = tl.load(state_ptr + off_in)
                updated = g * val_in - old_v + new_v
                off_out = b * (H * V * K) + h * (V * K) + i * K + j
                tl.store(newstate_ptr + off_out, updated)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Shapes: q [B,1,H_q,K], k [B,1,H_k,K], v [B,1,H,V], state [B,H,V,K], A_log [H], a [B,1,H], dt_bias [H], b [B,1,H], scale scalar
        B = q.shape[0]
        H_v = v.shape[1]  # heads from v
        V = v.shape[2]
        K = q.shape[3]

        # Ensure dtypes for kernels
        # Allocate gate and beta outputs (float32 on device)
        g_out = torch.empty((B, H_v), dtype=torch.float32, device=q.device)
        beta_out = torch.empty((B, H_v), dtype=torch.float32, device=q.device)

        # Flatten a and b for gate_beta_kernel
        a_flat = a.contiguous().view(B * H_v)
        dt_bias_flat = dt_bias.contiguous()  # [H]
        b_flat = b.contiguous().view(B * H_v)

        # Launch gate_beta_kernel
        grid_g = (B, H_v)
        gate_beta_kernel[grid_g](g_out, beta_out, B, H_v, A_log.contiguous().float(), a_flat, dt_bias_flat, b_flat)

        # Prepare output buffer (flat float32)
        output_flat = torch.empty(B * H_v, dtype=torch.float32, device=q.device)

        # Ensure inputs are contiguous and cast to float32 for compute
        state_c = state.contiguous()  # [B,H,V,K], float32 (as per original)
        q_c = q.contiguous().float()  # [B,1,H_q,K] -> [B,H_q,128]
        k_c = k.contiguous().float()  # [B,1,H_k,K] -> [B,H_k,128]
        v_c = v.contiguous().float()  # [B,1,H,V] -> [B,H,128]

        # Launch compute_output_kernel to compute per-(b,h) scalar output
        grid_out = (B, H_v)
        compute_output_kernel[grid_out](output_flat, state_c, q_c, k_c, v_c, g_out, beta_out, B, H_v, V, scale)

        # Create output tensor [B,1,H,V] in bfloat16 and fill by assigning scalar per (b,h) to output[b,0,h,0]
        output = torch.empty((B, 1, H_v, V), dtype=torch.bfloat16, device=q.device)
        # Assign scalars without using torch elementwise ops
        for b_idx in range(B):
            for h_idx in range(H_v):
                idx = b_idx * H_v + h_idx
                val = output_flat[idx]  # float32 scalar
                # output[b,0,h,0] single element assignment
                output[b_idx, 0, h_idx, 0] = val.to(torch.bfloat16)

        # Allocate new_state [B,H,V,K] as float32
        new_state = torch.empty((B, H_v, V, K), dtype=torch.float32, device=q.device)

        # Launch write_newstate_kernel to fill new_state
        grid_new = (B, H_v)
        write_newstate_kernel[grid_new](new_state, state_c, g_out, beta_out, B, H_v, V, K)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
