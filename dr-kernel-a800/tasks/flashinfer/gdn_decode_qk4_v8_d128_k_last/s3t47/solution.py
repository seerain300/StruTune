import torch
import triton
import triton.language as tl


@triton.jit
def output_dot_kernel(
    q_exp_ptr,       # [H*K] float32
    v_ptr,           # [B*H*V] float32
    k_ptr,           # [B*H*K] float32
    state_ptr,       # [B*H*V*K] float32
    a_ptr,           # [B*H] float32
    dt_bias_ptr,     # [H] float32
    A_log_ptr,       # [H] float32
    b_ptr,           # [B*H] float32
    new_state_ptr,   # [B*H*V*K] float32
    out_ptr,         # [B*H] float32
    scale,           # float32
    B: tl.constexpr, # int
    H: tl.constexpr, # int
    V: tl.constexpr, # int
    K: tl.constexpr, # int
):
    # Each program handles one (b,h)
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    # Compute g and beta for (b,h)
    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    dt_val = tl.load(dt_bias_ptr + h_idx)
    A_val = tl.load(A_log_ptr + h_idx)
    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    x = a_val + dt_val
    absx = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-absx)) + tl.maximum(x, 0.0)
    g = tl.exp(-tl.exp(A_val) * softplus)
    beta = 1.0 / (1.0 + tl.exp(-tl.load(b_ptr + b_idx * H + h_idx)))

    # Base pointers for (b,h)
    state_b_h_base = state_ptr + b_idx * (H * V * K) + h_idx * (V * K)
    new_state_b_h_base = new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K)

    # Prepare vectorized offsets
    j_offsets = tl.arange(0, K)  # [K]
    h_j_offsets = h_idx * K + j_offsets  # [K]

    # Output accumulator
    out_sum = 0.0

    # Update new_state row-wise over i in V and accumulate output
    for i in tl.static_range(0, V):
        row_state_base = state_b_h_base + i * K
        row_new_base = new_state_b_h_base + i * K

        # Load q_exp[h,:] and k[h,:]
        q_vec = tl.load(q_exp_ptr + h_j_offsets)  # [K]
        k_vec = tl.load(k_ptr + b_idx * (H * K) + h_j_offsets)  # [K]

        # Load v[b,h,i]
        v_val = tl.load(v_ptr + b_idx * (H * V) + h_idx * V + i)

        # Compute old_v = sum_j k[h,j] * state[b,h,i,j]
        old_v = 0.0
        for jj in tl.static_range(0, K):
            k_j = k_vec[jj]  # scalar
            state_j = tl.load(row_state_base + jj)  # scalar
            old_v += k_j * state_j

        # new_v = beta * v[b,h,i] + (1 - beta) * old_v
        new_v = beta * v_val + (1.0 - beta) * old_v

        # Update new_state row: new_state[b,h,i,j] = state[b,h,i,j] - old_v + k[h,j] * new_v
        for jj in tl.static_range(0, K):
            old_state = tl.load(row_new_base + jj)  # scalar
            new_state = old_state - old_v + k_vec[jj] * new_v
            tl.store(row_new_base + jj, new_state)

        # Accumulate output: sum_j q_exp[h,j] * new_state[b,h,j]
        new_state_col_j = tl.load(row_new_base + j_offsets)  # [K]
        out_sum += tl.sum(q_vec * new_state_col_j, axis=0)

    out_val = out_sum * scale
    tl.store(out_ptr + b_idx * H + h_idx, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure dtypes and contiguity
        q_f32 = q.to(torch.float32).contiguous()        # [B,1,num_q_heads,K]
        k_f32 = k.to(torch.float32).contiguous()        # [B,1,num_k_heads,K]
        v_f32 = v.to(torch.float32).contiguous()        # [B,1,num_v_heads,V]
        state_f32 = state.to(torch.float32).contiguous()  # [B,num_heads,V,K]
        A_log_f32 = A_log.to(torch.float32)             # [num_heads]
        a_f32 = a.to(torch.float32).squeeze(1).contiguous()  # [B,H]
        dt_bias_f32 = dt_bias.to(torch.float32)         # [H]
        b_f32 = b.to(torch.float32).squeeze(1).contiguous()  # [B,H]

        # Expand q and k heads by repeat_interleave ratio = num_v_heads // num_q_heads
        num_q_heads = q_f32.shape[1]
        num_v_heads = v_f32.shape[1]
        ratio = num_v_heads // num_q_heads
        q_exp = q_f32.repeat_interleave(ratio, dim=1)   # [B,num_v_heads,K]
        k_exp = k_f32.repeat_interleave(ratio, dim=1)   # [B,num_v_heads,K]

        B = q_f32.shape[0]
        H = q_f32.shape[1]
        V = state_f32.shape[2]
        K = state_f32.shape[3]

        # Allocate new_state and output buffer
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=q.device)
        out_flat = torch.empty((B * H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b,h)
        grid = (B * H,)
        output_dot_kernel[grid](
            q_exp.view(-1),   # [H*K]
            v_f32.view(-1),   # [B*H*V]
            k_exp.view(-1),   # [B*H*K]
            state_f32.view(-1),  # [B*H*V*K]
            a_f32.view(-1),  # [B*H]
            dt_bias_f32,      # [H]
            A_log_f32,        # [H]
            b_f32.view(-1),   # [B*H]
            new_state.view(-1),  # [B*H*V*K]
            out_flat,            # [B*H]
            float(scale),
            B=B, H=H, V=V, K=K,
        )

        # Return output cast to bfloat16 as [B,1,H,1], and new_state as [B,H,V,K] float32
        output = out_flat.view(B, H).unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        return output, new_state


def run(*args):
    return ModelNew()(*args)
