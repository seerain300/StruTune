import torch
import triton
import triton.language as tl


@triton.jit
def compute_g_beta_kernel(
    A_log_ptr,        # [H] float32
    a_ptr,            # [B,H] float32
    dt_bias_ptr,      # [H] float32
    b_ptr,            # [B,H] float32
    g_out_ptr,        # [B,H] float32
    beta_out_ptr,     # [B,H] float32
    H: tl.constexpr,  # number of heads
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b_idx * H + h_idx)          # a[b, h]
    dt_val = tl.load(dt_bias_ptr + h_idx)               # dt_bias[h]
    b_val = tl.load(b_ptr + b_idx * H + h_idx)          # b[b, h]

    # Compute softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    x = a_val + dt_val
    abs_x = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)
    # A_log is [H], so load scalar
    A_log_val = tl.load(A_log_ptr + h_idx)
    g = tl.exp(-tl.exp(A_log_val) * softplus)           # g = exp(-exp(A_log[h]) * softplus)
    beta = 1.0 / (1.0 + tl.exp(-b_val))                 # sigmoid(b)

    # Store results
    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta)


@triton.jit
def state_update_kernel(
    state_ptr,        # [B,H,V,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    k_ptr,            # [B,H,K] float32
    v_ptr,            # [B,H,V] float32
    beta_ptr,         # [B,H] float32
    # strides for state input
    stride_b, stride_h, stride_v, stride_k,
    # strides for new_state output
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,
    B, H, V, K,
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load k[h] as [K]
    k_vec = tl.load(k_ptr + b_idx * H + tl.arange(0, K))  # [K]
    beta_val = tl.load(beta_ptr + b_idx * H + h_idx)      # scalar

    # Iterate over i in V and j in K using static ranges
    for i in tl.static_range(0, V):
        # Load v[h,i] as scalar
        v_i = tl.load(v_ptr + b_idx * H + i)
        # Compute old_v = k @ state[b,h,i,:]
        old_v = tl.sum(k_vec * tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + tl.arange(0, K) * stride_k), axis=0)  # [K] dot
        new_v = beta_val * v_i + (1.0 - beta_val) * old_v  # scalar

        # Update new_state[b,h,i, :]
        for j in tl.static_range(0, K):
            old_ij = tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            # state_remove is old_v broadcasted along K, state_update is new_v broadcasted along K
            new_ij = old_ij - old_v + new_v
            tl.store(new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i + j * stride_new_j, new_ij)


@triton.jit
def output_dot_kernel(
    new_state_ptr,    # [B,H,V,K] float32
    q_exp_ptr,        # [B,H,K] float32 (we pass as [B*H, K] via reshape)
    out_ptr,          # [B,H] float32
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,
    stride_q_b, stride_q_h, stride_q_k,
    B, H, V, K,
    scale,            # float32 scalar
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load q_exp[h] as [K]
    q_vec = tl.load(q_exp_ptr + b_idx * H + tl.arange(0, K))

    # Compute out[b,h] = scale * sum_{i in V, j in K} q_vec[j] * new_state[b,h,i,j]
    total = 0.0
    for i in tl.static_range(0, V):
        for j in tl.static_range(0, K):
            val = tl.load(new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i + j * stride_new_j)
            total += q_vec[j] * val

    out_val = scale * total
    tl.store(out_ptr + b_idx * H + h_idx, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure float32 compute and contiguity
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()   # [B,1,4,K]
        k_f32 = k.to(torch.float32).contiguous()   # [B,1,4,K]
        v_f32 = v.to(torch.float32).contiguous()   # [B,1,8,V]
        state_f32 = state.to(torch.float32).contiguous()  # [B,H,V,K]

        # Repeat q and k heads by repeat_interleave along head dim (ratio = num_v_heads // num_q_heads = 2)
        q_exp = q_f32.repeat_interleave(2, dim=1)  # [B,8,K]
        k_exp = k_f32.repeat_interleave(2, dim=1)  # [B,8,K]

        # Dimensions
        B = q_f32.shape[0]  # batch
        H = q_f32.shape[1]  # num_q_heads (expanded to 8)
        V = v_f32.shape[2]  # num_v_heads
        K = state_f32.shape[3]  # K

        # Allocate outputs for g and beta
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        grid = (B * H,)
        compute_g_beta_kernel[grid](
            A_log.to(torch.float32), a.squeeze(1).to(torch.float32), dt_bias.to(torch.float32), b.squeeze(1).to(torch.float32),
            g_out, beta_out,
            H=H,
        )

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Get strides for tensors (in elements)
        stride_b = state_f32.stride(0)
        stride_h = state_f32.stride(1)
        stride_v = state_f32.stride(2)
        stride_k = state_f32.stride(3)

        stride_new_b = new_state.stride(0)
        stride_new_h = new_state.stride(1)
        stride_new_i = new_state.stride(2)
        stride_new_j = new_state.stride(3)

        # Launch Triton state update kernel over (B*H)
        state_update_kernel[grid](
            state_f32, new_state, k_exp, v_f32, beta_out,
            stride_b, stride_h, stride_v, stride_k,
            stride_new_b, stride_new_h, stride_new_i, stride_new_j,
            B=B, H=H, V=V, K=K,
        )

        # Compute output: out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        out = torch.empty((B, H), dtype=torch.float32, device=device)
        # Prepare q_exp for Triton: [B*H, K]
        q_exp_flat = q_exp.reshape(B * H, K).contiguous()

        # Launch Triton output dot kernel
        output_dot_kernel[grid](
            new_state, q_exp_flat, out,
            stride_new_b, stride_new_h, stride_new_i, stride_new_j,
            q_exp_flat.stride(0), q_exp_flat.stride(1), q_exp_flat.stride(2),  # q_exp_flat is 2D: [B*H, K]
            B=B, H=H, V=V, K=K,
            scale=scale,
        )

        # Return outputs: output as [B,1,H,1] bfloat16, new_state as [B,H,V,K] float32
        output = out.unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # shape [B,1,H,1]
        return (output, new_state)


def run(*args):
    return ModelNew()(*args)
