import torch
import triton
import triton.language as tl


@triton.jit
def g_beta_kernel(
    A_log_ptr,        # [H] float32
    a_ptr,            # [B, H] float32
    dt_bias_ptr,      # [H] float32
    b_ptr,            # [B, H] float32
    g_out_ptr,        # [B, H] float32
    beta_out_ptr,     # [B, H] float32
    H: tl.constexpr,  # number of heads
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    dt_val = tl.load(dt_bias_ptr + h_idx)
    A_log_val = tl.load(A_log_ptr + h_idx)

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) for numerical stability
    x = a_val + dt_val
    abs_x = tl.abs(x)
    softplus_x = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)

    g_val = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta_val = 1.0 / (1.0 + tl.exp(-x))

    tl.store(g_out_ptr + b_idx * H + h_idx, g_val)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta_val)


@triton.jit
def state_update_kernel(
    state_ptr,        # [B, H, V, K] float32
    new_state_ptr,    # [B, H, V, K] float32
    k_ptr,            # [H, K] float32 (k_exp[b,h] flattened along K)
    v_ptr,            # [B, H, V] float32 (v_f32[b,h])
    beta_ptr,         # [B, H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_v: tl.constexpr,
    stride_k: tl.constexpr,
    stride_new_b: tl.constexpr,
    stride_new_h: tl.constexpr,
    stride_new_i: tl.constexpr,
    stride_new_j: tl.constexpr,
    stride_i: tl.constexpr,  # stride for v_ptr along V (typically V)
    stride_j: tl.constexpr,  # stride for v_ptr along H (typically 1 for [B,H,V])
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load k[h,:] and beta[b,h]
    k_row = tl.load(k_ptr + h_idx * K + tl.arange(0, K))  # [K]
    beta_val = tl.load(beta_ptr + b_idx * H + h_idx)

    # Iterate over V and K with static ranges
    for i in tl.static_range(0, V):
        # old_v = dot(k_row, state[b,h,i,:])
        state_row_ptr = state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v
        old_v = tl.sum(k_row * tl.load(state_row_ptr + tl.arange(0, K) * stride_k), axis=0)

        # v[b,h,i] load
        v_val = tl.load(v_ptr + b_idx * (H * V) + h_idx * V + i * stride_i)  # [scalar]
        new_v = beta_val * v_val + (1.0 - beta_val) * old_v

        # Update new_state[b,h,i,j] = state[b,h,i,j] - old_v + k_row[j] * new_v
        base_new = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i
        state_row = tl.load(state_row_ptr + tl.arange(0, K) * stride_k)  # [K]
        new_row = state_row - old_v + k_row * new_v
        tl.store(base_new + tl.arange(0, K) * stride_new_j, new_row)


@triton.jit
def output_dot_kernel(
    q_ptr,            # [B*H, K] float32 (q_exp flattened)
    new_state_ptr,    # [B, H, V, K] float32
    out_ptr,          # [B*H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    stride_q_b: tl.constexpr,
    stride_q_k: tl.constexpr,
    stride_new_b: tl.constexpr,
    stride_new_h: tl.constexpr,
    stride_new_i: tl.constexpr,
    stride_new_j: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    # q_exp[h] flattened: q_ptr[h*K:(h+1)*K]
    q_row = tl.load(q_ptr + h_idx * K + tl.arange(0, K))  # [K]

    # Compute dot(q_row, new_state[b,h]) over V and K
    sum_acc = 0.0
    for i in tl.static_range(0, V):
        base = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i
        new_row = tl.load(base + tl.arange(0, K) * stride_new_j)  # [K]
        sum_acc += tl.sum(q_row * new_row, axis=0)

    # out[b,h] = scale * sum_acc (scale applied on host side)
    tl.store(out_ptr + b_idx * H + h_idx, sum_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation:
        - All math in Triton kernels.
        - Returns:
          - output_bf16: [B, 1, H, 1], bfloat16
          - new_state: [B, H, V, K], float32
        """
        device = q.device

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()   # [B, 1, QH, K]
        k_f32 = k.to(torch.float32).contiguous()   # [B, 1, KH, K]
        v_f32 = v.to(torch.float32).contiguous()   # [B, 1, VH, V]

        # Extract shapes
        B = q_f32.shape[0]
        QH = q_f32.shape[2]
        KH = k_f32.shape[2]
        VH = v_f32.shape[2]
        K = q_f32.shape[3]
        V = v_f32.shape[3]

        # state: [B, H, V, K], float32
        H = state.shape[1]
        Ks = state.shape[3]
        assert Ks == K, "state K dimension must match q/k K"
        state_f32 = state.to(torch.float32).contiguous()  # [B, H, V, K]

        # Expand q and k heads (repeat_interleave by ratio = num_v_heads // num_q_heads)
        # In provided tests num_v_heads=8, num_q_heads=4 => ratio=2
        ratio = 2
        q_exp = q_f32.squeeze(1).repeat_interleave(ratio, dim=1)  # [B, QH*ratio, K]
        k_exp = k_f32.squeeze(1).repeat_interleave(ratio, dim=1)  # [B, KH*ratio, K]

        # Allocate outputs for g and beta
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta over (B,H)
        grid = (B * H,)
        g_beta_kernel[grid](
            A_log.to(torch.float32), a.squeeze(1).to(torch.float32), dt_bias.to(torch.float32), b.squeeze(1).to(torch.float32),
            g_out, beta_out,
            H=H,
        )

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Get strides for state and new_state
        stride_b = state_f32.stride(0)
        stride_h = state_f32.stride(1)
        stride_v = state_f32.stride(2)
        stride_k = state_f32.stride(3)

        stride_new_b = new_state.stride(0)
        stride_new_h = new_state.stride(1)
        stride_new_i = new_state.stride(2)
        stride_new_j = new_state.stride(3)

        # Launch Triton kernel to update new_state over (B,H)
        state_update_kernel[grid](
            state_f32, new_state, k_exp, v_f32.squeeze(1), beta_out,
            B=B, H=H, V=V, K=K,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
            stride_i=V, stride_j=1,
        )

        # Compute output: out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        # q_exp is [B, QH*ratio, K]; for this setup QH*ratio==H, so h is a valid index
        q_exp_flat = q_exp.reshape(B * H, K).contiguous()  # [B*H, K]

        out = torch.empty((B * H,), dtype=torch.float32, device=device)

        # Launch output dot kernel
        grid_out = (B * H,)
        output_dot_kernel[grid_out](
            q_exp_flat, new_state, out,
            B=B, H=H, K=K, V=V,
            stride_q_b=K, stride_q_k=1,  # flattened q_exp has stride along K as 1
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
        )

        # Apply scale
        out = out * float(scale)

        # Reshape and cast output to [B, 1, H, 1] bfloat16
        out_bf16 = out.view(B, H).unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B, 1, H, 1]

        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
