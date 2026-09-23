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
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    dt_val = tl.load(dt_bias_ptr + h_idx)
    A_val = tl.load(A_log_ptr + h_idx)

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) for numerical stability
    x = a_val + dt_val
    softplus_x = tl.log(1.0 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0.0)
    g = tl.exp(-tl.exp(A_val) * softplus_x)
    # sigmoid(b_val) = 1 / (1 + exp(-b_val))
    b_val = tl.load(b_ptr + b_idx * H + h_idx)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta)


@triton.jit
def state_update_kernel(
    old_state_ptr,    # [B,H,V,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    k_ptr,            # [H,K] float32
    v_ptr,            # [H,V] float32
    beta_ptr,         # [H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,  # 128
    K: tl.constexpr,  # 128
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_i: tl.constexpr,
    stride_j: tl.constexpr,
    stride_new_b: tl.constexpr,
    stride_new_h: tl.constexpr,
    stride_new_i: tl.constexpr,
    stride_new_j: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    beta_val = tl.load(beta_ptr + h_idx)
    k_row = tl.load(k_ptr + h_idx * K + tl.arange(0, K))  # [K]
    for i in tl.static_range(0, V):
        # Load old_state[b,h,i,:] -> [K]
        old_row_ptr = old_state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_i
        old_vals = tl.load(old_row_ptr + tl.arange(0, K) * stride_j)  # [K]
        # old_v = dot(k_row, old_vals)
        old_v = tl.sum(old_vals * k_row)
        # new_v = beta * v[h,i] + (1 - beta) * old_v
        v_elem = tl.load(v_ptr + h_idx * V + i)
        new_v = beta_val * v_elem + (1.0 - beta_val) * old_v
        # Update new_state[b,h,i,j] = old_state + k[j] * new_v
        new_row_ptr = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i
        old_row_new_ptr = old_state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_i
        for j in tl.static_range(0, K):
            old_val = tl.load(old_row_new_ptr + j * stride_j)
            new_val = old_val - old_v + k_row[j] * new_v
            tl.store(new_row_ptr + j * stride_new_j, new_val)


@triton.jit
def output_dot_kernel(
    q_exp_ptr,        # [B*H,K] float32 flattened, contiguous
    new_state_ptr,    # [B,H,V,K] float32
    out_ptr,          # [B,H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,  # 128
    K: tl.constexpr,  # 128
    stride_new_b: tl.constexpr,
    stride_new_h: tl.constexpr,
    stride_new_i: tl.constexpr,
    stride_new_j: tl.constexpr,
    scale: tl.float32,
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    q_row = tl.load(q_exp_ptr + (b_idx * H + h_idx) * K + tl.arange(0, K))  # [K]
    acc = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(0, V):
        new_row_ptr = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i
        new_vals = tl.load(new_row_ptr + tl.arange(0, K) * stride_new_j)  # [K]
        acc += tl.sum(q_row * new_vals)
    tl.store(out_ptr + b_idx * H + h_idx, acc * scale)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device and dtype; compute in float32
        device = q.device
        q_f32 = q.to(torch.float32).squeeze(1).contiguous()      # [B,4,K]
        k_f32 = k.to(torch.float32).squeeze(1).contiguous()      # [B,4,K]
        v_f32 = v.to(torch.float32).squeeze(1).contiguous()      # [B,8,V]
        state_f32 = state.to(torch.float32).contiguous()         # [B,H,V,K], H=8,V=128,K=128

        # Repeat q and k heads by repeat_interleave (ratio 2)
        q_exp = q_f32.repeat_interleave(2, dim=1)                # [B,8,K]
        k_exp = k_f32.repeat_interleave(2, dim=1)                # [B,8,K]

        B = q_f32.shape[0]
        H = q_f32.shape[1]  # number of expanded heads (8 in tests)
        V = state_f32.shape[2]
        K = state_f32.shape[3]

        # Allocate outputs for g and beta
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        grid_gbeta = (B * H,)
        compute_g_beta_kernel[grid_gbeta](
            A_log.to(torch.float32), a.squeeze(1).to(torch.float32), dt_bias.to(torch.float32), b.squeeze(1).to(torch.float32),
            g_out, beta_out,
            H=H,
        )

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Get strides for tensors (elements)
        stride_b = state_f32.stride(0)
        stride_h = state_f32.stride(1)
        stride_i = state_f32.stride(2)
        stride_j = state_f32.stride(3)

        stride_new_b = new_state.stride(0)
        stride_new_h = new_state.stride(1)
        stride_new_i = new_state.stride(2)
        stride_new_j = new_state.stride(3)

        # Launch Triton state update kernel over (B*H)
        grid_update = (B * H,)
        state_update_kernel[grid_update](
            state_f32, new_state, k_exp, v_f32, beta_out,
            B=B, H=H, V=V, K=K,
            stride_b=stride_b, stride_h=stride_h, stride_i=stride_i, stride_j=stride_j,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
        )

        # Compute output: out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        # Flatten q_exp to [B*H, K] for the kernel
        q_exp_flat = q_exp.reshape(B * H, K).contiguous()

        out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton output dot kernel
        grid_out = (B * H,)
        output_dot_kernel[grid_out](
            q_exp_flat, new_state, out,
            B=B, H=H, V=V, K=K,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
            scale=float(scale),
        )

        # Cast output to bfloat16 and reshape to [B,1,H,1] to match original signature
        out_bf16 = out.unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
