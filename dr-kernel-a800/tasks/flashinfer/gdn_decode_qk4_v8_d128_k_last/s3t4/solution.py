import math
import torch
import triton
import triton.language as tl


@triton.jit
def softplus_exp_sigmoid_kernel(
    A_log_ptr,           # [H] float32
    a_ptr,               # [B,H] float32
    dt_bias_ptr,         # [H] float32
    b_ptr,               # [B,H] float32
    g_out_ptr,           # [B,H] float32
    beta_out_ptr,        # [B,H] float32
    B: tl.constexpr,     # runtime
    H: tl.constexpr,     # runtime
):
    # program id over (b,h)
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H
    if b_idx >= B or h_idx >= H:
        return

    a_val = tl.load(a_ptr + b_idx * H + h_idx)   # a[b,h]
    db_val = tl.load(dt_bias_ptr + h_idx)        # dt_bias[h]
    A_log_val = tl.load(A_log_ptr + h_idx)       # A_log[h]
    b_val = tl.load(b_ptr + b_idx * H + h_idx)   # b[b,h]

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    x = a_val + db_val
    abs_x = tl.abs(x)
    max_x0 = tl.maximum(x, 0.0)
    softplus = tl.log(1.0 + tl.exp(-abs_x)) + max_x0

    # g = exp(-exp(A_log) * softplus(a + dt_bias))
    exp_A = tl.exp(A_log_val)
    g = tl.exp(-exp_A * softplus)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    sig = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, sig)


@triton.jit
def state_update_kernel(
    old_state_ptr,       # [B,H,V,K] float32
    k_ptr,               # [H,K] float32
    v_ptr,               # [H,V] float32
    beta_ptr,            # [H] float32
    new_state_ptr,       # [B,H,V,K] float32
    B: tl.constexpr,     # runtime
    H: tl.constexpr,     # runtime
    V: tl.constexpr,     # compile-time (128)
    K: tl.constexpr,     # compile-time (128)
    stride_b: tl.constexpr,  # stride for dim 0 in elements
    stride_h: tl.constexpr,  # stride for dim 1
    stride_v: tl.constexpr,  # stride for dim 2
    stride_k: tl.constexpr,  # stride for dim 3
):
    # 1D grid over (b,h)
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    if b_idx >= B or h_idx >= H:
        return

    # Load k[h,:] and v[h,:]
    k_vec = tl.load(k_ptr + h_idx * K + tl.arange(0, K))  # [K]
    v_vec = tl.load(v_ptr + h_idx * V + tl.arange(0, V))  # [V]
    beta_val = tl.load(beta_ptr + h_idx)

    # For each row i in V, compute old_v and state_update, then update new_state
    for i in tl.static_range(0, V):
        # Compute old_v = dot(k[h], old_state[b,h,i,:]) = sum_{j=0}^{K-1} k[h,j] * old_state[b,h,i,j]
        old_v_i = tl.zeros((), dtype=tl.float32)
        for j in tl.static_range(0, K):
            old_val_i = tl.load(old_state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            old_v_i += old_val_i * k_vec[j]

        # new_v_i = beta[h] * v[h,i] + (1 - beta[h]) * old_v_i
        new_v_i = beta_val * v_vec[i] + (1.0 - beta_val) * old_v_i

        # state_remove = old_v_i (scalar), state_update = dot(k[h], new_v_i) = new_v_i (scalar)
        # Update new_state[b,h,i,j] = old_state[b,h,i,j] - state_remove + state_update
        for j in tl.static_range(0, K):
            old_val_i = tl.load(old_state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            new_val_i = old_val_i - old_v_i + new_v_i  # since both state_remove and state_update are scalars, new_v_i equals state_update
            tl.store(new_state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k, new_val_i)


@triton.jit
def output_dot_kernel(
    q_ptr,               # [B*8, K] float32
    new_state_ptr,       # [B,H,V,K] float32
    out_ptr,             # [B,H] float32
    B: tl.constexpr,     # runtime
    H: tl.constexpr,     # runtime
    V: tl.constexpr,     # compile-time (128)
    K: tl.constexpr,     # compile-time (128)
    stride_b: tl.constexpr,  # stride for dim 0 in elements
    stride_h: tl.constexpr,  # stride for dim 1
    stride_v: tl.constexpr,  # stride for dim 2
    stride_k: tl.constexpr,  # stride for dim 3
):
    # 1D grid over (b,h)
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    if b_idx >= B or h_idx >= H:
        return

    # q_exp_flat is [B*8, K]; index h_exp = b_idx * 8 + h_idx
    h_exp = b_idx * 8 + h_idx
    q_row = tl.load(q_ptr + h_exp * K + tl.arange(0, K))  # [K]

    # Compute output per (b,h) as dot(q_exp[h], new_state[b,h, :, :]) = sum_i q[h,i] * sum_j new_state[b,h,i,j]
    acc = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(0, K):
        col_sum = tl.zeros((), dtype=tl.float32)
        for i in tl.static_range(0, V):
            val = tl.load(new_state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            col_sum += val
        acc += q_row[j] * col_sum

    tl.store(out_ptr + b_idx * H + h_idx, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure float32 for computation
        q_f32 = q.to(torch.float32).contiguous()  # [B,1,4,128]
        k_f32 = k.to(torch.float32).contiguous()  # [B,1,4,128]
        v_f32 = v.to(torch.float32).contiguous()  # [B,1,8,128]
        state_f32 = state.to(torch.float32).contiguous()  # [B,8,128,128]

        B, _, num_q_heads, K = q_f32.shape
        _, _, num_k_heads, _ = k_f32.shape
        _, _, num_v_heads, V = v_f32.shape
        num_heads = num_v_heads

        # repeat_interleave q and k along head dim to match num_v_heads
        repeat_q = num_v_heads // num_q_heads
        repeat_k = num_v_heads // num_k_heads
        q_exp = q_f32.squeeze(1).repeat_interleave(repeat_q, dim=1)  # [B,8,128]
        k_exp = k_f32.squeeze(1).repeat_interleave(repeat_k, dim=1)  # [B,8,128]

        # Allocate outputs for g and beta
        g_out = torch.empty((B, num_heads), dtype=torch.float32, device=q.device)
        beta_out = torch.empty((B, num_heads), dtype=torch.float32, device=q.device)

        # Kernel 1: compute g and beta per (b,h)
        A_log_f32 = A_log.to(torch.float32).contiguous()  # [8]
        a_f32 = a.to(torch.float32).contiguous()          # [B,1,8]
        dt_bias_f32 = dt_bias.to(torch.float32).contiguous()  # [8]
        b_f32 = b.to(torch.float32).contiguous()          # [B,1,8]

        grid1 = (B * num_heads,)
        softplus_exp_sigmoid_kernel[grid1](
            A_log_f32, a_f32.squeeze(1), dt_bias_f32, b_f32.squeeze(1),
            g_out, beta_out,
            B=B, H=num_heads
        )

        # Prepare inputs for state update
        k_f32_ex = k_exp.squeeze(1)  # [B,8,128]
        v_f32_ex = v_f32.squeeze(1)  # [B,8,128]

        # Allocate new_state [B,8,128,128]
        new_state = torch.empty((B, num_heads, V, K), dtype=torch.float32, device=q.device)

        # Get strides for state tensors (PyTorch strides in elements)
        stride_b = state_f32.stride(0)   # stride for dim 0 (B)
        stride_h = state_f32.stride(1)   # stride for dim 1 (H)
        stride_v = state_f32.stride(2)   # stride for dim 2 (V)
        stride_k = state_f32.stride(3)   # stride for dim 3 (K)

        # Launch state_update_kernel over (B*H)
        grid2 = (B * num_heads,)
        state_update_kernel[grid2](
            state_f32,               # [B,H,V,K]
            k_f32_ex,                # [H,K]
            v_f32_ex,                # [H,V]
            beta_out,                # [B,H]
            new_state,               # [B,H,V,K]
            B=B, H=num_heads, V=128, K=128,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k
        )

        # Compute output: out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        q_exp_flat = q_exp.reshape(B * num_heads, K).contiguous()  # [B*8, K]
        out = torch.empty((B, num_heads), dtype=torch.float32, device=q.device)

        grid3 = (B * num_heads,)
        output_dot_kernel[grid3](
            q_exp_flat, new_state, out,
            B=B, H=num_heads, V=128, K=128,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k
        )

        # Cast output to bfloat16 and reshape to [B,1,H,1] to match original return signature
        out_bf16 = out.unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
