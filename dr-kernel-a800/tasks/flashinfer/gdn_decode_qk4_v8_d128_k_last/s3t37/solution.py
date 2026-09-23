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
    B, H,             # int scalars
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load scalars
    x = tl.load(a_ptr + b_idx * H + h_idx) + tl.load(dt_bias_ptr + h_idx)
    A = tl.load(A_log_ptr + h_idx)

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    abs_x = tl.abs(x)
    softplus_x = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)

    g = tl.exp(-tl.exp(A) * softplus_x)
    beta = 1.0 / (1.0 + tl.exp(-x))

    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta)


@triton.jit
def state_update_kernel(
    state_ptr,        # [B,H,V,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    g_ptr,            # [B,H] float32
    beta_ptr,         # [B,H] float32
    k_ptr,            # [B,H,K] float32
    v_ptr,            # [B,H,V] float32
    B, H, V, K,       # ints
    # strides for state input
    stride_b, stride_h, stride_v, stride_k,
    # strides for new_state output
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load per-(b,h) scalars
    g_val = tl.load(g_ptr + b_idx * H + h_idx)
    beta_val = tl.load(beta_ptr + b_idx * H + h_idx)

    # Iterate over rows i in V and columns j in K
    for i in tl.static_range(V):
        # old_v = dot(k[h], state[b,h,i,:])
        # Compute sum_j k[h,j] * state[b,h,i,j]
        acc_old = 0.0
        for j in tl.static_range(K):
            state_off = b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k
            state_val = tl.load(state_ptr + state_off)
            k_off = b_idx * H + h_idx * K + j  # k[h] is contiguous [K]
            k_val = tl.load(k_ptr + k_off)
            acc_old += state_val * k_val

        # new_v = beta * v[b,h,i] + (1 - beta) * old_v
        v_off = b_idx * H * V + h_idx * V + i
        v_val = tl.load(v_ptr + v_off)
        new_v = beta_val * v_val + (1.0 - beta_val) * acc_old  # scalar

        # Update new_state[b,h,i,j] = old_state - old_v + new_v
        # First set new_state[b,h,i,:] = old_state[b,h,i,:]
        for j in tl.static_range(K):
            state_off = b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k
            old_state_val = tl.load(state_ptr + state_off)
            new_state_off = b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i + j * stride_new_j
            # new_state += (new_v - acc_old) but since acc_old is contribution, set new_state to old_state
            tl.store(new_state_ptr + new_state_off, old_state_val)

        # Add contribution from new_v: new_state[b,h,i,j] += k[h,j] * new_v
        for j in tl.static_range(K):
            k_off = b_idx * H + h_idx * K + j
            k_val = tl.load(k_ptr + k_off)
            new_state_off = b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i + j * stride_new_j
            tl.store(new_state_ptr + new_state_off, tl.load(new_state_ptr + new_state_off) + k_val * new_v)


@triton.jit
def output_dot_kernel(
    new_state_ptr,    # [B,H,V,K] float32
    q_ptr,            # [B,H,K] float32 (q_exp)
    out_ptr,          # [B,H] float32
    B, H, V, K,       # ints
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,
    stride_q_b, stride_q_h,
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    acc = 0.0
    for j in tl.static_range(K):
        # q_exp[h,j] is a scalar
        q_off = b_idx * stride_q_b + h_idx * stride_q_h + j
        q_val = tl.load(q_ptr + q_off)
        # sum over V of new_state[b,h,i,j]
        sum_v = 0.0
        for i in tl.static_range(V):
            new_state_off = b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i + j * stride_new_j
            sum_v += tl.load(new_state_ptr + new_state_off)
        acc += q_val * sum_v

    tl.store(out_ptr + b_idx * H + h_idx, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure tensors are on same device
        device = q.device
        dtype_qkv = q.dtype
        dtype_state = state.dtype

        # Cast to float32 for computation
        q_f32 = q.squeeze(1).to(torch.float32).contiguous()   # [B, num_q_heads, K]
        k_f32 = k.squeeze(1).to(torch.float32).contiguous()   # [B, num_k_heads, K]
        v_f32 = v.squeeze(1).to(torch.float32).contiguous()   # [B, num_v_heads, V]
        state_f32 = state.to(torch.float32).contiguous()      # [B, num_heads, V, K]
        A_log = A_log.to(torch.float32)                       # [num_heads]
        a = a.squeeze(1).to(torch.float32)                    # [B, num_heads]
        dt_bias = dt_bias.to(torch.float32)                   # [num_heads]
        b = b.squeeze(1).to(torch.float32)                    # [B, num_heads]

        # Expand q and k by repeat_interleave (ratio = num_v_heads // num_q_heads = 2)
        # q_f32: [B, 4, K], v_f32: [B, 8, V], we expect num_heads = v.shape[1] // 2, but original uses 8/4 -> ratio 2
        q_exp = q_f32.repeat_interleave(2, dim=1)             # [B, 8, K]
        k_exp = k_f32.repeat_interleave(2, dim=1)             # [B, 8, K]

        B = q_f32.shape[0]
        H = q_f32.shape[1]  # num_q_heads
        V = v_f32.shape[2]  # V
        K = q_f32.shape[2]  # K

        # Allocate outputs for g and beta: [B, H]
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        grid_g = (B * H,)
        compute_g_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g_out, beta_out, B, H
        )

        # Allocate new_state: [B, H, V, K] float32
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Get strides (in elements)
        stride_b = state_f32.stride(0)
        stride_h = state_f32.stride(1)
        stride_v = state_f32.stride(2)
        stride_k = state_f32.stride(3)

        stride_new_b = new_state.stride(0)
        stride_new_h = new_state.stride(1)
        stride_new_i = new_state.stride(2)
        stride_new_j = new_state.stride(3)

        # Launch state update kernel
        grid_state = (B * H,)
        state_update_kernel[grid_state](
            state_f32, new_state, g_out, beta_out, k_exp, v_f32, B, H, V, K,
            stride_b, stride_h, stride_v, stride_k,
            stride_new_b, stride_new_h, stride_new_i, stride_new_j,
        )

        # Compute output: out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        out = torch.empty((B, H), dtype=torch.float32, device=device)

        stride_q_b = q_exp.stride(0)
        stride_q_h = q_exp.stride(1)

        grid_out = (B * H,)
        output_dot_kernel[grid_out](
            new_state, q_exp, out, B, H, V, K,
            stride_new_b, stride_new_h, stride_new_i, stride_new_j,
            stride_q_b, stride_q_h,
        )

        # Cast output to bfloat16 as [B,1,H,1]
        out_bf16 = out.unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
