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
    H: tl.constexpr,  # number of heads, e.g., 8
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b_idx * H + h_idx)   # float32
    dt_val = tl.load(dt_bias_ptr + h_idx)        # float32
    b_val = tl.load(b_ptr + b_idx * H + h_idx)   # float32

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) for numerical stability
    x = a_val + dt_val
    abs_x = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)

    A_log_val = tl.load(A_log_ptr + h_idx)       # float32
    exp_A = tl.exp(A_log_val)
    g = tl.exp(-exp_A * softplus)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta)


@triton.jit
def state_update_kernel(
    state_ptr,        # [B,H,V,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    k_exp_ptr,        # [B,H,K] float32
    v_ptr,            # [B,H,V] float32
    beta_ptr,         # [B,H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,  # 128
    K: tl.constexpr,  # 128
    stride_b,         # int: stride for B in state_ptr
    stride_h,         # int: stride for H in state_ptr
    stride_v,         # int: stride for V in state_ptr
    stride_k,         # int: stride for K in state_ptr
    new_stride_b,     # int: stride for B in new_state_ptr
    new_stride_h,     # int: stride for H in new_state_ptr
    new_stride_i,     # int: stride for V in new_state_ptr
    new_stride_j,     # int: stride for K in new_state_ptr
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Base pointers for this (b,h)
    base_state = state_ptr + b_idx * stride_b + h_idx * stride_h
    base_new = new_state_ptr + b_idx * new_stride_b + h_idx * new_stride_h

    # Loop over i in V and j in K with static_range for compile-time unrolling
    for i in tl.static_range(0, V):
        # Load old_v = dot(k_exp[b,h,:], state[b,h,i,:])
        sum_old = 0.0
        for j in tl.static_range(0, K):
            k_val = tl.load(k_exp_ptr + b_idx * H * K + h_idx * K + j)
            val = tl.load(base_state + i * stride_v + j * stride_k)
            sum_old += k_val * val
        # Load beta and v
        beta_val = tl.load(beta_ptr + b_idx * H + h_idx)
        v_val = tl.load(v_ptr + b_idx * H * V + h_idx * V + i)

        new_v = beta_val * v_val + (1.0 - beta_val) * sum_old

        # Update new_state[b,h,i,j] = state[b,h,i,j] - sum_old + new_v * k_exp[b,h,j]
        for j in tl.static_range(0, K):
            src = tl.load(base_state + i * stride_v + j * stride_k)
            k_val = tl.load(k_exp_ptr + b_idx * H * K + h_idx * K + j)
            dst = src - sum_old + new_v * k_val
            tl.store(base_new + i * new_stride_i + j * new_stride_j, dst)


@triton.jit
def output_dot_kernel(
    q_exp_ptr,        # [B,H,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    out_ptr,          # [B,H] float32
    scale,            # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,  # 128
    K: tl.constexpr,  # 128
    stride_b,         # int: stride for B in q_exp_ptr
    stride_h,         # int: stride for H in q_exp_ptr
    new_stride_b,     # int: stride for B in new_state_ptr
    new_stride_h,     # int: stride for H in new_state_ptr
    new_stride_i,     # int: stride for V in new_state_ptr
    new_stride_j,     # int: stride for K in new_state_ptr
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    base_q = q_exp_ptr + b_idx * stride_b + h_idx * stride_h
    base_new = new_state_ptr + b_idx * new_stride_b + h_idx * new_stride_h

    dot = 0.0
    for i in tl.static_range(0, V):
        q_row = tl.load(base_q + i * stride_k)
        row_new = tl.load(base_new + i * new_stride_i)
        # q_row is [K], row_new is [K], compute dot product over K
        for j in tl.static_range(0, K):
            dot += q_row[j] * row_new[j]

    out_val = scale * dot
    tl.store(out_ptr + b_idx * H + h_idx, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Cast inputs to float32 for stable compute
        device = q.device
        dtype_qk = torch.float32
        dtype_v = torch.float32

        q_f32 = q.to(dtype_qk).contiguous()  # [B,1,4,128]
        k_f32 = k.to(dtype_qk).contiguous()  # [B,1,4,128]
        v_f32 = v.to(dtype_v).contiguous()   # [B,1,8,128]

        # Repeat q and k heads by ratio 2 (num_v_heads // num_q_heads = 8 // 4 = 2)
        # q_exp and k_exp have shape [B,8,128]
        q_exp = q_f32.squeeze(1).repeat_interleave(2, dim=1)  # [B,8,128]
        k_exp = k_f32.squeeze(1).repeat_interleave(2, dim=1)  # [B,8,128]

        B = q_f32.shape[0]
        H = q_f32.shape[1]  # should be 4*2 = 8
        V = v_f32.shape[2]  # 128
        K = q_f32.shape[3]  # 128

        # Prepare state
        state_f32 = state.to(torch.float32).contiguous()  # [B,H,V,K], H=8 from original q_f32, but in inputs H=4; we align with original run: H=8
        # The original run uses H = num_v_heads = 8. In provided inputs, q has 4 heads but they expand to 8 heads via repeat_interleave in run; here we follow run: H=8.
        # Ensure H dimensions match: we will set H to be 8.
        assert q_f32.shape[1] == 4 and v_f32.shape[1] == 8, "This implementation assumes num_q_heads=4, num_v_heads=8"
        H_q = q_f32.shape[1]  # 4
        # The expanded heads after repeat_interleave is 8; we created q_exp and k_exp accordingly. So we proceed with H=8.
        H = 8
        B = q_f32.shape[0]
        V = v_f32.shape[2]  # 128
        K = q_f32.shape[3]  # 128

        # Allocate outputs for g and beta
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        grid = (B * H,)
        compute_g_beta_kernel[grid](
            A_log.to(torch.float32), a.squeeze(1).to(torch.float32), dt_bias.to(torch.float32), b.squeeze(1).to(torch.float32),
            g_out, beta_out,
            H=H,
            num_warps=4,
        )

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Get strides for state and new_state
        stride_b = state_f32.stride(0)  # in elements
        stride_h = state_f32.stride(1)
        stride_v = state_f32.stride(2)
        stride_k = state_f32.stride(3)

        new_stride_b = new_state.stride(0)
        new_stride_h = new_state.stride(1)
        new_stride_i = new_state.stride(2)
        new_stride_j = new_state.stride(3)

        # Launch Triton state update kernel over (B*H)
        state_update_kernel[grid](
            state_f32, new_state, k_exp, v_f32, beta_out,
            B=B, H=H, V=V, K=K,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k,
            new_stride_b=new_stride_b, new_stride_h=new_stride_h, new_stride_i=new_stride_i, new_stride_j=new_stride_j,
            num_warps=4,
        )

        # Compute output dot per (b,h): out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        out = torch.empty((B, H), dtype=torch.float32, device=device)

        q_exp_str_b = q_exp.stride(0)  # in elements
        q_exp_str_h = q_exp.stride(1)  # stride for H dimension in q_exp
        # Note: q_exp has shape [B,H,K], so stride for H is q_exp.stride(1), and stride for K is q_exp.stride(2)
        q_exp_str_k = q_exp.stride(2)

        state_dot_str_b = new_state.stride(0)
        state_dot_str_h = new_state.stride(1)
        state_dot_str_i = new_state.stride(2)
        state_dot_str_j = new_state.stride(3)

        output_dot_kernel[grid](
            q_exp, new_state, out, float(scale),
            B=B, H=H, V=V, K=K,
            stride_b=q_exp_str_b, stride_h=q_exp_str_h, stride_k=q_exp_str_k,
            new_stride_b=state_dot_str_b, new_stride_h=state_dot_str_h, new_stride_i=state_dot_str_i, new_stride_j=state_dot_str_j,
            num_warps=4,
        )

        # Return exactly two outputs: (output_bf16, new_state_f32)
        out_bf16 = out.to(torch.bfloat16)  # cast to bfloat16 as required by the original signature
        return (out_bf16, new_state)

# The following helper is not used by the harness, but provided for testing parity:
def get_inputs():
    q = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16)
    k = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16)
    v = torch.randn([1, 1, 8, 128], dtype=torch.bfloat16)
    state = torch.randn([1, 8, 128, 128], dtype=torch.float32)
    A_log = torch.randn([8], dtype=torch.float32)
    a = torch.randn([1, 1, 8], dtype=torch.bfloat16)
    dt_bias = torch.randn([8], dtype=torch.float32)
    b = torch.randn([1, 1, 8], dtype=torch.bfloat16)
    scale = 1.0  # float32 scalar
    return [q, k, v, state, A_log, a, dt_bias, b, scale]

# The original Model wraps run; we implement only ModelNew here.
# If a separate Model class is required, it can simply call ModelNew().forward(*args) inside __call__.


def run(*args):
    return ModelNew()(*args)
