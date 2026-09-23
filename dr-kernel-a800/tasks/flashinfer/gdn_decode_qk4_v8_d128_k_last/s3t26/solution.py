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
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b_idx * H + h_idx)       # float32
    dt_val = tl.load(dt_bias_ptr + h_idx)            # float32
    b_val = tl.load(b_ptr + b_idx * H + h_idx)       # float32

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) for numerical stability
    x = a_val + dt_val
    abs_x = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)

    # exp(A_log[h])
    exp_A = tl.exp(tl.load(A_log_ptr + h_idx))       # float32
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
    V: tl.constexpr,
    K: tl.constexpr,
    stride_b,         # int: stride for B in state_ptr
    stride_h,         # int: stride for H in state_ptr
    stride_v,         # int: stride for V in state_ptr
    stride_k,         # int: stride for K in state_ptr
    stride_new_b,     # int: stride for B in new_state_ptr
    stride_new_h,     # int: stride for H in new_state_ptr
    stride_new_i,     # int: stride for V in new_state_ptr
    stride_new_j,     # int: stride for K in new_state_ptr
):
    pid = tl.program_id(axis=0)  # 1D grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load k_exp for this (b,h) and beta
    k_ptr = k_exp_ptr + b_idx * H * K
    beta = tl.load(beta_ptr + b_idx * H + h_idx)

    # Loop over V and K (compile-time 128)
    for i in tl.static_range(0, V):
        # Compute old_v = dot(k[h], state[b,h,i,:])
        old_v = 0.0
        for j in tl.static_range(0, K):
            k_val = tl.load(k_ptr + j)  # k_exp[b,h,j]
            state_val = tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            old_v += k_val * state_val

        # new_v = beta * v[h,i] + (1 - beta) * old_v
        v_val = tl.load(v_ptr + b_idx * H * V + h_idx * V + i)  # v[b,h,i] but b_idx is redundant since v shape is [B,H,V]; use b_idx=0 indexing? We need per-b v, so v is [B,H,V].
        # Correction: v_ptr is [B,H,V], so v[b,h,i] = v_ptr + b*H*V + h*V + i
        v_val = tl.load(v_ptr + b_idx * H * V + h_idx * V + i)
        new_v = beta * v_val + (1.0 - beta) * old_v

        # Update new_state[b,h,i,j] = state_old - old_v + new_v * k[h,j]
        for j in tl.static_range(0, K):
            k_val = tl.load(k_ptr + j)
            state_old = tl.load(state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v + j * stride_k)
            new_state_val = state_old - old_v + k_val * new_v
            tl.store(new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i + j * stride_new_j, new_state_val)


@triton.jit
def output_dot_kernel(
    q_exp_ptr,        # [B,H,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    out_ptr,          # [B,H] float32
    scale,            # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    stride_q_b,       # int
    stride_q_h,       # int
    stride_new_b,     # int
    stride_new_h,     # int
    stride_new_i,     # int
    stride_new_j,     # int
):
    pid = tl.program_id(axis=0)  # 1D grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Compute out[b,h] = scale * sum_j q_exp[b,h,j] * new_state[b,h,0,j]
    acc = 0.0
    for j in tl.static_range(0, K):
        q_val = tl.load(q_exp_ptr + b_idx * H * K + h_idx * K + j)
        # For V=1, new_state index i=0
        state_val = tl.load(new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + 0 * stride_new_i + j * stride_new_j)
        acc += q_val * state_val

    out_val = acc * scale
    tl.store(out_ptr + b_idx * H + h_idx, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original 'run' function.
        Returns:
          - output: [B, 1, H, 1], bfloat16
          - new_state: [B, H, V, K], float32
        """
        device = q.device
        # Cast inputs to float32 for stable math and ensure contiguity
        q_f32 = q.float().contiguous()            # [B,1,num_q_heads,K]
        k_f32 = k.float().contiguous()            # [B,1,num_k_heads,K]
        v_f32 = v.float().contiguous()            # [B,1,num_v_heads,V]
        state_f32 = state.to(torch.float32).contiguous()  # [B,H,V,K]

        # Expand q and k heads by repeat_interleave: ratio = num_v_heads // num_q_heads
        ratio = v_f32.shape[1] // q_f32.shape[1]
        q_exp = q_f32.squeeze(1).repeat_interleave(ratio, dim=1)  # [B,num_v_heads,K]
        k_exp = k_f32.squeeze(1).repeat_interleave(ratio, dim=1)  # [B,num_v_heads,K]

        B = q_f32.shape[0]
        H = q_f32.shape[1]  # equals num_v_heads in this task (8)
        V = state_f32.shape[2]  # 128
        K = state_f32.shape[3]  # 128

        # Allocate g and beta outputs
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch g_beta kernel over grid (B*H)
        grid = (B * H,)
        compute_g_beta_kernel[grid](
            A_log.float(), a.squeeze(1).float(), dt_bias.float(), b.squeeze(1).float(),
            g_out, beta_out,
            H=H,
        )

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Launch state_update kernel over grid (B*H)
        state_update_kernel[grid](
            state_f32, new_state, k_exp, v_f32, beta_out,
            B=B, H=H, V=V, K=K,
            stride_b=state_f32.stride(0), stride_h=state_f32.stride(1), stride_v=state_f32.stride(2), stride_k=state_f32.stride(3),
            stride_new_b=new_state.stride(0), stride_new_h=new_state.stride(1), stride_new_i=new_state.stride(2), stride_new_j=new_state.stride(3),
        )

        # Launch output_dot kernel over grid (B*H)
        out = torch.empty((B, H), dtype=torch.float32, device=device)
        output_dot_kernel[grid](
            q_exp, new_state, out, float(scale),
            B=B, H=H, V=V, K=K,
            stride_q_b=q_exp.stride(0), stride_q_h=q_exp.stride(1),
            stride_new_b=new_state.stride(0), stride_new_h=new_state.stride(1), stride_new_i=new_state.stride(2), stride_new_j=new_state.stride(3),
        )

        # Return outputs with required shapes/dtypes
        # output: [B,1,H,1] in bfloat16, new_state: [B,H,V,K] in float32
        output_bf16 = out.unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        return [output_bf16], new_state


def run(*args):
    return ModelNew()(*args)
