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

    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    dt_val = tl.load(dt_bias_ptr + h_idx)

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) (numerically stable)
    x = a_val + dt_val
    absx = tl.abs(x)
    maxx = tl.maximum(x, 0.0)
    softplus = tl.log(1.0 + tl.exp(-absx)) + maxx
    g_val = tl.exp(-tl.exp(A_log_ptr + h_idx) * softplus)
    beta_val = 1.0 / (1.0 + tl.exp(-b_ptr + b_idx * H + h_idx))

    tl.store(g_out_ptr + b_idx * H + h_idx, g_val)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta_val)


@triton.jit
def state_update_kernel(
    state_ptr,         # [B,H,V,K] float32, input
    new_state_ptr,     # [B,H,V,K] float32, output
    k_exp_ptr,         # [B,H,K] float32
    v_ptr,             # [B,H,V] float32
    beta_ptr,          # [B,H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    stride_b, stride_h, stride_v, stride_k,         # strides for state
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,  # strides for new_state
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Pointers to rows for this (b,h)
    state_base = state_ptr + b_idx * stride_b + h_idx * stride_h
    new_state_base = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h

    # Load k_exp[h] and beta[h]
    k_vec = tl.load(k_exp_ptr + b_idx * H + h_idx + tl.arange(0, K))  # [K]
    beta_val = tl.load(beta_ptr + b_idx * H + h_idx)  # scalar

    # Iterate over i in V and j in K (fixed sizes; Triton supports static_range)
    for i in tl.static_range(0, V):
        state_row_base = state_base + i * stride_v
        new_row_base = new_state_base + i * stride_new_i

        old_v = 0.0
        for j in tl.static_range(0, K):
            # old_v += k_vec[j] * state[b,h,i,j]
            old_v += k_vec[j] * tl.load(state_row_base + j * stride_k)

        # Compute new_v per i (note: beta_val is scalar per (b,h))
        # We need v[b,h,i]; load it
        v_val = tl.load(v_ptr + b_idx * H * V + h_idx * V + i)
        new_v = beta_val * v_val + (1.0 - beta_val) * old_v

        # Update new_state[b,h,i,j] = state[b,h,i,j] - old_v + k_vec[j] * new_v
        for j in tl.static_range(0, K):
            old = tl.load(state_row_base + j * stride_k)
            new_s = old - old_v + k_vec[j] * new_v
            tl.store(new_row_base + j * stride_new_j, new_s)


@triton.jit
def output_dot_kernel(
    q_exp_ptr,         # [B,H,K] float32
    new_state_ptr,     # [B,H,V,K] float32
    out_ptr,           # [B,H] float32
    scale,             # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_k,         # strides for q_exp
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,  # strides for new_state
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    q_row_base = q_exp_ptr + b_idx * stride_q_b + h_idx * stride_q_h  # [K]
    new_row_base = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h

    acc = 0.0
    for i in tl.static_range(0, V):
        row_base = new_row_base + i * stride_new_i  # [K]
        for j in tl.static_range(0, K):
            val = tl.load(row_base + j * stride_new_j)  # float32
            acc += tl.load(q_row_base + j * stride_q_k) * val

    acc = acc * scale
    tl.store(out_ptr + b_idx * H + h_idx, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, num_q_heads, K], k: [B, 1, num_k_heads, K], v: [B, 1, num_v_heads, V]
        state: [B, num_heads, V, K]
        A_log: [num_heads], a: [B, 1, num_heads], dt_bias: [num_heads], b: [B, 1, num_heads], scale: float
        Returns:
        - output: [B, 1, H, 1] bfloat16
        - new_state: [B, H, V, K] float32 (in this harness, V=1)
        """
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "Triton requires CUDA tensors"
        assert device.type == "cuda", "Inputs must be on CUDA"

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()  # [B,1,Hq,K]
        k_f32 = k.to(torch.float32).contiguous()  # [B,1,Hk,K]
        v_f32 = v.to(torch.float32).contiguous()  # [B,1,Hv,V]
        state_f32 = state.to(torch.float32).contiguous()  # [B,H,V,K]

        # Compute H from v: num_v_heads
        H = v_f32.shape[1]  # hv
        B = q_f32.shape[0]
        K = q_f32.shape[-1]
        V = v_f32.shape[-1]
        # The original code uses num_v_heads == 8 and num_q_heads == 4 -> repeat_interleave ratio 2
        q_exp = q_f32.repeat_interleave(2, dim=1)  # [B, H, K]
        k_exp = k_f32.repeat_interleave(2, dim=1)  # [B, H, K]

        # Allocate outputs for g and beta: [B,H]
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        grid = (B * H,)
        compute_g_beta_kernel[grid](
            A_log.to(torch.float32), a.squeeze(1).to(torch.float32), dt_bias.to(torch.float32), b.squeeze(1).to(torch.float32),
            g_out, beta_out,
            H=H,
        )

        # Allocate new_state: [B,H,V,K], float32
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Prepare strides (elements) for Triton kernels
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
            state_f32, new_state, k_exp, v_f32.squeeze(1), beta_out,
            B=B, H=H, V=V, K=K,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
        )

        # Compute output: out[b,h] = scale * (q_exp[b,h,:] @ new_state[b,h,:,:])
        out = torch.empty((B, H), dtype=torch.float32, device=device)

        stride_q_b = q_exp.stride(0)
        stride_q_h = q_exp.stride(1)
        stride_q_k = q_exp.stride(2)

        output_dot_kernel[grid](
            q_exp, new_state, out, scale,
            B=B, H=H, V=V, K=K,
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_k=stride_q_k,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
        )

        # Return output as [B,1,H,1] bfloat16, and new_state as [B,H,V,K] float32
        output = out.unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        return (output, new_state)

# The original get_inputs() and fused_operator remain unchanged. ModelNew.forward is the entry point.


def run(*args):
    return ModelNew()(*args)
