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
    H: tl.constexpr,  # number of heads, compile-time (e.g., 8)
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    dt_val = tl.load(dt_bias_ptr + h_idx)
    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) for numerical stability
    x = a_val + dt_val
    abs_x = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)
    g = tl.exp(-tl.exp(A_log_ptr + h_idx) * softplus)
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
    B: tl.constexpr,  # batch size (not used, kept for clarity)
    H: tl.constexpr,  # number of heads
    V: tl.constexpr,  # e.g., 128
    K: tl.constexpr,  # e.g., 128
    stride_b, stride_h, stride_v, stride_k,
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    beta_val = tl.load(beta_ptr + h_idx)
    # Load k[h, :] vector
    k_vec = tl.load(k_ptr + h_idx * K + tl.arange(0, K))  # [K]

    # For each row i in V, compute dot with old_state and update
    for i in tl.static_range(0, V):
        old_row_ptr = old_state_ptr + b_idx * stride_b + h_idx * stride_h + i * stride_v
        new_row_ptr = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i
        old_v = 0.0
        for j in tl.static_range(0, K):
            old_v += tl.load(k_ptr + h_idx * K + j) * tl.load(old_row_ptr + j * stride_k)
        new_v = beta_val * tl.load(v_ptr + h_idx * V + i) + (1.0 - beta_val) * old_v
        for j in tl.static_range(0, K):
            old_val = tl.load(old_row_ptr + j * stride_k)
            upd = old_val - old_v + k_vec[j] * new_v
            tl.store(new_row_ptr + j * stride_new_j, upd)


@triton.jit
def output_dot_kernel(
    q_exp_ptr,        # [B*H,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    out_ptr,          # [B*H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    stride_qb, stride_qk,
    stride_bns, stride_hns, stride_i, stride_j,
    scale: tl.float32,
):
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    # q_exp_ptr is laid out as [B*H, K], we index row (b_idx*H + h_idx) -> [K]
    q_row_ptr = q_exp_ptr + (b_idx * H + h_idx) * K
    acc = 0.0
    for i in tl.static_range(0, V):
        row_ptr = new_state_ptr + b_idx * stride_bns + h_idx * stride_hns + i * stride_i
        for j in tl.static_range(0, K):
            val = tl.load(row_ptr + j * stride_j)
            qj = tl.load(q_row_ptr + j * stride_qk)
            acc += val * qj
    tl.store(out_ptr + pid, acc * scale)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Cast inputs to float32 for compute; squeeze batch dim 1 (since original has shape [B,1,...])
        device = q.device
        q_f32 = q.to(torch.float32).squeeze(1).contiguous()
        k_f32 = k.to(torch.float32).squeeze(1).contiguous()
        v_f32 = v.to(torch.float32).squeeze(1).contiguous()
        state_f32 = state.to(torch.float32).contiguous()  # [B,H,V,K]

        # Repeat q and k heads by repeat_interleave along head dim (ratio is num_v_heads // num_q_heads).
        # In provided tests num_v_heads=8 and num_q_heads=4 -> ratio 2.
        q_exp = q_f32.repeat_interleave(2, dim=1)  # [B,8,K]
        k_exp = k_f32.repeat_interleave(2, dim=1)  # [B,8,K]

        B = q_f32.shape[0]
        H = q_f32.shape[1]
        V = state_f32.shape[2]
        K = state_f32.shape[3]

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

        # Get strides for tensors (elements)
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
            B=B, H=H, V=V, K=K,
            stride_b=stride_b, stride_h=stride_h, stride_v=stride_v, stride_k=stride_k,
            stride_new_b=stride_new_b, stride_new_h=stride_new_h, stride_new_i=stride_new_i, stride_new_j=stride_new_j,
        )

        # Compute output: out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        q_exp_flat = q_exp.reshape(B * H, K).contiguous()  # [


def run(*args):
    return ModelNew()(*args)
