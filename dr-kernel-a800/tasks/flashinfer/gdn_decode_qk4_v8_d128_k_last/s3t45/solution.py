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
    A_val = tl.load(A_log_ptr + h_idx)

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    x = a_val + dt_val
    absx = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-absx)) + tl.maximum(x, 0.0)

    g = tl.exp(-tl.exp(A_val) * softplus)
    beta = 1.0 / (1.0 + tl.exp(-b_ptr[b_idx * H + h_idx]))

    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta)


@triton.jit
def state_update_kernel(
    state_ptr,        # [B,H,V,K] float32
    new_state_ptr,    # [B,H,V,K] float32
    k_ptr,            # [B,H,K] float32
    v_ptr,            # [B,H,V] float32
    beta_ptr,         # [B,H] float32
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    stride_b, stride_h, stride_v, stride_k,
    stride_new_b, stride_new_h, stride_new_i, stride_new_j,
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    # Load q_exp[h] and k[h] for this head
    # q_exp[h] is implicitly handled in output_dot_kernel, k[h] is k_ptr[b_idx, h_idx, :]
    k_row_ptr = k_ptr + b_idx * (H * K) + h_idx * K  # [K]
    old_state_ptr = state_ptr + b_idx * stride_b + h_idx * stride_h  # base for (b,h)

    for i in tl.static_range(0, V):
        old_state_row_ptr = old_state_ptr + i * stride_v  # [K]
        old_v = 0.0
        for j in tl.static_range(0, K):
            old_v += tl.load(k_row_ptr + j) * tl.load(old_state_row_ptr + j * stride_k)
        beta_val = tl.load(beta_ptr + b_idx * H + h_idx)
        v_val = tl.load(v_ptr + b_idx * (H * V) + h_idx * V + i)
        new_v = beta_val * v_val + (1.0 - beta_val) * old_v

        # Update new_state[b, h, i, :]
        new_state_row_ptr = new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_h + i * stride_new_i
        for j in tl.static_range(0, K):
            old_v_j = tl.load(old_state_row_ptr + j * stride_k)
            k_j = tl.load(k_row_ptr + j)
            # new_state[i, j] = old_state[i, j] - old_v + k_j * new_v
            new_val = tl.load(new_state_row_ptr + j * stride_new_j) - old_v + k_j * new_v
            tl.store(new_state_row_ptr + j * stride_new_j, new_val)


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
    stride_b, stride_h, stride_v, stride_k,
    stride_q_b, stride_q_h, stride_q_j,
):
    pid = tl.program_id(axis=0)  # grid over B*H
    b_idx = pid // H
    h_idx = pid % H

    q_row_ptr = q_exp_ptr + b_idx * stride_q_b + h_idx * stride_q_j  # [K]
    new_state_ptr_bh = new_state_ptr + b_idx * stride_b + h_idx * stride_h  # base for (b,h)
    acc = 0.0
    for j in tl.static_range(0, K):
        q_j = tl.load(q_row_ptr + j * stride_q_j)
        sum_row = 0.0
        for i in tl.static_range(0, V):
            row_ptr = new_state_ptr_bh + i * stride_v  # [K]
            for p in tl.static_range(0, K):
                val = tl.load(row_ptr + p * stride_k)
                sum_row += val * tl.load(q_row_ptr + p * stride_q_j)
        acc += q_j * sum_row
    tl.store(out_ptr + b_idx * H + h_idx, acc * scale)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Returns:
        - out: [B, 1, H, 1] in bfloat16
        - new_state: [B, H, V, K] in float32
        """
        device = q.device
        dtype = q.dtype

        # Cast to float32 for computation (Triton kernels assume float32)
        q_f32 = q.squeeze(1).to(torch.float32).contiguous()     # [B,num_q_heads,K]
        k_f32 = k.squeeze(1).to(torch.float32).contiguous()     # [B,num_k_heads,K]
        v_f32 = v.squeeze(1).to(torch.float32).contiguous()     # [B,num_v_heads,V]
        state_f32 = state.to(torch.float32).contiguous()        # [B,num_heads,V,K]

        # Repeat q and k heads to match v heads (ratio is 2 in provided tests)
        ratio = v_f32.shape[1] // q_f32.shape[1]  # 8 // 4 = 2
        q_exp = q_f32.repeat_interleave(ratio, dim=1)  # [B,8,K]
        k_exp = k_f32.repeat_interleave(ratio, dim=1)  # [B,8,K]

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
            A_log.to(torch.float32),
            a.squeeze(1).to(torch.float32),
            dt_bias.to(torch.float32),
            b.squeeze(1).to(torch.float32),
            g_out,
            beta_out,
            H=H,
        )

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Launch Triton state update kernel over (B*H)
        state_update_kernel[grid](
            state_f32, new_state, k_exp, v_f32, beta_out,
            B=B, H=H, V=V, K=K,
            stride_b=state_f32.stride(0), stride_h=state_f32.stride(1), stride_v=state_f32.stride(2), stride_k=state_f32.stride(3),
            stride_new_b=new_state.stride(0), stride_new_h=new_state.stride(1), stride_new_i=new_state.stride(2), stride_new_j=new_state.stride(3),
        )

        # Compute output: out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        out = torch.empty((B, H), dtype=torch.float32, device=device)
        # We need q_exp for this output; it is [B, H, K] after repeat_interleave
        # k_exp is [B, H, K], but q_exp is original q expanded. Build q_exp explicitly as [B,H,K] for output kernel.
        # We repeat q_f32 along head dim to get q_exp with H=num_v_heads
        # In this setup, q_exp[h] == q_f32[b, h % num_q_heads] after repeat. Build it explicitly for Triton:
        q_exp_for_output = torch.empty((B, H, K), dtype=torch.float32, device=device)
        for b in range(B):
            for h in range(H):
                q_exp_for_output[b, h] = q_f32[b, h % q_f32.shape[1]]

        # Launch output dot kernel
        output_dot_kernel[grid](
            q_exp_for_output,
            new_state,
            out,
            float(scale),
            B=B, H=H, V=V, K=K,
            stride_b=state_f32.stride(0), stride_h=state_f32.stride(1), stride_v=state_f32.stride(2), stride_k=state_f32.stride(3),
            stride_q_b=q_exp_for_output.stride(0), stride_q_h=q_exp_for_output.stride(1), stride_q_j=q_exp_for_output.stride(2),
        )

        # Return outputs as requested: [B, 1, H, 1] bfloat16, and [B, H, V, K] float32
        out_bf16 = out.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        # Note: new_state already has shape [B,H,V,K] float32; return it directly.
        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
