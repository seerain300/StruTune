import torch
import math
import triton
import triton.language as tl


@triton.jit
def g_beta_kernel(
    A_log_ptr,          # [H] float32
    a_ptr,              # [B,H] float32
    dt_bias_ptr,        # [H] float32
    b_ptr,              # [B,H] float32
    g_out_ptr,          # [B,H] float32
    beta_out_ptr,       # [B,H] float32
    B: tl.constexpr,    # runtime
    H: tl.constexpr     # runtime
):
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # Load a[b,h], dt_bias[h], A_log[h]
    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    dtb_val = tl.load(dt_bias_ptr + h_idx)
    A_val = tl.load(A_log_ptr + h_idx)

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    x = a_val + dtb_val
    absx = tl.abs(x)
    sp = tl.log(1.0 + tl.exp(-absx)) + tl.maximum(x, 0.0)

    # g = exp(-exp(A) * softplus(x))
    g = tl.exp(-tl.exp(A_val) * sp)

    # beta = sigmoid(b[b,h]) = 1 / (1 + exp(-b))
    bval = tl.load(b_ptr + b_idx * H + h_idx)
    beta = 1.0 / (1.0 + tl.exp(-bval))

    # store
    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, beta)


@triton.jit
def state_update_kernel(
    old_state_ptr,      # [B,H,V,K] float32
    k_ptr,              # [H,K] float32
    v_ptr,              # [H,V] float32
    beta_ptr,           # [B,H] float32
    new_state_ptr,      # [B,H,V,K] float32
    B: tl.constexpr,    # runtime
    H: tl.constexpr,    # runtime
    V: tl.constexpr,    # 128
    K: tl.constexpr     # 128
):
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # Load beta for this (b,h)
    beta_val = tl.load(beta_ptr + b_idx * H + h_idx)

    # Loop over rows i in V (update each row of old_state to new_state)
    for i in tl.static_range(0, V):
        # Compute dot(old_state[b,h,i,:], k[h,:]) = old_v
        old_v = tl.zeros((), dtype=tl.float32)
        for j in tl.static_range(0, K):
            s_val = tl.load(old_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j)
            k_val = tl.load(k_ptr + h_idx * K + j)
            old_v += s_val * k_val

        # Compute new_v = beta * v[h,i] + (1 - beta) * old_v
        v_val = tl.load(v_ptr + h_idx * V + i)
        new_v = beta_val * v_val + (1.0 - beta_val) * old_v

        # Update new_state[b,h,i,:] = old_state - old_v[:,None] + new_v[:,None]
        for j in tl.static_range(0, K):
            old_s_val = tl.load(old_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j)
            k_j = tl.load(k_ptr + h_idx * K + j)

            # state_remove contribution from this j: old_v * k_j
            state_remove = old_v * k_j

            # state_update contribution: new_v * k_j
            state_update = new_v * k_j

            new_elem = old_s_val - state_remove + state_update
            tl.store(new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j, new_elem)


@triton.jit
def output_dot_kernel(
    q_exp_ptr,          # [B*H, K] float32
    new_state_ptr,      # [B,H,V,K] float32
    out_ptr,            # [B*H] float32
    B: tl.constexpr,    # runtime
    H: tl.constexpr,    # runtime
    V: tl.constexpr,    # 128
    K: tl.constexpr,    # 128
    stride_qb, stride_qk
):
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # Load q_exp[b,h,:] row
    q_row = tl.load(q_exp_ptr + b_idx * stride_qb + h_idx * stride_qk)  # [K]

    # Compute dot(q_exp[h], new_state[b,h]) over V rows
    acc = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(0, V):
        row_ptr = new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K
        new_row = tl.zeros((K,), dtype=tl.float32)
        for j in tl.static_range(0, K):
            new_row[j] = tl.load(row_ptr + j)
        acc += tl.sum(new_row * q_row)

    tl.store(out_ptr + pid_bh, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Shapes: q: [B,1,4,K], k: [B,1,4,K], v: [B,1,8,V], state: [B,8,V,K]
        device = q.device

        # Cast and make contiguous for Triton
        q_f32 = q.squeeze(1).to(torch.float32).contiguous()   # [B,4,K]
        k_f32 = k.squeeze(1).to(torch.float32).contiguous()   # [B,4,K]
        v_f32 = v.squeeze(1).to(torch.float32).contiguous()   # [B,8,V]
        state_f32 = state.to(torch.float32).contiguous()      # [B,8,V,K]
        A_log_f32 = A_log.to(torch.float32).contiguous()      # [H]
        a_f32 = a.squeeze(1).to(torch.float32).contiguous()   # [B,H]
        dt_bias_f32 = dt_bias.to(torch.float32).contiguous()  # [H]
        b_f32 = b.squeeze(1).to(torch.float32).contiguous()   # [B,H]

        # H is number of heads in v => 8
        H = v_f32.shape[1]   # 8
        B = q_f32.shape[0]
        K = q_f32.shape[-1]  # 128
        V = v_f32.shape[-1]  # 128

        # Expand q and k heads by repeat_interleave (ratio 2)
        q_exp = q_f32.repeat_interleave(H // 4, dim=1)   # [B,8,K]
        k_exp = k_f32.repeat_interleave(H // 4, dim=1)   # [B,8,K]

        # Allocate outputs
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)       # [B,H]
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)    # [B,H]

        # Launch Triton kernel to compute g and beta
        grid1 = (B * H,)
        g_beta_kernel[grid1](
            A_log_f32,
            a_f32,
            dt_bias_f32,
            b_f32,
            g_out,
            beta_out,
            B=B, H=H
        )

        # Initialize new_state [B,H,V,K]
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Launch Triton kernel to update new_state for each (b,h)
        grid2 = (B * H,)
        state_update_kernel[grid2](
            state_f32,          # [B,H,V,K]
            k_exp,              # [B,8,K]
            v_f32,              # [B,8,V]
            beta_out,           # [B,H]
            new_state,          # [B,H,V,K]
            B=B, H=H, V=V, K=K
        )

        # Compute output per (b,h): out[b,h] = scale * (q_exp[h] @ new_state[b,h])
        q_exp_flat = q_exp.reshape(B * H, K).contiguous()  # [B*H, K]
        out = torch.empty((B * H), dtype=torch.float32, device=device)

        grid3 = (B * H,)
        output_dot_kernel[grid3](
            q_exp_flat,         # [B*H,K]
            new_state,          # [B,H,V,K]
            out,                # [B*H]
            B=B, H=H, V=V, K=K,
            stride_qb=1, stride_qk=K
        )

        # Return output as [B,1,H,1] in bfloat16 and new_state [B,H,V,K] in float32
        out_bf16 = out.unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)  # [B,1,H,1]
        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
