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
    B: tl.constexpr,     # batch count (for shape calculations)
    H: tl.constexpr,     # head count
):
    # 1D launch over B*H
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    db_val = tl.load(dt_bias_ptr + h_idx)
    A_log_val = tl.load(A_log_ptr + h_idx)
    b_val = tl.load(b_ptr + b_idx * H + h_idx)

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
    B: tl.constexpr,     # batch count
    H: tl.constexpr,     # head count
    V: tl.constexpr,     # 128
    K: tl.constexpr,     # 128
):
    # 1D grid over B*H
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # Compute scalar old_v = k[h] @ old_state[b,h]
    old_v = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(0, K):
        row_j = tl.load(old_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + j)  # [V]
        k_j = tl.load(k_ptr + h_idx * K + j)  # scalar
        old_v += k_j * tl.sum(row_j)

    # beta for this head
    beta_val = tl.load(beta_ptr + h_idx)

    # Compute scalar new_v_scalar = k[h] @ (beta * v + (1-beta) * old_v)
    new_v_scalar = beta_val * (1.0 - beta_val) * old_v
    for j in tl.static_range(0, K):
        k_j = tl.load(k_ptr + h_idx * K + j)
        v_j = tl.load(v_ptr + h_idx * V + j)  # scalar
        new_v_scalar += k_j * (beta_val * v_j + new_v_scalar)

    # Update new_state = old_state - old_v[:,None] + new_v_scalar[:,None]
    for i in tl.static_range(0, V):
        for j in tl.static_range(0, K):
            old_ij = tl.load(old_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j)
            remove_ij = old_v
            update_ij = new_v_scalar
            new_ij = old_ij - remove_ij + update_ij
            tl.store(new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j, new_ij)


@triton.jit
def output_dot_kernel(
    q_ptr,               # [H,K] float32 (q_exp for each head)
    new_state_ptr,       # [B,H,V,K] float32
    out_ptr,             # [B,1,H,1] float32 (we write out[b,0,h,0])
    B: tl.constexpr,     # batch count
    H: tl.constexpr,     # head count (8 in our setup)
    V: tl.constexpr,     # 128
    K: tl.constexpr,     # 128
):
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    acc = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(0, K):
        q_j = tl.load(q_ptr + h_idx * K + j)  # scalar
        sum_col_j = tl.zeros((), dtype=tl.float32)
        for i in tl.static_range(0, V):
            sum_col_j += tl.load(new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j)
        acc += q_j * sum_col_j

    # write to out[b,0,h,0]
    tl.store(out_ptr + b_idx * (1 * H + 1) + h_idx * 1, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward:
        - Compute g and beta in Triton.
        - Update state in Triton.
        - Compute output scalar per (b,h) in Triton.
        - Return output as [B,1,H,1] cast to bfloat16 and new_state as [B,H,V,K] (float32).
        """
        device = q.device

        # Shapes
        B, _, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        H = state.shape[1]  # num_heads, expected 8 in provided setup

        # Cast inputs to float32 for compute
        q_f32 = q.float()
        k_f32 = k.float()
        v_f32 = v.float()
        state_f32 = state.float()

        # Repeat q and k to align with num_v_heads
        repeat_q = num_v_heads // num_q_heads
        repeat_k = num_v_heads // num_k_heads
        q_exp = q_f32.repeat_interleave(repeat_q, dim=1)  # [B, 8, K]
        k_exp = k_f32.repeat_interleave(repeat_k, dim=1)  # [B, 8, K]

        # Prepare g and beta tensors [B,H]
        g_out = torch.empty(B * H, dtype=torch.float32, device=device)
        beta_out = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        grid_g = (B * H,)
        softplus_exp_sigmoid_kernel[grid_g](
            A_log.float(),
            a.float().squeeze(1),  # [B,H]
            dt_bias.float(),
            b.float().squeeze(1),  # [B,H]
            g_out,                 # [B,H]
            beta_out,              # [B,H]
            B=B, H=H
        )

        # Allocate output and new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)
        out = torch.empty((B, 1, H, 1), dtype=torch.float32, device=device)

        # Launch Triton kernel to update state
        grid_state = (B * H,)
        state_update_kernel[grid_state](
            state_f32,          # [B,H,V,K]
            k_f32.squeeze(1),   # [H,K]
            v_f32.squeeze(1),   # [H,V]
            beta_out,           # [B,H]
            new_state,          # [B,H,V,K]
            B=B, H=H, V=V, K=K
        )

        # Launch Triton kernel to compute output scalars per (b,h)
        grid_out = (B * H,)
        # Pass q_exp as [B*8,K] to the kernel
        q_exp_flat = q_exp.reshape(B * 8, K)  # [B*8, K]
        output_dot_kernel[grid_out](
            q_exp_flat,         # [B*8,K]
            new_state,          # [B,H,V,K]
            out,                # [B,1,H,1]
            B=B, H=8, V=128, K=128
        )

        # Cast output to bfloat16 and return [B,1,H,1], new_state [B,H,V,K]
        out_bf16 = out.to(torch.bfloat16)

        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
