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
    B: tl.constexpr,     # runtime but used in loops
    H: tl.constexpr,     # runtime but used in loops
):
    # One program per (b,h)
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H

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
    V: tl.constexpr,     # 128
    K: tl.constexpr,     # 128
    stride_bos, stride_hos, stride_vos, stride_kos,
    stride_kh, stride_kk,
    stride_vh, stride_vk,
    stride_bns, stride_hns, stride_vns, stride_kns,
):
    # 3D grid over (B*H, tiles over V, tiles over K)
    pid_bh = tl.program_id(axis=0)
    pid_vm = tl.program_id(axis=1)
    pid_kn = tl.program_id(axis=2)

    b_idx = pid_bh // H
    h_idx = pid_bh % H

    offs_v = pid_vm * 32 + tl.arange(0, 32)  # tile over V
    offs_k = pid_kn * 32 + tl.arange(0, 32)  # tile over K

    mask_v = offs_v < V
    mask_k = offs_k < K

    # Load k[h,:] and v[h,:] for this head
    k_vec = tl.load(k_ptr + h_idx * stride_kh + offs_k * stride_kk, mask=mask_k, other=0.0)      # [32]
    v_vec = tl.load(v_ptr + h_idx * stride_vh + offs_v * stride_vk, mask=mask_v, other=0.0)      # [32]

    # Load old_state tile [32, 32] from [B,H,V,K]
    old_state_tile = tl.load(
        old_state_ptr + b_idx * stride_bos + h_idx * stride_hos + (offs_v[:, None] * stride_vos) + (offs_k[None, :] * stride_kos),
        mask=mask_v[:, None] & mask_k[None, :],
        other=0.0
    )  # [32,32], float32

    # Compute old_v per row i in tile: sum_j k[j] * old_state[i,j]
    old_v_vec = tl.zeros((32,), dtype=tl.float32)
    for j in tl.static_range(0, K):
        # Load row i across K for each i in tile: [32] by broadcasting over j
        row_j = tl.load(
            old_state_ptr + b_idx * stride_bos + h_idx * stride_hos + j * stride_kos + offs_v * stride_vos,
            mask=mask_v,
            other=0.0
        )  # [32]
        k_j = tl.load(k_ptr + h_idx * stride_kh + j * stride_kk)
        old_v_vec += row_j * k_j

    beta_val = tl.load(beta_ptr + h_idx)

    # Compute new_v per row i in tile: beta * v[i] + (1 - beta) * old_v[i]
    new_v_vec = beta_val * v_vec + (1.0 - beta_val) * old_v_vec  # [32]

    # Compute state_update = dot(k[h,:], new_v_vec) = sum_j k[j] * new_v[j]
    state_update = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(0, K):
        state_update += new_v_vec[j] * tl.load(k_ptr + h_idx * stride_kh + j * stride_kk)

    # Compute new_state tile: old_state - old_v[:,None] + state_update[:,None]
    new_state_tile = old_state_tile - (old_v_vec[:, None]) + (state_update * tl.ones((32, 32), dtype=tl.float32))

    # Store new_state tile
    tl.store(
        new_state_ptr + b_idx * stride_bns + h_idx * stride_hns + (offs_v[:, None] * stride_vns) + (offs_k[None, :] * stride_kns),
        new_state_tile,
        mask=mask_v[:, None] & mask_k[None, :]
    )


@triton.jit
def output_dot_kernel(
    q_exp_ptr,           # [B*H, K] float32
    new_state_ptr,       # [B,H,V,K] float32
    out_ptr,             # [B,1,H,1] float32
    B: tl.constexpr,     # runtime
    H: tl.constexpr,     # runtime (H in kernel corresponds to num_heads, but here only b is used)
    V: tl.constexpr,     # 128
    K: tl.constexpr,     # 128
    stride_qb, stride_qk,
    stride_bns, stride_hns, stride_vns, stride_kns,
    scale: tl.float32,
):
    # One program per (b,h)
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # Load q_exp[b,h,:] row
    q_row = tl.load(q_exp_ptr + b_idx * stride_qb + h_idx * stride_qk)  # [128]

    # Accumulate dot with new_state[b,h] which is [V,K]
    acc = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(0, V):
        row_ptr = new_state_ptr + b_idx * stride_bns + h_idx * stride_hns + i * stride_vns
        new_row = tl.zeros((K,), dtype=tl.float32)
        for j in tl.static_range(0, K):
            new_row[j] = tl.load(row_ptr + j * stride_kns)
        acc += tl.sum(new_row * q_row)

    # Store into out[b,0,h,0] = scale * acc
    tl.store(out_ptr + b_idx * (1 * H * 1) + h_idx * (1 * 1) + 0, acc * scale)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device and dtype; compute in float32
        device = q.device
        q_f32 = q.to(torch.float32).squeeze(1).contiguous()    # [B,num_q_heads,K]
        k_f32 = k.to(torch.float32).squeeze(1).contiguous()    # [B,num_k_heads,K]
        v_f32 = v.to(torch.float32).squeeze(1).contiguous()    # [B,num_v_heads,V]
        state_f32 = state.to(torch.float32).contiguous()       # [B,num_heads,V,K]

        # Extract shapes
        B_q, _, K = q_f32.shape
        _, _, Kk = k_f32.shape
        _, _, V = v_f32.shape
        B_s, _, V_s, K_s = state_f32.shape
        assert B_q == B_s and K == Kk and K_s == 128 and V == 128 and V_s == 8, "Shapes must match the fixed configuration"

        # Compute repeat_interleave for q and k to align with num_v_heads
        # Here num_q_heads=4, num_k_heads=4, num_v_heads=8 => repeat=2
        repeat_q = 2  # 8 // 4
        repeat_k = 2  # 8 // 4
        q_exp = q_f32.repeat_interleave(repeat_q, dim=1)  # [B,8,K]
        k_exp = k_f32.repeat_interleave(repeat_k, dim=1)  # [B,8,K]

        # Compute g and beta via Triton kernel
        H = 8
        g_out = torch.empty((B, H), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=device)

        grid_ab = (B * H,)
        softplus_exp_sigmoid_kernel[grid_ab](
            A_log.to(torch.float32),         # [H]
            a.to(torch.float32).squeeze(1),  # [B,H]
            dt_bias.to(torch.float32),       # [H]
            b.to(torch.float32).squeeze(1),  # [B,H]
            g_out,                           # [B,H]
            beta_out,                        # [B,H]
            B=B, H=H,
            num_warps=1
        )

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Launch Triton state update kernel with 3D grid over (B*H, tiles over V, tiles over K)
        grid_state = (B * H, 4, 4)  # tiles: V=128 -> 4*32, K=128 -> 4*32
        state_update_kernel[grid_state](
            state_f32,            # [B,H,V,K]
            k_exp,                # [B,H,K]
            v_f32,                # [B,H,V]
            beta_out,             # [B,H]
            new_state,            # [B,H,V,K]
            B=B, H=H, V=V, K=K,
            stride_bos=B*H*V*K, stride_hos=V*K, stride_vos=K, stride_kos=1,
            stride_kh=H*K, stride_kk=1,
            stride_vh=H*V, stride_vk=1,
            stride_bns=B*H*V*K, stride_hns=V*K, stride_vns=K, stride_kns=1,
            num_warps=1
        )

        # Compute output per (b,h) via Triton kernel
        out = torch.empty((B, 1, H, 1), dtype=torch.float32, device=device)
        grid_out = (B * H,)
        output_dot_kernel[grid_out](
            q_exp,                # [B*H,K]
            new_state,            # [B,H,V,K]
            out,                  # [B,1,H,1]
            B=B, H=H, V=V, K=K,
            stride_qb=B*H*K, stride_qk=1,
            stride_bns=B*H*V*K, stride_hns=V*K, stride_vns=K, stride_kns=1,
            scale=scale,
            num_warps=1
        )

        # Cast output to bfloat16 as required by original: [B,1,H,V] -> [B,1,H,1]
        out_bf16 = out.to(torch.bfloat16)

        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
