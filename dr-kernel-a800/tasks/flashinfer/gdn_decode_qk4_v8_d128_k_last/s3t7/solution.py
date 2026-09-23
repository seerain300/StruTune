import math
import torch
import triton
import triton.language as tl


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
    stride_bos, stride_hos, stride_ious, stride_jos,   # strides for old_state
    stride_bns, stride_hns, stride_ious_new, stride_jns,  # strides for new_state
    BLOCK_M: tl.constexpr,  # tile size over V, e.g., 32
    BLOCK_N: tl.constexpr,  # tile size over K, e.g., 32
):
    # 3D grid: axis 0 over B*H, axis 1 over tiles of V, axis 2 over tiles of K
    pid_bh = tl.program_id(axis=0)
    pid_v = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    b_idx = pid_bh // H
    h_idx = pid_bh % H

    offs_m = pid_v * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in V
    offs_n = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in K

    mask_m = offs_m < V
    mask_n = offs_n < K

    # Load k[h,:] and v[h,:]
    k_vec = tl.load(k_ptr + h_idx * K + offs_n, mask=mask_n, other=0.0)  # [K]
    v_vec = tl.load(v_ptr + h_idx * V + offs_m, mask=mask_m, other=0.0)  # [V]

    # Compute old_v per row i in tile: sum_j k[j] * old_state[i,j]
    old_v = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for j in tl.static_range(0, K):
        row_j = tl.load(
            old_state_ptr + b_idx * stride_bos + h_idx * stride_hos + (offs_m[:, None] * stride_ious) + j * stride_jos,
            mask=mask_m[:, None],
            other=0.0
        )  # [BLOCK_M, 1]
        old_v += tl.sum(row_j, axis=1)  # sum across the [1] axis

    # beta for head h
    beta_val = tl.load(beta_ptr + h_idx)

    # Compute new_v per row i in tile: beta * v + (1 - beta) * old_v
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [BLOCK_M]

    # Compute state_update per column j: sum_i k[j] * new_v[i]
    state_update = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in tl.static_range(0, BLOCK_M):
        if mask_m[i]:
            new_v_i = new_v[i]  # scalar
            for j2 in tl.static_range(0, K):
                state_update += k_vec[j2] * new_v_i

    # Compute new_state = old_state - state_remove[:,None] + state_update[:,None]
    # state_remove = old_v (scalar) for each row; state_update is per column.
    for i in tl.static_range(0, BLOCK_M):
        if mask_m[i]:
            row_old = tl.load(
                old_state_ptr + b_idx * stride_bos + h_idx * stride_hos + (offs_m[i] * stride_ious) + (offs_n[None, :] * stride_jos),
                mask=mask_n[None, :],
                other=0.0
            )  # [1, BLOCK_N]
            row_new = row_old - old_v[i] + state_update[None, :]  # [1, BLOCK_N]
            tl.store(
                new_state_ptr + b_idx * stride_bns + h_idx * stride_hns + (offs_m[i] * stride_ious_new) + (offs_n[None, :] * stride_jns),
                row_new,
                mask=mask_n[None, :]
            )


@triton.jit
def output_dot_kernel(
    q_exp_ptr,           # [B*H,K] float32
    new_state_ptr,       # [B,H,V,K] float32
    out_ptr,             # [B,1,H,V] float32
    B: tl.constexpr,     # runtime
    H: tl.constexpr,     # runtime
    V: tl.constexpr,     # 128
    K: tl.constexpr,     # 128
    stride_qb, stride_qh, stride_qk,
    stride_bns, stride_hns, stride_ious_new, stride_jns,
    stride_bout, stride_hout, stride_vout, stride_kout,  # strides for out (we use V dimension; K is implicit)
):
    # Each program handles one (b,h)
    pid_bh = tl.program_id(axis=0)
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # Load q_exp[b,h,:] row
    q_row = tl.load(q_exp_ptr + b_idx * stride_qb + h_idx * stride_qh + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)  # [K]

    # Compute dot over j=0 column of new_state[b,h,:,0] (i.e., sum over V)
    dot_val = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(0, K):
        # sum over V of new_state[b,h,i,j]
        sum_V = tl.zeros((), dtype=tl.float32)
        for i in tl.static_range(0, V):
            val = tl.load(new_state_ptr + b_idx * stride_bns + h_idx * stride_hns + i * stride_ious_new + j * stride_jns)
            sum_V += val
        dot_val += sum_V * q_row[j]

    # Store scalar output[b,0,h,V] for all V entries (repeat scalar across V)
    base_out = b_idx * stride_bout + h_idx * stride_hout  # corresponds to out[b,0,h,:]
    for v_idx in tl.static_range(0, V):
        tl.store(out_ptr + base_out + v_idx * stride_vout, dot_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Inputs per provided tests:
        # q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
        device = q.device

        # Cast inputs to float32 for compute
        q_f32 = q.to(torch.float32).squeeze(1).contiguous()    # [B,4,128]
        k_f32 = k.to(torch.float32).squeeze(1).contiguous()    # [B,4,128]
        v_f32 = v.to(torch.float32).squeeze(1).contiguous()    # [B,8,128]
        state_f32 = state.to(torch.float32).contiguous()       # [B,8,128,128]

        # Repeat q and k heads to align with num_v_heads (ratio 2 for provided tests)
        B, Hq, Kdim = q_f32.shape
        assert Hq in (4, 8), "num_q_heads must be 4 or 8"
        assert Kdim == 128, "K must be 128"
        repeat_ratio = 8 // Hq  # 2 for Hq=4, 1 for Hq=8
        q_exp = q_f32.repeat_interleave(repeat_ratio, dim=1)   # [B,8,128]
        k_exp = k_f32.repeat_interleave(repeat_ratio, dim=1)   # [B,8,128]
        H = q_exp.shape[1]
        V = 128

        # Precompute g and beta scalars per (b,h) using torch
        a_flat = a.squeeze(1).to(torch.float32)   # [B,H]
        b_flat = b.squeeze(1).to(torch.float32)   # [B,H]
        A_log = A_log.to(torch.float32)           # [H]
        dt_bias = dt_bias.to(torch.float32)       # [H]

        # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
        x = a_flat + dt_bias                       # [B,H]
        softplus = torch.log1p(torch.exp(-torch.abs(x))) + torch.maximum(x, torch.tensor(0.0, device=device))
        g = torch.exp(-torch.exp(A_log) * softplus)     # [B,H]
        beta = torch.sigmoid(b_flat)                   # [B,H]

        # Allocate outputs
        new_state = torch.empty((B, H, V, Kdim), dtype=torch.float32, device=device)  # [B,H,V,K]
        output = torch.empty((B, 1, H, V), dtype=torch.float32, device=device)        # [B,1,H,V]

        # Launch Triton kernel to update state
        BLOCK_M = 32
        BLOCK_N = 32
        grid_state = (B * H, triton.cdiv(V, BLOCK_M), triton.cdiv(Kdim, BLOCK_N))
        # Strides for old_state
        stride_bos = state_f32.stride(0)
        stride_hos = state_f32.stride(1)
        stride_ious = state_f32.stride(2)
        stride_jos = state_f32.stride(3)
        # Strides for new_state
        stride_bns = new_state.stride(0)
        stride_hns = new_state.stride(1)
        stride_ious_new = new_state.stride(2)
        stride_jns = new_state.stride(3)

        state_update_kernel[grid_state](
            state_f32,            # old_state
            k_f32,                # k (H,K)
            v_f32,                # v (H,V)
            beta.to(torch.float32).squeeze(1),  # beta [B,H]
            new_state,            # new_state
            B=B, H=H, V=V, K=Kdim,
            stride_bos=stride_bos, stride_hos=stride_hos, stride_ious=stride_ious, stride_jos=stride_jos,
            stride_bns=stride_bns, stride_hns=stride_hns, stride_ious_new=stride_ious_new, stride_jns=stride_jns,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # Launch Triton kernel to compute output per (b,h) and store scalar at all V positions
        q_exp_flat = q_exp.reshape(B * H, Kdim).contiguous()  # [B*H, 128]
        out_strides = output.stride()
        stride_bout = out_strides[0]
        stride_hout = out_strides[2]
        stride_vout = out_strides[3]
        grid_out = (B * H,)

        output_dot_kernel[grid_out](
            q_exp_flat,           # [B*H, K]
            new_state,            # [B,H,V,K]
            output,               # [B,1,H,V]
            B=B, H=H, V=V, K=Kdim,
            stride_qb=q_exp_flat.stride(0), stride_qh=0, stride_qk=q_exp_flat.stride(1),
            stride_bns=new_state.stride(0), stride_hns=new_state.stride(1), stride_ious_new=new_state.stride(2), stride_jns=new_state.stride(3),
            stride_bout=stride_bout, stride_hout=stride_hout, stride_vout=stride_vout, stride_kout=0,
        )

        # Cast output to bfloat16 as required by original signature
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
