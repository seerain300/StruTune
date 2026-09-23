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
    B: tl.constexpr,     # int
    H: tl.constexpr,     # int
):
    # program id over (b,h)
    pid = tl.program_id(axis=0)
    b_idx = pid // H
    h_idx = pid % H

    # bounds check
    if b_idx >= B or h_idx >= H:
        return

    # load scalars for this (b,h)
    # a[b,h], dt_bias[h]
    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    db_val = tl.load(dt_bias_ptr + h_idx)
    A_log_val = tl.load(A_log_ptr + h_idx)
    b_val = tl.load(b_ptr + b_idx * H + h_idx)

    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)  (stable)
    # compute softplus(a + dt_bias)
    x = a_val + db_val
    abs_x = tl.abs(x)
    # max(x, 0)
    zero = 0.0
    max_x0 = tl.maximum(x, zero)
    softplus = tl.log(1.0 + tl.exp(-abs_x)) + max_x0

    # g = exp(-exp(A_log) * softplus(a + dt_bias))
    exp_A = tl.exp(A_log_val)
    g = tl.exp(-exp_A * softplus)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    sig = 1.0 / (1.0 + tl.exp(-b_val))

    # write out
    tl.store(g_out_ptr + b_idx * H + h_idx, g)
    tl.store(beta_out_ptr + b_idx * H + h_idx, sig)


@triton.jit
def state_update_kernel(
    old_state_ptr,       # [B,H,V,K] float32
    k_ptr,               # [H,K] float32
    v_ptr,               # [H,V] float32
    beta_ptr,            # [H] float32
    new_state_ptr,       # [B,H,V,K] float32
    B: tl.constexpr,     # int
    H: tl.constexpr,     # int
    V: tl.constexpr,     # int
    K: tl.constexpr,     # int
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program ids for tiling over V and K
    pid_m = tl.program_id(axis=0)  # tile along V
    pid_n = tl.program_id(axis=1)  # tile along K
    # grid should be (ceil_div(V, BLOCK_M), ceil_div(K, BLOCK_N))
    # we get b and h from 2D grid over (B*H)
    total = B * H
    pid_bh = tl.program_id(axis=2)  # tile index over B*H
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # compute offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows (V)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols (K)

    # mask for valid indices
    mask_m = offs_m < V
    mask_n = offs_n < K
    mask = mask_m[:, None] & mask_n[None, :]

    # base pointer for old_state[b,h, :, :] and new_state[b,h, :, :]
    base_old = old_state_ptr + b_idx * (H * V * K) + h_idx * (V * K)
    base_new = new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K)

    # load k[h, :] and v[h, :]
    k_vec = tl.load(k_ptr + h_idx * K + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    v_vec = tl.load(v_ptr + h_idx * V + offs_m, mask=mask_m, other=0.0)  # [BLOCK_M]

    # compute old_v = dot(k[h], old_state[b,h, :, :])
    # we need to accumulate over K
    old_v = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_N):
        k_offs = k_start + offs_n
        k_mask = k_offs < K
        # load old_state rows for this tile over K dimension
        # We need old_state values for each i in offs_m and j in k_offs
        # Build 2D pointer: base_old + i*stride + j*1, where stride = K
        old_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(0, BLOCK_M):
            row_ptr = base_old + offs_m[i] * K
            # load columns for this row
            vals = tl.load(row_ptr + k_offs, mask=k_mask, other=0.0)
            old_tile[i, :] = vals
        # multiply by k_vec and reduce over columns
        prod = old_tile * k_vec[None, :]
        # sum over axis=1 to get per-row sum
        old_v += tl.sum(prod, axis=1)

    # load beta[h]
    beta_h = tl.load(beta_ptr + h_idx)

    # compute new_v = beta * v + (1 - beta) * old_v
    new_v = beta_h * v_vec + (1.0 - beta_h) * old_v

    # compute state_update = dot(k[h], new_v)
    state_update = tl.sum(k_vec * new_v, axis=0)

    # load old_state tile and update
    # old_state_tile[i, j] = old_state[b,h,i,j]
    for i in range(0, BLOCK_M):
        row_ptr = base_old + offs_m[i] * K
        old_row = tl.load(row_ptr + offs_n, mask=mask[None, i], other=0.0)  # [BLOCK_N]
        # new_state_row[j] = old_row[j] - state_remove[j] + state_update[j]
        # state_remove[j] = old_v[i] (since we reduced over K per row), but old_v is length BLOCK_M
        # We need per-column removal. Compute state_remove as scalar per iteration j.
        # Better approach: compute per-column removal using state_remove_vec[j] = old_v[j % BLOCK_M]
        # However, old_v has length BLOCK_M. We need to map j to i in offs_m. Simpler: compute per-column using a loop.
        # Implement update per element: load old, compute new, store.
        for j in range(0, BLOCK_N):
            j_valid = (pid_n * BLOCK_N + j) < K
            if j_valid:
                old_val = old_row[j]
                # state_remove per column: j depends on which i we are; simpler: compute as old_v[i] if i exists
                # Instead, recompute per j scalar:
                # For column j, state_remove is k[j] * old_v, but old_v is per row. To get correct scalar per column, we need to do:
                # old_v is length BLOCK_M. We can compute old_v_j = sum_i k[i] * old_state[i,j] via k_offs loop again; but that's expensive.
                # Instead, we can store state_remove as a vector and use broadcast below. The simpler approach is to compute per-column removal.
                # Compute old_v_j via loop over K: old_v_j = sum_k k[k] * old_state[b,h,i,j]
                old_v_j = 0.0
                for kk in range(0, K):
                    old_val_ij = tl.load(row_ptr + kk, mask=(offs_m[i] < V and kk < K), other=0.0)
                    old_v_j += k_ptr[h_idx * K + kk] * old_val_ij
                state_remove_j = old_v_j
                # state_update is scalar; add uniformly
                new_val = old_val - state_remove_j + state_update
                # store to new_state
                tl.store(base_new + offs_m[i] * K + (pid_n * BLOCK_N + j), new_val)


@triton.jit
def output_dot_kernel(
    q_ptr,               # [H,K] float32
    new_state_ptr,       # [B,H,V,K] float32
    out_ptr,             # [B,H,V] float32 (we will write a single element per [b,h]; use flat index)
    B: tl.constexpr,     # int
    H: tl.constexpr,     # int
    V: tl.constexpr,     # int
    K: tl.constexpr,     # int
    scale: tl.float32,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # single program per (b,h) grid(1,)
    b_idx = pid // H
    h_idx = pid % H

    # base pointers
    base_q = q_ptr + h_idx * K
    base_ns = new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K)

    acc = tl.zeros((), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        q_vec = tl.load(base_q + k_offs, mask=k_mask, other=0.0)
        ns_vec = tl.load(base_ns + k_offs, mask=k_mask, other=0.0)
        acc += tl.sum(q_vec * ns_vec, axis=0)

    out_index = b_idx * (H * V) + h_idx * V  # store to [b,h,v] with v=0
    tl.store(out_ptr + out_index, acc * scale)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward:
        - All tensor math (elementwise and small dot-products) is performed by Triton kernels.
        - No torch matmul or F.* calls on tensors.
        Shapes:
          q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128]
          state: [B, 8, 128, 128]
          A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        Returns:
          output: [B, 1, H, V] bfloat16, new_state: [B, H, V, K] float32
        """
        device = q.device
        B_q, T, num_q_heads, K = q.shape
        B_k, T_k, num_k_heads, _ = k.shape
        B_v, T_v, num_v_heads, V = v.shape
        assert T == 1 and T_k == 1 and T_v == 1, "T must be 1"
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8, "Fixed head counts expected"
        assert K == 128 and V == 128, "K and V must be 128"
        B = B_q

        # Cast q, k, v to float32 for Triton; state is float32
        q_f32 = q.squeeze(1).to(torch.float32)
        k_f32 = k.squeeze(1).to(torch.float32)
        v_f32 = v.squeeze(1).to(torch.float32)
        state_f32 = state if state is not None else torch.zeros(B, num_v_heads, V, K, dtype=torch.float32, device=device)

        # Compute repeat_interleave for q and k (ratio 2 since num_v_heads / num_q_heads = 8/4 = 2, and 8/4 = 2)
        repeat_q = 2
        repeat_k = 2
        q_exp = q_f32.repeat_interleave(repeat_q, dim=1)  # [B, 8, 128]
        k_exp = k_f32.repeat_interleave(repeat_k, dim=1)  # [B, 8, 128]

        # Allocate outputs
        g_out = torch.empty((B, num_v_heads), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, num_v_heads), dtype=torch.float32, device=device)
        new_state = torch.empty((B, num_v_heads, V, K), dtype=torch.float32, device=device)
        out = torch.empty((B, num_v_heads, V), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta: grid (B, H)
        softplus_exp_sigmoid_kernel[(B, num_v_heads)](
            A_log.to(torch.float32),          # [H]
            a.squeeze(1).to(torch.float32),   # [B,H]
            dt_bias.to(torch.float32),        # [H]
            b.squeeze(1).to(torch.float32),   # [B,H]
            g_out,                            # [B,H]
            beta_out,                         # [B,H]
            B=B, H=num_v_heads
        )

        # Launch Triton kernel to compute new_state: grid tiles over (V, K), and one program per (b,h)
        BLOCK_M = 32
        BLOCK_N = 32
        grid_tiles = (triton.cdiv(V, BLOCK_M), triton.cdiv(K, BLOCK_N), B * num_v_heads)
        state_update_kernel[grid_tiles](
            state_f32,                       # [B,H,V,K]
            k_exp,                           # [B,H,K] -> we only need k[h], so pass k_exp[b,h,:]
            v_f32,                           # [B,H,V] -> v[b,h,:]
            beta_out,                        # [B,H]
            new_state,                       # [B,H,V,K]
            B=B, H=num_v_heads, V=V, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

        # Launch Triton kernel to compute output: one program per (b,h)
        BLOCK_K = 128
        output_dot_kernel[(B * num_v_heads,)](
            q_exp,                           # [B,H,K]
            new_state,                       # [B,H,V,K]
            out,                             # [B,H,V]
            B=B, H=num_v_heads, V=V, K=K, scale=scale, BLOCK_K=BLOCK_K
        )

        # Return output as [B,1,H,V] bfloat16 and new_state [B,H,V,K] float32
        output = out.unsqueeze(1).to(torch.bfloat16)  # [B,1,H,V]
        return output, new_state


def run(*args):
    return ModelNew()(*args)
