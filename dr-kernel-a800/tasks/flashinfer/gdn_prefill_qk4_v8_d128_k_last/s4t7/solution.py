import torch
import triton
import triton.language as tl


@triton.jit
def softplus_torch_like(x_ptr, out_ptr, N: tl.constexpr):
    # x_ptr: [N] flat input, out_ptr: [N] flat output
    # softplus(x) = log(1 + exp(x))
    i = tl.program_id(0)
    x = tl.load(x_ptr + i)
    # Numerically stable softplus
    zero = 0.0
    sp = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, sp)


@triton.jit
def g_kernel(a_ptr, dt_ptr, A_ptr, out_ptr, N: tl.constexpr, A_len: tl.constexpr):
    # a_ptr: [N], dt_ptr: [A_len], A_ptr: [A_len], out_ptr: [N]
    # g = exp(-exp(A) * softplus(a + dt))
    i = tl.program_id(0)
    a = tl.load(a_ptr + i)
    dt = tl.load(dt_ptr + (i % A_len))
    # Note: a and dt are scalars per i. Triton supports broadcasting of scalars.
    g_val = tl.exp(-tl.exp(A_ptr[i // A_len]) * tl.log(1.0 + tl.exp(a + dt)))
    tl.store(out_ptr + i, g_val)


@triton.jit
def sigmoid_kernel(b_ptr, out_ptr, N: tl.constexpr):
    i = tl.program_id(0)
    b = tl.load(b_ptr + i)
    sig = 1.0 / (1.0 + tl.exp(-b))
    tl.store(out_ptr + i, sig)


@triton.jit
def repeat2_expand(x_ptr, out_ptr, L: tl.constexpr, D: tl.constexpr, factor: tl.constexpr):
    # Expand q/k along dim=1 by repeat_interleave(2): x is [L, 4, D] -> out [L, 8, D]
    pid = tl.program_id(0)  # iterate over L*D*factor
    idx = pid
    base = L * 4 * D
    if idx < base:
        l = idx // (4 * D)
        rem = idx % (4 * D)
        head = rem // D
        d = rem % D
        val = tl.load(x_ptr + l * 4 * D + head * D + d)
        out_h = head * factor + (head % 2)  # mapping to expanded head index
        tl.store(out_ptr + l * 8 * D + out_h * D + d, val)


@triton.jit
def gemv_kernel(q_ptr, state_ptr, out_ptr, scale, L: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    # Compute out[t, h, :] = scale * q_exp[t, h, :] @ state_new[h, :, :]
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    if t < L and h < H:
        q_vec = tl.load(q_ptr + t * H * D + h * D + tl.arange(0, D))
        # state_ptr points to state_new[h] which is [D, D]
        state_mat = tl.load(state_ptr + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :], mask=tl.arange(0, D)[:, None] < D and tl.arange(0, D)[None, :] < D, other=0.0)
        # acc: [D]
        acc = tl.zeros((D,), dtype=tl.float32)
        # Loop over K dimension. Since q_vec is [D], and state_mat is [D, D], we can compute dot via sum over axis=1.
        # But Triton needs explicit loop: we'll treat D as 128 and loop.
        for j in range(D):
            vec = state_mat[j, :]
            acc += q_vec[j] * vec
        out_vec = acc * scale
        # Write out_vec to out_ptr[t, h, :]
        # out_ptr is [L, H, D] flattened as L*H*D
        out_off = t * H * D + h * D
        for j in range(D):
            tl.store(out_ptr + out_off + j, out_vec[j])


@triton.jit
def update_state_kernel(state_old_ptr, state_new_ptr, k_ptr, v_ptr, beta_ptr, g_ptr,
                         seq_idx: tl.constexpr, t: tl.constexpr, h: tl.constexpr, D: tl.constexpr):
    # Update state_new[h] = g * state_old[h] - dot(k, dot(k, state_old[h])) + dot(k, dot(k, beta * v + (1 - beta) * dot(k, state_old[h])))
    # state_old_ptr points to [D, D] block for head h; similarly for state_new.
    # We need to load state_old[h, :, :], compute old_v = k @ state_old[h], new_v = beta * v + (1 - beta) * old_v,
    # then compute contributions.
    # Load state_old[h, :, :] (as 2D)
    # Create indices
    i = tl.arange(0, D)
    j = tl.arange(0, D)
    S = tl.load(state_old_ptr + i[:, None] * D + j[None, :], mask=(i[:, None] < D) & (j[None, :] < D), other=0.0).to(tl.float32)
    k_vec = tl.load(k_ptr + t * 8 * D + h * D + tl.arange(0, D)).to(tl.float32)
    old_v = tl.sum(S * k_vec[None, :], axis=1)  # [D]
    # v_vec
    v_vec = tl.load(v_ptr + t * 8 * D + h * D + tl.arange(0, D)).to(tl.float32)
    beta_val = tl.load(beta_ptr + t * 8 + h).to(tl.float32)
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [D]

    # contributions
    # Cremove = k^T @ old_v = dot(k, old_v)
    Cremove = tl.sum(k_vec * old_v, axis=0)
    Cadd = tl.sum(k_vec * new_v, axis=0)

    g_val = tl.load(g_ptr + t * 8 + h).to(tl.float32)

    # Update state_new[h, :, :] = S * g - Cremove * S + Cadd * S
    # Multiply S elementwise by (g - Cremove + Cadd)
    factor = g_val - Cremove + Cadd  # scalar
    S_new = S * factor

    # Store S_new back
    for ii in range(D):
        for jj in range(D):
            tl.store(state_new_ptr + ii * D + jj, S_new[ii, jj])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure CUDA and contiguity
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and A_log.is_cuda and a.is_cuda and b.is_cuda and dt_bias.is_cuda, "All inputs must be CUDA tensors."
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        A_log = A_log.contiguous()
        a = a.contiguous()
        b = b.contiguous()
        device = q.device

        # Dimensions
        L, Hq, K = q.shape  # q: [L, 4, 128]
        Kk, Hk, Kk2 = k.shape  # k: [L, 4, 128]
        Lv, Hv, V = v.shape  # v: [L, 8, 128]
        assert Hq == Hk and K == Kk and K == 128 and V == 128 and Hq == 4, "Expected q[k,4,128], k[k,4,128], v[k,8,128]"
        D = K  # head_size

        # Flatten inputs for Triton kernels
        a_flat = a.view(-1)           # [L*32]
        b_flat = b.view(-1)           # [L*32]
        N = a_flat.numel()

        # 1) Compute g and beta via Triton kernels
        # g_flat: [N] = exp(-exp(A_log) * softplus(a + dt_bias))
        g_flat = torch.empty(N, dtype=torch.float32, device=device)
        grid_g = (N,)
        triton.run(g_kernel, grid_g, a_flat, dt_bias, A_log, g_flat, N=N, A_len=A_log.numel())
        g_per_t = g_flat.view(L, 32)[:, [0, 4, 8, 12, 16, 20, 24, 28]].squeeze(1)  # [L, 8]

        # beta_flat: [N] = sigmoid(b)
        beta_flat = torch.empty(N, dtype=torch.float32, device=device)
        grid_beta = (N,)
        triton.run(sigmoid_kernel, grid_beta, b_flat, beta_flat, N=N)
        beta_per_t = beta_flat.view(L, 32)[:, [0, 4, 8, 12, 16, 20, 24, 28]].squeeze(1)  # [L, 8]

        # 2) Expand q and k to 8 heads via repeat_interleave(2)
        q_exp = torch.empty((L, 8, D), dtype=q.dtype, device=device)
        k_exp = torch.empty((L, 8, D), dtype=k.dtype, device=device)
        grid_expand = (L * 4 * D * 2,)  # 8 heads
        triton.run(repeat2_expand, grid_expand, q, q_exp, L, D, 2)
        triton.run(repeat2_expand, grid_expand, k, k_exp, L, D, 2)

        # 3) Compute output[t, h, :] using GEMV in Triton
        output = torch.empty((L, 8, D), dtype=torch.bfloat16, device=device)
        # We need state_new for each (t, h) to compute GEMV. Initialize state_new_tmp zeros and update in Triton below.
        state_new_tmp = torch.zeros((8, D, D), dtype=torch.float32, device=device)

        # 4) Update state for each sequence, time, head using Triton kernel
        num_seqs = cu_seqlens.shape[0] - 1
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue
            # Initialize state_old per head: load state[seq_idx, h, :, :]
            for h in range(8):
                # state_old[h] = state[seq_idx, h, :, :]
                state_old_ptr = state[seq_idx, h]  # shape [D, D], contiguous
                state_new_ptr = state_new_tmp[h]    # shape [D, D]
                k_vec = k_exp[seq_start, h]         # [D]
                v_vec = v[seq_start, h]             # [D]
                beta_val = beta_per_t[seq_start, h].item()
                g_val = g_per_t[seq_start, h].item()
                # Launch update kernel for this (seq_idx, t=seq_start, h)
                # Triton requires grid to be a tuple; we launch per iteration.
                grid_update = (1,)
                triton.run(update_state_kernel, grid_update, state_old_ptr, state_new_ptr, k_vec, v_vec, beta_val, g_val,
                           seq_idx=seq_idx, t=seq_start, h=h, D=D)

        # 5) Now compute output[t, h, :] for t in [seq_start..seq_end-1]
        # We have state_new_tmp updated. Use GEMV Triton kernel
        for t in range(seq_start, seq_end):
            for h in range(8):
                # q_exp[t, h, :] -> q_ptr, state_new_tmp[h, :, :] -> state_ptr
                q_ptr = q_exp[t, h]  # [D]
                state_ptr = state_new_tmp[h]  # [D, D]
                out_off = t * 8 * D + h * D
                grid_gemv = (1,)
                triton.run(gemv_kernel, grid_gemv, q_ptr, state_ptr, output, scale=scale,
                           L=L, H=8, D=D)

        return output, state_new_tmp


def run(*args):
    return ModelNew()(*args)
