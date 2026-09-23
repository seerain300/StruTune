import torch
import math
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# ---------------------------
# Triton kernels
# ---------------------------

# Elementwise gate and beta computation (kept in PyTorch for simplicity since it's light).
# If you want, we can implement this in Triton too, but it's not performance-critical.

# Matmul kernel: C = A @ B
# A: [M, K], B: [K, N], C: [M, N]
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # program ids for tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak)
        b_ptrs = B_ptr + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)

        # cast to float32 for accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    # apply scaling if provided
    if scale != 1.0:
        acc = acc * scale

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton kernel to compute the scalar output per step:
# Output = scale * q[t] @ new_state, with q[t]: [H, 128], new_state: [H, 128, 128]
# We implement a per-H matmul with M=H, N=128, K=128
@triton.jit
def _output_scalar_kernel(
    q_ptr, state_ptr, out_ptr,
    H, N,  # N is head_size (128)
    stride_qh, stride_qk,
    stride_sh, stride_sk, stride_sv,  # state layout: [H, K, V] == [H, 128, 128]
    stride_oh, stride_ok,
    scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # grid: (H, 1) or (H,) is fine
    pid_h = tl.program_id(0)
    # M=H, N=128, K=128
    BLOCK_M = 1  # we only process one row (head) per program
    BLOCK_N = 128
    BLOCK_K = 32

    offs_m = pid_h  # scalar M
    offs_n = tl.arange(0, BLOCK_N)  # [0..127]
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, 128, BLOCK_K):
        # A = q[pid_h, :] -> shape [1, K]
        a_ptrs = q_ptr + (offs_m * stride_qh + (k + offs_k) * stride_qk)
        # B = state[pid_h, :, :] -> shape [K, N]
        b_ptrs = state_ptr + (offs_m * stride_sh + (k + offs_k)[:, None] * stride_sk + offs_n[None, :] * stride_sv)

        a = tl.load(a_ptrs, mask=(offs_k + k < 128), other=0.0)  # [32]
        b = tl.load(b_ptrs, mask=(offs_k + k < 128) & (offs_n < 128), other=0.0)  # [32, 128]

        a = a.to(tl.float32)[:, None]  # [1, 32]
        b = b.to(tl.float32)            # [32, 128]

        acc += tl.dot(a, b)  # [1, 128]

    # scale
    acc = acc * scale

    # store as bfloat16
    out_ptrs = out_ptr + (offs_m * stride_oh + offs_n * stride_ok)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=(offs_n < 128))


# Update state kernel per step:
# Computes:
#   old_v   = k[t] @ state_old
#   new_v   = beta * v + (1 - beta) * old_v
#   remove  = k^T @ old_v
#   update  = k^T @ new_v
#   new_state = g * state_old + update - remove
# Input pointers:
#   q_exp_ptr, k_ptr, v_ptr: 1x128 vectors (bfloat16)
#   state_old_ptr: [H, K, N] float32, where K=N=128, H=num_q_heads=4
# Output:
#   new_state_ptr: [H, K, N] float32
@triton.jit
def _update_state_single_kernel(
    q_ptr, k_ptr, v_ptr, state_old_ptr, new_state_ptr,
    g_scale: tl.float32, beta_scale: tl.float32,
    H, K, N,  # K=N=128; H=4
    stride_qh, stride_qk,
    stride_kh, stride_kk,
    stride_vh, stride_vk,
    stride_sh, stride_sk, stride_sv,  # state_old layout: [H, K, N]
    stride_nsh, stride_nsk, stride_nsv,  # new_state layout: [H, N, K] (we'll write as [H, K, N])
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # We will do per-H blocks and iterate over K and N tiles.
    # grid: (H, tiles_N, tiles_K). tiles_N = ceil_div(N, BLOCK_N), tiles_K = ceil_div(K, BLOCK_K).
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # process one head pid_h
    # We need to compute four matmuls for this head:
    # 1) old_v = k @ state_old  -> [N] = [K] @ [K, N]
    # 2) new_v = beta*v + (1-beta)*old_v -> [N]
    # 3) state_remove = k^T @ old_v -> [K]
    # 4) state_update = k^T @ new_v -> [K]
    # 5) new_state = g*state_old + state_update - state_remove -> [K, N]

    # Constants
    BLOCK_M = 1  # M dimension is 1 in these vec ops
    BLOCK_N = 64  # tile N
    BLOCK_K = 32  # tile K

    # 1) old_v = k @ state_old
    # A = k[t], shape [K]
    k_ptrs = k_ptr + (k * stride_kh + k * stride_kk)  # k is scalar index 0..K-1
    a_k = tl.load(k_ptrs, mask=(k < K), other=0.0)  # [K]
    a_k = a_k.to(tl.float32)  # [K] float32

    # B = state_old[pid_h, :, :] -> [K, N]
    b_ptrs = state_old_ptr + (pid_h * stride_sh + k * stride_sk + n * stride_sv)
    # We need to load B matrix in tiles: loop over k and n tiles
    # Allocate acc_old_v[N] for this head
    old_v_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Loop over K tiles for the vector dot
    for kk in range(0, K, BLOCK_K):
        a_k_tile = tl.load(k_ptrs + kk, mask=(kk + tl.arange(0, BLOCK_K) < K), other=0.0)  # [BLOCK_K]
        # Now we need to multiply a_k_tile (BLOCK_K) with each N tile
        for nn in range(0, N, BLOCK_N):
            b_tile_ptrs = state_old_ptr + (pid_h * stride_sh + (kk + tl.arange(0, BLOCK_K))[:, None] * stride_sk + (nn + tl.arange(0, BLOCK_N))[None, :] * stride_sv)
            b_tile = tl.load(b_tile_ptrs, mask=((kk + tl.arange(0, BLOCK_K))[:, None] < K) & ((nn + tl.arange(0, BLOCK_N))[None, :] < N), other=0.0)  # [BLOCK_K, BLOCK_N]
            b_tile = b_tile.to(tl.float32)
            # acc_old_v += sum_k a_k_tile[k] * b_tile[k, :]
            # We can do it by reducing dot of [1,BLOCK_K] and [BLOCK_K,BLOCK_N]
            a_1 = a_k_tile[None, :]  # [1,BLOCK_K]
            partial = tl.dot(a_1, b_tile)  # [1,BLOCK_N]
            old_v_acc += partial[0, :]

    # 2) new_v = beta * v + (1 - beta) * old_v
    v_ptrs = v_ptr + (pid_h * stride_vh + k * stride_vk)  # vector v[k] -> scalar v per k
    v_vals = tl.load(v_ptrs, mask=(k < N), other=0.0)  # [N]
    v_vals = v_vals.to(tl.float32)
    new_v = beta_scale * v_vals + (1.0 - beta_scale) * old_v_acc  # [N]

    # 3) state_remove = k^T @ old_v
    # A = k[t] (shape [K]), B = old_v (shape [N])
    # result is [K]
    state_remove_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        a_k_tile = tl.load(k_ptrs + kk, mask=(kk + tl.arange(0, BLOCK_K) < K), other=0.0)  # [BLOCK_K]
        old_v_sub = old_v_acc[tl.arange(0, BLOCK_N)]  # reuse as scalar vector? better loop:
        # We need old_v values at N tile positions; use direct scalar load:
        for nn in range(0, N, BLOCK_N):
            # compute old_v_acc for each nn sub-block by slicing? Instead, compute per element
            # Simpler: since N is 128, we can compute dot directly with scalar loads.
            pass  # Implement scalar dot below

    # Implement scalar dot for state_remove:
    # Loop over N to accumulate dot with old_v
    for n_idx in range(0, N):
        val = tl.load(v_ptr + (pid_h * stride_vh + n_idx * stride_vk), mask=(n_idx < N), other=0.0)
        val = val.to(tl.float32)
        # k values at each kk contribute to dot: sum_k k[kk] * old_v[n_idx]
        for kk in range(0, K):
            kk_val = tl.load(k_ptr + (kk * stride_kh + kk * stride_kk), mask=(kk < K), other=0.0)
            kk_val = kk_val.to(tl.float32)
            state_remove_acc[kk] += kk_val * val

    # 4) state_update = k^T @ new_v
    state_update_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for n_idx in range(0, N):
        val = new_v[n_idx]  # scalar
        for kk in range(0, K):
            kk_val = tl.load(k_ptr + (kk * stride_kh + kk * stride_kk), mask=(kk < K), other=0.0)
            kk_val = kk_val.to(tl.float32)
            state_update_acc[kk] += kk_val * val

    # 5) new_state = g * state_old + state_update - state_remove
    # C = state_old[pid_h, :, :] -> [K, N]
    for kk in range(0, K, BLOCK_K):
        for nn in range(0, N, BLOCK_N):
            c_ptrs = state_old_ptr + (pid_h * stride_sh + (kk + tl.arange(0, BLOCK_K))[:, None] * stride_sk + (nn + tl.arange(0, BLOCK_N))[None, :] * stride_sv)
            c_tile = tl.load(c_ptrs, mask=((kk + tl.arange(0, BLOCK_K))[:, None] < K) & ((nn + tl.arange(0, BLOCK_N))[None, :] < N), other=0.0)
            c_tile = c_tile.to(tl.float32)
            # new state tile
            new_c_tile = g_scale * c_tile + state_update_acc[:, None] - state_remove_acc[:, None]
            new_c_ptrs = new_state_ptr + (pid_h * stride_nsh + (nn + tl.arange(0, BLOCK_N))[None, :] * stride_nsv + (kk + tl.arange(0, BLOCK_K))[:, None] * stride_nsk)
            tl.store(new_c_ptrs, new_c_tile, mask=((kk + tl.arange(0, BLOCK_K))[:, None] < K) & ((nn + tl.arange(0, BLOCK_N))[None, :] < N))

    # Note: above we only handled one head per program. The grid dims should cover all H, N, K tiles.
    # To cover all heads, we need to launch H programs. Since H=4 in the given code, we set grid = (H, tiles_N, tiles_K).


# ---------------------------
# ModelNew: Triton-optimized forward
# ---------------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; forward uses Triton kernels

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized forward matching the original run signature.
        Returns:
          - output: [total_seq_len, num_sab_heads, head_size], dtype bfloat16
          - new_state: [num_seqs, num_sab_heads, head_size, head_size], dtype float32
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        device = q.device
        dtype = q.dtype  # bfloat16
        head_size = q.shape[2]
        total_seq_len = q.shape[0]
        num_q_heads = q.shape[1]
        num_k_heads = k.shape[1]
        num_v_heads = v.shape[1]
        num_seqs = cu_seqlens.shape[0] - 1

        # Compute g and beta once (elementwise, on GPU via PyTorch ops)
        # Ensure dtypes: a is bfloat16, A_log, dt_bias, b are float32
        # g: [T, HV], beta: [T, HV]
        # We need HV = num_v_heads (8), T = total_seq_len
        # Note: The original code repeats q/k by repeat_interleave(num_v_heads // num_q_heads, dim=1),
        #       which is 2x here, but our Triton update will handle 4 q heads and 8 v heads directly.
        # We will compute g/beta for the expanded a/b (which are [T, HV] where HV=num_v_heads).
        # However, the original repeats q/k from 4 -> 8, so we must expand a/b accordingly:
        # a_exp: [T, 8], b_exp: [T, 8]
        # But in our Triton update, q and k inputs are already the expanded ones from run (shape [T,8,128]).
        # We will compute g and beta accordingly by passing A_log (shape [8]) and a, b (shape [T,8]).
        # Expand heads from q/k to v heads (2x), matching original behavior.
        a_exp = a.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [T, 8]
        b_exp = b.repeat_interleave(num_v_heads // num_k_heads, dim=1)  # [T, 8]
        A_log_exp = A_log.repeat(1)  # repeat 8 times for v heads
        g = torch.exp(-torch.exp(A_log_exp.float()) * F.softplus(a_exp.float() + dt_bias.float()).unsqueeze(2))  # [T, 8, 1]
        beta = torch.sigmoid(b_exp.float().unsqueeze(2))  # [T, 8, 1]
        # We don't need to store g/beta beyond computing per-step; we pass scalars to Triton per step.

        # Prepare output and new_state
        output = torch.empty((total_seq_len, num_v_heads, head_size), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((num_seqs, num_v_heads, head_size, head_size), dtype=torch.float32, device=device)

        # Process each sequence
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Expand q/k to v heads
            q_exp = q[seq_start:seq_end].repeat_interleave(num_v_heads // num_q_heads, dim=1).contiguous()  # [seq_len, 8, 128]
            k_exp = k[seq_start:seq_end].repeat_interleave(num_v_heads // num_k_heads, dim=1).contiguous()  # [seq_len, 8, 128]
            v_seq = v[seq_start:seq_end].contiguous()  # [seq_len, 8, 128]

            # Handle initial state: original uses state[seq_idx] if provided, else zeros.
            # State layout expected: [H, V, K] == [8, 128, 128]
            if state is None:
                state_seq = torch.zeros((num_v_heads, head_size, head_size), dtype=torch.float32, device=device)
            else:
                state_seq = state[seq_idx].transpose(-1, -2).contiguous()  # [8, 128, 128]

            # new_state buffer for this sequence
            new_state[seq_idx] = torch.zeros((num_v_heads, head_size, head_size), dtype=torch.float32, device=device)

            # Loop over timesteps and update state, computing output
            for i in range(seq_len):
                t = seq_start + i
                q_t = q_exp[i]  # [8, 128]
                k_t = k_exp[i]  # [8, 128]
                v_t = v_seq[i]  # [8, 128]

                # Compute per-step gate and beta scalars (already computed above as [T,8,1], we take t-th row)
                # Extract g_scalar and beta_scalar for head 0 (since state_dim=8 and original uses heads 0..7)
                # For simplicity and correctness, we'll pass g[t, 0], beta[t, 0] as scalars. But since g/beta are per-head,
                # we need per-head values. We'll compute g/beta for each head separately and pass them.
                # Given A_log is [8] and a_exp is [T,8], we can compute g/beta per head by extracting head id.
                # However, the original run computes g and beta for a_exp (shape [T,8]) and uses them for each head.
                # We'll compute g_scalar = torch.exp(-torch.exp(A_log.float()) * softplus(a_exp[t, h] + dt_bias.float())) for each head h.
                # Implement using PyTorch (fast and simple):
                g_scalar_list = []
                beta_scalar_list = []
                for h in range(num_v_heads):
                    a_h = a_exp[t, h]  # scalar
                    b_h = b_exp[t, h]  # scalar
                    # g for head h
                    g_h = torch.exp(-torch.exp(A_log[h].float()) * F.softplus(a_h.float() + dt_bias[h].float()))
                    # beta for head h
                    beta_h = torch.sigmoid(b_h.float())
                    g_scalar_list.append(g_h.item())
                    beta_scalar_list.append(beta_h.item())

                # Now compute new_state per head and output
                # We'll run Triton kernels per head. For simplicity, we can call Triton kernel 8 times (one per head).
                # But Triton launch requires tensor inputs, not scalars. We'll implement per-head in a loop calling Triton update and output kernels.

                # Initialize per-head new_state for this sequence
                new_state_sub = torch.zeros((head_size, head_size), dtype=torch.float32, device=device)

                # Triton update per head
                # We need q_t, k_t, v_t, state_seq[:, h, :], then update to new_state_sub and write into new_state[seq_idx, h, :, :]
                # q_t, k_t, v_t are [8, 128], state_seq is [8, 128, 128]
                # Triton kernel expects 1x128 vectors for q/k/v and [H, K, N] for state_old, returning [H, K, N] new_state.
                # Since we have H=8 heads, we’ll do per-head loop with host-side Triton calls (Triton supports scalar loop via while).
                # But Triton kernels are launched per grid; for a per-step update, we can call Triton for each head.

                # Prepare per-head state slices
                state_old_h = state_seq[h].clone()  # [128, 128]
                new_state_h = torch.zeros((head_size, head_size), dtype=torch.float32, device=device)

                # Triton update for this head
                # We need to pass pointers to q_t[:, h], k_t[:, h], v_t[:, h], and state_old_h, and write new_state_h.
                # However, Triton kernel signature expects [H, 128] tensors; our q_t/k_t/v_t are [8, 128].
                # To keep it general, we can select the h-th "row" from the 8 by loading the h-th 1x128 vector from q_t/k_t/v_t by viewing.
                # Simpler: just pass the entire 8x128 vectors and rely on Triton to handle h index by passing h as an argument and slicing inside Triton? Triton doesn't support slicing with dynamic h in the kernel; we'll implement per-head using PyTorch for q_t/k_t/v_t vectors and Triton for matmuls.

                # For Triton update, we need to construct A=[1,128] vectors for q/k/v by selecting h-th row. Since Triton can't index dynamically like that, we'll do per-head using PyTorch indexing to construct the required 1x128 vectors and call Triton kernels accordingly.

                # Instead of per-head, we can write a Triton kernel that takes q_t, k_t, v_t (8x128), state_old (8x128x128), and returns new_state (8x128x128).
                # That requires dynamic indexing per head in Triton. Since Triton lacks Python-side vector indexing, we will implement per-head loop in Python and call Triton kernels accordingly.

                # Implement per-head Triton call using PyTorch indexing:
                # We'll run Triton kernels in a loop over h, constructing A,B,C as needed by passing q_t[:,h], k_t[:,h], v_t[:,h], and state_old[:, :, h].
                # But Triton kernels require tensor pointers; we cannot create tensors on the fly per h in a loop with Triton pointers. Therefore, we'll keep Triton for the matmul update and output scalar, and per-head loop will be in Python. This is acceptable for correctness and simplicity.

                # We'll compute new_state for each head using Triton matmul update and Triton output scalar. For brevity


def run(*args):
    return ModelNew()(*args)
