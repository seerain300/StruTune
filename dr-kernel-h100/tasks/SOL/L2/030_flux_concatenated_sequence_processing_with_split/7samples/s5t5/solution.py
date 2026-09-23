import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,                # *const T: [B, T, H]
    i_ptr,                # *const T: [B, I, H]
    out_ptr,              # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,     # strides for e_ptr
    i_s0, i_s1, i_s2,     # strides for i_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence index in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden feature index

    mask_l = l < (T + I)
    mask_h = h < H

    # Determine if this l corresponds to encoder or image stream
    is_encoder = l < T
    # Compute source l indices for each side
    src_l_e = l
    src_l_i = l - T

    # Build 2D offsets for loads from each source
    # Offsets shape: [BLOCK_l, BLOCK_h]
    # For encoder: e_off = pid_b*e_s0 + src_l_e*e_s1 + h[None, :]*e_s2
    # For image:   i_off = pid_b*i_s0 + src_l_i*i_s1 + h[None, :]*i_s2
    # We need a common offset matrix to pass to tl.load; we can't branch, so we load both and select via masks.
    # However, Triton requires a single pointer for tl.load; we will load with masks and combine using tl.where by reconstructing values from masked loads.

    # Prepare offsets
    e_off = pid_b * e_s0 + src_l_e[:, None] * e_s1 + h[None, :] * e_s2
    i_off = pid_b * i_s0 + src_l_i[:, None] * i_s1 + h[None, :] * i_s2

    # Masked loads: we need values for both sides, but only one is valid per element.
    # Use a neutral element for the invalid side (0.0). Then select using is_encoder.
    val_e = tl.load(e_ptr + e_off, mask=mask_l[:, None] & mask_h[None, :], other=0.0)
    val_i = tl.load(i_ptr + i_off, mask=mask_l[:, None] & mask_h[None, :], other=0.0)
    # Select based on is_encoder. Note: is_encoder is a vector [BLOCK_l]; we broadcast across h.
    selected = tl.where(is_encoder[:, None], val_e, val_i)

    # Compute output offsets
    out_off = pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2
    out_mask = (mask_l[:, None] & mask_h[None, :])
    tl.store(out_ptr + out_off, selected, mask=out_mask)


@triton.jit
def matmul_kernel(
    A_ptr,                # *const T: [B, L, H], where L = T+I
    W_ptr,                # *const T: [H, H] (process_weight)
    C_ptr,                # *T: [B, L, H]
    B: tl.int32, L: tl.int32, H: tl.int32,
    a_s0, a_s1, a_s2,     # strides for A_ptr
    w_s0, w_s1, w_s2,     # strides for W_ptr (H, H)
    c_s0, c_s1, c_s2,     # strides for C_ptr
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid dims: (M, N, K) tiles, but we flatten (B, L) into M. Here we treat 3D grid as (B*L, H_tiles, K_tiles).
    # Triton supports 1D/2D/3D grid; we use 3D where pid0 = b*l, pid1 = h tile, pid2 = k tile.
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    # Recover b and l from pid0
    # Note: B is not directly available here, but we can infer l from L and B only if we pass B. Simpler: pass grid sizes accordingly and compute l via pid0 and a count.
    # Since Triton kernels receive only program_id, we need to map pid0 to (b, l). We'll pass grid=(B*L, cdiv(H,BLOCK_N), cdiv(H,BLOCK_K)). Then:
    # b = pid0 // L, l = pid0 % L. But B is not known. Instead, we'll pass B as a runtime scalar argument (as tl.int32) to compute b,l.
    # However, Triton kernels don't have access to external Python B here. So we'll use a 2D grid: (B, cdiv(L,BLOCK_M)), and within kernel assume pid0 is index over B.
    # To handle general, we assume grid0 = B*L. Then:
    # b = pid0 // L, l = pid0 % L. We need L as argument.
    L_arg = L
    b = pid0 // L_arg
    l = pid0 % L_arg

    # h tile
    h = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)
    k = pid2 * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_h = h < H
    mask_k = k < H

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # A is [L, H], row index is l, columns k
    # A offsets: a_off = b*a_s0 + l*a_s1 + k[a, :]*a_s2. But we iterate rows (a) as l and cols as k.
    # We want to load A rows for each (b, l), which is vectorized over BLOCK_M = 1 (since we flatten), but Triton expects 2D.
    # Better approach: compute a_off = b*a_s0 + l*a_s1 + kk[None, :]*a_s2. Here kk is k for the K tile.

    # We need A[b, l, k] values for the K tile. Since we flattened (b,l) into pid0, we can reconstruct:
    # For a single row vector across K: a_off_vec = b*a_s0 + l*a_s1 + k[None, :]*a_s2, shape [1, BLOCK_K]
    # For dot: we need A_tile [BLOCK_M, BLOCK_K]. Since we have only one row (for this pid0), we can set BLOCK_M=1. To support multiple rows, we'd need to loop over m dimension in Python grid. Simpler: restrict grid0=B and compute per-batch, per-seq index manually by looping over B in Python.

    # Fix: We'll make grid0 = B*L, and inside kernel assume 3D tiling over (L, H, K). To do that, pass B as kernel arg and compute b,l explicitly:
    # We previously passed B as a runtime scalar via grid, but Triton doesn't receive it. To avoid confusion, we'll instead use a 2D grid: (B, cdiv(L,BLOCK_M)).
    # For simplicity and correctness, we redefine kernel signature to take B as an argument (Triton supports scalar args), and map grid0 = B, grid1 = cdiv(L,BLOCK_M), grid2 = cdiv(H,BLOCK_N). Then we still need K tiling; we'll set grid2 to cdiv(H,BLOCK_K) as well. However, Triton grid is 3D and we can pass cdiv(H, BLOCK_K) as the third.

    # Final approach: define grid as (B, cdiv(L,BLOCK_M), cdiv(H,BLOCK_N)), and compute b = tl.program_id(0), l = pid_m * BLOCK_M + tl.arange(0, BLOCK_M). But we only have 3D grid. So we will pass B as a runtime scalar argument and compute b = pid0 (since grid0 = B), l = pid1 * BLOCK_M + tl.arange(0, BLOCK_M).

    # To make this robust, we redefine the kernel to accept B and map grid0 = B, grid1 = cdiv(L,BLOCK_M), grid2 = cdiv(H,BLOCK_N).
    # However, Triton launch grid is static; we can't create a kernel with B in signature unless we also pass cdiv() values. Simpler: restructure matmul to 3D grid over (B, L tiles, H tiles), and loop over K inside the kernel.

    # Therefore, we implement matmul kernel with grid = (B, cdiv(L,BLOCK_M), cdiv(H,BLOCK_N)), and loop over K in kernel. We'll pass B as an argument.

    # Kernel signature: (A_ptr, W_ptr, C_ptr, B, L, H, a_s0/a_s1/a_s2, w_s0/w_s1/w_s2, c_s0/c_s1/c_s2, BLOCK_M, BLOCK_N, BLOCK_K)

    # Compute b and l from pid0. grid0 = B, so b = pid0; grid1 = cdiv(L, BLOCK_M).
    # We need to recover l from pid1? The earlier approach was wrong. Simpler: use a 2D grid: (B, L) where each program handles one row (b, l). Then we tile H across pid2. However, Triton prefers fixed block tiling. So we'll implement 3D grid: (B, H tiles, K tiles) and use l as a runtime scalar via grid mapping? Not ideal.

    # Fix: We'll implement a robust matmul kernel with 3D grid: (B, L tiles, H tiles), and loop over K. Triton allows loops; we can have kernel take B, L, H, and compute b = pid0 % B is not applicable; since grid0 = B, b = pid0. l is covered by pid1. We'll restructure.

    # Define grid: (B, cdiv(L, BLOCK_M), cdiv(H, BLOCK_N)). Inside kernel: b = pid0, l = pid1 * BLOCK_M + tl.arange(0, BLOCK_M). Then for each tile h, k, we load A rows for this (b, l) across k, and W across k,n, accumulate.

    # Initialize h and k for this program
    h = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)
    k = pid2 * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_h = h < H
    mask_k = k < H

    # Accumulator for this (b, l) and h tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each k tile, load A[b, l, k] and W[k, h], and accumulate
    # Note: A is [B, L, H] with strides a_s0, a_s1, a_s2; W is [H, H] with strides w_s0, w_s1, w_s2. We need W[k, h] -> offset k*w_s0 + h*w_s1.
    # We iterate kk over K tiles, load A vector for this (b,l) across kk, and W tile [kk, h], then do dot: acc += A_vec[:, None] * W_tile[None, :].

    # Loop over kk in K tiles
    kk = k
    for kk_start in range(0, H, BLOCK_K):
        kk = kk_start + tl.arange(0, BLOCK_K)
        mask_kk = kk < H

        # Load A[b, l, kk] as a vector of length BLOCK_K
        A_off = b * a_s0 + l * a_s1 + kk * a_s2  # vector of offsets
        A_vec = tl.load(A_ptr + A_off, mask=mask_kk, other=0.0)  # shape [BLOCK_K]

        # Load W[kk, h] as a tile [BLOCK_K, BLOCK_N]
        W_off = kk[:, None] * w_s0 + h[None, :] * w_s1  # shape [BLOCK_K, BLOCK_N]
        W_tile = tl.load(W_ptr + W_off, mask=mask_kk[:, None] & mask_h[None, :], other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate: acc += outer product of A_vec and W_tile
        # A_vec: [BLOCK_K], W_tile: [BLOCK_K, BLOCK_N]
        # We need acc: [BLOCK_M, BLOCK_N], and we're loading one row. To keep shape, we can broadcast A_vec across N.
        # But we only have one (b,l). We need to accumulate into acc of shape [1, BLOCK_N] if BLOCK_M=1. In Triton, we can keep acc as [BLOCK_M, BLOCK_N] with BLOCK_M=1.

        # However, Triton doesn't support dynamic row; we must keep BLOCK_M as 1 for this pattern. Simpler: per-(b,l) program handles H tile and K loop.

        # So we compute outer product and accumulate
        # acc += A_vec[:, None] * W_tile[None, :]
        # But Triton requires tensors of same shape for elementwise ops; we'll use a loop over kk to update acc row by row. Alternatively, use tl.dot.

        # Compute dot for each kk across N: acc += A_vec[kk] * W_tile[kk, :]
        # Implement by iterating over kk within the tile
        for kk_idx in range(0, BLOCK_K):
            valid_kk = kk_idx < (H - kk_start)  # mask for kk within bounds
            a_val = A_vec[kk_idx] * valid_kk
            W_row = W_tile[kk_idx, :]  # shape [BLOCK_N]
            acc += a_val[:, None] * W_row[None, :]

    # Store result to C[b, l, h]
    C_off = b * c_s0 + l * c_s1 + h[None, :] * c_s2
    C_mask = mask_h[None, :]
    tl.store(C_ptr + C_off, acc, mask=C_mask)


@triton.jit
def slice_first_part_kernel_with_T(
    processed_ptr,        # *const T: [B, L, H], L=T+I
    out_ptr,              # *T: [B, T, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    p_s0, p_s1, p_s2,     # strides for processed_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # l in [0, T)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # h in [0, H)

    mask_l = l < T
    mask_h = h < H

    # Offsets for processed and output
    p_off = pid_b * p_s0 + l[:, None] * p_s1 + h[None, :] * p_s2
    o_off = pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2

    mask = mask_l[:, None] & mask_h[None, :]

    vals = tl.load(processed_ptr + p_off, mask=mask, other=0.0)
    tl.store(out_ptr + o_off, vals, mask=mask)


@triton.jit
def slice_second_part_kernel_with_T(
    processed_ptr,        # *const T: [B, L, H], L=T+I
    out_ptr,              # *T: [B, I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    p_s0, p_s1, p_s2,     # strides for processed_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # l in [T, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # h in [0, H)

    mask_l = l < (T + I)
    mask_h = h < H

    # Adjust source l to start at T
    src_l = l - T

    p_off = pid_b * p_s0 + l[:, None] * p_s1 + h[None, :] * p_s2
    o_off = pid_b * o_s0 + src_l[:, None] * o_s1 + h[None, :] * o_s2

    mask = mask_l[:, None] & mask_h[None, :]

    vals = tl.load(processed_ptr + p_off, mask=mask, other=0.0)
    tl.store(out_ptr + o_off, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension (Triton).
        2) Apply linear projection (concatenated @ process_weight.T) using Triton matmul.
        3) Split back into separate encoder and image streams (Triton).
        """
        # Shapes
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate [B, T, H] and [B, I, H] into [B, T+I, H] using Triton
        concatenated = torch.empty((B, T + I, H), dtype=dtype, device=device)
        e = encoder_hidden_states
        i = hidden_states

        BLOCK_l = 128
        BLOCK_h = 64
        grid = (B, triton.cdiv(T + I, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_encoder_image_kernel[grid](
            e, i, concatenated,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: concatenated [B, L, H] @ process_weight.T [H, H] -> processed [B, L, H], where L = T+I
        L = T + I
        processed = torch.empty((B, L, H), dtype=dtype, device=device)
        # We use the matmul kernel with grid over (B, cdiv(L, BLOCK_M), cdiv(H, BLOCK_N)).
        BLOCK_M = 1  # one row per program; we tile only across H and loop over K inside
        BLOCK_N = 64
        BLOCK_K = 64

        grid_matmul = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        # Note: The kernel loops over K internally; we don't need a 3D grid over K. We just set grid0=B and cover H tiling.
        matmul_kernel[grid_matmul](
            concatenated, process_weight, processed,
            B, L, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.stride(0), process_weight.stride(1), process_weight.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split processed into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)

        BLOCK_l_split = 128
        BLOCK_h_split = 64

        grid_first = (B, triton.cdiv(T, BLOCK_l_split), triton.cdiv(H, BLOCK_h_split))
        slice_first_part_kernel_with_T[grid_first](
            processed, processed_encoder,
            B, T, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_l=BLOCK_l_split, BLOCK_h=BLOCK_h_split,
            num_warps=4, num_stages=2,
        )

        grid_second = (B, triton.cdiv(I, BLOCK_l_split), triton.cdiv(H, BLOCK_h_split))
        slice_second_part_kernel_with_T[grid_second](
            processed, processed_hidden,
            B, T, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_l=BLOCK_l_split, BLOCK_h=BLOCK_h_split,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
