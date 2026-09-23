import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,                # *const float: [B, T, H]
    i_ptr,                # *const float: [B, I, H]
    out_ptr,              # *float: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,     # strides for e_ptr
    i_s0, i_s1, i_s2,     # strides for i_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D launch grid: (B, ceil((T+I)/BLOCK_l), ceil(H/BLOCK_h))
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    L = T + I

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)   # sequence indices in [0, L)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)   # hidden dims

    # Compute linear indices for output tensor using strides
    # Address: out[b, l, h] with strides o_s0, o_s1, o_s2
    # We use broadcasting to form a [BLOCK_l, BLOCK_h] tile
    out_idx = (
        pid_b * o_s0
        + l[:, None] * o_s1
        + h[None, :] * o_s2
    )

    # Masks to avoid OOB
    mask = (l[:, None] < L) & (h[None, :] < H)

    # Select source: if l < T -> from encoder; else from image at (l - T)
    from_encoder = l[:, None] < T  # [BLOCK_l, 1] -> broadcast over h

    # Compute source indices
    # e[i, j, k] index: i*se0 + j*se1 + k*se2
    e_idx = (
        pid_b * e_s0
        + l[:, None] * e_s1
        + h[None, :] * e_s2
    )
    i_idx = (
        pid_b * i_s0
        + (l[:, None] - T) * i_s1
        + h[None, :] * i_s2
    )

    # Load values (masked)
    e_vals = tl.load(e_ptr + e_idx, mask=mask & from_encoder, other=0.0)
    i_vals = tl.load(i_ptr + i_idx, mask=mask & (~from_encoder), other=0.0)
    # Combine
    vals = tl.where(from_encoder, e_vals, i_vals)

    # Store
    tl.store(out_ptr + out_idx, vals, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,        # *const float: [B*L, H], A = concatenated
    W_ptr,        # *const float: [H, H], weight
    C_ptr,        # *float: [B*L, H], output
    B: tl.int32,  # not used directly, kept for potential future use
    L: tl.int32,  # sequence length after concatenation
    H: tl.int32,  # hidden dimension
    A_stride0,    # stride for A rows (M dimension)
    A_stride1,    # stride for A columns (K dimension)
    W_stride0,    # stride for W rows (K dimension)
    W_stride1,    # stride for W columns (N dimension)
    C_stride0,    # stride for C rows (M dimension)
    C_stride1,    # stride for C columns (N dimension)
    BLOCK_M: tl.constexpr,   # tile over rows (B*L)
    BLOCK_N: tl.constexpr,   # tile over columns (H)
    BLOCK_K: tl.constexpr,   # reduction tile (H)
):
    # Each program handles a tile [BLOCK_M, BLOCK_N] of (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = B * L  # total rows in A/C

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # row indices in [0, M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # column indices in [0, H)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # K indices in [0, H)

        # Load A tile: A[m, k] with strides (A_stride0 for rows, A_stride1 for cols)
        a_ptrs = A_ptr + m[:, None] * A_stride0 + k[None, :] * A_stride1
        a_mask = (m[:, None] < M) & (k[None, :] < H)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype follows A_ptr (float32 here)

        # Load W^T tile: we need W[k, n] (because W is [K, N])
        w_ptrs = W_ptr + k[:, None] * W_stride0 + n[None, :] * W_stride1
        w_mask = (k[:, None] < H) & (n[None, :] < H)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # dtype follows W_ptr

        # Accumulate
        acc += tl.dot(a, w)

    # Store result to C[m, n]
    c_ptrs = C_ptr + m[:, None] * C_stride0 + n[None, :] * C_stride1
    c_mask = (m[:, None] < M) & (n[None, :] < H)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_kernel(
    src_ptr,         # *const float: source tensor [B, L, H]
    dst_ptr,         # *float: destination tensor [B, L_copy, H]
    B: tl.int32, L: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.constexpr,    # starting row index to copy from src
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D grid: (B, ceil(L_copy/BLOCK_l), ceil(H/BLOCK_h))
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # row indices in [0, L_copy)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden dims

    # Compute source indices: we copy rows starting at ROW_START
    src_idx = (
        pid_b * src_s0
        + (ROW_START + l) * src_s1
        + h[None, :] * src_s2
    )
    dst_idx = (
        pid_b * dst_s0
        + l[:, None] * dst_s1
        + h[None, :] * dst_s2
    )

    mask = (l[:, None] < L) & (h[None, :] < H)
    vals = tl.load(src_ptr + src_idx, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_idx, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Ensure inputs are float32 and contiguous for simplicity
        # (You can remove .float()/.contiguous() if you want to preserve dtype; here we match typical usage and keep it simple.)
        # If you need to preserve dtype, comment out .float() and keep the rest; kernels operate in the tensor's dtype.
        e = encoder_hidden_states.contiguous().float()
        i = hidden_states.contiguous().float()
        W = process_weight.contiguous().float()

        # Allocate concatenated output
        concatenated = torch.empty((B, L, H), dtype=torch.float32, device=e.device)

        # Launch concatenation kernel
        BLOCK_l = 64
        BLOCK_h = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_encoder_image_kernel[grid_concat](
            e, i, concatenated,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # Allocate processed output
        processed = torch.empty((B, L, H), dtype=torch.float32, device=concatenated.device)

        # Launch matmul kernel: A = concatenated, W = process_weight (H x H)
        M = B * L
        # Choose block sizes; H is usually moderate (e.g., 128-1024). We tile across M and N and reduce over K=H.
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_kernel[grid_matmul](
            concatenated, W, processed,
            B, L, H,
            concatenated.stride(0), concatenated.stride(2),
            W.stride(0), W.stride(1),
            processed.stride(0), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=processed.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=processed.device)

        # Launch copy kernels to split
        # Copy first T rows: rows 0..T-1
        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, L, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows: rows T..T+I-1
        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, L, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Return as requested (note: original returns tensors of shape [B, seq, H]; we keep float32)
        return processed_encoder, processed_hidden