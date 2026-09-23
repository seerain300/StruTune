import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,        # *fp32, [B, T, H]
    img_ptr,        # *fp32, [B, I, H]
    A_ptr,          # *fp32, [M, H], M = B*(T+I)
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_m_a, stride_n_a,               # A strides
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row m and a tile of columns n
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row index and column tile
    m = pid_m  # 0 .. M-1
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_n = n < H

    # Compute batch and sequence indices
    TpI = T + I
    b = m // TpI
    s = m % TpI

    # Decide source tensor and load
    use_encoder = s < T
    # Compute source indices
    # enc row index: b, col index: s
    enc_row = b
    enc_col = s
    # img row index: b, col index: s - T
    img_row = b
    img_col = s - T

    # Compute element pointers for enc and img
    # enc_ptr + b*stride_b_e + s*stride_t_e + n*stride_h_e
    enc_el_ptrs = enc_ptr + b * stride_b_e + enc_col * stride_t_e + n * stride_h_e  # shape [BLOCK_N]
    # img_ptr + b*stride_b_i + (s - T)*stride_i_i + n*stride_h_i
    img_el_ptrs = img_ptr + b * stride_b_i + img_col * stride_i_i + n * stride_h_i  # shape [BLOCK_N]

    # Select element values based on use_encoder
    # Create a mask for n
    mask_n1 = mask_n
    # Build values
    vals = tl.zeros((BLOCK_N,), dtype=tl.float32)
    vals += tl.load(enc_el_ptrs, mask=mask_n1 & use_encoder, other=0.0)
    vals += tl.load(img_el_ptrs, mask=mask_n1 & (~use_encoder), other=0.0)

    # Store into A[m, n]
    A_row_ptr = A_ptr + m * stride_m_a + n * stride_n_a
    tl.store(A_row_ptr, vals, mask=mask_n1)


@triton.jit
def batched_matmul_kernel(
    A_ptr,           # *fp32, [M, H]
    Wt_ptr,          # *fp32, [H, H] (process_weight.T)
    C_ptr,           # *fp32, [M, H]
    M: tl.int32, H: tl.int32,  # M = B*(T+I)
    stride_am, stride_ak,      # A strides: (row, col)
    stride_wk, stride_wn,      # Wt strides: (row=k, col=n)
    stride_cm, stride_cn,      # C strides: (row, col)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (tiles over M, tiles over N, tiles over K)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Tile coordinates
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    k0 = pid_k * BLOCK_K

    # Ranges for this tile
    m = m0 + tl.arange(0, BLOCK_M)       # [BLOCK_M]
    n = n0 + tl.arange(0, BLOCK_N)       # [BLOCK_N]
    k = k0 + tl.arange(0, BLOCK_K)       # [BLOCK_K]

    # Masks for boundaries
    mask_m = m < M
    mask_n = n < H
    mask_k = k < H

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    # k ranges from k0 to k0 + BLOCK_K - 1
    for kk in range(BLOCK_K):
        kk_val = k0 + kk
        # A_tile: [BLOCK_M, 1] by broadcasting
        # A[m, kk_val]
        A_ptrs = A_ptr + m[:, None] * stride_am + kk_val * stride_ak
        A_mask = mask_m[:, None] & (kk_val < H)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # shape [BM, 1]

        # W_tile: [1, BLOCK_N] by broadcasting
        # W^T[kk_val, n]
        W_ptrs = Wt_ptr + kk_val * stride_wk + n[None, :] * stride_wn
        W_mask = (kk_val < H) & mask_n[None, :]
        W_tile = tl.load(W_ptrs, mask=W_mask, other=0.0)  # shape [1, BN]

        # Accumulate
        acc += A_tile @ W_tile  # [BM, BN]

    # Store result to C[m, n]
    C_ptrs = C_ptr + m[:, None] * stride_cm + n[None, :] * stride_cn
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,           # *fp32, [M, H]
    out_ptr,         # *fp32, [B, T, H]
    B: tl.int32, T: tl.int32, H: tl.int32,
    stride_c_m, stride_c_h,
    stride_out_b, stride_out_t, stride_out_h,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, cdiv(H, BLOCK_N))
    b = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Rows this batch owns: 0 .. B*T-1
    start_row = b * (T * 1 + 0)  # 0 is fine, but we can compute start as b*T more robustly below
    # We need start_row = b*(T+I) ? No: encoder rows are m in [0, B*T), independent of I.
    # For encoder, m = b*T + t, t in [0, T).
    # Compute t = pid_n * BLOCK_N + arange and use t, but we can iterate over t directly in another grid if needed.
    # Here, we set grid to (B, T) and simplify: launch with grid=(B, T) for exact one row copy per program, or grid=(B, cdiv(H,BLOCK_N)) and loop t, but that adds complexity.
    # Instead, we set grid=(B, T) and re-define kernel to handle t. To keep simple, we relaunch with appropriate grid.
    # To avoid complexity, we implement a grid over (B, T) and copy one row at a time in separate kernel (already done via separate kernel below).
    # Therefore, we redefine the launch as batched over t with separate kernel; this kernel is not used in final forward below.

    # Note: In practice, we will use a different kernel for encoder rows that maps directly over t.
    # Placeholder: just return (kept for signature consistency).
    return


# We will use a dedicated copy kernel for encoder rows that maps each program to a single row m = b*T + t.
@triton.jit
def copy_row_encoder_kernel(
    C_ptr,           # *fp32, [M, H]
    out_ptr,         # *fp32, [B, T, H]
    B: tl.int32, T: tl.int32, H: tl.int32,
    stride_c_m, stride_c_h,
    stride_out_b, stride_out_t, stride_out_h,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B*T, cdiv(H, BLOCK_N))
    m = tl.program_id(0)
    pid_n = tl.program_id(1)

    b = m // T
    t = m % T

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < H

    # Load C[m, :]
    C_row_ptrs = C_ptr + m * stride_c_m + n * stride_c_h
    vals = tl.load(C_row_ptrs, mask=mask_n, other=0.0)

    # Store into out[b, t, :]
    out_row_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + n * stride_out_h
    tl.store(out_row_ptrs, vals, mask=mask_n)


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,           # *fp32, [M, H]
    out_ptr,         # *fp32, [B, I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_c_m, stride_c_h,
    stride_out_b, stride_out_i, stride_out_h,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, cdiv(H, BLOCK_N))
    b = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Rows this batch owns in hidden: m in [b*(T+I) + T, (b+1)*(T+I))
    start_row = b * (T + I) + T
    end_row = (b + 1) * (T + I)

    # We need to copy all rows from start_row to end_row. Do it by looping t over [0, I) per batch.
    # To keep a 2D grid, we compute m = start_row + t for each t in [0, I). Use a separate kernel per t is not ideal.
    # Instead, we iterate over t inside the kernel by launching grid=(B, I) and compute m per program.

    # Compute t from program_id(0) and program_id(1): we need only pid_n; pid_m is b. To avoid confusion, we set grid=(B, I).
    # However, Triton grid is fixed; so we cannot change per-program logic to depend on I. Therefore, we launch this kernel
    # with grid=(B, 1) and iterate t in host. But Triton kernels are static; we will instead use a specialized kernel for hidden rows.

    # Placeholder: redefine with proper grid=(B, I) and copy per t. We'll do that below using a separate hidden kernel.

    return


# Specialized hidden copy kernel: grid=(B, I), copy C[b*(T+I)+t, :] into out[b, t, :]
@triton.jit
def copy_row_hidden_kernel(
    C_ptr,           # *fp32, [M, H]
    out_ptr,         # *fp32, [B, I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_c_m, stride_c_h,
    stride_out_b, stride_out_i, stride_out_h,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, I)
    b = tl.program_id(0)
    t = tl.program_id(1)

    m = b * (T + I) + t
    n = tl.arange(0, BLOCK_N)
    mask_n = n < H

    # Load C[m, :]
    C_row_ptrs = C_ptr + m * stride_c_m + n * stride_c_h
    vals = tl.load(C_row_ptrs, mask=mask_n, other=0.0)

    # Store into out[b, t, :]
    out_row_ptrs = out_ptr + b * stride_out_b + t * stride_out_i + n * stride_out_h
    tl.store(out_row_ptrs, vals, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension into A using Triton.
        - Compute C = A @ process_weight.T using Triton batched matmul.
        - Split C into processed_encoder and processed_hidden using Triton copy kernels per batch.
        Returns: (processed_encoder [B, T, H], processed_hidden [B, I, H])
        """
        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        Wt = process_weight.t().contiguous()  # process_weight.T: [H, H]
        B, T, H = enc.shape
        _, I, _ = img.shape
        M = B * (T + I)

        # Allocate A and C
        A = torch.empty((M, H), dtype=torch.float32, device=enc.device)
        C = torch.empty((M, H), dtype=torch.float32, device=enc.device)

        # Launch concatenation kernel: grid over rows and column tiles
        BLOCK_N = 128  # tuneable
        grid_concat = (M, triton.cdiv(H, BLOCK_N))
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_N=BLOCK_N,
        )

        # Launch batched matmul kernel: grid over (M tiles, N tiles, K tiles)
        # Choose block sizes; H is hidden_dim. For typical H up to a few thousands, 64 is reasonable.
        BLOCK_M = 64
        BLOCK_N_m = 64
        BLOCK_K = 64
        grid_mm = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N_m), triton.cdiv(H, BLOCK_K))
        batched_matmul_kernel[grid_mm](
            A, Wt, C,
            M, H,
            A.stride(0), A.stride(1),
            Wt.stride(0), Wt.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_m, BLOCK_K=BLOCK_K,
        )

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=enc.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=enc.device)

        # Copy rows for encoder: m in [0, B*T)
        # We launch a 2D grid over (B*T rows, column tiles). But to keep exact one row per program, use grid=(B*T, 1) and loop n in kernel.
        # Instead, use a per-row kernel for clarity: copy_row_encoder_kernel with grid=(B*T, cdiv(H, BLOCK_N))
        BLOCK_N_copy = 128
        grid_encoder = (B * T, triton.cdiv(H, BLOCK_N_copy))
        copy_row_encoder_kernel[grid_encoder](
            C, processed_encoder,
            B, T, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_N=BLOCK_N_copy,
        )

        # Copy rows for hidden: m in [B*T, M)
        grid_hidden = (B, triton.cdiv(H, BLOCK_N_copy))
        copy_row_hidden_kernel[grid_hidden](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_N=BLOCK_N_copy,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
