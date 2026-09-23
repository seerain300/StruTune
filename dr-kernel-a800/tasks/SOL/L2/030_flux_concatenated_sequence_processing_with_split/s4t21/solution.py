import torch
import triton
import triton.language as tl


@triton.jit
def concat_to_A_rows_kernel(
    enc_ptr,     # *ptr to [B, T, H]
    img_ptr,     # *ptr to [B, I, H]
    A_ptr,       # *ptr to [M, H], M = B*(T+I)
    B, T, I, H,  # dimensions
    stride_b_e, stride_t_e, stride_h_e,   # strides for encoder
    stride_b_i, stride_i_i, stride_h_i,   # strides for image
    stride_b_a, stride_h_a,                # strides for A
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each program handles a tile of rows (BLOCK_M) and columns (BLOCK_N)
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    M = B * (T + I)

    # Compute batch and seq for each row
    b = rows // (T + I)           # [BLOCK_M]
    seq = rows % (T + I)          # [BLOCK_M]

    # Determine which source tensor (encoder or image)
    is_encoder = seq < T          # [BLOCK_M]

    # Offsets in source tensors
    e_off = b[:, None] * stride_b_e + seq[:, None] * stride_t_e
    i_off = b[:, None] * stride_b_i + (seq[:, None] - T) * stride_i_i

    # Column offsets in A
    col = tl.arange(0, BLOCK_N)
    col = col[None, :]            # shape [1, BLOCK_N]

    # Masks for loads
    mask_rows = rows < M
    mask_cols = col < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Load from appropriate source based on is_encoder
    vals = tl.load(enc_ptr + e_off, mask=(mask & is_encoder[:, None]), other=0.0)
    vals = vals + tl.load(img_ptr + i_off, mask=(mask & (~is_encoder)[:, None]), other=0.0)

    # Store into A
    a_off = rows[:, None] * stride_b_a + col * stride_h_a
    tl.store(A_ptr + a_off, vals, mask=mask)


@triton.jit
def matmul_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,        # A strides: row-major [M, K]
    stride_bk, stride_bn,        # B strides: row-major [K, N]
    stride_cm, stride_cn,        # C strides: row-major [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in A/C
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in B/C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for tiles
        a_ptrs = A_ptr + m[:, None] * stride_am + k[None, :] * stride_ak
        b_ptrs = B_ptr + k[:, None] * stride_bk + n[None, :] * stride_bn

        a_mask = (m[:, None] < M) & (k[None, :] < K)
        b_mask = (k[:, None] < K) & (n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Write back to C
    c_ptrs = C_ptr + m[:, None] * stride_cm + n[None, :] * stride_cn
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_to_output_kernel(
    src_ptr, out_ptr,
    B, T, I, H,
    src_stride0, src_stride1, src_stride2,
    out_stride0, out_stride1, out_stride2,
    start_row,  # starting row offset within src for this batch
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = start_row + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in src
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)             # cols in hidden_dim

    # Batch index for each row
    b = rows // (T + I)
    # Output batch slice
    out_b = b[0]  # single batch for this kernel invocation

    # Masks
    M_total = B * (T + I)
    mask_rows = rows < M_total
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Compute src pointers: src_ptr[rows, cols]
    src_ptrs = src_ptr + rows[:, None] * src_stride0 + cols[None, :] * src_stride1
    # Compute out pointers: out_ptr[out_b, rows - start_row, cols]
    out_rows = rows - start_row
    out_ptrs = out_ptr + out_b * out_stride0 + out_rows[:, None] * out_stride1 + cols[None, :] * out_stride2

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Fuses concatenation, matmul, and splitting into Triton kernels.
        Returns (processed_encoder, processed_hidden) with shapes [B, T, H] and [B, I, H].
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, L, H]"
        assert hidden_states.shape[2] == encoder_hidden_states.shape[2] == process_weight.shape[0], "Hidden dim mismatch"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = B * (T + I)

        # Ensure inputs are contiguous and on the same device
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight_T = process_weight.t().contiguous()  # [H, H]
        device = enc.device

        # 1) Concatenate along sequence dimension into A [M, H] using Triton
        A = torch.empty((M, H), dtype=enc.dtype, device=device)
        BLOCK_M = 128
        BLOCK_N = 64
        grid_concat = (triton.cdiv(M, BLOCK_M),)
        concat_to_A_rows_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: C = A @ weight_T, C is [M, H]
        C = torch.empty((M, H), dtype=enc.dtype, device=device)
        K = H
        grid_gemm = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_gemm_kernel[grid_gemm](
            A, weight_T, C,
            M, H, K,
            A.stride(0), A.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # 3) Split C into processed_encoder and processed_hidden using Triton per-batch copies

        # For encoder: rows [0, B*T)
        processed_encoder = torch.empty((B, T, H), dtype=enc.dtype, device=device)
        grid_encoder = (triton.cdiv(B * T, BLOCK_M), triton.cdiv(H, BLOCK_N))
        copy_rows_to_output_kernel[grid_encoder](
            C, processed_encoder,
            B, T, I, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=0,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # For hidden: rows [B*T, M)
        processed_hidden = torch.empty((B, I, H), dtype=enc.dtype, device=device)
        grid_hidden = (triton.cdiv(M - B * T, BLOCK_M), triton.cdiv(H, BLOCK_N))
        copy_rows_to_output_kernel[grid_hidden](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row=B * T,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
