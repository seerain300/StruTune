import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # *f32, [B, T, H]
    img_ptr,       # *f32, [B, I, H]
    A_ptr,         # *f32, [M_total, H], M_total = B*(T+I)
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_b_e: tl.int32, stride_t_e: tl.int32, stride_h_e: tl.int32,  # enc strides
    stride_b_i: tl.int32, stride_i_i: tl.int32, stride_h_i: tl.int32,  # img strides
    stride_m_a: tl.int32, stride_n_a: tl.int32,                        # A strides
    num_warps: tl.constexpr,
):
    # Each program handles one output row m in [0, M_total)
    m = tl.program_id(0)
    M_total = B * (T + I)
    if m >= M_total:
        return

    # Compute batch and sequence index
    b = m // (T + I)
    seq = m % (T + I)

    offs_n = tl.arange(0, H)
    mask_n = offs_n < H

    # Choose source tensor and build pointer
    src_ptr = None
    if seq < T:
        base = enc_ptr + b * stride_b_e + seq * stride_t_e
        src_ptr = base + offs_n * stride_h_e
    else:
        s_i = seq - T
        base = img_ptr + b * stride_b_i + s_i * stride_i_i
        src_ptr = base + offs_n * stride_h_i

    # Store to A[m, :]
    A_row_ptr = A_ptr + m * stride_m_a
    tl.store(A_row_ptr + offs_n * stride_n_a, tl.load(src_ptr, mask=mask_n, other=0.0))


@triton.jit
def matmul_rows_kernel(
    A_ptr,          # *f32, [M_total, H]
    B_ptr,          # *f32, [H, H] (process_weight.T)
    C_ptr,          # *f32, [M_total, H]
    M_total: tl.int32, H: tl.int32,
    stride_am_row: tl.int32, stride_am_col: tl.int32,  # A strides
    stride_b_row: tl.int32, stride_b_col: tl.int32,    # B strides
    stride_cm_row: tl.int32, stride_cm_col: tl.int32,  # C strides
    num_warps: tl.constexpr,
):
    # Each program handles one output row m in [0, M_total) and columns in chunks of 64
    m = tl.program_id(0)
    if m >= M_total:
        return

    # Precompute column offsets for this tile
    offs_n64 = tl.arange(0, 64)
    offs_n64_mask = offs_n64 < H

    # Accumulator for this row chunk
    acc = tl.zeros([64], dtype=tl.float32)

    # Reduction over K in chunks of 32
    K = H
    k0 = 0
    while k0 < K:
        offs_k32 = k0 + tl.arange(0, 32)
        mask_k32 = offs_k32 < K

        # Load A[m, k] vector (length 32)
        A_ptrs = A_ptr + m * stride_am_row + offs_k32 * stride_am_col
        a = tl.load(A_ptrs, mask=mask_k32, other=0.0)  # shape [32]

        # Load B[k, n] matrix (32x64)
        B_ptrs = B_ptr + offs_k32[:, None] * stride_b_row + offs_n64[None, :] * stride_b_col
        mask_B = mask_k32[:, None] & offs_n64_mask[None, :]
        b = tl.load(B_ptrs, mask=mask_B, other=0.0)  # shape [32, 64]

        # Accumulate: acc += a[kk] * b[kk, :]
        # Using tl.dot on (32,) and (32,64) yields (64,)
        acc += tl.dot(a, b)

        k0 += 32

    # Store result to C[m, :]
    C_row_ptr = C_ptr + m * stride_cm_row
    tl.store(C_row_ptr + offs_n64 * stride_cm_col, acc, mask=offs_n64_mask)


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,               # *f32, [M_total, H]
    out_ptr,             # *f32, [B, T, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    start_row: tl.int32,   # start row in C for encoder: 0
    stride_c_row: tl.int32, stride_c_col: tl.int32,
    out_stride_b: tl.int32, out_stride_t: tl.int32, out_stride_h: tl.int32,
    num_warps: tl.constexpr,
):
    # Grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if b >= B or t >= T:
        return

    m = b * (T + I) + t
    offs_n = tl.arange(0, 64)
    mask_n = offs_n < H

    C_row_ptr = C_ptr + m * stride_c_row + offs_n * stride_c_col
    out_row_ptr = out_ptr + b * out_stride_b + t * out_stride_t + offs_n * out_stride_h
    tl.store(out_row_ptr, tl.load(C_row_ptr, mask=mask_n, other=0.0))


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,               # *f32, [M_total, H]
    out_ptr,             # *f32, [B, I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    start_row: tl.int32,   # start row in C for hidden: B*T
    stride_c_row: tl.int32, stride_c_col: tl.int32,
    out_stride_b: tl.int32, out_stride_i: tl.int32, out_stride_h: tl.int32,
    num_warps: tl.constexpr,
):
    # Grid: (B, I)
    b = tl.program_id(0)
    i = tl.program_id(1)
    if b >= B or i >= I:
        return

    m = b * (T + I) + (T + i)
    offs_n = tl.arange(0, 64)
    mask_n = offs_n < H

    C_row_ptr = C_ptr + m * stride_c_row + offs_n * stride_c_col
    out_row_ptr = out_ptr + b * out_stride_b + i * out_stride_i + offs_n * out_stride_h
    tl.store(out_row_ptr, tl.load(C_row_ptr, mask=mask_n, other=0.0))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim into A [B*(T+I), H] using Triton.
        - Computes C = A @ process_weight.T using a Triton GEMM kernel (rows loop, column tiles).
        - Splits C into processed_encoder [B, T, H] and processed_hidden [B, I, H] using Triton copy kernels.

        Args:
            hidden_states: [B, I, H]
            encoder_hidden_states: [B, T, H]
            process_weight: [H, H]
        Returns:
            (processed_encoder, processed_hidden)
        """
        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight = process_weight.contiguous()

        B, T, H = enc.shape
        I = img.shape[1]
        M_total = B * (T + I)

        # 1) Concatenate into A [M_total, H]
        A = torch.empty((M_total, H), device=enc.device, dtype=enc.dtype)
        stride_b_e, stride_t_e, stride_h_e = enc.stride()
        stride_b_i, stride_i_i, stride_h_i = img.stride()
        stride_m_a, stride_n_a = A.stride()

        grid_concat = (M_total,)
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            stride_b_e, stride_t_e, stride_h_e,
            stride_b_i, stride_i_i, stride_h_i,
            stride_m_a, stride_n_a,
            num_warps=4,
        )

        # 2) Compute C = A @ process_weight.T using Triton GEMM over rows (columns in tiles of 64)
        B_T = weight.t().contiguous()  # [H, H]
        C = torch.empty((M_total, H), device=A.device, dtype=A.dtype)

        stride_am_row, stride_am_col = A.stride()
        stride_b_row, stride_b_col = B_T.stride()
        stride_cm_row, stride_cm_col = C.stride()

        grid_rows = (M_total,)
        matmul_rows_kernel[grid_rows](
            A, B_T, C,
            M_total, H,
            stride_am_row, stride_am_col,
            stride_b_row, stride_b_col,
            stride_cm_row, stride_cm_col,
            num_warps=4,
        )

        # 3) Split C into processed_encoder [B, T, H] and processed_hidden [B, I, H]
        processed_encoder = torch.empty((B, T, H), device=C.device, dtype=C.dtype)
        processed_hidden = torch.empty((B, I, H), device=C.device, dtype=C.dtype)

        # For outputs, use contiguous strides
        out_stride_b_e, out_stride_t_e, out_stride_h_e = processed_encoder.stride()
        out_stride_b_h, out_stride_i_h, out_stride_h_h = processed_hidden.stride()

        # Copy encoder rows 0..B*T-1
        grid_copy_e = (B, T)
        copy_rows_encoder_kernel[grid_copy_e](
            C, processed_encoder,
            B, T, I, H,
            0,  # start_row
            C.stride(0), C.stride(1),
            out_stride_b_e, out_stride_t_e, out_stride_h_e,
            num_warps=4,
        )

        # Copy hidden rows B*T..M_total-1
        start_hidden = B * (T + I)
        grid_copy_h = (B, I)
        copy_rows_hidden_kernel[grid_copy_h](
            C, processed_hidden,
            B, T, I, H,
            start_hidden,  # start_row
            C.stride(0), C.stride(1),
            out_stride_b_h, out_stride_i_h, out_stride_h_h,
            num_warps=4,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
