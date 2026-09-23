import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # [B, T, H]
    img_ptr,       # [B, I, H]
    A_ptr,         # [M_total, H], M_total = B*(T+I)
    B, T, I, H,    # dimensions
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_b_a, stride_h_a,               # A strides
    num_warps: tl.constexpr,
):
    # Grid: (M_total,)
    m = tl.program_id(0)
    M_total = B * (T + I)
    if m >= M_total:
        return

    # Compute batch and sequence index
    b = m // (T + I)
    seq = m % (T + I)

    offs = tl.arange(0, H)
    mask = offs < H

    # Choose source tensor
    use_img = seq >= T
    if use_img:
        src_ptr = img_ptr + b * stride_b_i + (seq - T) * stride_i_i + offs * stride_h_i
    else:
        src_ptr = enc_ptr + b * stride_b_e + seq * stride_t_e + offs * stride_h_e

    # Store to A[m, :]
    A_row_ptr = A_ptr + m * stride_b_a + offs * stride_h_a
    tl.store(A_row_ptr, tl.load(src_ptr, mask=mask, other=0.0))


@triton.jit
def triton_gemm_AWT_kernel(
    A_ptr,           # [M_total, H] input rows
    B_ptr,           # [H, H] process_weight.T
    C_ptr,           # [M_total, H] output
    M_total, H,      # dims
    stride_a_row, stride_a_col,       # A strides
    stride_b_row, stride_b_col,       # B strides (rows = H, cols = H)
    stride_c_row, stride_c_col,       # C strides
    num_warps: tl.constexpr,
):
    # Grid: (M_total, H)
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Bounds check
    if m >= M_total or n >= H:
        return

    # Accumulator for a single row m over H columns
    acc = tl.zeros((H,), dtype=tl.float32)

    # Loop over reduction dimension in tiles of BLOCK_K
    BLOCK_K = 64  # tuneable; 64 works well for typical H up to 4096
    k = 0
    while k < H:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # Load a chunk of A_row: A[m, offs_k]
        a_ptrs = A_ptr + m * stride_a_row + offs_k * stride_a_col
        a_vals = tl.load(a_ptrs, mask=mask_k, other=0.0)  # dtype follows A
        a_vals = a_vals.to(tl.float32)  # ensure fp32 accumulation

        # Load a chunk of B: B[offs_k, n]
        b_ptrs = B_ptr + offs_k * stride_b_row + n * stride_b_col
        b_vals = tl.load(b_ptrs, mask=mask_k, other=0.0)  # dtype follows B
        b_vals = b_vals.to(tl.float32)

        # Accumulate dot product
        acc += tl.sum(a_vals[:, None] * b_vals[None, :], axis=0)

        k += BLOCK_K

    # Store result to C[m, n]
    c_ptrs = C_ptr + m * stride_c_row + n * stride_c_col
    tl.store(c_ptrs, acc.to(tl.float32))  # store as fp32; if C was allocated as fp32 this is fine


@triton.jit
def copy_rows_to_encoder_kernel(
    C_ptr,               # [M_total, H]
    out_ptr,             # [B, T, H]
    B, T, I, H,          # dims
    stride_c_row, stride_c_col,  # C strides
    out_stride_b, out_stride_t, out_stride_h,  # out strides
    num_warps: tl.constexpr,
):
    # Grid: (B, T)
    b = tl.program_id(0)
    seq = tl.program_id(1)

    # Each program copies a single row: C[b*T + seq, :] -> out[b, seq, :]
    start_row = b * T + seq
    M_total = B * (T + I)

    # Iterate over columns in chunks of 128
    n = 0
    while n < H:
        offs_n = n + tl.arange(0, 128)
        mask_n = offs_n < H
        c_ptrs = C_ptr + start_row * stride_c_row + offs_n * stride_c_col
        values = tl.load(c_ptrs, mask=mask_n, other=0.0)
        out_ptrs = out_ptr + b * out_stride_b + seq * out_stride_t + offs_n * out_stride_h
        tl.store(out_ptrs, values, mask=mask_n)
        n += 128


@triton.jit
def copy_rows_to_hidden_kernel(
    C_ptr,               # [M_total, H]
    out_ptr,             # [B, I, H]
    B, T, I, H,          # dims
    stride_c_row, stride_c_col,  # C strides
    out_stride_b, out_stride_i, out_stride_h,  # out strides
    num_warps: tl.constexpr,
):
    # Grid: (B, I)
    b = tl.program_id(0)
    seq = tl.program_id(1)

    # Each program copies a single row: C[b*(T+I) + T + seq, :] -> out[b, seq, :]
    start_row = b * (T + I) + T + seq
    M_total = B * (T + I)

    # Iterate over columns in chunks of 128
    n = 0
    while n < H:
        offs_n = n + tl.arange(0, 128)
        mask_n = offs_n < H
        c_ptrs = C_ptr + start_row * stride_c_row + offs_n * stride_c_col
        values = tl.load(c_ptrs, mask=mask_n, other=0.0)
        out_ptrs = out_ptr + b * out_stride_b + seq * out_stride_i + offs_n * out_stride_h
        tl.store(out_ptrs, values, mask=mask_n)
        n += 128


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states into A [B*(T+I), H] via Triton.
        - Computes C = A @ process_weight.T via Triton GEMM.
        - Copies rows into processed_encoder and processed_hidden via Triton.
        """
        # Ensure contiguous tensors
        B, T, H = encoder_hidden_states.shape
        B_img, I, H_img = hidden_states.shape
        assert B_img == B, "Batch sizes must match"
        assert H == H_img, "Hidden dimensions must match"

        # Allocate A: [M_total, H], M_total = B * (T + I)
        M_total = B * (T + I)
        A = torch.empty((M_total, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch Triton kernel to build A
        concat_rows_to_A_kernel[(M_total,)](
            encoder_hidden_states, hidden_states, A,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            A.stride(0), A.stride(1),
            num_warps=4,
        )

        # Prepare weight^T and output C. Use fp32 for robust accumulation, then we can cast outputs back.
        Wt = process_weight.t().contiguous()  # [H, H]
        # Allocate C as fp32 for numeric stability in Triton kernel
        C = torch.empty((M_total, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton GEMM kernel: C = A @ Wt
        triton_gemm_AWT_kernel[(M_total, H)](
            A, Wt, C,
            M_total, H,
            A.stride(0), A.stride(1),
            Wt.stride(0), Wt.stride(1),
            C.stride(0), C.stride(1),
            num_warps=4,
        )

        # Prepare outputs (fp32 for now; final cast if needed)
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton copy kernels per batch to split C
        # Copy first B*T rows into processed_encoder
        grid_e = (B, T)
        copy_rows_to_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            num_warps=4,
        )

        # Copy remaining B*I rows into processed_hidden
        grid_h = (B, I)
        copy_rows_to_hidden_kernel[grid_h](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=4,
        )

        # Cast outputs back to original dtype if desired (original uses default float32, so keep as is)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
