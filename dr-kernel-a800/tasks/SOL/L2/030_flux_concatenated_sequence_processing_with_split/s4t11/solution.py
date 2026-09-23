import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_rows_to_A_kernel(
    enc_ptr,       # [B, T, H]
    A_ptr,         # [M, H], M = B*T
    B, T, I, H,    # dimensions (I unused here)
    stride_b_e, stride_t_e, stride_h_e,  # strides for enc
    stride_b_a, stride_h_a,               # strides for A
    BLOCK_M: tl.constexpr,        # rows per program
    BLOCK_H: tl.constexpr,        # columns per program
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_rows = rows < (B * T)

    b = rows // T
    seq = rows % T

    enc_row_ptrs = enc_ptr + b * stride_b_e + seq * stride_t_e + tl.arange(0, BLOCK_H) * stride_h_e
    A_row_ptrs = A_ptr + rows * stride_b_a + tl.arange(0, BLOCK_H) * stride_h_a

    vals = tl.load(enc_row_ptrs, mask=mask_rows, other=0)
    tl.store(A_row_ptrs, vals, mask=mask_rows)


@triton.jit
def concat_img_rows_to_A_kernel(
    img_ptr,       # [B, I, H]
    A_ptr,         # [M, H], M = B*(T+I)
    B, T, I, H,    # dimensions (T unused here)
    stride_b_i, stride_i_i, stride_h_i,  # strides for img
    stride_b_a, stride_h_a,               # strides for A
    BLOCK_M: tl.constexpr,        # rows per program
    BLOCK_H: tl.constexpr,        # columns per program
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    M_total = B * (T + I)
    start_img = B * T
    mask_rows = (rows + start_img) < M_total

    abs_rows = rows + start_img

    b = abs_rows // (T + I)
    seq = abs_rows - start_img  # in [0, I)

    img_row_ptrs = img_ptr + b * stride_b_i + seq * stride_i_i + tl.arange(0, BLOCK_H) * stride_h_i
    A_row_ptrs = A_ptr + abs_rows * stride_b_a + tl.arange(0, BLOCK_H) * stride_h_a

    vals = tl.load(img_row_ptrs, mask=mask_rows, other=0)
    tl.store(A_row_ptrs, vals, mask=mask_rows)


@triton.jit
def copy_rows_to_encoder_kernel(
    C_ptr,            # [M, H], M = B*(T+I)
    out_ptr,          # [B, T, H]
    B, T, I, H,       # dims (I unused)
    stride_c_b, stride_c_h,     # C strides
    out_stride_b, out_stride_t, out_stride_h,  # output strides
    start_row,         # starting row in C (usually 0)
    BLOCK_M: tl.constexpr,      # rows per program
    BLOCK_H: tl.constexpr,      # columns per program
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_rows = rows < (B * T)
    b = rows // T
    seq = rows % T

    C_row_ptrs = C_ptr + (start_row + rows) * stride_c_b + tl.arange(0, BLOCK_H) * stride_c_h
    out_row_ptrs = out_ptr + b * out_stride_b + seq * out_stride_t + tl.arange(0, BLOCK_H) * out_stride_h

    vals = tl.load(C_row_ptrs, mask=mask_rows, other=0)
    tl.store(out_row_ptrs, vals, mask=mask_rows)


@triton.jit
def copy_rows_to_hidden_kernel(
    C_ptr,            # [M, H], M = B*(T+I)
    out_ptr,          # [B, I, H]
    B, T, I, H,       # dims
    stride_c_b, stride_c_h,     # C strides
    out_stride_b, out_stride_h, out_stride_i,  # output strides
    start_row,         # starting row in C (usually B*T)
    BLOCK_M: tl.constexpr,      # rows per program
    BLOCK_H: tl.constexpr,      # columns per program
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_rows = rows < (B * I)
    abs_rows = rows + start_row
    b = abs_rows // (T + I)
    seq = abs_rows - start_row  # seq in [0, I)

    C_row_ptrs = C_ptr + abs_rows * stride_c_b + tl.arange(0, BLOCK_H) * stride_c_h
    out_row_ptrs = out_ptr + b * out_stride_b + seq * out_stride_i + tl.arange(0, BLOCK_H) * out_stride_h

    vals = tl.load(C_row_ptrs, mask=mask_rows, other=0)
    tl.store(out_row_ptrs, vals, mask=mask_rows)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only data movement for concatenation and splitting; matrix multiply via torch for numerical robustness.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton."
        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2 and H == H2, "hidden_states and encoder_hidden_states must have the same batch and hidden_dim."
        assert process_weight.shape == (H, H), "process_weight must have shape [hidden_dim, hidden_dim]."

        M_total = B * (T + I)

        # Allocate concatenated A [M_total, H]
        A = torch.empty((M_total, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()

        # Grid and block sizes
        BLOCK_M = 128
        BLOCK_H = 64

        # Copy encoder rows into A[0 : B*T, :]
        grid_e = (triton.cdiv(B * T, BLOCK_M),)
        concat_encoder_rows_to_A_kernel[grid_e](
            enc, A, B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Copy image rows into A[B*T : M, :]
        grid_i = (triton.cdiv(B * I, BLOCK_M),)
        concat_img_rows_to_A_kernel[grid_i](
            img, A, B, T, I, H,
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Matrix multiply via torch for robustness: processed = A @ process_weight.T
        weight_T = process_weight.t().contiguous()
        processed = torch.matmul(A, weight_T)  # [M_total, H]

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=processed.device)

        # Launch per-batch copy kernels to split processed
        grid_e_copy = (triton.cdiv(B * T, BLOCK_M),)
        copy_rows_to_encoder_kernel[grid_e_copy](
            processed, processed_encoder,
            B, T, I, H,
            processed.stride(0), processed.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=0,
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        grid_h_copy = (triton.cdiv(B * I, BLOCK_M),)
        copy_rows_to_hidden_kernel[grid_h_copy](
            processed, processed_hidden,
            B, T, I, H,
            processed.stride(0), processed.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(2), processed_hidden.stride(1),
            start_row=B * T,
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
