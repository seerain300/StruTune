import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_to_A_BTI_kernel_for_enc(
    enc_ptr, A_ptr,
    B, T, H,
    stride_b_e, stride_t_e, stride_h_e,
    stride_b_a, stride_s_a, stride_h_a,
    BLOCK_COLS: tl.constexpr,
):
    # Grid: (B, ceil(T / BLOCK_COLS))
    b = tl.program_id(0)
    col_block = tl.program_id(1)

    col_off = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask_cols = col_off < H

    # sequence index for encoder part
    s = 0  # we process s=0..T-1 in separate program instances
    # Base pointers
    enc_row_ptr = enc_ptr + b * stride_b_e + s * stride_t_e
    A_row_ptr = A_ptr + b * stride_b_a + s * stride_s_a

    vals = tl.load(enc_row_ptr + col_off * stride_h_e, mask=mask_cols, other=0.0)
    tl.store(A_row_ptr + col_off * stride_h_a, vals, mask=mask_cols)


@triton.jit
def copy_rows_to_A_BTI_kernel_for_img(
    img_ptr, A_ptr,
    B, I, H,
    stride_b_i, stride_i_i, stride_h_i,
    stride_b_a, stride_s_a, stride_h_a,
    start_img,  # starting index in A's sequence dim (T)
    BLOCK_COLS: tl.constexpr,
):
    # Grid: (B, ceil(I / BLOCK_COLS))
    b = tl.program_id(0)
    col_block = tl.program_id(1)

    col_off = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask_cols = col_off < H

    i = 0  # we process i=0..I-1 in separate program instances
    s_in_A = start_img + i  # position in A's sequence dim

    img_row_ptr = img_ptr + b * stride_b_i + i * stride_i_i
    A_row_ptr = A_ptr + b * stride_b_a + s_in_A * stride_s_a

    vals = tl.load(img_row_ptr + col_off * stride_h_i, mask=mask_cols, other=0.0)
    tl.store(A_row_ptr + col_off * stride_h_a, vals, mask=mask_cols)


@triton.jit
def copy_rows_to_output_batch_kernel(
    C_ptr, out_ptr,
    M_rows,  # number of rows to copy into 'out'
    B, T, H,
    stride_m_c, stride_h_c,
    out_stride_b, out_stride_t, out_stride_h,
    start_row,  # starting row in C to copy
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (B,)
    b = tl.program_id(0)
    # row indices in C to copy
    row_off = tl.arange(0, BLOCK_M)
    # we will iterate rows by masking; here we just handle one tile per batch
    mask_m = row_off < M_rows

    # Load tile of rows from C
    c_rows = tl.load(C_ptr + (start_row + row_off) * stride_m_c + tl.arange(0, BLOCK_N) * stride_h_c,
                     mask=mask_m[:, None], other=0.0)

    # Store into out[b, row_off, :]
    out_row_base = b * out_stride_b
    tl.store(out_ptr + out_row_base + row_off[:, None] * out_stride_t + tl.arange(0, BLOCK_N)[None, :] * out_stride_h,
             c_rows, mask=mask_m[:, None])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Inputs: hidden_states [B, I, H], encoder_hidden_states [B, T, H], process_weight [H, H]
        B, I, H = hidden_states.shape
        B_e, T, H_e = encoder_hidden_states.shape
        assert B_e == B and H == H_e, "Batch or hidden_dim mismatch"

        S = T + I

        # Allocate A [B, S, H]
        A = torch.empty((B, S, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # 1) Build A via Triton kernels: copy encoder rows to A[:, :T, :], image rows to A[:, T:, :]
        BLOCK_COLS = 128
        grid_enc = (B, triton.cdiv(T, BLOCK_COLS))
        copy_rows_to_A_BTI_kernel_for_enc[grid_enc](
            encoder_hidden_states, A,
            B, T, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            BLOCK_COLS=BLOCK_COLS,
            num_warps=4, num_stages=2,
        )

        grid_img = (B, triton.cdiv(I, BLOCK_COLS))
        copy_rows_to_A_BTI_kernel_for_img[grid_img](
            hidden_states, A,
            B, I, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            start_img=T,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = A @ process_weight.T using torch for robustness
        BT = process_weight.t().contiguous()  # [H, H]
        processed_cat = torch.matmul(A, BT)  # [B, S, H]
        processed = processed_cat.reshape(B * S, H)  # [B*S, H]

        # 3) Split into encoder and hidden via Triton copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=processed.device)

        # Copy encoder rows: rows [0 : B*T)
        M_encoder = B * T
        copy_rows_to_output_batch_kernel[(B,)](
            processed, processed_encoder,
            M_encoder, B, T, H,
            processed.stride(0), processed.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=0,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # Copy hidden rows: rows [B*T : B*S)
        start_row_hidden = B * T
        M_hidden = B * I
        copy_rows_to_output_batch_kernel[(B,)](
            processed, processed_hidden,
            M_hidden, B, I, H,
            processed.stride(0), processed.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row=start_row_hidden,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
