import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr, i_ptr, out_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_ob, stride_om, stride_on,
    BLOCK_N: tl.constexpr,  # tile along hidden dim H
):
    # Grid: (B, M) where M = T + I
    b = tl.program_id(0)
    m = tl.program_id(1)  # row index in concatenated matrix

    # column offsets
    n_offsets = tl.arange(0, BLOCK_N)
    mask_cols = n_offsets < H

    # Determine source: encoder if m < T, else image at row (m - T)
    from_encoder = m < T
    # Base offsets for source rows
    e_row_offset = b * stride_eb + m * stride_et
    i_row_offset = b * stride_ib + (m - T) * stride_it

    # Load selected row (masked for columns)
    e_vals = tl.load(e_ptr + e_row_offset + n_offsets * stride_eh, mask=mask_cols, other=0.0)
    i_vals = tl.load(i_ptr + i_row_offset + n_offsets * stride_ih, mask=mask_cols, other=0.0)
    selected = tl.where(from_encoder, e_vals, i_vals)

    # Store into output
    out_row_offset = b * stride_ob + m * stride_om
    tl.store(out_ptr + out_row_offset + n_offsets * stride_on, selected, mask=mask_cols)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA and contiguous
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Convert to float32 for numerical stability in Triton
        e = encoder_hidden_states.contiguous().to(torch.float32)
        i = hidden_states.contiguous().to(torch.float32)
        w = process_weight.contiguous().to(torch.float32)

        # Allocate output for concatenated rows: [B, M, H]
        X_cat = torch.empty((B, M, H), device=e.device, dtype=torch.float32)

        # Launch Triton kernel to build concatenated matrix
        BLOCK_N = 128  # tile along hidden dim; H is often <= 1024
        grid = (B, M)
        cat_rows_kernel[grid](
            e, i, X_cat,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Linear projection using torch (allowed): Y = X_cat @ process_weight
        # process_weight is [H, H], no bias
        Y = torch.matmul(X_cat, w)

        # Split results
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        # Cast back to original dtype to match original function
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(encoder_hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
