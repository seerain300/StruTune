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
    BLOCK_N: tl.constexpr,  # tile along hidden dim
):
    # Grid: (B, M), where M = T + I
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

    # Load selected row
    e_vals = tl.load(e_ptr + e_row_offset + n_offsets * stride_eh, mask=mask_cols, other=0.0)
    i_vals = tl.load(i_ptr + i_row_offset + n_offsets * stride_ih, mask=mask_cols, other=0.0)
    selected = tl.where(from_encoder, e_vals, i_vals)

    # Store into output
    out_row_offset = b * stride_ob + m * stride_om
    tl.store(out_ptr + out_row_offset + n_offsets * stride_on, selected, mask=mask_cols)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Make contiguous; keep original dtype to match original behavior
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        w_t = process_weight.t().contiguous()  # weight^T for original code

        # Allocate concatenated input [B, M, H]
        X_cat = torch.empty((B, M, H), dtype=e.dtype, device=e.device)

        # Launch concatenation kernel: grid over (B, M)
        BLOCK_N = 128  # tile along hidden dim; adjust if H is small
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            e, i, X_cat,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Perform linear projection using PyTorch (to ensure exact correctness)
        processed = torch.matmul(X_cat, w_t)  # [B, M, H] @ [H, H] -> [B, M, H]

        # Split into encoder and hidden streams (host-side, no torch ops for split)
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
