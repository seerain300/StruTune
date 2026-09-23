import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,         # *const float, [B, T, H]
    h_ptr,         # *const float, [B, I, H]
    out_ptr,       # *float,       [B, M, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_m, stride_o_h,
    M: tl.constexpr,  # M = T + I
):
    b = tl.program_id(0)
    p = tl.program_id(1)  # row index in concatenated sequence

    # Determine source: first T rows from e, remaining from h
    is_text = p < T

    # Compute source and destination base pointers for this row
    if is_text:
        src_row = e_ptr + b * stride_e_b + p * stride_e_t
    else:
        src_row = h_ptr + b * stride_h_b + (p - T) * stride_h_i

    dst_row = out_ptr + b * stride_o_b + p * stride_o_m

    # Copy H elements from src_row to dst_row
    offs = tl.arange(0, H)
    vals = tl.load(src_row + offs * stride_e_h if is_text else src_row + offs * stride_h_h)
    tl.store(dst_row + offs * stride_o_h, vals)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only forward for concatenation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim using Triton (no torch.cat).
        - Apply linear projection using torch.matmul (PyTorch).
        - Split back into separate streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Ensure contiguous for simple stride math
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        # Allocate concatenated tensor X_cat [B, M, H]
        X_cat = torch.empty((B, M, H), dtype=e.dtype, device=e.device)

        # Launch Triton kernel to fill X_cat
        grid = (B, M)
        cat_rows_kernel[grid](
            e, h, X_cat,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            M,
            num_warps=1, num_stages=1,
        )

        # Linear projection: [B, M, H] @ [H, H] -> [B, M, H]
        processed = torch.matmul(X_cat, w)

        # Split results into encoder and image streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
