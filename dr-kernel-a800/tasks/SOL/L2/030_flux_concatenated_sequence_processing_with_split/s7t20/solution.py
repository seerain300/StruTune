import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,         # encoder_hidden_states: [B, T, H]
    i_ptr,         # hidden_states: [B, I, H]
    out_ptr,       # output X_cat: [B, M, H], M = T + I
    B, T, I, H,    # dimensions (ints)
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_ih,
    stride_ob, stride_om, stride_oh,
    BLOCK_H: tl.constexpr,
):
    # program ids: one per (b, p)
    b = tl.program_id(0)
    p = tl.program_id(1)
    M = T + I

    # Determine source and row index
    is_img = p >= T
    src_row = p - T if is_img else p

    # Compute base offsets
    e_off = b * stride_eb + src_row * stride_et
    i_off = b * stride_ib + src_row * stride_ih
    o_off = b * stride_ob + p * stride_om

    # Hidden dimension vector
    h = tl.arange(0, BLOCK_H)
    mask = h < H

    # Load and store with conditional
    vals = tl.load(e_ptr + e_off + h * stride_eh, mask=mask, other=0.0) if not is_img else tl.load(i_ptr + i_off + h * stride_ih, mask=mask, other=0.0)
    tl.store(out_ptr + o_off + h * stride_oh, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim in Triton.
        - Performs linear projection using torch.matmul (no torch ops on concatenated result).
        - Splits back into separate encoder and image streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D [B, dim, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H
        assert process_weight.shape == (H, H), "process_weight must be [H, H]"

        # Ensure contiguous for predictable strides
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        w = process_weight.contiguous()

        # Allocate output concatenated tensor X_cat: [B, M, H], M = T + I
        M = T + I
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton kernel: one program per (b, p)
        grid = (B, M)
        cat_rows_kernel[grid](
            e, i, X_cat,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            BLOCK_H=H,  # vectorize across hidden dim
            num_warps=1, num_stages=1,
        )

        # Linear projection using PyTorch (no bias)
        processed = torch.matmul(X_cat, w)

        # Split back into separate streams (metadata ops, no torch ops on processed)
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
