import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    in_ptr,       # *f32, [B, T, H] (concatenated sequence rows)
    wT_ptr,       # *f32, [H, H] (transposed process_weight: rows = input hidden, cols = output hidden)
    out_ptr,      # *f32, [B, T, H] (result)
    B, T, H,
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_b, stride_w_k, stride_w_h,  # wT is [H, H], k is row index, h is col index
    stride_out_b, stride_out_t, stride_out_h,
):
    # 2D grid: (B, T)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    b = pid_b
    t = pid_t

    in_row_ptr = in_ptr + b * stride_in_b + t * stride_in_t
    out_row_ptr = out_ptr + b * stride_out_b + t * stride_out_t

    # Compute out[b, t, :] = W_T @ in[b, t, :]
    # Explicit outer product across the hidden dimension
    for k in range(0, H):
        a_k = tl.load(in_row_ptr + k * stride_in_h)  # scalar
        h_vec = tl.arange(0, H)
        b_vec = tl.load(wT_ptr + k * stride_w_k + h_vec * stride_w_h, mask=h_vec < H, other=0.0)
        out_vec = a_k * b_vec
        tl.store(out_row_ptr + h_vec * stride_out_h, out_vec, mask=h_vec < H)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, Simg, H]
        encoder_hidden_states: torch.Tensor,  # [B, Stext, H]
        process_weight: torch.Tensor,     # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along the sequence dimension (torch.cat).
        - Apply linear projection process_weight.T to the concatenated sequence using a Triton kernel (per row).
        - Split the processed output back into encoder and hidden parts based on original sequence lengths.
        """
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # 1) Concatenate along sequence dimension (torch for robustness)
        out_cat = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T, H]

        # 2) Transpose process_weight to [H, H] for Triton read (no copy, view)
        W_T = process_weight  # [H, H]; ensure contiguous for simple strides

        # 3) Allocate output
        out = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # 4) Launch Triton kernel to compute out[b, t, :] = W_T @ out_cat[b, t, :]
        grid_gemm = (B, T)
        matvec_row_kernel[grid_gemm](
            out_cat, W_T, out,
            B, T, H,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            W_T.stride(0), W_T.stride(1), W_T.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Split back along sequence dimension
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
