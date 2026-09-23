import torch
import triton
import triton.language as tl


@triton.jit
def concat_by_row(e_ptr, h_ptr, out_ptr,
                   B, Stext, Simg, H,
                   stride_e_b, stride_e_t, stride_e_h,
                   stride_h_b, stride_h_t, stride_h_h,
                   stride_out_b, stride_out_t, stride_out_h):
    """
    For each (b, t), if t < Stext: out[b, t, :] = e[b, t, :]
    else: out[b, t, :] = h[b, t - Stext, :]
    """
    b = tl.program_id(0)
    t = tl.program_id(1)
    h_offsets = tl.arange(0, H)  # vector along hidden dim

    mask_e = t < Stext
    # Base pointers for the row
    e_row_ptr = e_ptr + b * stride_e_b + t * stride_e_t + h_offsets * stride_e_h
    h_row_ptr = h_ptr + b * stride_h_b + (t - Stext) * stride_h_t + h_offsets * stride_h_h
    out_row_ptr = out_ptr + b * stride_out_b + t * stride_out_t + h_offsets * stride_out_h

    # Choose source pointer based on t
    src_row_ptr = tl.where(mask_e, e_row_ptr, h_row_ptr)
    vals = tl.load(src_row_ptr, mask=tl.full([H], True, dtype=tl.int1), other=0.0)
    tl.store(out_row_ptr, vals, mask=tl.full([H], True, dtype=tl.int1))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        1) Concatenate [B, Stext, H] and [B, Simg, H] -> [B, T, H], T = Stext + Simg (Triton).
        2) Apply linear projection via PyTorch matmul: processed = cat @ process_weight.T
        3) Split back: processed_encoder = processed[:, :Stext, :], processed_hidden = processed[:, Stext:, :]
        """
        # Ensure CUDA and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernels."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        H = hidden_states.shape[2]  # hidden_dim
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg

        # 1) Concatenate along sequence dimension using Triton (no torch.cat)
        out_cat = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)

        # Launch concat kernel over (B, T)
        grid_concat = (B, T)
        concat_by_row[grid_concat](
            encoder_hidden_states, hidden_states, out_cat,
            B, Stext, Simg, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=1, num_stages=1
        )

        # 2) Linear projection via PyTorch matmul to ensure exact numerical match
        weights_T = process_weight.transpose(0, 1).contiguous()  # [H, H]
        processed = torch.matmul(out_cat, weights_T)  # [B, T, H]

        # 3) Split back along sequence dimension
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
