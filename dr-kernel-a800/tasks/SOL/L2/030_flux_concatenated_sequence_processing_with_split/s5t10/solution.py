import torch
import triton
import triton.language as tl


@triton.jit
def compute_out_elem(e_ptr, h_ptr, wT_ptr, out_ptr,
                     B: tl.constexpr, Stext: tl.constexpr, Simg: tl.constexpr, H: tl.constexpr):
    """
    Elementwise compute: for each (b, t, h),
      if t < Stext:
        src = e[b, t, h]
      else:
        src = h[b, t - Stext, h]
      out[b, t, h] = sum_{k=0..H-1} src * wT[k, h]
    wT_ptr is process_weight.T with shape [H, H] (row-major).
    """
    b = tl.program_id(0)
    t = tl.program_id(1)
    h = tl.program_id(2)

    # Select source based on t < Stext
    if t < Stext:
        src = tl.load(e_ptr + b * H + t * H + h)
    else:
        src = tl.load(h_ptr + b * H + (t - Stext) * H + h)

    # Compute dot with process_weight.T[:, h]
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, H):
        w = tl.load(wT_ptr + k * H + h)  # wT[k, h] for fixed h
        acc += src * w

    # Store result to out[b, t, h] where out has shape [B, T, H]
    tl.store(out_ptr + b * (Stext + Simg) * H + t * H + h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run():
        1) Emulate concatenation along sequence dimension inside Triton.
        2) Apply linear projection (batched matvec) using process_weight.T.
        3) Split back into separate encoder and image streams.
        """
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

        # Allocate output: out [B, T, H]
        out = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)

        # Prepare process_weight.T as [H, H] contiguous
        weights_T = process_weight.transpose(0, 1).contiguous()  # [H, H]

        # Launch elementwise kernel over (B, T, H)
        grid = (B, T, H)
        compute_out_elem[grid](
            encoder_hidden_states, hidden_states, weights_T, out,
            B=B, Stext=Stext, Simg=Simg, H=H,
            num_warps=1, num_stages=1
        )

        # Split: processed_encoder [B, Stext, H], processed_hidden [B, Simg, H]
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
