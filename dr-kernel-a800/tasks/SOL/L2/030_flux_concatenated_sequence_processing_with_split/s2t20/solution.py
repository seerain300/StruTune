import torch
import triton
import triton.language as tl


# Triton kernel: concatenate [B, T, H] and [B, I, H] along sequence dimension into [B, L, H], with L = T + I.
# 3D grid over (batch, sequence, hidden). Each program writes one element out[b, m, n].
@triton.jit
def concat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_m, stride_o_h
):
    b = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)

    src_ptr = None
    if m < T:
        src_ptr = encoder_ptr + b * stride_e_b + m * stride_e_t + n * stride_e_h
    else:
        src_ptr = hidden_ptr + b * stride_h_b + (m - T) * stride_h_i + n * stride_h_h

    val = tl.load(src_ptr)
    dst_ptr = out_ptr + b * stride_o_b + m * stride_o_m + n * stride_o_h
    tl.store(dst_ptr, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward: concatenates [B, T, H] and [B, I, H] using Triton,
        applies linear projection via PyTorch matmul, and splits outputs back.
        """
        # Validate shapes
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be [B, L, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Ensure CUDA and contiguity
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors."
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        # 1) Concatenate along sequence dimension using Triton kernel: out_cat [B, L, H]
        out_cat = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        grid = (B, L, H)
        concat_seq_kernel[grid](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=1, num_stages=1
        )

        # 2) Linear projection using PyTorch matmul (GPU): processed = out_cat @ process_weight.T
        WT = process_weight.t().contiguous()  # [H, H]
        processed = torch.matmul(out_cat, WT)  # [B, L, H], same dtype as inputs

        # 3) Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
