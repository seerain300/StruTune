import torch
import triton
import triton.language as tl

@triton.jit
def _concat_sequences_kernel(
    out_ptr,        # *float32
    left_ptr,       # *float32
    right_ptr,      # *float32
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    sL_b, sL_t, sL_d,
    sR_b, sR_i, sR_d,
    sO_b, sO_t, sO_d,
    BLOCK_M: tl.constexpr = 128,
):
    # Each program instance handles one (batch, tile-of-t) pair
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)  # batch index

    # Compute offsets for 'left' (encoder) part
    t_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_t = t_offsets < T

    # Pointers for left: out[b, t, :] where t in [0, T)
    out_ptrs_left = out_ptr + pid_b * sO_b + t_offsets * sO_t + tl.arange(0, D) * sO_d
    left_ptrs = left_ptr + pid_b * sL_b + t_offsets * sL_t + tl.arange(0, D) * sL_d

    # Load, store
    left_vals = tl.load(left_ptrs, mask=mask_t[:, None], other=0.0)
    tl.store(out_ptrs_left, left_vals, mask=mask_t[:, None])

    # Compute offsets for 'right' (image) part, starting at T
    i_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_i = i_offsets < I

    out_ptrs_right = out_ptr + pid_b * sO_b + (i_offsets + T) * sO_t + tl.arange(0, D) * sO_d
    right_ptrs = right_ptr + pid_b * sR_b + i_offsets * sR_i + tl.arange(0, D) * sR_d

    right_vals = tl.load(right_ptrs, mask=mask_i[:, None], other=0.0)
    tl.store(out_ptrs_right, right_vals, mask=mask_i[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim into a single tensor using Triton.
        - Apply linear projection via torch.matmul (to ensure numerical consistency with PyTorch).
        - Split back into two outputs.
        """
        # Ensure dtype is float32 for numerical parity with PyTorch default
        # Make tensors contiguous for simpler stride handling
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        device = hidden_states.device

        # Allocate the concatenated tensor: [B, T + I, D]
        concatenated = torch.empty((B, T + I, D), device=device, dtype=torch.float32)

        # Launch Triton kernel to copy left [B, T, D] into concatenated[:, :T, :]
        grid_left = (triton.cdiv(T, 128), B)
        _concat_sequences_kernel[grid_left](
            concatenated,
            encoder_hidden_states,
            hidden_states,
            B=B, T=T, I=I, D=D,
            sL_b=encoder_hidden_states.stride(0), sL_t=encoder_hidden_states.stride(1), sL_d=encoder_hidden_states.stride(2),
            sR_b=hidden_states.stride(0), sR_i=hidden_states.stride(1), sR_d=hidden_states.stride(2),
            sO_b=concatenated.stride(0), sO_t=concatenated.stride(1), sO_d=concatenated.stride(2),
            BLOCK_M=128,
            num_warps=4, num_stages=2,
        )

        # Ensure process_weight is [D, D] float32 on the same device
        W = process_weight.to(device=device, dtype=torch.float32).contiguous()  # [D, D]

        # Matmul on concatenated: [B, T+I, D] @ [D, D] -> [B, T+I, D]
        processed = torch.matmul(concatenated, W.t())

        # Split back
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
