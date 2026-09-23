import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    input_ptr,       # *f32, in [B, T, H]
    weightT_ptr,     # *f32, process_weight.T [H, H]
    output_ptr,      # *f32, out [B, T, H]
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    stride_in_b: tl.constexpr,
    stride_in_t: tl.constexpr,
    stride_in_h: tl.constexpr,
    stride_w_k: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
):
    # One program computes the whole output vector for a single (b, t)
    b = tl.program_id(0)
    t = tl.program_id(1)

    # Bounds check (in case grid is larger than B or T)
    if (b >= B) or (t >= T):
        return

    # Base pointers for this row
    base_in = input_ptr + b * stride_in_b + t * stride_in_t
    base_out = output_ptr + b * stride_out_b + t * stride_out_t

    # For each output hidden dimension h, compute the dot with the weight row
    for h in range(0, H):
        acc = 0.0
        for k in range(0, H):
            in_val = tl.load(base_in + k * stride_in_h)
            w_val = tl.load(weightT_ptr + k * stride_w_k + h * stride_w_h)  # weight_T[k, h]
            acc += in_val * w_val
        tl.store(base_out + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along the sequence dimension using torch.cat (host-side).
        - Apply linear projection via Triton matvec_row_kernel: out = concatenated @ process_weight.T
        - Split back into separate encoder and image streams.
        """
        # Ensure tensors are on the same CUDA device and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        # The original code uses float32 tensors; we keep float32
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"

        hidden = hidden_states.contiguous()        # [B, Simg, H]
        encoder = encoder_hidden_states.contiguous()  # [B, Stext, H]
        weight = process_weight.contiguous()      # [H, H]

        B = hidden.shape[0]
        H = hidden.shape[2]
        T = encoder.shape[1] + hidden.shape[1]    # sequence length after concatenation

        # Concatenate along sequence dimension (host-side op)
        concatenated = torch.cat([encoder, hidden], dim=1)  # [B, T, H]

        # Prepare transposed weight for Triton: process_weight.T -> [H, H]
        weight_T = weight.transpose(0, 1).contiguous()  # [H, H]

        # Allocate output
        out = torch.empty((B, T, H), dtype=torch.float32, device=concatenated.device)

        # Launch Triton kernel: grid over (B, T)
        grid = (B, T)
        matvec_row_kernel[grid](
            concatenated, weight_T, out,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Split back along sequence dimension
        processed_encoder = out[:, :encoder.shape[1], :]  # first Stext tokens
        processed_hidden = out[:, encoder.shape[1]:, :]   # remaining Simg tokens

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
