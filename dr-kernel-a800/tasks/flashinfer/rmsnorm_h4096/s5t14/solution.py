import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _row_sum_squares_kernel(
    hidden_ptr,     # *ptr to hidden states (original dtype)
    inv_rms_ptr,    # *ptr to output [B] inv_rms (float32)
    B,              # int: batch size
    H,              # int: hidden size (4096)
    stride_hs,      # int: row stride for hidden (for contiguous, == H)
):
    row = tl.program_id(0)
    sumsq = 0.0
    # Loop over all columns and accumulate sum of squares (scalar accumulator)
    for col in range(0, H):
        x = tl.load(hidden_ptr + row * stride_hs + col)
        x32 = x.to(tl.float32)
        sumsq += x32 * x32
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)


@triton.jit
def _scale_apply_weight_kernel(
    hidden_ptr,       # *ptr to hidden states (original dtype)
    inv_rms_ptr,      # *ptr to [B] inv_rms (float32)
    weight_ptr,       # *ptr to weight (original dtype, e.g., bfloat16)
    out_ptr,          # *ptr to final output (float32)
    B,                # int
    H,                # int
    stride_hs,        # int: row stride for hidden (for contiguous, == H)
    stride_out,       # int: row stride for output (for contiguous, == H)
):
    row = tl.program_id(0)
    inv_rms = tl.load(inv_rms_ptr + row)  # scalar float32
    # Elementwise compute y[row, col] = hidden[row, col].float() * inv_rms * weight[col].float()
    for col in range(0, H):
        x = tl.load(hidden_ptr + row * stride_hs + col).to(tl.float32)
        w = tl.load(weight_ptr + col).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_ptr + row * stride_out + col, y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors. The evaluation harness provides CUDA tensors.
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B, H = hidden_states.shape
        assert H == HIDDEN_SIZE, f"hidden_size must be {HIDDEN_SIZE}"

        # Ensure contiguous for simple row-major access
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        # Kernel 1: compute inv_rms per row (float32)
        inv_rms = torch.empty(B, dtype=torch.float32, device=hidden_states.device)
        grid = (B,)
        _row_sum_squares_kernel[grid](
            hidden_states, inv_rms,
            B, H,
            hidden_states.stride(0),
            num_warps=1,  # simple kernel, 1 warp is fine
            num_stages=1,
        )

        # Kernel 2: scale and apply weight
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)
        _scale_apply_weight_kernel[grid](
            hidden_states, inv_rms, weight.to(torch.float32), out_fp32,
            B, H,
            hidden_states.stride(0), out_fp32.stride(0),
            num_warps=1,
            num_stages=1,
        )

        # Match original behavior: cast back to original dtype of hidden_states
        return out_fp32.to(hidden_states.dtype)


# Entry point required by the evaluation harness: return a nn.Module via get()
class Model:
    @staticmethod
    def get():
        return ModelNew()


def run(*args):
    return ModelNew()(*args)
