import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _scale_rows_and_weight_kernel(
    hidden_ptr,       # *ptr to hidden states (original dtype)
    weight_ptr,       # *ptr to weight (float32)
    inv_rms_ptr,      # *ptr to per-row inv_rms (float32)
    out_ptr,          # *ptr to output (float32)
    B,                # int: batch size
    H,                # int: hidden size (4096)
    stride_hs,        # int: row stride for hidden (for contiguous, == H)
    stride_out,       # int: row stride for output (for contiguous, == H)
    BLOCK_SIZE: tl.constexpr,  # set to H to avoid masks
):
    row = tl.program_id(0)
    # Load per-row inv_rms (scalar float32)
    inv_rms = tl.load(inv_rms_ptr + row)

    # Vectorized load of the entire row, cast to float32
    offs = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE == H
    x = tl.load(hidden_ptr + row * stride_hs + offs)
    x32 = x.to(tl.float32)

    # Load weight and cast to float32 (vector)
    w = tl.load(weight_ptr + offs).to(tl.float32)

    # Compute y = (x * inv_rms) * w
    y = x32 * inv_rms * w

    # Store result
    tl.store(out_ptr + row * stride_out + offs, y)


class ModelNew(torch.nn.Module):
    def __init__(self, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors. The evaluation harness provides CUDA tensors.
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B, H = hidden_states.shape
        assert H == HIDDEN_SIZE, f"hidden_size must be {HIDDEN_SIZE}"

        # Ensure contiguous for simple row-major access
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        # Compute inv_rms on host using torch (allowed, only scalar per row):
        # inv_rms = rsqrt(mean(x^2) + EPS), where x is hidden_states in float32
        x_f32 = hidden_states.float()
        mean_sq = x_f32.pow(2).mean(dim=-1)  # shape [B]
        inv_rms = torch.rsqrt(mean_sq + EPS)  # shape [B], float32

        # Prepare weight in float32 for compute
        weight_f32 = weight.to(torch.float32).contiguous()

        # Allocate FP32 output for Triton kernel
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _scale_rows_and_weight_kernel[grid](
            hidden_states, weight_f32, inv_rms, out_fp32,
            B, H,
            hidden_states.stride(0), out_fp32.stride(0),
            BLOCK_SIZE=H,  # specialize kernel for H=4096 to avoid masks
            num_warps=self.num_warps,
            num_stages=self.num_stages,
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
