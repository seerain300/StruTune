import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _compute_inv_rms_per_row_kernel(
    hidden_ptr,    # *ptr to hidden states (original dtype)
    inv_rms_ptr,   # *ptr to per-row inv_rms (float32)
    B,             # int: batch size
    H,             # int: hidden size (4096)
    stride_hs,     # int: row stride for hidden (for contiguous, == H)
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    sumsq = 0.0
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(hidden_ptr + row * stride_hs + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sumsq += tl.sum(x32 * x32, axis=0)
        col += BLOCK_SIZE
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)


@triton.jit
def _scale_apply_weight_kernel(
    hidden_ptr,       # *ptr to hidden states (original dtype)
    inv_rms_ptr,      # *ptr to per-row inv_rms (float32)
    weight_ptr,       # *ptr to weight (original dtype)
    out_ptr,          # *ptr to final output (float32)
    B,                # int
    H,                # int
    stride_hs,        # int
    stride_out,       # int
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    inv_rms = tl.load(inv_rms_ptr + row)  # scalar float32

    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(hidden_ptr + row * stride_hs + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x32 * inv_rms * w
        tl.store(out_ptr + row * stride_out + offs, y, mask=mask)
        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        self.block_size = block_size
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

        # Compute per-row inv_rms in FP32
        inv_rms = torch.empty(B, dtype=torch.float32, device=hidden_states.device)
        _compute_inv_rms_per_row_kernel[(B,)](
            hidden_states, inv_rms,
            B, H,
            hidden_states.stride(0),
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Scale and apply weight, write final FP32 output
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)
        _scale_apply_weight_kernel[(B,)](
            hidden_states, inv_rms, weight.to(torch.float32), out_fp32,
            B, H,
            hidden_states.stride(0), out_fp32.stride(0),
            BLOCK_SIZE=self.block_size,
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
