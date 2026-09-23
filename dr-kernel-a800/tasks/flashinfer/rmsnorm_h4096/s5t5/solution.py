import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def _normalize_scale_weight_block_kernel(
    hidden_ptr,     # *ptr to hidden states (original dtype, e.g., bfloat16)
    weight_ptr,     # *ptr to weight (original dtype, e.g., bfloat16)
    out_ptr,        # *ptr to output (float32)
    B,              # int: batch size
    H,              # int: hidden size (4096)
    stride_hs,      # int: row stride for hidden (for contiguous, == H)
    stride_out,     # int: row stride for output (for contiguous, == H)
    BLOCK_ROWS: tl.constexpr,  # number of rows per program (e.g., 128)
    BLOCK_SIZE: tl.constexpr,  # tile size over columns (e.g., 1024)
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < B

    # Pass 1: compute sum of squares per row
    sumsq = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        col_mask = offs < H
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(hidden_ptr + rows[:, None] * stride_hs + offs[None, :],
                    mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        x32_valid = tl.where(mask, x32, 0.0)
        sumsq += tl.sum(x32_valid * x32_valid, axis=1)
        col += BLOCK_SIZE

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)  # per-row float32

    # Pass 2: compute y and store
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        col_mask = offs < H
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(hidden_ptr + rows[:, None] * stride_hs + offs[None, :],
                    mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=col_mask, other=0.0).to(tl.float32)
        y32 = x32 * inv_rms[:, None] * w[None, :]
        tl.store(out_ptr + rows[:, None] * stride_out + offs[None, :],
                 y32, mask=mask)
        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def __init__(self, block_rows: int = 128, block_size: int = 1024, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        self.block_rows = block_rows
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

        # Output will be computed in FP32 inside the kernel; cast to original dtype after.
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program handles BLOCK_ROWS rows
        grid = (triton.cdiv(B, self.block_rows),)
        _normalize_scale_weight_block_kernel[grid](
            hidden_states, weight, out_fp32,
            B, H,
            hidden_states.stride(0), out_fp32.stride(0),
            BLOCK_ROWS=self.block_rows,
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
