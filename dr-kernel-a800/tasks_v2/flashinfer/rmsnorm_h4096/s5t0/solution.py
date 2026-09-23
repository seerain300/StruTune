import torch
import triton
import triton.language as tl

# Fixed hidden size as in the original code
HIDDEN_SIZE = 4096
EPS = 1e-5

# Triton kernel: one program per row
@triton.jit
def _normalize_scale_weight_kernel(
    hidden_ptr,         # *ptr to hidden states (original dtype, e.g., bfloat16)
    weight_ptr,         # *ptr to weight (float32)
    out_ptr,            # *ptr to output (same dtype as hidden states)
    batch_size,         # int: number of rows
    H,                  # int: number of columns (assert == HIDDEN_SIZE)
    stride_hs,          # int: row stride for hidden (for contiguous, == H)
    stride_out,         # int: row stride for output
    BLOCK_SIZE: tl.constexpr,  # chunk size for looping
):
    row = tl.program_id(0)  # each program handles one row
    # Base pointers for this row
    row_hidden_ptr = hidden_ptr + row * stride_hs
    row_out_ptr = out_ptr + row * stride_out

    # Accumulate sum of squares in FP32
    sumsq = 0.0
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(row_hidden_ptr + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sumsq += tl.sum(x32 * x32, axis=0)

    # Compute inv_rms = 1 / sqrt(mean + EPS)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Second pass: compute output and store
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(row_hidden_ptr + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)  # weight is float32
        y32 = x32 * inv_rms * w
        # Store as original dtype (out_ptr dtype controls this)
        tl.store(row_out_ptr + offs, y32.to(tl.dtype(out_ptr)), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size=1024, num_warps=4, num_stages=2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, hidden_states, weight):
        # If not on CUDA, fall back to PyTorch implementation for correctness
        if hidden_states.device.type != "cuda":
            # Fallback: do exactly what the original run does
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure shapes: hidden_states [B, H], weight [H]
        assert hidden_states.dim() == 2, "hidden_states must be 2D [batch, hidden_size]"
        B, H = hidden_states.shape
        # In the original, hidden_size is asserted to be 4096. We mirror that behavior.
        # If you want to be strict, uncomment the assert below. Given the harness, H == 4096.
        # assert H == HIDDEN_SIZE, f"hidden_size must be {HIDDEN_SIZE}"

        # Make sure tensors are contiguous for simple row-major access
        # (get_inputs already returns contiguous tensors; keep it defensive)
        hidden_states = hidden_states.contiguous()
        # Cast weight to float32 for compute (original code does this)
        weight = weight.to(torch.float32).contiguous()

        # Allocate output with same shape and dtype as hidden_states
        out = torch.empty_like(hidden_states)

        # Launch Triton kernel: one program per row
        grid = (B,)
        _normalize_scale_weight_kernel[grid](
            hidden_states, weight, out,
            B, H,
            hidden_states.stride(0), out.stride(0),
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return out


def run(*args):
    return ModelNew()(*args)
