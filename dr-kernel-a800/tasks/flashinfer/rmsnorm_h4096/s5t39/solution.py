import torch
import triton
import triton.language as tl

# Fixed hidden size from the original code
HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_and_scale_row_kernel(
    hidden_ptr,       # *ptr to hidden states (B, H)
    weight_ptr,       # *ptr to weight (H,)
    out_ptr,          # *ptr to output (B, H), float32
    B: tl.constexpr,  # batch size (not used directly in kernel)
    H: tl.constexpr,  # hidden size, compile-time constant
):
    row = tl.program_id(0)
    # Compute base offset for the current row (assuming contiguous [B, H])
    # We pass row directly and rely on strides in pointer arithmetic via tl.load with row * H + offs.
    # However, Triton kernels expect linear indexing. Use 1D indexing by computing linear index.
    # For simplicity, ensure we load a contiguous row by using row * H + offs.
    # But since we use 2D tensors, we will index as hidden[row, offs] using the pointer arithmetic below.
    # We'll load the entire row vectorized.
    offs = tl.arange(0, H)
    mask = offs < H  # always true for H=4096, but keep mask for safety

    # Load hidden row, cast to float32
    x = tl.load(hidden_ptr + row * H + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    # Zero out masked lanes to avoid contribution (mask is true for all lanes when H covers the row)
    x32 = tl.where(mask, x32, 0.0)

    # Compute sum of squares
    sumsq = tl.sum(x32 * x32, axis=0)

    # Compute inv_rms per row: rsqrt(mean + EPS)
    inv_rms = tl.rsqrt(sumsq / H + EPS)

    # Load weight vector, cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute output: (hidden.float() * inv_rms) * weight.float()
    y32 = x32 * inv_rms * w

    # Store result (FP32)
    tl.store(out_ptr + row * H + offs, y32, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        assert H == HIDDEN_SIZE, f"Expected hidden_size == {HIDDEN_SIZE}, got {H}"
        assert weight.numel() == H, "weight must have length equal to hidden_size"

        # Allocate FP32 output for the Triton kernel
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        normalize_and_scale_row_kernel[grid](hidden, weight, out_fp32, B=B, H=HIDDEN_SIZE, num_warps=4)

        # Cast back to original dtype to match original behavior
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
