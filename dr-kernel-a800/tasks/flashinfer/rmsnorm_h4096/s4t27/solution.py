import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const float32 (host provides weight as float32)
    out_ptr,          # *output (float32)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096 in this task)
    EPS: tl.float32,  # epsilon
):
    # One Triton program per row
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    # Column offsets for this row
    cols = tl.arange(0, H)

    # Compute base pointers for this row (assuming row-major contiguous: stride = H)
    hidden_row_ptr = hidden_ptr + row_id * H + cols

    # Load hidden row (original dtype), cast to float32 for math
    x = tl.load(hidden_row_ptr)
    x = tl.cast(x, tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector as float32 (host ensures weight_ptr is float32)
    w = tl.load(weight_ptr + cols)

    # Compute output in float32
    y = x * inv_rms * w

    # Store output as float32
    tl.store(out_ptr + row_id * H + cols, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA device."
        hidden = hidden_states.contiguous()
        B, H = hidden.shape
        assert H == 4096, "Hidden size must be 4096."

        # Prepare weight as float32 (matches original PyTorch behavior where weight is cast to float32)
        weight_f32 = weight.to(torch.float32).contiguous()

        # Allocate output as float32 for robust math
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        # Empirically best parameters for H=4096 in this environment
        _layernorm_weight_scale_kernel[grid](
            hidden, weight_f32, out_f32,
            B, H, 1e-5,
            num_warps=8,
            num_stages=2,
        )

        # Return in the original hidden_states dtype (matches original PyTorch code behavior)
        return out_f32.to(hidden.dtype)


def run(*args):
    return ModelNew()(*args)
