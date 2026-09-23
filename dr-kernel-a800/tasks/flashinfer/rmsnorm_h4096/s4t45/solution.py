import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input tensor (e.g., bfloat16/float16)
    weight_ptr,       # *const weight tensor (e.g., bfloat16/float16)
    out_ptr,          # *output tensor (same shape as hidden; dtype from allocation)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (expect 4096)
    EPS: tl.float32,  # epsilon
):
    # One Triton program per row
    row_id = tl.program_id(0)
    cols = tl.arange(0, H)  # H is 4096 in this task
    # Load the entire row
    x = tl.load(hidden_ptr + row_id * H + cols)
    # Compute in float32 for stability
    x_f32 = tl.cast(x, tl.float32)
    # Sum of squares
    sumsq = tl.sum(x_f32 * x_f32, axis=0)
    # Mean and inverse RMS
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    # Load weight and cast
    w = tl.load(weight_ptr + cols)
    w_f32 = tl.cast(w, tl.float32)
    # Compute scaled output
    y_f32 = x_f32 * inv_rms * w_f32
    # Store; Triton will cast to out_ptr dtype if needed
    tl.store(out_ptr + row_id * H + cols, y_f32)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight, eps=1e-5):
        """
        hidden_states: [B, 4096], bfloat16 or float16 (CUDA)
        weight: [4096], same dtype/device as hidden for performance (CUDA)
        """
        # Triton-only path: ensure CUDA tensors
        if not hidden_states.is_cuda or not weight.is_cuda:
            raise RuntimeError("ModelNew.forward requires CUDA tensors for Triton kernels.")
        # Ensure contiguous layout
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()
        B, H = hidden.shape
        if H != 4096:
            # This kernel is specialized for H=4096; adjust upstream to use 4096.
            raise AssertionError("ModelNew expects hidden size H=4096.")
        # Allocate output with same shape and dtype as hidden
        out = torch.empty_like(hidden)
        # Launch kernel: one program per row
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, eps,
            num_warps=8, num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
