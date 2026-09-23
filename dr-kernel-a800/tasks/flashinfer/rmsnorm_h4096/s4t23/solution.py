import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16)
    weight_ptr,       # *const weight (float32, host provides)
    out_ptr,          # *output
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (expected 4096)
    EPS: tl.float32,  # epsilon
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per program (4096)
):
    row_id = tl.program_id(axis=0)
    if row_id >= B:
        return

    # Row base pointers
    hidden_row_ptr = hidden_ptr + row_id * H
    out_row_ptr = out_ptr + row_id * H

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    # Load x row and weight vector, cast to float32 for math
    x = tl.load(hidden_row_ptr + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0).to(tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Compute output: y = x * inv_rms * w
    y = x * inv_rms * w

    # Store result; Triton will cast to the output pointer's element type if needed
    tl.store(out_row_ptr + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden.shape
        # Output tensor: same dtype as hidden
        out = torch.empty_like(hidden)

        # Launch one program per row, using BLOCK_SIZE=4096 (compile-time constant)
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden,
            weight.to(torch.float32),
            out,
            B,
            H,
            EPS=1e-5,
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
