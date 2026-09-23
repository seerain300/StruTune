import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_weight_scale_kernel(
    hidden_ptr,       # *const input (e.g., bfloat16/float16/float32)
    weight_ptr,       # *const float32 (host provides weight as float32)
    out_ptr,          # *output (same dtype as hidden, e.g., bfloat16/float16/float32)
    B: tl.int32,      # batch size
    H: tl.int32,      # hidden size (4096)
    EPS: tl.float32,  # epsilon
    hidden_stride0: tl.int32,  # stride along batch dim for hidden
    hidden_stride1: tl.int32,  # stride along hidden dim for hidden
    out_stride0: tl.int32,     # stride along batch dim for output
    out_stride1: tl.int32,     # stride along hidden dim for output
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per program (4096 here)
):
    row = tl.program_id(0)  # one program per row
    offs = tl.arange(0, BLOCK_SIZE)

    hidden_row_ptr = hidden_ptr + row * hidden_stride0
    out_row_ptr = out_ptr + row * out_stride0

    # Load x row (masked by H) and cast to float32 for robust math
    x = tl.load(hidden_row_ptr + offs, mask=offs < H, other=0).to(tl.float32)

    # Compute sum of squares and inv_rms
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)

    # Load weight vector as float32
    w = tl.load(weight_ptr + offs, mask=offs < H, other=0.0)

    # Compute output in float32: y = x * inv_rms * w
    y = x * inv_rms * w

    # Store result; Triton will cast to out_ptr element type if needed
    tl.store(out_row_ptr + offs, y, mask=offs < H)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two inputs: hidden_states [B, 4096] and weight [4096]
        # Model signature matches the original run(hidden_states, weight).
        assert len(args) == 2, "ModelNew.forward expects hidden_states and weight"
        hidden_states, weight = args

        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors"
        hidden = hidden_states.contiguous()
        # We compute in float32 in-kernel; weight is provided as float32 to avoid repeated casting
        weight_f32 = weight.to(torch.float32).contiguous()

        B, H = hidden.shape
        # Output tensor same shape and dtype as input
        out = torch.empty_like(hidden)

        # Launch one program per row
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden, weight_f32, out,
            B, H, 1e-5,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=H,
            num_warps=8,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
