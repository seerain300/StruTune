import torch
import triton
import triton.language as tl

# Triton kernel: LayerNorm over each row (per patch), affine with ln_weight and ln_bias
@triton.jit
def layernorm_affine_kernel(
    x_ptr,          # *bf16, input [NUM_PATCHES, HIDDEN_SIZE]
    out_ptr,        # *fp32, output [NUM_PATCHES, HIDDEN_SIZE]
    ln_weight_ptr,  # *fp32, [HIDDEN_SIZE]
    ln_bias_ptr,    # *fp32, [HIDDEN_SIZE]
    HIDDEN_SIZE: tl.constexpr,   # hidden_size (e.g., 1536)
    NUM_PATCHES: tl.constexpr,   # number of rows
    EPS: tl.constexpr,           # epsilon
    BLOCK_SIZE: tl.constexpr,    # tile size (set to HIDDEN_SIZE)
):
    pid = tl.program_id(0)  # one program per row (patch)
    # Offsets for this row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HIDDEN_SIZE

    # Load input row (bf16), convert to fp32 for math
    x_bf16 = tl.load(x_ptr + pid * HIDDEN_SIZE + offs, mask=mask, other=0)
    x = x_bf16.to(tl.float32)

    # Compute mean and variance over the row
    # sum(x)
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / HIDDEN_SIZE
    # sum((x - mean)^2)
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / HIDDEN_SIZE
    inv_std = 1.0 / tl.math.sqrt(var + EPS)

    # Normalize and apply affine
    norm = diff * inv_std
    w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0)  # fp32
    b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0)    # fp32
    out = norm * w + b

    # Store as fp32
    tl.store(out_ptr + pid * HIDDEN_SIZE + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int = 1536, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float = None):
        # hidden: [num_patches, hidden_size] in bfloat16
        # ln_weight, ln_bias: [hidden_size] in bfloat16 or float32, we will use fp32 in kernel
        assert hidden.is_cuda, "ModelNew.forward expects CUDA tensors"
        assert hidden.dtype == torch.bfloat16, "hidden must be torch.bfloat16"
        assert ln_weight.dtype in (torch.bfloat16, torch.float32) and ln_bias.dtype in (torch.bfloat16, torch.float32), "ln_weight/ln_bias must be bf16 or fp32"

        num_patches, hidden_size = hidden.shape
        assert hidden_size == self.hidden_size, f"hidden_size must be {self.hidden_size}"

        # Output in fp32 for numerical stability
        out = torch.empty((num_patches, hidden_size), dtype=torch.float32, device=hidden.device)

        # Cast ln_weight/ln_bias to fp32 for kernel
        ln_weight_fp32 = ln_weight.to(torch.float32).contiguous()
        ln_bias_fp32 = ln_bias.to(torch.float32).contiguous()

        # Launch Triton kernel: one program per row
        grid = (num_patches,)
        layernorm_affine_kernel[grid](
            hidden, out, ln_weight_fp32, ln_bias_fp32,
            HIDDEN_SIZE=self.hidden_size,
            NUM_PATCHES=num_patches,
            EPS=(self.eps if eps is None else float(eps)),
            BLOCK_SIZE=self.hidden_size,  # cover full hidden_size
            num_warps=4,
        )
        return out


def run(*args):
    return ModelNew()(*args)
