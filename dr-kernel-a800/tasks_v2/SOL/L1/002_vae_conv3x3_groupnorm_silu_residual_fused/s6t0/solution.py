import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel to compute per-(N, group) sums and sum of squares for GroupNorm
# out_ptr: float32* of shape (N, num_groups), stores (sum, sumsq) per (n, g)
@triton.jit
def _groupnorm_reduce_sums(
    x_ptr,                    # * (float16/float32), input after first conv
    sums_ptr,                 # * float32, output (sum, sumsq) per (n,g)
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    C_per_group: tl.constexpr,
    eps: tl.constexpr,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_s_n, stride_s_g,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)  # batch
    g = tl.program_id(1)  # group id

    # Accumulators in float32
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over channels in this group
    c_start = g * C_per_group
    for c_off in range(0, C_per_group):
        c = c_start + c_off
        # Iterate over spatial HW in chunks
        hw = H * W
        # We'll iterate linearly over HW
        # Triton does not have vectorized 2D loads easily; do per-element loop
        # We'll compute base address and then loop i from 0..hw-1
        # Using a static loop here is fine for moderate sizes
        for i in range(0, hw):
            h = i // W
            w = i % W
            # Load x[n, c, h, w] as float32 for accumulation
            x_val = tl.load(x_ptr + n * stride_x_n + c * stride_x_c + h * stride_x_h + w * stride_x_w, eviction_policy='evict_last')
            x_val = x_val.to(tl.float32)
            total_sum += x_val
            total_sumsq += x_val * x_val

    # Store (sum, sumsq) for this (n, g)
    tl.store(sums_ptr + n * stride_s_n + g * stride_s_g, total_sum)
    tl.store(sums_ptr + n * stride_s_n + g * stride_s_g + 1, total_sumsq)

# Triton kernel to apply GroupNorm (with affine) and SiLU on output tensor
# x_ptr: input after conv (and possibly previous SiLU), float16/float32
# y_ptr: output tensor
# norm_weight_ptr: (C,) float32
# norm_bias_ptr: (C,) float32
# N, C, H, W, num_groups, C_per_group, eps, strides, and pointer to sums to compute mean/rstd
@triton.jit
def _groupnorm_apply_silu(
    x_ptr, y_ptr,
    norm_weight_ptr, norm_bias_ptr,
    sums_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    C_per_group: tl.constexpr,
    eps: tl.constexpr,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    stride_s_n, stride_s_g,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)  # linear over N*C*H*W
    Ntotal = N * C * H * W
    # Compute indices
    HW = H * W
    c = pid // (N * HW)
    rem = pid % (N * HW)
    n = rem // (C * HW)
    hw = rem % (C * HW)
    h = hw // W
    w = hw % W

    # Determine which group channel c belongs to
    g = c // C_per_group

    # Compute mean and rstd for this group
    # sums_ptr[n, g] holds (sum, sumsq)
    total_sum = tl.load(sums_ptr + n * stride_s_n + g * stride_s_g)
    total_sumsq = tl.load(sums_ptr + n * stride_s_n + g * stride_s_g + 1)
    group_size = C_per_group * HW
    mean = total_sum / group_size
    var = total_sumsq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load input
    x_val = tl.load(x_ptr + n * stride_x_n + c * stride_x_c + h * stride_x_h + w * stride_x_w)
    x_f32 = x_val.to(tl.float32)
    # Normalize
    norm = (x_f32 - mean) * rstd
    # Affine: scale and bias
    scale = tl.load(norm_weight_ptr + c)
    bias = tl.load(norm_bias_ptr + c)
    y_before_silu = norm * scale + bias

    # SiLU activation
    sig = 1.0 / (1.0 + tl.exp(-y_before_silu))
    y = y_before_silu * sig

    # Cast back to original dtype of x_ptr if needed (assume y_ptr dtype matches x_ptr)
    # Store
    tl.store(y_ptr + n * stride_y_n + c * stride_y_c + h * stride_y_h + w * stride_y_w, y.to(x_val.dtype))

# Triton kernel: SiLU elementwise on tensor
@triton.jit
def _silu_elementwise(
    x_ptr, y_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
):
    pid = tl.program_id(0)
    total = N * C * H * W
    # Compute indices
    HW = H * W
    c = pid // (N * HW)
    rem = pid % (N * HW)
    n = rem // (C * HW)
    hw = rem % (C * HW)
    h = hw // W
    w = hw % W

    x_val = tl.load(x_ptr + n * stride_x_n + c * stride_x_c + h * stride_x_h + w * stride_x_w)
    x_f32 = x_val.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_f32))
    y = x_f32 * sig
    tl.store(y_ptr + n * stride_y_n + c * stride_y_c + h * stride_y_h + w * stride_y_w, y.to(x_val.dtype))

# Triton kernel: elementwise add (out = x1 + x2), used for residual
@triton.jit
def _add_elementwise(
    x1_ptr, x2_ptr, y_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    stride1_n, stride1_c, stride1_h, stride1_w,
    stride2_n, stride2_c, stride2_h, stride2_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
):
    pid = tl.program_id(0)
    total = N * C * H * W
    HW = H * W
    c = pid // (N * HW)
    rem = pid % (N * HW)
    n = rem // (C * HW)
    hw = rem % (C * HW)
    h = hw // W
    w = hw % W

    v1 = tl.load(x1_ptr + n * stride1_n + c * stride1_c + h * stride1_h + w * stride1_w)
    v2 = tl.load(x2_ptr + n * stride2_n + c * stride2_c + h * stride2_h + w * stride2_w)
    # Accumulate in fp32 then cast back
    res = (v1.to(tl.float32) + v2.to(tl.float32)).to(v1.dtype)
    tl.store(y_ptr + n * stride_y_n + c * stride_y_c + h * stride_y_h + w * stride_y_w, res)

def _triton_groupnorm_apply_silu(x, norm_weight, norm_bias, num_groups, eps, out):
    """
    GroupNorm + affine + SiLU using Triton. x: (N,C,H,W) tensor, out: same shape.
    """
    assert x.is_cuda, "Triton kernels require CUDA tensors"
    N, C, H, W = x.shape
    assert C % num_groups == 0, "C must be divisible by num_groups"
    C_per_group = C // num_groups

    # Allocate sums buffer for (N, num_groups), we store (sum, sumsq) so 2*N*num_groups elements
    # In Triton kernel, we store two floats per (n,g). We'll use a flat buffer and compute offsets.
    # We'll create a (N, num_groups, 2) float32 buffer.
    sums = torch.empty((N, num_groups, 2), dtype=torch.float32, device=x.device)

    # Launch reduction kernel: grid over (N, num_groups)
    BLOCK_HW = 4096  # arbitrary large chunk for HW; loop handles any H*W
    grid = (N, num_groups)
    _groupnorm_reduce_sums[grid](
        x, sums,
        N, C, H, W,
        num_groups, C_per_group, eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        sums.stride(0), sums.stride(1),
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )

    # Apply normalization + affine + SiLU
    grid_elems = N * C * H * W
    _groupnorm_apply_silu[(grid_elems,)](
        x, norm_weight, norm_bias, sums,
        N, C, H, W,
        num_groups, C_per_group, eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sums.stride(0), sums.stride(1),
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )

def _triton_silu(x, out):
    """
    Elementwise SiLU using Triton.
    """
    N, C, H, W = x.shape
    grid_elems = N * C * H * W
    _silu_elementwise[(grid_elems,)](
        x, out,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_warps=4,
    )

def _triton_add(x1, x2, out):
    """
    Elementwise add using Triton (out = x1 + x2).
    """
    N, C, H, W = x1.shape
    grid_elems = N * C * H * W
    _add_elementwise[(grid_elems,)](
        x1, x2, out,
        N, C, H, W,
        x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_warps=4,
    )

@torch.no_grad()
def run_triton(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    """
    Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
    Uses Triton for GroupNorm (with affine) and SiLU, and for elementwise residual add.
    """
    assert x.is_cuda, "Triton implementation requires CUDA tensors"
    # Ensure tensors are contiguous for Triton
    x = x.contiguous()
    conv1_weight = conv1_weight.contiguous()
    norm1_weight = norm1_weight.contiguous()
    norm1_bias = norm1_bias.contiguous()
    conv2_weight = conv2_weight.contiguous()
    norm2_weight = norm2_weight.contiguous()
    norm2_bias = norm2_bias.contiguous()

    num_groups = 32

    # 1) First conv
    out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)  # (N, C, H, W)
    # 2) GroupNorm + affine
    out = out.contiguous()
    _triton_groupnorm_apply_silu(out, norm1_weight, norm1_bias, num_groups, eps, out)
    # 3) SiLU
    out = out.contiguous()
    _triton_silu(out, out)  # in-place

    # 4) Second conv
    out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
    out = out.contiguous()

    # 5) GroupNorm + affine
    _triton_groupnorm_apply_silu(out, norm2_weight, norm2_bias, num_groups, eps, out)
    # 6) SiLU
    out = out.contiguous()
    _triton_silu(out, out)  # in-place

    # 7) Residual add (out = out + x)
    out = out.contiguous()
    x_in = x.contiguous()
    out_out = torch.empty_like(out)
    _triton_add(out, x_in, out_out)

    return out_out

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure everything is on same device
        if x.device.type != 'cuda':
            # Fallback to original PyTorch if not on CUDA, though evaluator uses CUDA
            # But here we strictly require Triton, so raise
            raise RuntimeError("ModelNew requires CUDA tensors for Triton execution.")
        return run_triton(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
