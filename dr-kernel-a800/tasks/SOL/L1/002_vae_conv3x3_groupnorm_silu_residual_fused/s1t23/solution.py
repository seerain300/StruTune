import torch
import triton
import triton.language as tl

# First conv: conv3x3 NCHW, stride=1, padding=1, no bias
@triton.jit
def conv3x3_nchw_fp32(
    inp_ptr,           # *float32, shape [N, C_in, H, W]
    weight_ptr,        # *float32, shape [C_out, C_in, 3, 3]
    out_ptr,           # *float32, shape [N, C_out, H, W]
    N, C_in, H, W, C_out,
    BLOCK_IN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Map pid -> (n, c_out, h_out, w_out)
    total_hw = H * W
    n = pid // (C_out * total_hw)
    tmp = pid % (C_out * total_hw)
    c_out = tmp // total_hw
    tmp2 = tmp % total_hw
    h_out = tmp2 // W
    w_out = tmp2 % W

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks
    for ci_start in range(0, C_in, BLOCK_IN):
        ci_range = ci_start + tl.arange(0, BLOCK_IN)
        ci_mask = ci_range < C_in

        # For each (kh, kw), compute corresponding input indices and masked loads
        for kh in range(3):
            ih = h_out + kh - 1  # padding
            for kw_in in range(3):
                iw = w_out + kw_in - 1
                # valid spatial positions
                valid_hw = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                # base pointer for input tensor at (n, :, ih, iw)
                inp_base = inp_ptr + n * C_in * H * W + ih * W + iw
                # load input values for all ci in chunk
                vals = tl.load(inp_base + ci_range, mask=ci_mask & valid_hw, other=0.0)

                # load weight vector for this (c_out, ci chunk, kh, kw)
                w_base = weight_ptr + c_out * C_in * 9 + ci_range * 9 + kh * 3 + kw_in
                w_vec = tl.load(w_base, mask=ci_mask, other=0.0)

                # accumulate dot product
                acc += tl.sum(vals * w_vec, axis=0)

    # Store result
    out_index = out_ptr + n * C_out * H * W + c_out * H * W + h_out * W + w_out
    tl.store(out_index, acc)


# GroupNorm with affine: per (n, group), C_in % 32 == 0
@triton.jit
def groupnorm_affine_fp32(
    inp_ptr,        # *float32, shape [N, C_in, H, W]
    gamma_ptr,      # *float32, shape [C_in] (norm weight)
    beta_ptr,       # *float32, shape [C_in] (norm bias)
    out_ptr,        # *float32, shape [N, C_in, H, W]
    N, C_in, H, W,
    group_id: tl.constexpr,  # 0..num_groups-1
    group_size: tl.constexpr,  # C_in // 32
    num_groups: tl.constexpr,  # 32
    eps,                        # float32
):
    # Compute mean and variance over this group
    sum_ = tl.zeros((), dtype=tl.float32)
    sumsq_ = tl.zeros((), dtype=tl.float32)
    total = H * W

    # Pass 1: accumulate sum and sumsq
    ci = 0
    while ci < C_in:
        # channel index within group
        for gi in range(group_size):
            ch = ci + gi  # valid since ci+group_size=C_in
            # spatial loop
            s = 0
            while s < total:
                hw = s
                h = hw // W
                w = hw % W
                base = inp_ptr + n * C_in * H * W + ch * H * W + h * W + w
                x = tl.load(base, mask=True, other=0.0)
                sum_ += x
                sumsq_ += x * x
                s += 1
        ci += group_size

    # Compute mean and variance
    M = C_in * total
    mean = sum_ / M
    var = sumsq_ / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, write to out
    ci = 0
    while ci < C_in:
        for gi in range(group_size):
            ch = ci + gi
            gamma = tl.load(gamma_ptr + ch)
            beta = tl.load(beta_ptr + ch)
            s = 0
            while s < total:
                hw = s
                h = hw // W
                w = hw % W
                base_in = inp_ptr + n * C_in * H * W + ch * H * W + h * W + w
                x = tl.load(base_in, mask=True, other=0.0)
                y = (x - mean) * rstd
                y = y * gamma + beta
                base_out = out_ptr + n * C_in * H * W + ch * H * W + h * W + w
                tl.store(base_out, y)
                s += 1
        ci += group_size


# Elementwise SiLU kernel
@triton.jit
def silu_kernel_fp32(x_ptr, y_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


# Elementwise residual addition kernel: add x (original input) to out
@triton.jit
def add_residual_kernel_fp32(x_ptr, out_ptr, res_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    a = tl.load(out_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(res_ptr + offsets, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_groups = 32
        self.eps = 1e-5

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # Ensure dtype and contiguity; compute in float32
        device = x.device
        dtype = torch.float32

        # Cast and make contiguous
        x_in = x.to(dtype).contiguous()
        conv1_w = conv1_weight.to(dtype).contiguous()
        norm1_g = norm1_weight.to(dtype).contiguous()
        norm1_b = norm1_bias.to(dtype).contiguous()
        conv2_w = conv2_weight.to(dtype).contiguous()
        norm2_g = norm2_weight.to(dtype).contiguous()
        norm2_b = norm2_bias.to(dtype).contiguous()

        N, C, H, W = x_in.shape
        # Check GroupNorm group divisibility
        if C % self.num_groups != 0:
            raise RuntimeError(f"Input channels {C} must be divisible by num_groups {self.num_groups}.")

        group_size = C // self.num_groups

        # First conv: output [N, C, H, W]
        y1 = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
        # Grid: one program per output element
        total = N * C * H * W
        grid_conv1 = (total,)
        conv3x3_nchw_fp32[grid_conv1](
            x_in, conv1_w, y1,
            N, C, H, W, C,
            BLOCK_IN=16,
        )

        # GroupNorm1
        y1_gn = torch.empty_like(y1, device=device, dtype=torch.float32)
        for g in range(self.num_groups):
            groupnorm_affine_fp32[(1,)](
                y1, norm1_g, norm1_b, y1_gn,
                N, C, H, W,
                group_id=g, group_size=group_size, num_groups=self.num_groups,
                eps=self.eps,
            )

        # SiLU1
        y1_silu = torch.empty_like(y1_gn, device=device, dtype=torch.float32)
        total_silu1 = y1_gn.numel()
        grid_silu1 = (triton.cdiv(total_silu1, 1024),)
        silu_kernel_fp32[grid_silu1](y1_gn, y1_silu, total_silu1, BLOCK=1024)

        # Second conv: output [N, C, H, W]
        y2 = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (N * C * H * W,)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_w, y2,
            N, C, H, W, C,
            BLOCK_IN=16,
        )

        # GroupNorm2
        y2_gn = torch.empty_like(y2, device=device, dtype=torch.float32)
        for g in range(self.num_groups):
            groupnorm_affine_fp32[(1,)](
                y2, norm2_g, norm2_b, y2_gn,
                N, C, H, W,
                group_id=g, group_size=group_size, num_groups=self.num_groups,
                eps=self.eps,
            )

        # SiLU2
        y2_silu = torch.empty_like(y2_gn, device=device, dtype=torch.float32)
        total_silu2 = y2_gn.numel()
        grid_silu2 = (triton.cdiv(total_silu2, 1024),)
        silu_kernel_fp32[grid_silu2](y2_gn, y2_silu, total_silu2, BLOCK=1024)

        # Residual addition: add original input x to the final output
        x_for_add = x_in  # shape (N, C, H, W), matches y2_silu
        res = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        total_add = y2_silu.numel()
        grid_add = (triton.cdiv(total_add, 1024),)
        add_residual_kernel_fp32[grid_add](y2_silu, x_for_add, res, total_add, BLOCK=1024)

        return res


# Example helper to generate inputs (not used by evaluator, provided for completeness)
def get_inputs():
    # Default: CPU tensors; evaluator may move to CUDA
    N, C, H, W = 1, 128, 64, 64  # Example; actual axes come from JSON config
    x = torch.randn(N, C, H, W, dtype=torch.float32)
    # Weights and norms are random init; evaluator passes real weights
    conv1_w = torch.randn(C, C, 3, 3, dtype=torch.float32)
    norm1_g = torch.randn(C, dtype=torch.float32)
    norm1_b = torch.randn(C, dtype=torch.float32)
    conv2_w = torch.randn(C, C, 3, 3, dtype=torch.float32)
    norm2_g = torch.randn(C, dtype=torch.float32)
    norm2_b = torch.randn(C, dtype=torch.float32)
    eps = 1e-5
    return x, conv1_w, norm1_g, norm1_b, conv2_w, norm2_g, norm2_b, eps


def run(*args):
    return ModelNew()(*args)
