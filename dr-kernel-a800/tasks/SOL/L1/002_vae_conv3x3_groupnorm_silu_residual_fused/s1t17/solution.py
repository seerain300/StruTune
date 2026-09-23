import triton
import triton.language as tl

# Triton kernels
# Conv3x3 NCHW, stride=1, padding=1, no bias
@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,            # *float32, input [B, C_in, H, W]
    w_ptr,            # *float32, weight [C_out, C_in, 3, 3]
    y_ptr,            # *float32, output [B, C_out, H, W]
    B: tl.constexpr,  # int
    C_in: tl.constexpr,  # int
    H_in: tl.constexpr,  # int
    W_in: tl.constexpr,  # int
    C_out: tl.constexpr,  # int
    H_out: tl.constexpr,  # int
    W_out: tl.constexpr,  # int
    BLOCK_IN: tl.constexpr,  # chunk size for input channels
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    acc = 0.0
    # Iterate input channels in chunks
    for c_start in range(0, C_in, BLOCK_IN):
        c_idx = c_start + tl.arange(0, BLOCK_IN)
        mask_c = c_idx < C_in
        # Accumulate over 3x3 window with padding
        for kh in range(3):
            for kw in range(3):
                h_in = h_out * 1 + kh - 1  # stride=1, padding=1
                w_in = w_out * 1 + kw - 1
                in_bounds = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                # Base offset for this output location and channel chunk
                base = ((n * C_in + c_idx) * H_in + h_in) * W_in + w_in
                x_vals = tl.load(x_ptr + base, mask=mask_c & in_bounds, other=0.0)
                # Loop over chunk to gather per-channel values and multiply by corresponding weight
                for i in range(BLOCK_IN):
                    if mask_c[i]:
                        for ci in range(C_in):
                            # Load weight for (c_out, ci, kh, kw)
                            # w layout: [C_out, C_in, 3, 3]
                            w_off = c_out * (C_in * 9) + ci * 9 + kh * 3 + kw
                            w_val = tl.load(w_ptr + w_off)
                            acc += x_vals[i] * w_val
    # Store the accumulated result
    y_off = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out
    tl.store(y_ptr + y_off, acc)

# GroupNorm with affine per (batch, group)
@triton.jit
def groupnorm_affine_fp32(
    x_ptr,            # *float32, input flattened per (B, group) over channels and H*W
    scale_ptr,        # *float32, per-channel scale (C,)
    bias_ptr,         # *float32, per-channel bias (C,)
    y_ptr,            # *float32, output flattened per (B, group)
    B: tl.constexpr,  # int
    C: tl.constexpr,  # int (channels per group times group_count)
    HW: tl.constexpr,  # int (H*W)
    num_groups: tl.constexpr,  # int (e.g., 32)
    group_size: tl.constexpr,  # int (C // num_groups)
    eps: tl.constexpr,  # float
    BLOCK_HW: tl.constexpr,  # chunk size for HW
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c_start = group * group_size

    # Compute sum and sum of squares over this group's channels and HW
    sum_val = 0.0
    sum_sq = 0.0
    # Loop over channels in the group
    for c in range(group_size):
        ch = c_start + c
        # Loop over HW in chunks
        for hw_start in range(0, HW, BLOCK_HW):
            hw_idx = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < HW
            x_off = ((n * C) + ch) * HW + hw_idx
            x_vals = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
            sum_val += tl.sum(x_vals, axis=0)
            sum_sq += tl.sum(x_vals * x_vals, axis=0)
    mean = sum_val / (C * HW)
    var = sum_sq / (C * HW) - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for c in range(group_size):
        ch = c_start + c
        for hw_start in range(0, HW, BLOCK_HW):
            hw_idx = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < HW
            x_off = ((n * C) + ch) * HW + hw_idx
            x_vals = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
            # Per-channel scale and bias
            scale = tl.load(scale_ptr + ch)
            bias = tl.load(bias_ptr + ch)
            y_vals = (x_vals - mean) * rstd
            y_vals = y_vals * scale + bias
            y_off = ((n * C) + ch) * HW + hw_idx
            tl.store(y_ptr + y_off, y_vals, mask=mask_hw)

# SiLU elementwise over flattened tensor
@triton.jit
def silu_fp32_elementwise(
    x_ptr,            # *float32
    y_ptr,            # *float32
    total_elems: tl.constexpr,  # int
    BLOCK: tl.constexpr,        # chunk size
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total_elems
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + idx, y, mask=mask)

# Residual addition elementwise: y = x + res (both fp32)
@triton.jit
def add_residual_fp32(
    x_ptr,            # *float32
    res_ptr,          # *float32
    y_ptr,            # *float32
    total_elems: tl.constexpr,  # int
    BLOCK: tl.constexpr,        # chunk size
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total_elems
    a = tl.load(x_ptr + idx, mask=mask, other=0.0)
    b = tl.load(res_ptr + idx, mask=mask, other=0.0)
    y = a + b
    tl.store(y_ptr + idx, y, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,   # (C, C, 3, 3)
        norm1_weight: torch.Tensor,   # (C,)
        norm1_bias: torch.Tensor,     # (C,)
        conv2_weight: torch.Tensor,   # (C, C, 3, 3)
        norm2_weight: torch.Tensor,   # (C,)
        norm2_bias: torch.Tensor,     # (C,)
        eps: float,
    ):
        # Ensure device is CUDA and dtype float32 for Triton kernels
        device = x.device
        assert device.type == 'cuda', "Triton kernels require CUDA tensors"

        B, C_in, H, W = x.shape
        # First conv: y1 = conv3x3(x)
        C1_out = conv1_weight.shape[0]
        H1_out = H
        W1_out = W
        x_contig = x.contiguous().to(torch.float32)
        w1_contig = conv1_weight.contiguous().to(torch.float32)
        y1 = torch.empty((B, C1_out, H1_out, W1_out), device=device, dtype=torch.float32)

        grid_conv1 = (B, C1_out, H1_out, W1_out)
        conv3x3_nchw_fp32[grid_conv1](
            x_contig, w1_contig, y1,
            B, C_in, H, W, C1_out, H1_out, W1_out,
            BLOCK_IN=64, num_warps=4, num_stages=2
        )

        # First GroupNorm (num_groups=32)
        C = C1_out
        assert C % self.num_groups == 0, "Channels must be divisible by num_groups"
        group_size = C // self.num_groups
        y1_flat = y1.view(B, C, H * W).contiguous()
        y1_norm = torch.empty((B, C, H * W), device=device, dtype=torch.float32)

        grid_gn1 = (B, self.num_groups)
        groupnorm_affine_fp32[grid_gn1](
            y1_flat, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
            y1_norm, B, C, H * W, self.num_groups, group_size, eps,
            BLOCK_HW=256, num_warps=4, num_stages=2
        )
        y1_norm = y1_norm.view(B, C, H, W)

        # SiLU on y1_norm
        total1 = B * C * H * W
        y1_silu = torch.empty_like(y1_norm, device=device, dtype=torch.float32)
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_fp32_elementwise[grid_silu1](y1_norm, y1_silu, total1, BLOCK=1024, num_warps=4, num_stages=2)

        # Second conv: y2 = conv3x3(y1_silu)
        C_in2 = C
        C2_out = conv2_weight.shape[0]
        H2_out = H
        W2_out = W
        y2 = torch.empty((B, C2_out, H2_out, W2_out), device=device, dtype=torch.float32)

        grid_conv2 = (B, C2_out, H2_out, W2_out)
        w2_contig = conv2_weight.contiguous().to(torch.float32)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, w2_contig, y2,
            B, C_in2, H2_out, W2_out, C2_out, H2_out, W2_out,
            BLOCK_IN=64, num_warps=4, num_stages=2
        )

        # Second GroupNorm (num_groups=32)
        C = C2_out
        assert C % self.num_groups == 0, "Channels must be divisible by num_groups"
        group_size2 = C // self.num_groups
        y2_flat = y2.view(B, C, H * W).contiguous()
        y2_norm = torch.empty((B, C, H * W), device=device, dtype=torch.float32)

        grid_gn2 = (B, self.num_groups)
        groupnorm_affine_fp32[grid_gn2](
            y2_flat, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
            y2_norm, B, C, H * W, self.num_groups, group_size2, eps,
            BLOCK_HW=256, num_warps=4, num_stages=2
        )
        y2_norm = y2_norm.view(B, C, H, W)

        # SiLU on y2_norm
        total2 = B * C * H * W
        y2_silu = torch.empty_like(y2_norm, device=device, dtype=torch.float32)
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_fp32_elementwise[grid_silu2](y2_norm, y2_silu, total2, BLOCK=1024, num_warps=4, num_stages=2)

        # Residual addition: original input x (cast to fp32) added to final output
        # Shapes must match (B, C, H, W); original x has this shape
        x_fp32 = x.contiguous().to(torch.float32)
        total_final = total2
        final_out = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        grid_add = (triton.cdiv(total_final, 1024),)
        add_residual_fp32[grid_add](x_fp32, y2_silu, final_out, total_final, BLOCK=1024, num_warps=4, num_stages=2)

        return final_out


def run(*args):
    return ModelNew()(*args)
