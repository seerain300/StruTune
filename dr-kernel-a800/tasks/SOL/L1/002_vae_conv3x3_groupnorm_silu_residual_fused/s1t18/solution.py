import triton
import triton.language as tl


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
    BLOCK_IN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h_out = tl.program_id(2)
    pid_w_out = tl.program_id(3)

    acc = 0.0

    # Loop over input channels in chunks
    for cin_start in range(0, C_in, BLOCK_IN):
        c_range = cin_start + tl.arange(0, BLOCK_IN)
        mask_c = c_range < C_in

        # 3x3 kernel window with padding
        for kh in range(3):
            for kw in range(3):
                h_in = pid_h_out * 1 + kh - 1  # padding=1
                w_in = pid_w_out * 1 + kw - 1
                in_h_ok = (h_in >= 0) & (h_in < H_in)
                in_w_ok = (w_in >= 0) & (w_in < W_in)
                in_ok = in_h_ok & in_w_ok

                x_offset = pid_n * (C_in * H_in * W_in) + c_range[:, None] * (H_in * W_in) + h_in * W_in + w_in
                x_mask = mask_c[:, None] & in_ok

                x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0).to(tl.float32)
                w_offset = pid_c_out * (C_in * 9) + c_range * 9 + (kh * 3 + kw)
                w_vals = tl.load(w_ptr + w_offset, mask=mask_c, other=0.0).to(tl.float32)

                acc += tl.sum(x_vals * w_vals[:, None], axis=0)

    y_offset = pid_n * (C_out * H_out * W_out) + pid_c_out * (H_out * W_out) + pid_h_out * W_out + pid_w_out
    tl.store(y_ptr + y_offset, acc)


# GroupNorm with affine per (batch, group). C is number of channels in the input tensor (post-conv).
@triton.jit
def groupnorm_affine_fp32(
    x_ptr,          # *float32, input flattened [B, C, H*W]
    gamma_ptr,      # *float32, norm weight [C]
    beta_ptr,       # *float32, norm bias [C]
    y_ptr,          # *float32, output flattened [B, C, H*W]
    B: tl.constexpr, C: tl.constexpr,
    HW: tl.constexpr, num_groups: tl.constexpr, group_size: tl.constexpr, eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)  # group id in [0, num_groups)

    c_start = pid_g * group_size
    c_end = c_start + group_size

    # Pass 1: compute sum and sum of squares over channels in group and all spatial positions
    sum_val = 0.0
    sum_sq = 0.0
    for c in range(c_start, c_end):
        for hw_start in range(0, HW, BLOCK_HW):
            idx = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = idx < HW
            x_offset = pid_n * (C * HW) + c * HW + idx
            x_vals = tl.load(x_ptr + x_offset, mask=mask_hw, other=0.0).to(tl.float32)
            sum_val += tl.sum(x_vals, axis=0)
            sum_sq += tl.sum(x_vals * x_vals, axis=0)

    group_elems = group_size * HW
    mean = sum_val / group_elems
    var = sum_sq / group_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, then store
    for c in range(c_start, c_end):
        gamma_c = tl.load(gamma_ptr + c).to(tl.float32)
        beta_c = tl.load(beta_ptr + c).to(tl.float32)
        for hw_start in range(0, HW, BLOCK_HW):
            idx = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = idx < HW
            x_offset = pid_n * (C * HW) + c * HW + idx
            x_vals = tl.load(x_ptr + x_offset, mask=mask_hw, other=0.0).to(tl.float32)
            y_vals = (x_vals - mean) * inv_std
            y_vals = y_vals * gamma_c + beta_c
            y_offset = pid_n * (C * HW) + c * HW + idx
            tl.store(y_ptr + y_offset, y_vals, mask=mask_hw)


# Elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_fp32_elementwise(x_ptr, y_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Elementwise residual addition: y = x1 + x2 (float32)
@triton.jit
def add_residual_fp32(x1_ptr, x2_ptr, y_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(x1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a + b
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,                      # input tensor (B, C, H, W)
        conv1_weight: torch.Tensor,           # (C, C, 3, 3)
        norm1_weight: torch.Tensor,           # (C,)
        norm1_bias: torch.Tensor,             # (C,)
        conv2_weight: torch.Tensor,           # (C, C, 3, 3)
        norm2_weight: torch.Tensor,           # (C,)
        norm2_bias: torch.Tensor,             # (C,)
        eps: float,
    ):
        device = x.device
        B, C_in, H, W = x.shape

        # conv1: (C_in, C, 3, 3) -> output shape: (B, C, H, W)
        conv1_weight = conv1_weight.to(device=device, dtype=torch.float32).contiguous()
        C = conv1_weight.shape[0]  # output channels of conv1
        y1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)

        grid_conv1 = (B, C, H, W)
        conv3x3_nchw_fp32[grid_conv1](
            x.contiguous().to(torch.float32), conv1_weight, y1,
            B, C_in, H, W, C, H, W, BLOCK_IN=64, num_warps=4,
        )

        # GroupNorm 1 (num_groups=32) over y1, affine
        assert C % self.num_groups == 0, "C must be divisible by num_groups (32)"
        group_size1 = C // self.num_groups
        y1_flat = y1.view(B, C, H * W).contiguous()
        y1_norm = torch.empty_like(y1_flat, device=device, dtype=torch.float32)

        grid_gn1 = (B, self.num_groups)
        groupnorm_affine_fp32[grid_gn1](
            y1_flat, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
            y1_norm, B, C, H * W, self.num_groups, group_size1, self.eps,
            BLOCK_HW=256, num_warps=4,
        )
        y1_norm = y1_norm.view(B, C, H, W)

        # SiLU on y1_norm
        y1_silu = torch.empty_like(y1_norm, device=device, dtype=torch.float32)
        total1 = B * C * H * W
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_fp32_elementwise[grid_silu1](y1_norm, y1_silu, total1, BLOCK=1024, num_warps=4)

        # conv2: (C, C, 3, 3) -> output shape: (B, C, H, W)
        conv2_weight = conv2_weight.to(device=device, dtype=torch.float32).contiguous()
        y2 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)

        grid_conv2 = (B, C, H, W)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight, y2,
            B, C, H, W, C, H, W, BLOCK_IN=64, num_warps=4,
        )

        # GroupNorm 2 (num_groups=32) over y2, affine
        assert C % self.num_groups == 0, "C must be divisible by num_groups (32)"
        group_size2 = C // self.num_groups
        y2_flat = y2.view(B, C, H * W).contiguous()
        y2_norm = torch.empty_like(y2_flat, device=device, dtype=torch.float32)

        grid_gn2 = (B, self.num_groups)
        groupnorm_affine_fp32[grid_gn2](
            y2_flat, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
            y2_norm, B, C, H * W, self.num_groups, group_size2, self.eps,
            BLOCK_HW=256, num_warps=4,
        )
        y2_norm = y2_norm.view(B, C, H, W)

        # SiLU on y2_norm
        y2_silu = torch.empty_like(y2_norm, device=device, dtype=torch.float32)
        total2 = B * C * H * W
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_fp32_elementwise[grid_silu2](y2_norm, y2_silu, total2, BLOCK=1024, num_warps=4)

        # Residual addition: add original x (cast to fp32) to y2_silu. Shapes must match (B, C, H, W).
        x_fp32 = x.contiguous().to(torch.float32)
        out = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        total = B * C * H * W
        grid_add = (triton.cdiv(total, 1024),)
        add_residual_fp32[grid_add](x_fp32, y2_silu, out, total, BLOCK=1024, num_warps=4)

        return out


def run(*args):
    return ModelNew()(*args)
