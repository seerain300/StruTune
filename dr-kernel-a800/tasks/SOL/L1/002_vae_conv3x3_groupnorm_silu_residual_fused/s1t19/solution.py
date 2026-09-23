import triton
import triton.language as tl

# Conv3x3 NCHW, stride=1, padding=1, no bias
@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,            # *float32, input [B, C_in, H, W]
    w_ptr,            # *float32, weight [C_out, C_in, 3, 3]
    y_ptr,            # *float32, output [B, C_out, H, W]
    B: tl.constexpr,     # batch size
    C_in: tl.constexpr,  # input channels
    H_in: tl.constexpr,  # input height
    W_in: tl.constexpr,  # input width
    C_out: tl.constexpr, # output channels
    H_out: tl.constexpr, # output height
    W_out: tl.constexpr, # output width
    BLOCK_IN: tl.constexpr,
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    # Accumulator for one output element
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks
    for cin_start in range(0, C_in, BLOCK_IN):
        cin_offsets = cin_start + tl.arange(0, BLOCK_IN)
        mask_cin = cin_offsets < C_in

        # Accumulate over 3x3 window
        for kh in range(3):
            h_in = h_out + kh - 1  # -1 due to padding=1, kh in [0,2]
            valid_h = (h_in >= 0) & (h_in < H_in)
            for kw in range(3):
                w_in = w_out + kw - 1  # -1 due to padding=1
                valid_w = (w_in >= 0) & (w_in < W_in)

                base_in = ((n * C_in + cin_offsets[:, None]) * H_in + h_in) * W_in + w_in
                mask_load = mask_cin[:, None] & valid_h & valid_w

                # Load input vector for this (n, cin chunk, h_in, w_in)
                x_vals = tl.load(x_ptr + base_in, mask=mask_load, other=0.0)
                x_vals = x_vals.to(tl.float32)

                # Load weight vector for this (c_out, cin chunk, kh, kw)
                w_base = ((c_out * C_in + cin_offsets) * 9) + (kh * 3 + kw)
                w_vals = tl.load(w_ptr + w_base, mask=mask_cin, other=0.0)
                w_vals = w_vals.to(tl.float32)

                # Outer product accumulate: [BLOCK_IN] * [BLOCK_IN] -> scalar
                # Sum over cin chunk
                acc += tl.sum(x_vals * w_vals[None, :], axis=0)

    # Store the result
    y_index = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out
    tl.store(y_ptr + y_index, acc)

# GroupNorm with affine per (batch, group)
@triton.jit
def groupnorm_affine_fp32(
    x_ptr,           # *float32, input flattened [B, C, H*W]
    scale_ptr,       # *float32, per-channel scale [C]
    bias_ptr,        # *float32, per-channel bias [C]
    y_ptr,           # *float32, output flattened [B, C, H*W]
    B: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,       # H*W
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)

    # Compute sum and sum of squares over this group
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over channels in the group
    for g_start in range(0, C, group_size):
        c_idx = g_start + tl.arange(0, group_size)
        mask_c = c_idx < C

        # Loop over spatial positions
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = offs < HW

            # Base pointers for x and y: each channel has a contiguous HW block
            base = n * C * HW + c_idx[:, None] * HW + offs[None, :]
            mask = mask_c[:, None] & mask_hw[None, :]

            x_vals = tl.load(x_ptr + base, mask=mask, other=0.0)  # [group_size, BLOCK_HW]
            x_vals = x_vals.to(tl.float32)

            # Sum over channels and spatial
            sum_val += tl.sum(x_vals, axis=None)
            sum_sq += tl.sum(x_vals * x_vals, axis=None)

    # Compute mean and variance for this (n, group)
    m = C * HW
    mean = sum_val / m
    var = sum_sq / m - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for g_start in range(0, C, group_size):
        c_idx = g_start + tl.arange(0, group_size)
        mask_c = c_idx < C

        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = offs < HW

            base = n * C * HW + c_idx[:, None] * HW + offs[None, :]
            mask = mask_c[:, None] & mask_hw[None, :]

            x_vals = tl.load(x_ptr + base, mask=mask, other=0.0)
            x_vals = x_vals.to(tl.float32)

            # Normalize
            y_norm = (x_vals - mean) * rstd

            # Affine per channel
            scale = tl.load(scale_ptr + c_idx, mask=mask_c, other=1.0)
            bias = tl.load(bias_ptr + c_idx, mask=mask_c, other=0.0)
            y_vals = y_norm * scale[:, None] + bias[:, None]

            tl.store(y_ptr + base, y_vals, mask=mask)

# SiLU elementwise over flattened tensors
@triton.jit
def silu_fp32_elementwise(x_ptr, y_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)

# Residual addition: elementwise add two tensors
@triton.jit
def add_residual_fp32(a_ptr, b_ptr, y_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    y = a + b
    tl.store(y_ptr + offsets, y, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        # Ensure float32 and contiguous
        device = x.device
        x_fp32 = x.contiguous().to(torch.float32)

        # First conv: y1 = conv3x3(x)
        B, C_in, H, W = x_fp32.shape
        C_out1 = conv1_weight.shape[0]
        H_out1 = H
        W_out1 = W
        y1 = torch.empty((B, C_out1, H_out1, W_out1), device=device, dtype=torch.float32)

        grid_conv1 = (B, C_out1, H_out1, W_out1)
        conv3x3_nchw_fp32[grid_conv1](
            x_fp32, conv1_weight.contiguous().to(torch.float32), y1,
            B, C_in, H, W, C_out1, H_out1, W_out1,
            BLOCK_IN=64,
            num_warps=4,
        )

        # GroupNorm 1 (num_groups=32) over y1
        C = C_out1
        assert C % 32 == 0, "C must be divisible by num_groups (32)"
        group_size = C // 32
        y1_flat = y1.view(B, C, H * W).contiguous()
        y1_norm = torch.empty((B, C, H * W), device=device, dtype=torch.float32)

        grid_gn1 = (B, 32)
        groupnorm_affine_fp32[grid_gn1](
            y1_flat, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
            y1_norm, B, C, H * W, 32, group_size, eps,
            BLOCK_HW=256,
            num_warps=4,
        )
        y1_norm = y1_norm.view(B, C, H, W)

        # SiLU on y1_norm
        y1_silu = torch.empty_like(y1_norm, device=device, dtype=torch.float32)
        total1 = B * C * H * W
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_fp32_elementwise[grid_silu1](y1_norm, y1_silu, total1, BLOCK=1024, num_warps=4)

        # Second conv: y2 = conv3x3(y1_silu)
        C_in2 = C  # output channels of first conv == input channels of second conv
        C_out2 = conv2_weight.shape[0]
        H_out2 = H
        W_out2 = W
        y2 = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=torch.float32)

        grid_conv2 = (B, C_out2, H_out2, W_out2)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight.contiguous().to(torch.float32), y2,
            B, C_in2, H_out2, W_out2, C_out2, H_out2, W_out2,
            BLOCK_IN=64,
            num_warps=4,
        )

        # GroupNorm 2 (num_groups=32) over y2
        C = C_out2
        assert C % 32 == 0, "C must be divisible by num_groups (32)"
        group_size = C // 32
        y2_flat = y2.view(B, C, H * W).contiguous()
        y2_norm = torch.empty((B, C, H * W), device=device, dtype=torch.float32)

        grid_gn2 = (B, 32)
        groupnorm_affine_fp32[grid_gn2](
            y2_flat, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
            y2_norm, B, C, H * W, 32, group_size, eps,
            BLOCK_HW=256,
            num_warps=4,
        )
        y2_norm = y2_norm.view(B, C, H, W)

        # SiLU on y2_norm
        y2_silu = torch.empty_like(y2_norm, device=device, dtype=torch.float32)
        total2 = B * C * H * W
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_fp32_elementwise[grid_silu2](y2_norm, y2_silu, total2, BLOCK=1024, num_warps=4)

        # Residual addition: add original x (cast to fp32) to y2_silu. Shapes must match (B, C, H, W).
        out = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        total = B * C * H * W
        grid_add = (triton.cdiv(total, 1024),)
        add_residual_fp32[grid_add](x_fp32, y2_silu, out, total, BLOCK=1024, num_warps=4)

        return out


def run(*args):
    return ModelNew()(*args)
