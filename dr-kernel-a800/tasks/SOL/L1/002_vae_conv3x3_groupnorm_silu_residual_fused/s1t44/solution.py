import torch
import triton
import triton.language as tl

# Conv3x3 NCHW: output y = conv2d(x, weight, stride=1, padding=1, no bias)
# Each program computes one output element y[n, c_out, h_out, w_out]
@triton.jit
def conv3x3_nchw_fp32(x_ptr, w_ptr, y_ptr,
                       B, C_in, H, W, C_out,
                       BLOCK_IN: tl.constexpr):
    total = B * C_out * H * W
    pid = tl.program_id(0)
    # Map pid to (n, c_out, h_out, w_out)
    HW = H * W
    n = pid // (C_out * HW)
    rem1 = pid % (C_out * HW)
    c_out = rem1 // HW
    rem2 = rem1 % HW
    h_out = rem2 // W
    w_out = rem2 % W

    # Accumulator for output element
    acc = 0.0

    # Iterate over input channels in chunks
    for c_in_start in range(0, C_in, BLOCK_IN):
        c_in_range = c_in_start + tl.arange(0, BLOCK_IN)
        mask_c = c_in_range < C_in

        # Base offsets for x and w
        # x offset: ((n * C_in + c_in) * H + h) * W + w
        # w offset: ((c_out * C_in + c_in) * 3 * 3 + kh * 3 + kw)
        # We will compute acc += sum_{ci in chunk} sum_{kh,kw in 3x3} x[n, ci, h_out+kh, w_out+kw] * w[ci, c_out, kh, kw]
        for kh in range(0, 3):
            h_in = h_out + kh
            in_bounds_h = (h_in >= 0) & (h_in < H)
            for kw in range(0, 3):
                w_in = w_out + kw
                in_bounds_w = (w_in >= 0) & (w_in < W)
                # Load x vector for all ci in chunk at (h_in, w_in)
                # Address for x: ((n * C_in + c_in) * H + h_in) * W + w_in
                x_offset = ((n * C_in + c_in_range[:, None]) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_offset, mask=mask_c, other=0.0)  # shape [BLOCK_IN]
                # Load w vector for all ci in chunk at (kh, kw): ((c_out * C_in + c_in) * (3*3) + kh*3 + kw)
                w_offset = ((c_out * C_in + c_in_range) * 9 + kh * 3 + kw)
                w_val = tl.load(w_ptr + w_offset, mask=mask_c, other=0.0)  # shape [BLOCK_IN]
                # Outer product accumulate: acc += sum over ci of x_val[ci] * w_val[ci]
                # To do elementwise multiply then reduce: tl.sum(x_val[:, None] * w_val[None, :], axis=0)
                acc += tl.sum(x_val[:, None] * w_val[None, :], axis=0)

    # Store result
    y_offset = ((n * C_out + c_out) * H + h_out) * W + w_out
    tl.store(y_ptr + y_offset, acc)

# Second conv: identical to the first conv, same signature
@triton.jit
def conv3x3_nchw_fp32_2(x_ptr, w_ptr, y_ptr,
                        B, C_in, H, W, C_out,
                        BLOCK_IN: tl.constexpr):
    total = B * C_out * H * W
    pid = tl.program_id(0)
    HW = H * W
    n = pid // (C_out * HW)
    rem1 = pid % (C_out * HW)
    c_out = rem1 // HW
    rem2 = rem1 % HW
    h_out = rem2 // W
    w_out = rem2 % W

    acc = 0.0

    for c_in_start in range(0, C_in, BLOCK_IN):
        c_in_range = c_in_start + tl.arange(0, BLOCK_IN)
        mask_c = c_in_range < C_in

        for kh in range(0, 3):
            h_in = h_out + kh
            in_bounds_h = (h_in >= 0) & (h_in < H)
            for kw in range(0, 3):
                w_in = w_out + kw
                in_bounds_w = (w_in >= 0) & (w_in < W)
                x_offset = ((n * C_in + c_in_range[:, None]) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_offset, mask=mask_c, other=0.0)
                w_offset = ((c_out * C_in + c_in_range) * 9 + kh * 3 + kw)
                w_val = tl.load(w_ptr + w_offset, mask=mask_c, other=0.0)
                acc += tl.sum(x_val[:, None] * w_val[None, :], axis=0)

    y_offset = ((n * C_out + c_out) * H + h_out) * W + w_out
    tl.store(y_ptr + y_offset, acc)

# GroupNorm with affine: per (n, group), two-pass approach
@triton.jit
def groupnorm_affine_kernel(x_ptr, y_ptr, weight_ptr, bias_ptr,
                             B, C, H, W, num_groups, eps,
                             BLOCK_HW: tl.constexpr):
    # One program per (n, group)
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    group_size = C // num_groups
    start_c = g * group_size
    # Compute sum and sum of squares over channels in group and all spatial positions
    sum_val = 0.0
    sum_sq = 0.0

    for c in range(start_c, start_c + group_size):
        for hw in range(0, H * W, BLOCK_HW):
            hw_idx = hw + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < (H * W)
            h = hw_idx // W
            w = hw_idx % W
            # Linearized index for x: ((n * C + c) * H + h) * W + w
            idx = ((n * C + c) * H + h) * W + w
            x_val = tl.load(x_ptr + idx, mask=mask_hw, other=0.0)
            sum_val += tl.sum(x_val, axis=0)
            sum_sq += tl.sum(x_val * x_val, axis=0)

    count = group_size * (H * W)
    mean = sum_val / count
    var = sum_sq / count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine per channel
    for c in range(start_c, start_c + group_size):
        for hw in range(0, H * W, BLOCK_HW):
            hw_idx = hw + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < (H * W)
            h = hw_idx // W
            w = hw_idx % W
            idx = ((n * C + c) * H + h) * W + w
            x_val = tl.load(x_ptr + idx, mask=mask_hw, other=0.0)
            y = (x_val - mean) * inv_std
            scale = tl.load(weight_ptr + c)
            bias = tl.load(bias_ptr + c)
            y = y * scale + bias
            tl.store(y_ptr + idx, y, mask=mask_hw)

# SiLU elementwise: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
@triton.jit
def silu_kernel(x_ptr, y_ptr, N):
    pid = tl.program_id(0)
    # N is total number of elements; we assume 1D launch
    x = tl.load(x_ptr + pid)
    # sigmoid
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + pid, y)

# Residual addition: add x_res to y
@triton.jit
def add_residual_kernel(y_ptr, x_res_ptr, out_ptr, N):
    pid = tl.program_id(0)
    y = tl.load(y_ptr + pid)
    x = tl.load(x_res_ptr + pid)
    tl.store(out_ptr + pid, y + x)

class ModelNew(torch.nn.Module):
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
        dtype = torch.float32
        x_fp32 = x.to(dtype).contiguous()
        conv1_weight = conv1_weight.to(dtype).contiguous()
        norm1_weight = norm1_weight.to(dtype).contiguous()
        norm1_bias = norm1_bias.to(dtype).contiguous()
        conv2_weight = conv2_weight.to(dtype).contiguous()
        norm2_weight = norm2_weight.to(dtype).contiguous()
        norm2_bias = norm2_bias.to(dtype).contiguous()

        B, C_in, H, W = x_fp32.shape
        # First conv: y1 = conv3x3(x, conv1_weight)
        C_out = conv1_weight.shape[0]
        y1 = torch.empty((B, C_out, H, W), device=device, dtype=dtype)
        total1 = B * C_out * H * W
        conv3x3_nchw_fp32[(total1,)](
            x_fp32, conv1_weight, y1,
            B, C_in, H, W, C_out,
            BLOCK_IN=8,
            num_warps=4,
        )

        # First GroupNorm with affine
        y1_gn = torch.empty_like(y1, device=device, dtype=dtype)
        num_groups = 32
        groupnorm_affine_kernel[(B * num_groups,)](
            y1, y1_gn, norm1_weight, norm1_bias,
            B, C_out, H, W, num_groups, eps,
            BLOCK_HW=1024,
            num_warps=4,
        )

        # SiLU
        silu_out1 = torch.empty_like(y1_gn, device=device, dtype=dtype)
        total_silu = y1_gn.numel()
        silu_kernel[(total_silu,)](
            y1_gn, silu_out1,
            total_silu,
            num_warps=4,
        )

        # Second conv: y2 = conv3x3(silu_out1, conv2_weight)
        C_out2 = conv2_weight.shape[0]
        y2 = torch.empty((B, C_out2, H, W), device=device, dtype=dtype)
        total2 = B * C_out2 * H * W
        conv3x3_nchw_fp32_2[(total2,)](
            silu_out1, conv2_weight, y2,
            B, C_out, H, W, C_out2,  # NOTE: C_in for second conv is C_out of first output (which equals conv1_weight.shape[0])
            BLOCK_IN=8,
            num_warps=4,
        )

        # Second GroupNorm with affine
        y2_gn = torch.empty_like(y2, device=device, dtype=dtype)
        groupnorm_affine_kernel[(B * num_groups,)](
            y2, y2_gn, norm2_weight, norm2_bias,
            B, C_out2, H, W, num_groups, eps,
            BLOCK_HW=1024,
            num_warps=4,
        )

        # SiLU
        silu_out2 = torch.empty_like(y2_gn, device=device, dtype=dtype)
        total_silu2 = y2_gn.numel()
        silu_kernel[(total_silu2,)](
            y2_gn, silu_out2,
            total_silu2,
            num_warps=4,
        )

        # Residual connection: add original x (cast to fp32) to the final output
        x_res = x_fp32  # already fp32 contiguous
        final_out = torch.empty_like(silu_out2, device=device, dtype=dtype)
        add_residual_kernel[(silu_out2.numel(),)](
            silu_out2, x_res, final_out,
            silu_out2.numel(),
            num_warps=4,
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
