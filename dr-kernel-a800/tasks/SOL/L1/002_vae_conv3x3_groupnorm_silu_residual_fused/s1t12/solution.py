import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,         # *const float
    w_ptr,         # *const float
    y_ptr,         # *float
    B: tl.constexpr, C_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    C_out: tl.constexpr, K: tl.constexpr,  # K = 3
    BLOCK_IN: tl.constexpr,
):
    # program id maps to (n, c_out, h_out, w_out)
    pid = tl.program_id(0)
    HW = H_in * W_in
    N = B
    C_OUT = C_out

    n = pid // (C_OUT * HW)
    rem = pid % (C_OUT * HW)
    c_out = rem // HW
    rem2 = rem % HW
    h_out = rem2 // W_in
    w_out = rem2 % W_in

    # initialize accumulator
    acc = 0.0

    # loop over input channels in chunks
    for cin_base in range(0, C_in, BLOCK_IN):
        cin_chunk = tl.arange(0, BLOCK_IN)
        cin = cin_base + cin_chunk  # vector of input channels for this chunk
        mask_cin = cin < C_in

        # accumulate over 3x3 neighborhood
        for kh in range(0, K):
            h = h_out * 1 + kh - 1  # padding=1
            in_h_ok = (h >= 0) & (h < H_in)
            for kw in range(0, K):
                w = w_out * 1 + kw - 1  # padding=1
                in_w_ok = (w >= 0) & (w < W_in)

                # compute input index for vector of (n, cin, h, w)
                in_index = ((n * C_in + cin) * H_in + h) * W_in + w
                x_vals = tl.load(x_ptr + in_index, mask=mask_cin & in_h_ok & in_w_ok, other=0.0)

                # load weights for this chunk: w_ptr has layout [C_out, C_in, 3, 3]
                w_index = (c_out * C_in + cin) * (K * K) + (kh * K + kw)
                w_vals = tl.load(w_ptr + w_index, mask=mask_cin, other=0.0)  # scalar per cin

                # accumulate
                acc += tl.sum(x_vals * w_vals, axis=0)

    # store result
    out_index = ((n * C_out + c_out) * H_in + h_out) * W_in + w_out
    tl.store(y_ptr + out_index, acc)


@triton.jit
def groupnorm_affine_fp32(
    x_ptr,          # *const float, input tensor flattened
    y_ptr,          # *float, output tensor flattened
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    group_size: tl.constexpr, eps: tl.constexpr,
    scale_ptr,      # *const float, shape (C,)
    bias_ptr,       # *const float, shape (C,)
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    groups = C // group_size
    n = pid // groups
    group = pid % groups

    # compute sum and sum of squares over channels in group and all spatial positions
    sum_val = 0.0
    sum_sq = 0.0

    # iterate over channels in the group
    for c in range(0, group_size):
        c_abs = group * group_size + c
        for h in range(0, H):
            for w in range(0, W):
                index = ((n * C + c_abs) * H + h) * W + w
                x_val = tl.load(x_ptr + index)
                sum_val += x_val
                sum_sq += x_val * x_val

    # compute mean and variance
    total_elems = group_size * H * W
    mean = sum_val / total_elems
    var = sum_sq / total_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize and apply affine
    for c in range(0, group_size):
        c_abs = group * group_size + c
        scale = tl.load(scale_ptr + c_abs)
        bias = tl.load(bias_ptr + c_abs)
        for h in range(0, H):
            for w in range(0, W):
                index = ((n * C + c_abs) * H + h) * W + w
                x_val = tl.load(x_ptr + index)
                y_val = (x_val - mean) * inv_std
                y_val = y_val * scale + bias
                tl.store(y_ptr + index, y_val)


@triton.jit
def silu_fp32(x_ptr, y_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    # elementwise: y = x * sigmoid(x)
    for i in range(0, total_elems, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < total_elems
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(y_ptr + idx, y, mask=mask)


@triton.jit
def add_residual_fp32(x_ptr, out_ptr, y_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    # elementwise addition: y = out + x
    for i in range(0, total_elems, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < total_elems
        out = tl.load(out_ptr + idx, mask=mask, other=0.0)
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = out + x
        tl.store(y_ptr + idx, y, mask=mask)


class Model(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # Ensure float32 and contiguous
        x = x.to(torch.float32).contiguous()
        device = x.device

        # 1) First conv: NCHW, stride=1, padding=1, no bias
        B, C_in, H_in, W_in = x.shape
        C_out1 = conv1_weight.shape[0]
        K = 3
        y1 = torch.empty((B, C_out1, H_in, W_in), device=device, dtype=torch.float32)

        grid1 = (B * C_out1 * H_in * W_in,)
        conv3x3_nchw_fp32[grid1](
            x, conv1_weight.to(torch.float32).contiguous(), y1,
            B, C_in, H_in, W_in, C_out1, K,
            BLOCK_IN=32, num_warps=4
        )

        # 2) First GroupNorm (num_groups=32), affine
        groups1 = 32
        channels_per_group1 = 32
        y1n = torch.empty_like(y1)  # normalized output
        grid_gn1 = (B * groups1,)
        groupnorm_affine_fp32[grid_gn1](
            y1, y1n,
            B, C_out1, H_in, W_in,
            channels_per_group1, eps,
            norm1_weight.to(torch.float32).contiguous(), norm1_bias.to(torch.float32).contiguous(),
            BLOCK_HW=1024, num_warps=4
        )

        # 3) SiLU 1
        y1n_flat = y1n.flatten()
        y1n_out = torch.empty_like(y1n_flat, device=device, dtype=torch.float32)
        total1 = y1n_flat.numel()
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_fp32[grid_silu1](y1n_flat, y1n_out, total1, 1024, num_warps=4)
        y1silu = y1n_out.view_as(y1n)

        # 4) Second conv: NCHW, stride=1, padding=1, no bias
        C_in2 = y1silu.shape[1]
        C_out2 = conv2_weight.shape[0]
        y2 = torch.empty((B, C_out2, H_in, W_in), device=device, dtype=torch.float32)
        grid2 = (B * C_out2 * H_in * W_in,)
        conv3x3_nchw_fp32[grid2](
            y1silu, conv2_weight.to(torch.float32).contiguous(), y2,
            B, C_in2, H_in, W_in, C_out2, K,
            BLOCK_IN=32, num_warps=4
        )

        # 5) Second GroupNorm (num_groups=32), affine
        groups2 = 32
        channels_per_group2 = 32
        y2n = torch.empty_like(y2)
        grid_gn2 = (B * groups2,)
        groupnorm_affine_fp32[grid_gn2](
            y2, y2n,
            B, C_out2, H_in, W_in,
            channels_per_group2, eps,
            norm2_weight.to(torch.float32).contiguous(), norm2_bias.to(torch.float32).contiguous(),
            BLOCK_HW=1024, num_warps=4
        )

        # 6) SiLU 2
        y2n_flat = y2n.flatten()
        y2n_out = torch.empty_like(y2n_flat, device=device, dtype=torch.float32)
        total2 = y2n_flat.numel()
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_fp32[grid_silu2](y2n_flat, y2n_out, total2, 1024, num_warps=4)
        y2silu = y2n_out.view_as(y2n)

        # 7) Residual addition: add original input x to the final output (elementwise in Triton)
        total_final = y2silu.numel()
        out_final = torch.empty_like(y2silu, device=device, dtype=torch.float32)
        grid_add = (triton.cdiv(total_final, 1024),)
        add_residual_fp32[grid_add](y2silu, x.contiguous().view(-1), out_final.view(-1), total_final, 1024, num_warps=4)

        return out_final


# Optional: also expose ModelNew as requested in the original prompt
class ModelNew(Model):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        return super().forward(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
