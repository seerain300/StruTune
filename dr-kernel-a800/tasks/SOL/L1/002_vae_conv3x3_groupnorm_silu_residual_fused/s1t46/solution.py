import torch
import triton
import triton.language as tl


# Conv3x3 NCHW, stride=1, padding=1, no bias
# Each program computes one output element y[n, c_out, h_out, w_out]
@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,  # float32 input (B, C_in, H, W)
    w_ptr,  # float32 weight (C_out, C_in, 3, 3)
    y_ptr,  # float32 output (B, C_out, H, W)
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,  # strides of x
    w_s0, w_s1, w_s2, w_s3,  # strides of w
    y_s0, y_s1, y_s2, y_s3   # strides of y
):
    pid = tl.program_id(0)
    total = C_out * H * W
    if pid >= B * total:
        return
    n = pid // total
    tmp = pid % total
    c_out = tmp // (H * W)
    tmp2 = tmp % (H * W)
    h_out = tmp2 // W
    w_out = tmp2 % W

    acc = 0.0

    # Loop over input channels in chunks
    for cin_start in range(0, C_in, 64):  # BLOCK_IN = 64 (tunable)
        cin_idx = cin_start + tl.arange(0, 64)
        mask_cin = cin_idx < C_in

        # 3x3 window with padding=1, stride=1
        for kh in range(3):
            h_in = h_out - kh
            in_h_ok = (h_in >= 0) & (h_in < H)
            for kw in range(3):
                w_in = w_out - kw
                in_w_ok = (w_in >= 0) & (w_in < W)
                valid = in_h_ok & in_w_ok

                # Load x values for all cin in chunk: shape [64, 1]
                addr_x = n * x_s0 + cin_idx[:, None] * x_s1 + h_in * x_s2 + w_in * x_s3
                mask_x = mask_cin[:, None] & valid
                x_vals = tl.load(x_ptr + addr_x, mask=mask_x, other=0.0)  # [64,1], float32

                # Load corresponding weights for this (c_out, cin chunk, kh, kw): shape [64]
                addr_w = c_out * w_s0 + cin_idx * w_s1 + kh * w_s2 + kw * w_s3
                mask_w = mask_cin
                w_vals = tl.load(w_ptr + addr_w, mask=mask_w, other=0.0)  # [64], float32

                # Accumulate contribution
                acc += tl.sum(w_vals[:, None] * x_vals, axis=0)

    # Store result
    addr_y = n * y_s0 + c_out * y_s1 + h_out * y_s2 + w_out * y_s3
    tl.store(y_ptr + addr_y, acc)


# GroupNorm with affine per (n, group): out = (x - mean) * scale + bias
# Two-pass: compute mean/var, then normalize + affine
@triton.jit
def group_norm_affine_kernel(
    x_ptr,  # float32 input (B, C, H, W)
    out_ptr,  # float32 output (B, C, H, W)
    scale_ptr,  # float32 per-channel scale (C,)
    bias_ptr,   # float32 per-channel bias (C,)
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,  # strides of x: N, C, H, W
    out_s0, out_s1, out_s2, out_s3  # strides of out
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    group_size = C // num_groups
    c_start = g * group_size

    # First pass: compute mean and variance over channels [c_start:c_start+group_size] and all H*W
    sum_val = 0.0
    sum_sq = 0.0
    for c in range(c_start, c_start + group_size):
        for h in range(0, H):
            for w in range(0, W):
                addr = n * x_s0 + c * x_s1 + h * x_s2 + w * x_s3
                x_val = tl.load(x_ptr + addr)
                sum_val += x_val
                sum_sq += x_val * x_val
    M = group_size * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, store
    for c in range(c_start, c_start + group_size):
        scale = tl.load(scale_ptr + c)
        bias = tl.load(bias_ptr + c)
        for h in range(0, H):
            for w in range(0, W):
                addr_x = n * x_s0 + c * x_s1 + h * x_s2 + w * x_s3
                x_val = tl.load(x_ptr + addr_x)
                y_val = (x_val - mean) * inv_std
                y_val = y_val * scale + bias
                addr_out = n * out_s0 + c * out_s1 + h * out_s2 + w * out_s3
                tl.store(out_ptr + addr_out, y_val)


# SiLU elementwise: y = x * sigmoid(x)
@triton.jit
def silu_kernel(x_ptr, y_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Elementwise residual addition: out = out + x
@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Cast to float32 and contiguous for Triton
        device = x.device
        dtype = torch.float32

        B, C, H, W = x.shape

        # First conv: (B, C, H, W) -> (B, C, H, W) with padding=1, stride=1, no bias
        x_in1 = x.to(dtype).contiguous()
        w1 = conv1_weight.to(dtype).contiguous()  # (C, C, 3, 3)
        out1 = torch.empty((B, C, H, W), dtype=dtype, device=device)

        grid_conv1 = (B * C * H * W,)
        conv3x3_nchw_fp32[grid_conv1](
            x_in1, w1, out1,
            B, C, H, W, C,
            x_in1.stride(0), x_in1.stride(1), x_in1.stride(2), x_in1.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            BLOCK_IN=64, H_out=H, W_out=W
        )

        # First GroupNorm (num_groups=32, affine)
        num_groups = 32
        assert C % num_groups == 0, "C must be divisible by num_groups"
        group_size = C // num_groups
        y1 = torch.empty((B, C, H, W), dtype=dtype, device=device)
        grid_gn1 = (B, num_groups)
        group_norm_affine_kernel[grid_gn1](
            out1, y1, norm1_weight.to(dtype).contiguous(), norm1_bias.to(dtype).contiguous(),
            B, C, H, W, num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3)
        )

        # SiLU
        total1 = y1.numel()
        y1_silu = torch.empty_like(y1, device=device, dtype=dtype)
        BLOCK_SILU = 1024
        grid_silu1 = (triton.cdiv(total1, BLOCK_SILU),)
        silu_kernel[grid_silu1](y1, y1_silu, total1, BLOCK=BLOCK_SILU)

        # Second conv: (B, C, H, W) -> (B, C, H, W) with padding=1, stride=1, no bias
        w2 = conv2_weight.to(dtype).contiguous()  # (C, C, 3, 3)
        out2 = torch.empty((B, C, H, W), dtype=dtype, device=device)
        grid_conv2 = (B * C * H * W,)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, w2, out2,
            B, C, H, W, C,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            BLOCK_IN=64, H_out=H, W_out=W
        )

        # Second GroupNorm
        y2 = torch.empty((B, C, H, W), dtype=dtype, device=device)
        grid_gn2 = (B, num_groups)
        group_norm_affine_kernel[grid_gn2](
            out2, y2, norm2_weight.to(dtype).contiguous(), norm2_bias.to(dtype).contiguous(),
            B, C, H, W, num_groups, self.eps,
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3)
        )

        # SiLU
        total2 = y2.numel()
        y2_silu = torch.empty_like(y2, device=device, dtype=dtype)
        grid_silu2 = (triton.cdiv(total2, BLOCK_SILU),)
        silu_kernel[grid_silu2](y2, y2_silu, total2, BLOCK=BLOCK_SILU)

        # Residual addition: add original x to final output (shapes match: (B, C, H, W))
        x0 = x.to(dtype).contiguous()
        total_final = x0.numel()
        final_out = torch.empty_like(x0, device=device, dtype=dtype)
        grid_add = (triton.cdiv(total_final, BLOCK_SILU),)
        add_residual_kernel[grid_add](y2_silu, x0, final_out, total_final, BLOCK=BLOCK_SILU)

        return final_out


# Optional quick check
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H, W = 2, 64, 32, 32
    conv1_weight = torch.randn(C, C, 3, 3, device=device, dtype=torch.float32)
    norm1_weight = torch.randn(C, device=device, dtype=torch.float32)
    norm1_bias = torch.randn(C, device=device, dtype=torch.float32)
    conv2_weight = torch.randn(C, C, 3, 3, device=device, dtype=torch.float32)
    norm2_weight = torch.randn(C, device=device, dtype=torch.float32)
    norm2_bias = torch.randn(C, device=device, dtype=torch.float32)
    x = torch.randn(B, C, H, W, device=device, dtype=torch.float32)

    model = ModelNew(eps=1e-5).to(device)
    with torch.no_grad():
        y = model(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias)
    print(y.shape)  # (B, C, H, W)


def run(*args):
    return ModelNew()(*args)
