import torch
import triton
import triton.language as tl


# Conv3x3 NCHW, stride=1, padding=1, no bias, float32 compute
@triton.jit
def conv3x3_nchw_fp32(x_ptr, w_ptr, y_ptr,
                       B, C_in, H, W, C_out,
                       BLOCK_CIN: tl.constexpr):
    pid = tl.program_id(0)  # one program per output element
    # map pid -> (n, c_out, h_out, w_out)
    N = B
    H_out = H
    W_out = W

    # We flatten (n, c_out, h_out, w_out) space
    # total programs = N * C_out * H_out * W_out
    tmp = pid
    w_out = tmp % W_out
    tmp = tmp // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    c_out = tmp % C_out
    n = tmp // C_out

    # Accumulator
    acc = tl.zeros([1], dtype=tl.float32)

    # Loop over input channels in chunks
    for c_start in range(0, C_in, BLOCK_CIN):
        c_offsets = c_start + tl.arange(0, BLOCK_CIN)
        mask_c = c_offsets < C_in

        # Accumulate over 3x3 window with padding
        for kh in range(3):
            for kw in range(3):
                h_in = h_out + kh - 1  # kh in [0..2], so h_in in [h_out-1..h_out+1]
                w_in = w_out + kw - 1  # same for w

                # Valid if h_in in [0..H-1], w_in in [0..W-1]
                valid_hw = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

                # For each input channel c_in
                for ci in range(BLOCK_CIN):
                    c_in_val = c_start + ci
                    if mask_c[ci]:
                        # Load x[n, c_in_val, h_in, w_in]
                        x_off = (((n * C_in + c_in_val) * H + h_in) * W + w_in)
                        x_val = tl.load(x_ptr + x_off, mask=valid_hw, other=0.0)

                        # Load weight w[c_out, c_in_val, kh, kw]
                        w_off = (((c_out * C_in + c_in_val) * 3 + kh) * 3 + kw)
                        w_val = tl.load(w_ptr + w_off, mask=True, other=0.0)

                        acc += x_val * w_val

    # Store output y[n, c_out, h_out, w_out]
    y_off = (((n * C_out + c_out) * H_out + h_out) * W_out + w_out)
    tl.store(y_ptr + y_off, acc)


# GroupNorm with affine, per (n, group), NCHW, float32
@triton.jit
def groupnorm_affine_nchw_fp32(x_ptr, y_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr,
                               B, C, H, W, num_groups, eps,
                               BLOCK_HW: tl.constexpr):
    n = tl.program_id(0)
    group = tl.program_id(1)

    group_size = C // num_groups
    c_start = group * group_size
    C_group = group_size
    H_out = H
    W_out = W

    # Compute sum and sum of squares over group channels and all spatial positions
    sum_val = 0.0
    sum_sq = 0.0

    # Pass 1: compute mean/var
    for c in range(C_group):
        c_chan = c_start + c
        for h in range(0, H_out, BLOCK_HW):
            for w in range(0, W_out, BLOCK_HW):
                hw_offsets = h * W_out + (w + tl.arange(0, BLOCK_HW))
                mask_hw = (hw_offsets < (H_out * W_out))
                # linearize index: ((n*C + c_chan)*H + h)*W + w
                base = ((n * C + c_chan) * H_out) * W_out
                x_vals = tl.load(x_ptr + base + hw_offsets, mask=mask_hw, other=0.0)
                sum_val += tl.sum(x_vals, axis=0)
                sum_sq += tl.sum(x_vals * x_vals, axis=0)

    total = H_out * W_out * C_group
    mean = sum_val / total
    var = sum_sq / total - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + (n * num_groups + group), mean)
    tl.store(invstd_ptr + (n * num_groups + group), invstd)

    # Pass 2: normalize and apply affine
    for c in range(C_group):
        c_chan = c_start + c
        for h in range(0, H_out, BLOCK_HW):
            for w in range(0, W_out, BLOCK_HW):
                hw_offsets = h * W_out + (w + tl.arange(0, BLOCK_HW))
                mask_hw = (hw_offsets < (H_out * W_out))
                base = ((n * C + c_chan) * H_out) * W_out
                x_vals = tl.load(x_ptr + base + hw_offsets, mask=mask_hw, other=0.0)
                normed = (x_vals - mean) * invstd
                scale = tl.load(weight_ptr + c_chan)
                bias = tl.load(bias_ptr + c_chan)
                y_vals = normed * scale + bias
                tl.store(y_ptr + base + hw_offsets, y_vals, mask=mask_hw)


# SiLU elementwise Triton kernel
@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


# Residual addition elementwise Triton kernel
@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a + b
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5,
                 conv_block_cin: int = 32, groupnorm_block_hw: int = 256):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        # Tunables for Triton loops (must be constexpr at JIT time)
        self.conv_block_cin = conv_block_cin    # e.g., 32
        self.groupnorm_block_hw = groupnorm_block_hw  # e.g., 256

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Ensure dtype float32, contiguous NCHW
        B, C, H, W = x.shape
        device = x.device

        # Cast to float32 for compute; original code uses float32 params anyway
        x_fp32 = x.contiguous().to(torch.float32)
        conv1_w_fp32 = conv1_weight.contiguous().to(torch.float32)
        conv2_w_fp32 = conv2_weight.contiguous().to(torch.float32)
        norm1_w_fp32 = norm1_weight.contiguous().to(torch.float32)
        norm1_b_fp32 = norm1_bias.contiguous().to(torch.float32)
        norm2_w_fp32 = norm2_weight.contiguous().to(torch.float32)
        norm2_b_fp32 = norm2_bias.contiguous().to(torch.float32)

        C_in = C  # input channels
        C_out = C  # output channels after conv (same as input channels)

        # Output buffers
        out1 = torch.empty((B, C_out, H, W), device=device, dtype=torch.float32)
        out1_silu = torch.empty((B, C_out, H, W), device=device, dtype=torch.float32)

        # Launch conv1
        total_out1 = B * C_out * H * W
        grid_conv = (total_out1,)
        conv3x3_nchw_fp32[grid_conv](
            x_fp32, conv1_w_fp32, out1,
            B, C_in, H, W, C_out,
            BLOCK_CIN=self.conv_block_cin,
            num_warps=4, num_stages=2
        )

        # Launch GroupNorm1 + affine
        out1_norm = torch.empty((B, C_out, H, W), device=device, dtype=torch.float32)
        mean1 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        invstd1 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)

        grid_gn1 = (B, self.num_groups)
        groupnorm_affine_nchw_fp32[grid_gn1](
            out1, out1_norm, mean1, invstd1, norm1_w_fp32, norm1_b_fp32,
            B, C_out, H, W, self.num_groups, self.eps,
            BLOCK_HW=self.groupnorm_block_hw,
            num_warps=4, num_stages=2
        )

        # SiLU1
        out1_silu.copy_(out1_norm)  # temporary holder, we will compute via kernel
        total_silu = B * C_out * H * W
        silu_out1 = torch.empty_like(out1_norm)
        grid_silu = (triton.cdiv(total_silu, 1024),)
        silu_kernel[grid_silu](
            out1_norm, silu_out1, total_silu, BLOCK=1024,
            num_warps=4, num_stages=2
        )
        out1_silu = silu_out1

        # Launch conv2
        out2 = torch.empty((B, C_out, H, W), device=device, dtype=torch.float32)
        total_conv2 = B * C_out * H * W
        grid_conv2 = (total_conv2,)
        conv3x3_nchw_fp32[grid_conv2](
            out1_silu, conv2_w_fp32, out2,
            B, C_out, H, W, C_out,
            BLOCK_CIN=self.conv_block_cin,
            num_warps=4, num_stages=2
        )

        # Launch GroupNorm2 + affine
        out2_norm = torch.empty((B, C_out, H, W), device=device, dtype=torch.float32)
        mean2 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        invstd2 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)

        grid_gn2 = (B, self.num_groups)
        groupnorm_affine_nchw_fp32[grid_gn2](
            out2, out2_norm, mean2, invstd2, norm2_w_fp32, norm2_b_fp32,
            B, C_out, H, W, self.num_groups, self.eps,
            BLOCK_HW=self.groupnorm_block_hw,
            num_warps=4, num_stages=2
        )

        # SiLU2
        out2_silu = torch.empty_like(out2_norm)
        total_silu2 = B * C_out * H * W
        grid_silu2 = (triton.cdiv(total_silu2, 1024),)
        silu_kernel[grid_silu2](
            out2_norm, out2_silu, total_silu2, BLOCK=1024,
            num_warps=4, num_stages=2
        )

        # Final residual add: out = out2_silu + x_fp32
        final_out = torch.empty((B, C_out, H, W), device=device, dtype=torch.float32)
        grid_add = (triton.cdiv(total_silu2, 1024),)
        add_residual_kernel[grid_add](
            out2_silu, x_fp32, final_out, total_silu2, BLOCK=1024,
            num_warps=4, num_stages=2
        )

        return final_out


# If you want to quickly test locally:
# model = ModelNew(num_groups=32, eps=1e-5)
# x = torch.randn(1, 384, 128, 128, device='cuda', dtype=torch.float32)
# conv1 = torch.randn(384, 384, 3, 3, device='cuda', dtype=torch.float32)
# norm1_w = torch.randn(384, device='cuda', dtype=torch.float32)
# norm1_b = torch.randn(384, device='cuda', dtype=torch.float32)
# conv2 = torch.randn(384, 384, 3, 3, device='cuda', dtype=torch.float32)
# norm2_w = torch.randn(384, device='cuda', dtype=torch.float32)
# norm2_b = torch.randn(384, device='cuda', dtype=torch.float32)
# y = model(x, conv1, norm1_w, norm1_b, conv2, norm2_w, norm2_b)
# print(y.shape)


def run(*args):
    return ModelNew()(*args)
