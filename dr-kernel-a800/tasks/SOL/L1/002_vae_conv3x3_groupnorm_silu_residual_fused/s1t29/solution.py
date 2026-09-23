import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,           # *const float, input tensor (B, C_in, H, W)
    w_ptr,           # *const float, weight tensor (C_out, C_in, 3, 3)
    y_ptr,           # *float, output tensor (B, C_out, H_out, W_out)
    B, C_in, H, W,   # int32 sizes
    C_out,           # int32
    H_out, W_out,    # int32 output spatial sizes
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,   # int32 strides for x
    w_stride_cout, w_stride_cin, w_stride_kh, w_stride_kw,  # int32 strides for w
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,   # int32 strides for y
    BLOCK_IN: tl.constexpr, BLOCK_HW: tl.constexpr
):
    # program ids: pid_n, pid_cout, pid_t over tiles of H_out * W_out
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_t = tl.program_id(2)

    # vector of HW indices for this tile
    start = pid_t * BLOCK_HW
    offs = start + tl.arange(0, BLOCK_HW)
    mask_hw = offs < (H_out * W_out)

    # map offs to (h_out, w_out)
    h_out_vec = offs // W_out
    w_out_vec = offs % W_out

    # accumulator for output
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # loop over input channels in chunks
    for cin0 in range(0, C_in, BLOCK_IN):
        cin_range = cin0 + tl.arange(0, BLOCK_IN)
        mask_cin = cin_range < C_in

        # loop over 3x3 kernel window, with padding
        for kh in range(1, 4):  # kh in [1, 2, 3]
            for kw in range(1, 4):  # kw in [1, 2, 3]
                # compute corresponding input indices with padding
                h_in_vec = h_out_vec + (1 - kh)  # padding=1
                w_in_vec = w_out_vec + (1 - kw)
                mask_in = (h_in_vec >= 0) & (h_in_vec < H) & (w_in_vec >= 0) & (w_in_vec < W) & mask_hw

                # load x[n, cin, h_in, w_in] for all cin in chunk
                # broadcast x_stride_* for vectorized indexing
                x_idx = (
                    pid_n * x_stride_n
                    + cin_range[:, None] * x_stride_c
                    + h_in_vec[None, :] * x_stride_h
                    + w_in_vec[None, :] * x_stride_w
                )
                # mask for 2D: mask_cin[:, None] & mask_in[None, :]
                mask_load = (mask_cin[:, None]) & (mask_in[None, :])
                x_vals = tl.load(x_ptr + x_idx, mask=mask_load, other=0.0)  # [BLOCK_IN, BLOCK_HW]

                # load w[c_out, cin, kh, kw] as vector over cin
                w_idx = (
                    pid_cout * w_stride_cout
                    + cin_range * w_stride_cin
                    + (kh - 1) * w_stride_kh
                    + (kw - 1) * w_stride_kw
                )
                w_vals = tl.load(w_ptr + w_idx, mask=mask_cin, other=0.0)  # [BLOCK_IN]

                # outer product accumulate
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)  # [BLOCK_HW]

    # store results to y[n, c_out, h_out, w_out]
    y_idx = (
        pid_n * y_stride_n
        + pid_cout * y_stride_c
        + h_out_vec * y_stride_h
        + w_out_vec * y_stride_w
    )
    tl.store(y_ptr + y_idx, acc, mask=mask_hw)


@triton.jit
def groupnorm_affine_kernel(
    x_ptr,           # *const float, input tensor (B, C, H_out, W_out)
    gamma_ptr,       # *const float, per-channel scale (C,)
    beta_ptr,        # *const float, per-channel bias (C,)
    y_ptr,           # *float, output tensor (B, C, H_out, W_out)
    B, C, H_out, W_out, G, eps,  # int32 and float32
    BLOCK_HW: tl.constexpr
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size_c = (C + G - 1) // G  # number of channels per group
    c_start = g * group_size_c

    # pass 1: compute sum and sumsq over channels in group and all HW
    sum_val = 0.0
    sum_sq = 0.0

    for ch in range(0, group_size_c):
        c = c_start + ch
        # loop HW in chunks
        for hw0 in range(0, H_out * W_out, BLOCK_HW):
            offs = hw0 + tl.arange(0, BLOCK_HW)
            mask = offs < (H_out * W_out)
            h = offs // W_out
            w = offs % W_out
            x_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h * W_out + w
            x_vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            sum_val += tl.sum(x_vals)
            sum_sq += tl.sum(x_vals * x_vals)

    Nch = group_size_c  # channels in this group
    Nsp = H_out * W_out
    mean = sum_val / (Nch * Nsp)
    var = sum_sq / (Nch * Nsp) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # pass 2: normalize and affine, write back
    for ch in range(0, group_size_c):
        c = c_start + ch
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for hw0 in range(0, H_out * W_out, BLOCK_HW):
            offs = hw0 + tl.arange(0, BLOCK_HW)
            mask = offs < (H_out * W_out)
            h = offs // W_out
            w = offs % W_out
            x_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h * W_out + w
            x_vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            y_vals = (x_vals - mean) * inv_std
            y_vals = y_vals * gamma + beta
            y_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h * W_out + w
            tl.store(y_ptr + y_idx, y_vals, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    # elementwise y = x * sigmoid(x)
    for i in range(0, n_elements, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-x))
        y = x * s
        tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # elementwise addition: out = a + b
    for i in range(0, n_elements, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        out = a + b
        tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def pad_to_out_fp32(x_ptr, y_ptr, B, C, H, W, H_out, W_out, BLOCK_HW: tl.constexpr):
    # pad x (B, C, H, W) to y (B, C, H_out, W_out) with zeros around borders
    for n in range(0, B):
        for c in range(0, C):
            for h in range(0, H_out):
                for w in range(0, W_out):
                    if (h >= 1 and h <= H) and (w >= 1 and w <= W):
                        x_idx = n * (C * H * W) + c * (H * W) + (h - 1) * W + (w - 1)
                        y_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h * W_out + w
                        val = tl.load(x_ptr + x_idx, mask=True, other=0.0)
                        tl.store(y_ptr + y_idx, val)
                    else:
                        y_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h * W_out + w
                        tl.store(y_ptr + y_idx, 0.0)


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                 conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        super().__init__()
        # store weights and params
        self.conv1_weight = conv1_weight
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.conv2_weight = conv2_weight
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # Ensure float32 and contiguous
        x = x.to(torch.float32).contiguous()
        device = x.device
        B, C, H, W = x.shape

        # Compute conv1
        C_out = self.conv1_weight.shape[0]  # (C_out, C_in, 3, 3)
        H_out1 = H + 2  # padding=1, stride=1
        W_out1 = W + 2
        conv1_out = torch.empty((B, C_out, H_out1, W_out1), device=device, dtype=torch.float32)

        conv1_w = self.conv1_weight.contiguous().to(torch.float32)
        conv1_w = conv1_w.view(C_out, -1, 3, 3)  # ensure (C_out, C_in, 3, 3)

        grid_conv1 = (B, C_out, triton.cdiv(H_out1 * W_out1, 128))
        conv3x3_nchw_fp32[grid_conv1](
            x, conv1_w, conv1_out,
            B, C, H, W, C_out, H_out1, W_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), conv1_w.stride(3),
            conv1_out.stride(0), conv1_out.stride(1), conv1_out.stride(2), conv1_out.stride(3),
            BLOCK_IN=32, BLOCK_HW=128,
            num_warps=4, num_stages=2
        )

        # GroupNorm 1 with affine
        gn1_out = torch.empty_like(conv1_out)
        # num_groups is not passed; infer or use 32? The original uses num_groups=32.
        # To match original behavior, we assume num_groups=32. If C isn't divisible by 32, fall back to C.
        num_groups1 = min(32, C if C >= 32 else C)
        grid_gn1 = (B * num_groups1,)
        groupnorm_affine_kernel[grid_gn1](
            conv1_out, self.norm1_weight, self.norm1_bias, gn1_out,
            B, conv1_out.shape[1], H_out1, W_out1, num_groups1, self.eps,
            BLOCK_HW=1024,
            num_warps=4, num_stages=2
        )

        # SiLU 1
        silu1_out = torch.empty_like(gn1_out)
        n_elements1 = gn1_out.numel()
        grid_silu1 = (triton.cdiv(n_elements1, 1024),)
        silu_kernel[grid_silu1](gn1_out, silu1_out, n_elements1, 1024,
                                num_warps=4, num_stages=2)

        # conv2: same as conv1, preserves shape (B, C_out, H_out1, W_out1)
        conv2_w = self.conv2_weight.contiguous().to(torch.float32).view(C_out, -1, 3, 3)
        conv2_out = torch.empty((B, C_out, H_out1, W_out1), device=device, dtype=torch.float32)

        grid_conv2 = (B, C_out, triton.cdiv(H_out1 * W_out1, 128))
        conv3x3_nchw_fp32[grid_conv2](
            silu1_out, conv2_w, conv2_out,
            B, C_out, H_out1, W_out1, C_out, H_out1, W_out1,
            silu1_out.stride(0), silu1_out.stride(1), silu1_out.stride(2), silu1_out.stride(3),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), conv2_w.stride(3),
            conv2_out.stride(0), conv2_out.stride(1), conv2_out.stride(2), conv2_out.stride(3),
            BLOCK_IN=32, BLOCK_HW=128,
            num_warps=4, num_stages=2
        )

        # GroupNorm 2 with affine
        num_groups2 = min(32, conv2_out.shape[1] if conv2_out.shape[1] >= 32 else conv2_out.shape[1])
        gn2_out = torch.empty_like(conv2_out)
        grid_gn2 = (B * num_groups2,)
        groupnorm_affine_kernel[grid_gn2](
            conv2_out, self.norm2_weight, self.norm2_bias, gn2_out,
            B, conv2_out.shape[1], H_out1, W_out1, num_groups2, self.eps,
            BLOCK_HW=1024,
            num_warps=4, num_stages=2
        )

        # SiLU 2
        silu2_out = torch.empty_like(gn2_out)
        n_elements2 = gn2_out.numel()
        grid_silu2 = (triton.cdiv(n_elements2, 1024),)
        silu_kernel[grid_silu2](gn2_out, silu2_out, n_elements2, 1024,
                                num_warps=4, num_stages=2)

        # Residual addition: add original input x to final output
        # Create residual padded to (B, C, H_out1, W_out1) as fp32 zeros except center region
        residual = torch.empty((B, C, H_out1, W_out1), device=device, dtype=torch.float32)
        # We need residual to match original input values in center. Use pad kernel to fill center with x.
        # First, pad x to residual with zeros (out already zeros), then write center values via pad kernel.
        # However, pad kernel is costly; instead, build residual = x cast to fp32 and padding via Triton.
        # Implement pad_to_out_fp32: write x into residual at (h-1, w-1)
        grid_pad = (B, C, H, W)
        pad_to_out_fp32[grid_pad](
            x, residual,
            B, C, H, W, H_out1, W_out1,
            BLOCK_HW=1024,
            num_warps=4, num_stages=2
        )

        total_final = silu2_out.numel()
        final_out = torch.empty_like(silu2_out)
        grid_add = (triton.cdiv(total_final, 1024),)
        add_kernel[grid_add](silu2_out, residual, final_out, total_final, 1024,
                             num_warps=4, num_stages=2)

        return final_out


# Example usage:
# model = ModelNew(conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)
# x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
# y = model(x)


def run(*args):
    return ModelNew()(*args)
