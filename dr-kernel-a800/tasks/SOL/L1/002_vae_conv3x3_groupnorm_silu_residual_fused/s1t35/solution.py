import torch
import triton
import triton.language as tl


# -------------------------------
# Triton Kernels
# -------------------------------

@triton.jit
def conv3x3_nchw_fp32(x_ptr, w_ptr, out_ptr,
                      B, C_in, C_out, H_in, W_in, H_out, W_out,
                      x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                      w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
                      BLOCK_IN: tl.constexpr):
    """
    NCHW conv3x3 stride=1, padding=1, no bias.
    Each program computes one output element y[n, c_out, h_out, w_out].
    """
    pid = tl.program_id(axis=0)
    HW_out = H_out * W_out
    n = pid // (C_out * HW_out)
    rem = pid % (C_out * HW_out)
    c_out = rem // HW_out
    rem2 = rem % HW_out
    h_out = rem2 // W_out
    w_out = rem2 % rem2  # redundant, but keeps symmetry

    # Accumulator scalar
    acc = 0.0

    # Loop over input channels in chunks
    for c_start in range(0, C_in, BLOCK_IN):
        # Partial accumulator for this chunk (scalar)
        partial = 0.0
        # Iterate over input channels within the chunk
        for ci in range(BLOCK_IN):
            c = c_start + ci
            if c >= C_in:
                break
            # For each 3x3 neighborhood
            for kh in range(3):
                for kw in range(3):
                    h_in = h_out + kh - 1
                    w_in = w_out + kw - 1
                    valid_spatial = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                    # Load x[n, c, h_in, w_in] as scalar with mask
                    x_offset = n * x_stride_n + c * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                    x_val = tl.load(x_ptr + x_offset, mask=valid_spatial, other=0.0)
                    # Load w[c_out, c, kh, kw] as scalar
                    w_offset = c_out * w_stride_co + c * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                    w_val = tl.load(w_ptr + w_offset)
                    partial += x_val * w_val
        acc += partial

    # Store result to out[n, c_out, h_out, w_out]
    out_offset = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out
    tl.store(out_ptr + out_offset, acc)


@triton.jit
def groupnorm_affine_fp32(inp_ptr, out_ptr, weight_ptr, bias_ptr,
                          B, C, H, W, num_groups, eps, BLOCK_HW: tl.constexpr):
    """
    GroupNorm with affine per channel, NCHW layout.
    Two-pass per (n, group):
      pass 1: compute mean and var over group's channels and all spatial positions
      pass 2: normalize and apply affine
    """
    channels_per_group = C // num_groups
    n = tl.program_id(axis=0)  # one program per batch
    g = tl.program_id(axis=1)  # one program per group
    c_start = g * channels_per_group
    c_end = (g + 1) * channels_per_group

    total_hw = H * W
    sum_total = 0.0
    sumsq_total = 0.0

    # Pass 1: compute sum and sum of squares
    for c in range(c_start, c_end):
        base = inp_ptr + n * (C * H * W) + c * (H * W)
        for hw_start in range(0, total_hw, BLOCK_HW):
            hw_range = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = hw_range < total_hw
            x = tl.load(base + hw_range, mask=mask_hw, other=0.0)
            sum_total += tl.sum(x, axis=0)
            sumsq_total += tl.sum(x * x, axis=0)

    count = (c_end - c_start) * total_hw
    mean = sum_total / count
    var = sumsq_total / count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine
    for c in range(c_start, c_end):
        gamma = tl.load(weight_ptr + c)  # scale
        beta = tl.load(bias_ptr + c)     # bias
        base = inp_ptr + n * (C * H * W) + c * (H * W)
        out_base = out_ptr + n * (C * H * W) + c * (H * W)
        for hw_start in range(0, total_hw, BLOCK_HW):
            hw_range = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = hw_range < total_hw
            x = tl.load(base + hw_range, mask=mask_hw, other=0.0)
            y = (x - mean) * inv_std
            y = y * gamma + beta
            tl.store(out_base + hw_range, y, mask=mask_hw)


@triton.jit
def silu_fp32(x_ptr, out_ptr, total_elems, BLOCK: tl.constexpr):
    """
    Elementwise SiLU: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    """
    for start in range(0, total_elems, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < total_elems
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(out_ptr + idx, y, mask=mask)


@triton.jit
def add_residual_fp32(a_ptr, b_ptr, out_ptr, total_elems, BLOCK: tl.constexpr):
    """
    Elementwise addition: out = a + b
    """
    for start in range(0, total_elems, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < total_elems
        a = tl.load(a_ptr + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0)
        y = a + b
        tl.store(out_ptr + idx, y, mask=mask)


# -------------------------------
# ModelNew: Triton-only forward
# -------------------------------

class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        """
        Triton implementation of the original fused residual block:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Inputs and outputs are NCHW. Computation is done in Triton kernels.
        """
        assert x.is_cuda, "Triton kernels require CUDA tensors."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be CUDA tensors."

        B, C, H, W = x.shape
        C1 = conv1_weight.shape[1]  # conv1: (C1, C, 3, 3)
        C2 = conv2_weight.shape[1]  # conv2: (C2, C1, 3, 3)
        H_out1 = H
        W_out1 = W
        H_out2 = H_out1
        W_out2 = W_out1

        device = x.device
        dtype = torch.float32

        # Cast and make contiguous
        x0 = x.to(dtype).contiguous()
        conv1_w = conv1_weight.to(dtype).contiguous()  # shape (C1, C, 3, 3)
        conv2_w = conv2_weight.to(dtype).contiguous()  # shape (C2, C1, 3, 3)
        norm1_weight_t = norm1_weight.to(dtype).contiguous()
        norm1_bias_t = norm1_bias.to(dtype).contiguous()
        norm2_weight_t = norm2_weight.to(dtype).contiguous()
        norm2_bias_t = norm2_bias.to(dtype).contiguous()

        # First conv: y1 = conv(x, conv1_w) -> shape (B, C1, H, W)
        y1 = torch.empty((B, C1, H_out1, W_out1), device=device, dtype=dtype)
        grid1 = (B * C1 * H_out1 * W_out1,)
        conv3x3_nchw_fp32[grid1](
            x0, conv1_w, y1,
            B, C, C1, H, W, H_out1, W_out1,
            x0.stride(0), x0.stride(1), x0.stride(2), x0.stride(3),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), conv1_w.stride(3),
            BLOCK_IN=32
        )

        # First GroupNorm: y1_gn over num_groups (default 32)
        y1_gn = torch.empty_like(y1, device=device, dtype=dtype)
        grid_gn1 = (B, self.num_groups)
        groupnorm_affine_fp32[grid_gn1](
            y1, y1_gn, norm1_weight_t, norm1_bias_t,
            B, C1, H_out1, W_out1, self.num_groups, eps,
            BLOCK_HW=256
        )

        # First SiLU
        y1_silu = torch.empty_like(y1_gn, device=device, dtype=dtype)
        total_elems1 = y1_gn.numel()
        silu_fp32[(triton.cdiv(total_elems1, 1024),)](
            y1_gn, y1_silu, total_elems1, BLOCK=1024
        )

        # Second conv: y2 = conv(y1_silu, conv2_w) -> shape (B, C2, H, W)
        y2 = torch.empty((B, C2, H_out2, W_out2), device=device, dtype=dtype)
        grid2 = (B * C2 * H_out2 * W_out2,)
        conv3x3_nchw_fp32[grid2](
            y1_silu, conv2_w, y2,
            B, C1, C2, H_out1, W_out1, H_out2, W_out2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), conv2_w.stride(3),
            BLOCK_IN=32
        )

        # Second GroupNorm
        y2_gn = torch.empty_like(y2, device=device, dtype=dtype)
        grid_gn2 = (B, self.num_groups)
        groupnorm_affine_fp32[grid_gn2](
            y2, y2_gn, norm2_weight_t, norm2_bias_t,
            B, C2, H_out2, W_out2, self.num_groups, eps,
            BLOCK_HW=256
        )

        # Second SiLU
        y2_silu = torch.empty_like(y2_gn, device=device, dtype=dtype)
        total_elems2 = y2_gn.numel()
        silu_fp32[(triton.cdiv(total_elems2, 1024),)](
            y2_gn, y2_silu, total_elems2, BLOCK=1024
        )

        # Residual addition: add original input x to final output (cast to fp32 and flatten)
        total_elems_add = x0.numel()
        out_add = torch.empty_like(x0, device=device, dtype=dtype)
        add_residual_fp32[(triton.cdiv(total_elems_add, 1024),)](
            x0, y2_silu, out_add, total_elems_add, BLOCK=1024
        )

        return out_add


def run(*args):
    return ModelNew()(*args)
