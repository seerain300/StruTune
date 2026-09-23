import torch
import triton
import triton.language as tl

# Triton kernels
@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,         # *const float, input (B, C_in, H, W)
    w_ptr,         # *const float, weight (C_out, C_in, 3, 3)
    y_ptr,         # *float, output (B, C_out, H, W)
    B: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    n = tl.program_id(0)
    co = tl.program_id(1)
    ho = tl.program_id(2)
    wo = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks
    for c0 in range(0, C_in, BLOCK_IN):
        cin_offsets = c0 + tl.arange(0, BLOCK_IN)  # [BLOCK_IN]
        cin_mask = cin_offsets < C_in

        # Accumulate across 3x3 neighborhood with padding
        for kh in range(3):
            for kw in range(3):
                hi = ho + kh - 1
                wi = wo + kw - 1
                valid_h = (hi >= 0) & (hi < H)
                valid_w = (wi >= 0) & (wi < W)
                valid_hw = valid_h & valid_w

                for ci in tl.static_range(BLOCK_IN):
                    ci_idx = c0 + ci
                    if ci_idx >= C_in:
                        continue
                    # NCHW strides for contiguous tensors: x offset = n*(C_in*H*W) + ci_idx*(H*W) + hi*W + wi
                    x_offset = n * (C_in * H * W) + ci_idx * (H * W) + hi * W + wi
                    # Weight offset = co*(C_in*9) + ci_idx*9 + kh*3 + kw
                    w_offset = co * (C_in * 9) + ci_idx * 9 + kh * 3 + kw

                    x_val = tl.load(x_ptr + x_offset, mask=cin_mask[ci] & valid_hw, other=0.0)
                    w_val = tl.load(w_ptr + w_offset, mask=True, other=0.0)
                    acc += x_val * w_val

    # Store result y[n, co, ho, wo]
    y_offset = n * (C_out * H_out * W_out) + co * (H_out * W_out) + ho * W_out + wo
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def groupnorm_affine_fp32(
    x_ptr,           # *const float, input (B, C, H*W) contiguous
    weight_ptr,      # *const float, scale (C,)
    bias_ptr,        # *const float, bias (C,)
    y_ptr,           # *float, output (B, C, H*W) contiguous
    B: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # C // NUM_GROUPS
    EPS: tl.constexpr,
):
    # Each program handles one (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)

    # Compute mean and variance over this group across all HW elements
    sum_all = tl.zeros((), dtype=tl.float32)
    sumsq_all = tl.zeros((), dtype=tl.float32)

    # Pass 1: accumulate sum and sumsq
    for c in range(0, C, GROUP_SIZE):
        for ci in range(GROUP_SIZE):
            ch = c + ci
            if ch >= C:
                break
            base = (n * C + ch) * HW
            for off in range(0, HW, 256):
                idx = off + tl.arange(0, 256)
                mask = idx < HW
                x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
                sum_all += tl.sum(x, axis=0)
                sumsq_all += tl.sum(x * x, axis=0)

    m = sum_all / (C * HW)
    var = sumsq_all / (C * HW) - m * m
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Pass 2: normalize and apply affine, store
    for c in range(0, C, GROUP_SIZE):
        for ci in range(GROUP_SIZE):
            ch = c + ci
            if ch >= C:
                break
            scale = tl.load(weight_ptr + ch)
            bias = tl.load(bias_ptr + ch)
            base = (n * C + ch) * HW
            for off in range(0, HW, 256):
                idx = off + tl.arange(0, 256)
                mask = idx < HW
                x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
                y = (x - m) * rstd
                y = y * scale + bias
                tl.store(y_ptr + base + idx, y, mask=mask)


@triton.jit
def silu_fp32_elementwise(y_ptr, N, BLOCK: tl.constexpr):
    # Elementwise y = x * sigmoid(x) over length N
    for off in range(0, N, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < N
        x = tl.load(y_ptr + idx, mask=mask, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(y_ptr + idx, y, mask=mask)


@triton.jit
def add_residual_fp32(y_ptr, r_ptr, out_ptr, N, BLOCK: tl.constexpr):
    # out = y + r, elementwise
    for off in range(0, N, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < N
        y = tl.load(y_ptr + idx, mask=mask, other=0.0)
        r = tl.load(r_ptr + idx, mask=mask, other=0.0)
        out = y + r
        tl.store(out_ptr + idx, out, mask=mask)


# ModelNew: Triton-only forward (no PyTorch ops)
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
        """
        Triton implementation of:
            out = Conv3x3(x) -> GroupNorm(num_groups=32, affine) -> SiLU
                   Conv3x3(out) -> GroupNorm(num_groups=32, affine) -> SiLU
                   + x  (residual connection)

        All computation is performed by Triton kernels. No PyTorch ops in forward.
        """
        device = x.device

        # Ensure fp32 and contiguous
        x_f = x.contiguous().to(torch.float32)

        # First conv: y1 = conv3x3(x_f)
        B, C_in, H, W = x_f.shape
        C_out1 = conv1_weight.shape[0]  # output channels for conv1
        C_in1 = conv1_weight.shape[1]   # input channels for conv1 (should equal C_in)
        assert C_in1 == C_in, "conv1_weight second dim must match x channels"
        H_out1 = H
        W_out1 = W
        y1 = torch.empty((B, C_out1, H_out1, W_out1), device=device, dtype=torch.float32)

        grid_conv1 = (B, C_out1, H_out1, W_out1)
        conv3x3_nchw_fp32[grid_conv1](
            x_f, conv1_weight.contiguous().to(torch.float32), y1,
            B, C_in, H, W, C_out1, H_out1, W_out1,
            BLOCK_IN=64,
            num_warps=4,
        )

        # GroupNorm 1 (num_groups=32) over y1
        num_groups = 32
        C = C_out1
        assert C % num_groups == 0, "C must be divisible by num_groups (32)"
        group_size = C // num_groups

        y1_flat = y1.view(B, C, H_out1 * W_out1).contiguous()
        y1_norm = torch.empty_like(y1_flat, device=device, dtype=torch.float32)

        grid_gn1 = (B, num_groups)
        groupnorm_affine_fp32[grid_gn1](
            y1_flat, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
            y1_norm, B, C, H_out1 * W_out1, num_groups, group_size, eps,
            num_warps=4,
        )
        y1_norm = y1_norm.view(B, C, H_out1, W_out1)

        # SiLU on y1_norm
        y1_silu = torch.empty_like(y1_norm, device=device, dtype=torch.float32)
        total1 = B * C * H_out1 * W_out1
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_fp32_elementwise[grid_silu1](y1_norm, total1, BLOCK=1024, num_warps=4)

        # Second conv: y2 = conv3x3(y1_silu)
        C_in2 = conv2_weight.shape[1]  # input channels should equal C (output channels of first conv)
        assert C_in2 == C, "conv2_weight second dim must match output channels of first conv"
        C_out2 = conv2_weight.shape[0]
        H_out2 = H_out1
        W_out2 = W_out1
        y2 = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=torch.float32)

        grid_conv2 = (B, C_out2, H_out2, W_out2)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight.contiguous().to(torch.float32), y2,
            B, C, H_out2, W_out2, C_out2, H_out2, W_out2,
            BLOCK_IN=64,
            num_warps=4,
        )

        # GroupNorm 2 (num_groups=32) over y2
        C = C_out2
        assert C % num_groups == 0, "C must be divisible by num_groups (32)"
        group_size = C // num_groups
        y2_flat = y2.view(B, C, H_out2 * W_out2).contiguous()
        y2_norm = torch.empty_like(y2_flat, device=device, dtype=torch.float32)

        grid_gn2 = (B, num_groups)
        groupnorm_affine_fp32[grid_gn2](
            y2_flat, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
            y2_norm, B, C, H_out2 * W_out2, num_groups, group_size, eps,
            num_warps=4,
        )
        y2_norm = y2_norm.view(B, C, H_out2, W_out2)

        # SiLU on y2_norm
        y2_silu = torch.empty_like(y2_norm, device=device, dtype=torch.float32)
        total2 = B * C * H_out2 * W_out2
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_fp32_elementwise[grid_silu2](y2_norm, total2, BLOCK=1024, num_warps=4)

        # Residual addition: add original x (cast to fp32) to y2_silu. Shapes must match (B, C, H, W).


def run(*args):
    return ModelNew()(*args)
