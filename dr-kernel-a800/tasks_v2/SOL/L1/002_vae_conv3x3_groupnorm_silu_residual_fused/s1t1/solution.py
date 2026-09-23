import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_3x3_stride1_pad1_kernel(
    x_ptr,  # *float32, input: (B, C_in, H, W)
    w_ptr,  # *float32, weight: (C_out, C_in, 3, 3)
    y_ptr,  # *float32, output: (B, C_out, H, W)
    B, C_in, C_out, H, W, H_out, W_out,
    BLOCK_IN: tl.constexpr,
):
    # program id maps to (n, c_out, h_out, w_out)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # bounds check
    if pid_n >= B or pid_c >= C_out or pid_h >= H_out or pid_w >= W_out:
        return

    # accumulator for output channel pid_c
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels in chunks
    for ic_start in range(0, C_in, BLOCK_IN):
        c_range = ic_start + tl.arange(0, BLOCK_IN)
        mask_c = c_range < C_in

        # accumulate over 3x3 neighborhood
        sum_vec = tl.zeros((BLOCK_IN,), dtype=tl.float32)

        # iterate 3x3 kernel positions
        for kh in range(3):
            for kw in range(3):
                # compute input coordinates with padding
                ih = pid_h + kh  # padding=1 => no offset; stride=1
                iw = pid_w + kw
                # validity masks
                valid_hw = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                # broadcast masks for channel
                mask = mask_c & valid_hw

                # load input vector for these channels and positions
                # x[n, c_range, ih, iw] -> linear index: ((n*C_in + c)*H + ih)*W + iw
                # base for n and c_range: n*C_in*H*W + c_range*H*W
                # then + ih*W + iw
                x_off = (pid_n * C_in * H * W) + c_range * H * W + ih * W + iw
                x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0)

                # load weight vector for these channels at (kh, kw)
                # w[c_out=pid_c, c_range, kh, kw] -> linear index: ((pid_c*C_in + c)*3 + kh)*3 + kw
                w_off = (pid_c * C_in + c_range) * 9 + kh * 3 + kw
                w_vals = tl.load(w_ptr + w_off, mask=mask_c, other=0.0)

                # multiply and reduce across channels
                sum_vec += x_vals * w_vals

        # reduce vector to scalar and accumulate
        acc += tl.sum(sum_vec, axis=0)

    # store result
    y_off = (pid_n * C_out + pid_c) * H_out * W_out + pid_h * W_out + pid_w
    tl.store(y_ptr + y_off, acc)


@triton.jit
def groupnorm_affine_kernel(
    in_ptr, out_ptr,
    weight_ptr, bias_ptr,
    N, C, H, W, NUM_GROUPS, EPS,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    GROUP_SIZE = C // NUM_GROUPS
    c0 = g * GROUP_SIZE

    # accumulate sum and sum of squares over all elements of this (n, group)
    sum_val = 0.0
    sum_sq = 0.0

    # loop over channels in the group
    for ic in range(GROUP_SIZE):
        c = c0 + ic
        # loop over spatial HW in chunks
        hw = H * W
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            # map flat offs to (h, w): h = offs // W, w = offs % W
            h = offs // W
            w = offs % W
            # NCHW linear index for (n, c, h, w)
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            sum_val += tl.sum(x, axis=0)
            sum_sq += tl.sum(x * x, axis=0)

    hw = H * W
    mean = sum_val / (GROUP_SIZE * hw)
    var = sum_sq / (GROUP_SIZE * hw) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # write normalized outputs with affine
    for ic in range(GROUP_SIZE):
        c = c0 + ic
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            h = offs // W
            w = offs % W
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            y = (x - mean) * inv_std
            # affine scale and bias
            scale = tl.load(weight_ptr + c, mask=True, other=1.0)
            bias = tl.load(bias_ptr + c, mask=True, other=0.0)
            y = y * scale + bias
            out_offs = (n * C + c) * hw + offs
            tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def silu_kernel(in_ptr, out_ptr, TOTAL_ELEMS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, TOTAL_ELEMS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight, conv2_weight, norm1_weight, norm1_bias, norm2_weight, norm2_bias, eps: float):
        super().__init__()
        self.conv1_weight = torch.nn.Parameter(conv1_weight, requires_grad=False)
        self.conv2_weight = torch.nn.Parameter(conv2_weight, requires_grad=False)
        self.norm1_weight = torch.nn.Parameter(norm1_weight, requires_grad=False)
        self.norm1_bias = torch.nn.Parameter(norm1_bias, requires_grad=False)
        self.norm2_weight = torch.nn.Parameter(norm2_weight, requires_grad=False)
        self.norm2_bias = torch.nn.Parameter(norm2_bias, requires_grad=False)
        self.eps = eps
        self.num_groups = 32

    def forward(self, x: torch.Tensor):
        """
        Compute:
          y = SiLU( GroupNorm( Conv3x3(x, conv2_weight, stride=1, pad=1) )(norm2_weight, norm2_bias, eps) ) + residual
          residual = x
        But implement all steps in Triton kernels.
        """
        assert x.dim() == 4, "Input must be 4D (N, C, H, W)"
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "GroupNorm requires channels divisible by num_groups"

        # Ensure contiguous tensors
        x = x.contiguous()
        device = x.device
        dtype = torch.float32  # we operate in fp32 inside Triton

        # First conv: out1 = conv(x, conv1_weight)
        C_in = C  # channels after first conv == input channels to next conv
        C_out = C  # assume same as input (typical in these blocks)
        H_out = H  # stride=1, padding=1, no change in spatial size
        W_out = W

        # Allocate output for first conv
        out1 = torch.empty((B, C_out, H_out, W_out), device=device, dtype=torch.float32)

        # Launch conv1 kernel
        # Grid: (B, C_out, H_out, W_out)
        grid_conv = (B, C_out, H_out, W_out)
        conv2d_nchw_3x3_stride1_pad1_kernel[grid_conv](
            x, self.conv1_weight.to(dtype), out1,
            B, C_in, C_out, H, W, H_out, W_out,
            BLOCK_IN=32,
            num_warps=4,
            num_stages=2,
        )

        # First GroupNorm
        out1_flat = out1.view(B, C, H * W).contiguous()
        gn1_out = torch.empty_like(out1_flat)

        groupnorm_affine_kernel[(B, self.num_groups)](
            out1_flat, gn1_out,
            self.norm1_weight.to(torch.float32), self.norm1_bias.to(torch.float32),
            B, C, H, W, self.num_groups, self.eps,
            BLOCK_HW=1024,
            num_warps=4,
            num_stages=2,
        )

        # First SiLU
        total1 = B * C * H * W
        silu_out1 = torch.empty(total1, device=device, dtype=torch.float32)
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_kernel[grid_silu1](gn1_out, silu_out1, total1, BLOCK=1024, num_warps=4, num_stages=2)

        # Second conv: conv(silu_out1.view(N,C,H,W), conv2_weight)
        # We need to reshape silu_out1 back to (B,C,H,W) layout. But silu_out1 is flat across N,C,H*W.
        # Let's compute H_out2, W_out2 for conv input which is (B,C,H,W). For conv2 input tensor we need to think:
        # silu_out1 is (B,C,H,W), but we only have flattened values. We can reconstruct by assuming (B,C,H,W).
        # However, silu_out1 is flat; we can't directly reshape to (B,C,H,W) unless we keep original layout. Since we
        # convolved x -> out1 -> silu -> out2, the intermediate is not strictly (B,C,H,W) after flatten, but we can
        # emulate by using a temporary tensor of shape (B,C,H,W) with the same flattened values via view_as if we keep track.
        # Simpler: We will allocate a tensor conv_in of shape (B,C,H,W) and fill it using view_as and the flat data.
        # But since we don't have original (B,C,H,W), we cannot reconstruct. Therefore, we must keep track by not flattening.
        # Fix: Keep conv1 output as (B,C,H,W) then run SiLU in-place on out1. Let's redefine silu as operating on out1 directly.

        # Instead, perform SiLU on out1 in-place using a temporary tensor with same shape.
        # We will do SiLU directly on out1 by launching a Triton kernel over N,C,H,W. To do that, we need a flattened view.
        # We will create conv_in with shape (B,C,H,W) and copy the flattened gn1_out back using a view. But that requires
        # mapping flat index to (n,c,h,w) then assign. Triton can't index arbitrary positions without copying; so we must
        # create conv_in and copy elements from gn1_out to conv_in using a temporary kernel (not necessary if we keep
        # out1 as the target for SiLU).

        # Simpler approach: Since out1 is (B,C,H,W), we can run SiLU kernel over out1.view(-1). But out1 is not flat.
        # So we create silu_out1 = out1.clone(), then do silu over silu_out1.view(-1).

        silu_out1 = out1.clone()
        total1 = silu_out1.numel()
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_kernel[grid_silu1](silu_out1.view(-1), silu_out1.view(-1), total1, BLOCK=1024, num_warps=4, num_stages=2)

        # Now silu_out1 holds post-SiLU output from first path. This conv_in tensor for second conv is actually silu_out1.
        # However, silu_out1 is modified in-place. We need to use its values to compute next conv. So we will use
        # silu_out1 as the input to conv2. But we already applied conv2 on silu_out1 earlier? Wait, we must apply conv2 on
        # post-groupnorm output, not after SiLU.

        # Let's clarify: After GroupNorm1, we had gn1_out flattened. We can't directly SiLU on gn1_out because it's flat and
        # not mapped back. We need to do SiLU on the tensor of shape (B,C,H,W). We'll apply SiLU to gn1_out by copying
        # gn1_out into a tensor with shape (B,C,H,W) and then apply elementwise kernel. But we don't have mapping unless
        # we keep shape.

        # Fix: Instead of flattening for GroupNorm, apply GroupNorm and then apply SiLU elementwise on out1 directly:
        # We will store conv1 output to out1, then apply GroupNorm to out1, then SiLU elementwise on out1 (by cloning and
        # launching Triton kernel), then conv2, then GroupNorm2, SiLU, add residual.

        # First conv result is out1; after GroupNorm, we have gn1_out; now apply SiLU on out1 (in-place) by launching
        # elementwise kernel on out1.view(-1). Then conv2 on out1 (which is now silu result), GroupNorm2, SiLU, add residual.

        # Let's do that:

        # First conv: out1 = conv(x, conv1_weight)
        C_in2 = C  # channels for second conv input equals C (silu_out has same C)
        C_out2 = C
        H_out2 = H
        W_out2 = W

        # Second conv input: out1 (post-GroupNorm). We need to conv on out1 (which is GroupNorm result).
        out2 = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=torch.float32)

        # Launch conv2 kernel: conv(out1, conv2_weight)
        conv2d_nchw_3x3_stride1_pad1_kernel[(B, C_out2, H_out2, W_out2)](
            out1, self.conv2_weight.to(torch.float32), out2,
            B, C_in2, C_out2, H_out2, W_out2, H_out2, W_out2,
            BLOCK_IN=32,
            num_warps=4,
            num_stages=2,
        )

        # Second GroupNorm
        out2_flat = out2.view(B, C, H * W).contiguous()
        gn2_out = torch.empty_like(out2_flat)

        groupnorm_affine_kernel[(B, self.num_groups)](
            out2_flat, gn2_out,
            self.norm2_weight.to(torch.float32), self.norm2_bias.to(torch.float32),
            B, C, H, W, self.num_groups, self.eps,
            BLOCK_HW=1024,
            num_warps=4,
            num_stages=2,
        )

        # Second SiLU (elementwise)
        total2 = B * C * H * W
        silu_out2 = torch.empty(total2, device=device, dtype=torch.float32)
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_kernel[grid_silu2](gn2_out, silu_out2, total2, BLOCK=1024, num_warps=4, num_stages=2)

        # Reshape back to (B, C, H, W)
        out2 = silu_out2.view(B, C, H, W)

        # Add residual (x) elementwise
        total_elems = B * C * H * W
        add_out = torch.empty(total_elems, device=device, dtype=torch.float32)
        grid_add = (triton.cdiv(total_elems, 1024),)
        add_residual_kernel[out2.view(-1), x.view(-1), add_out](total_elems, BLOCK=1024, num_warps=4, num_stages=2)
        out2 = add_out.view(B, C, H, W)

        return out2


def run(*args):
    return ModelNew()(*args)
