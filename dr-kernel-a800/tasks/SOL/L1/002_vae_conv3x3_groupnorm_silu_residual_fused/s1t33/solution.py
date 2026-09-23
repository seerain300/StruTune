import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,         # *fp32, input (B, C_in, H, W), contiguous
    w_ptr,         # *fp32, weight (C_out, C_in, 3, 3), contiguous
    y_ptr,         # *fp32, output (B, C_out, H, W), contiguous
    B, C_in, C_out, H, W,
    stride_n, stride_cin, stride_h, stride_w,      # input strides
    w_stride_cout, w_stride_cin, w_stride_kh, w_stride_kw,  # weight strides
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,          # output strides
    BLOCK_CIN: tl.constexpr,
):
    # Program id maps to (n, c_out, h_out, w_out)
    pid = tl.program_id(axis=0)
    total = C_out * H * W
    n = pid // total
    rem = pid % total
    c_out = rem // (H * W)
    spatial = rem % (H * W)
    h_out = spatial // W
    w_out = spatial % W

    # Accumulator for output
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks
    for cin_start in range(0, C_in, BLOCK_CIN):
        cin_offsets = cin_start + tl.arange(0, BLOCK_CIN)
        # Valid mask for cin
        cin_mask = cin_offsets < C_in

        # Accumulate over 3x3 neighborhood
        for kh in range(3):
            for kw in range(3):
                # Compute input h, w indices for this output position and kernel offset
                h_in = h_out + kh - 1  # padding=1
                w_in = w_out + kw - 1  # padding=1

                # Mask for valid input spatial location
                valid_hw = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

                # Base input pointer offset for (n, cin, h_in, w_in), vectorized over cin
                # offset = n*stride_n + cin*stride_cin + h_in*stride_h + w_in*stride_w
                # Broadcast cin_offsets to vector
                input_offset = n * stride_n + cin_offsets * stride_cin + h_in * stride_h + w_in * stride_w

                # Load input vector for this kh,kw and cin chunk; mask combines cin and spatial validity
                x_vec = tl.load(x_ptr + input_offset, mask=cin_mask & valid_hw, other=0.0)

                # Load corresponding weight vector for this (c_out, cin chunk) at (kh, kw)
                # weight index: w_ptr[c_out, cin, kh, kw]
                w_offset = c_out * w_stride_cout + cin_offsets * w_stride_cin + kh * w_stride_kh + kw * w_stride_kw
                w_vec = tl.load(w_ptr + w_offset, mask=cin_mask, other=0.0)

                # Accumulate outer product: acc += sum_cin x_vec[cin] * w_vec[cin]
                acc += tl.sum(x_vec * w_vec, axis=0)

    # Store result
    y_offset = n * y_stride_n + c_out * y_stride_c + h_out * y_stride_h + w_out * y_stride_w
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def groupnorm_affine_nchw_fp32(
    x_ptr,         # *fp32, input tensor (B, C, H, W), contiguous
    y_ptr,         # *fp32, output tensor (B, C, H, W), contiguous
    weight_ptr,    # *fp32, per-channel scale (C,)
    bias_ptr,      # *fp32, per-channel bias (C,)
    B, C, H, W, num_groups, eps,
    BLOCK_HW: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(axis=0)
    n = pid // num_groups
    group = pid % num_groups
    group_size = C // num_groups

    # Compute channel range for this group: channels[group*group_size : (group+1)*group_size]
    c_start = group * group_size
    c_end = (group + 1) * group_size

    # Pass 1: compute sum and sum of squares over group channels and all spatial positions
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for c in range(c_start, c_end):
        # Flatten (H, W) dimension and iterate in chunks
        total_hw = H * W
        for hw_start in range(0, total_hw, BLOCK_HW):
            hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
            mask = hw_offsets < total_hw
            h = hw_offsets // W
            w = hw_offsets % W

            # Load x[n, c, h, w] vector
            x_offset = n * (C * H * W) + c * (H * W) + h * W + w
            x_vec = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            # Accumulate sum and sum of squares
            sum_val += tl.sum(x_vec, axis=0)
            sum_sq += tl.sum(x_vec * x_vec, axis=0)

    # Compute mean and variance (unbiased=False)
    hw_count = total_hw
    channels_in_group = group_size
    count = channels_in_group * hw_count
    mean = sum_val / count
    var = sum_sq / count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, write back
    for c in range(c_start, c_end):
        total_hw = H * W
        for hw_start in range(0, total_hw, BLOCK_HW):
            hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
            mask = hw_offsets < total_hw
            h = hw_offsets // W
            w = hw_offsets % W

            x_offset = n * (C * H * W) + c * (H * W) + h * W + w
            x_vec = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            # Normalize
            y_vec = (x_vec - mean) * inv_std

            # Load scale and bias for this channel
            scale = tl.load(weight_ptr + c)
            bias = tl.load(bias_ptr + c)

            y_vec = y_vec * scale + bias

            # Store
            y_offset = n * (C * H * W) + c * (H * W) + h * W + w
            tl.store(y_ptr + y_offset, y_vec, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(out_ptr + offsets, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        # Tunable block sizes
        self.conv_block_cin = 64
        self.groupnorm_block_hw = 256

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float = None):
        # Ensure float32 and contiguous for Triton
        x_fp32 = x.contiguous().to(torch.float32)

        B, C, H, W = x_fp32.shape

        # Kernel 1: conv1
        # Prepare input/output tensors
        x_in = x_fp32
        C_in = C  # same as C_in for conv1
        C_out1 = C
        H1 = H
        W1 = W

        y1 = torch.empty((B, C_out1, H1, W1), device=x.device, dtype=torch.float32)

        # Compute strides for NCHW
        # For contiguous tensors, strides are:
        # input strides: stride_n = C*H*W, stride_cin = H*W, stride_h = W, stride_w = 1
        # weight strides: w_stride_cout = C_in*3*3, w_stride_cin = 3*3, w_stride_kh = 3, w_stride_kw = 1
        # output strides: y_stride_n = C_out*H*W, y_stride_c = H*W, y_stride_h = W, y_stride_w = 1
        stride_n = C_in * H1 * W1
        stride_cin = H1 * W1
        stride_h = W1
        stride_w = 1
        # weight strides
        w_stride_cout = C_in * 3 * 3
        w_stride_cin = 3 * 3
        w_stride_kh = 3
        w_stride_kw = 1
        # output strides
        y_stride_n = C_out1 * H1 * W1
        y_stride_c = H1 * W1
        y_stride_h = W1
        y_stride_w = 1

        # Launch conv1
        total_out = C_out1 * H1 * W1
        grid_conv = (B * total_out,)
        conv3x3_nchw_fp32[grid_conv](
            x_in, conv1_weight.contiguous().to(torch.float32), y1,
            B, C_in, C_out1, H1, W1,
            stride_n, stride_cin, stride_h, stride_w,
            w_stride_cout, w_stride_cin, w_stride_kh, w_stride_kw,
            y_stride_n, y_stride_c, y_stride_h, y_stride_w,
            BLOCK_CIN=self.conv_block_cin,
        )

        # GroupNorm1 + affine
        y1_gn = torch.empty_like(y1)
        grid_gn1 = (B * self.num_groups,)
        groupnorm_affine_nchw_fp32[grid_gn1](
            y1, y1_gn, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
            B, C_out1, H1, W1, self.num_groups, self.eps if eps is None else eps,
            BLOCK_HW=self.groupnorm_block_hw,
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_gn)
        total1 = B * C_out1 * H1 * W1
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_kernel[grid_silu1](y1_gn, y1_silu, total1, BLOCK=1024)

        # Conv2 on y1_silu (same shape and channels)
        C_out2 = C  # output channels same as input channels
        H2 = H1
        W2 = W1
        y2 = torch.empty((B, C_out2, H2, W2), device=x.device, dtype=torch.float32)

        # Launch conv2: same strides setup as conv1
        # For conv2, input is y1_silu and weight is conv2_weight
        stride_n2 = C_out2 * H2 * W2  # but we need x2 stride? Actually we need input stride for conv2 which is y1_silu's layout: (B, C_out1=64, H1, W1)
        # Note: conv2 input is the silu output of 64 channels; we need to map accordingly
        # To keep it simple, reuse the NCHW stride logic. Here x_in for conv2 is y1_silu.
        # We already computed strides for conv1 based on x (B, C, H, W). Now conv2 input is (B, 64, H, W).
        # We'll recompute with C_in=64, C_out=64, H=H1, W=W1.

        stride_n2 = C_out2 * H2 * W2  # still equals C*H*W but now C_out2=C and H2=H1, W2=W1 -> equals C*H*W? For conv2 input, C_in=64, so input has C_in=64; but here conv2 weight maps to C_out2=C, and input after silu has channels=64. We need to map strides correctly.
        # Better approach: treat conv2 input as (B, C_in2, H2, W2) where C_in2 equals y1_silu channels. However, y1_silu is (B, 64, H, W). To avoid confusion, we can set stride_n2 = y2.stride(0) => y2.stride() returns contiguous strides, but we passed pointers; for Triton, we pass strides explicitly. Let's recompute using actual tensor strides.

        # We can get strides directly from y1_silu as input for conv2: input tensor is y1_silu
        # y1_silu is (B, 64, H, W) -> strides: stride_n2 = 64*H*W, stride_cin = H*W, stride_h = W, stride_w = 1
        # But to simplify, we will pass explicit strides based on shape:
        # Conv2 input shape: (B, C_in2, H2, W2) where C_in2 is number of channels of y1_silu after GroupNorm and SiLU. In original pipeline, after GroupNorm and SiLU, channels remain 384 (GroupNorm doesn't change C). But code suggests conv2 uses conv2_weight (C, C, 3, 3) applied to output of first conv which has C channels. Here, the first conv's output after GroupNorm+SiLU has same C as input (384), because weight is (C, C, 3, 3) and conv output channels=C. So conv2 input has C channels (384).
        # Correction: conv2 input is the silu output of conv1, which has C channels (384). So C_in2=C=384. Our y1_silu has shape (B, 384, H, W). But our previous y1 had shape (B, 384, H, W). GroupNorm with 32 groups maps 384->32 groups, each with 12 channels. However, the code applies conv2 on the GroupNorm output; GroupNorm doesn't change channel count. Therefore, conv2 input should have same C as original x, i.e., 384. In earlier steps, we set C_out1=C, but we need conv2 input to have C_in2 equal to number of channels of y1_silu. Since y1_silu is after GroupNorm and SiLU, it retains 384 channels. Hence, conv2 weight (C_out=384, C_in2=384, 3, 3) is applied to input of 384 channels.

        # Fix: conv2 input is y1_silu which has shape (B, 384, H, W). Therefore, for conv2:
        # x_in2 = y1_silu, C_in2 = 384, C_out2 = 384, H2 = H, W2 = W. We must pass corresponding strides for conv2.

        # We'll compute strides for conv2 input:
        # y1_silu shape (B, C_in2, H2, W2) -> strides: stride_n2 = C_in2*H2*W2, stride_cin = H2*W2, stride_h = W2, stride_w = 1
        # For output y2: y2 strides: y_stride_n2 = C_out2*H2*W2, y_stride_c2 = H2*W2, y_stride_h2 = W2, y_stride_w2 = 1

        # Recompute with correct C_in2 (channels of y1_silu). From pipeline, y1 after GroupNorm and SiLU has same C as original x (384). Therefore conv2 input has 384 channels. We need to know shape of y1 after GroupNorm: it remains (B, 384, H, W). Our code above applied GroupNorm on y1 (shape (B, 384, H, W)). So conv2 input is y1_gn_silu of shape (B, 384, H, W). We previously set y1_silu = silu(y1_gn), which still has 384 channels.

        # Therefore, for conv2:
        C_in2 = C  # 384
        C_out2 = C  # 384
        H2 = H1  # H
        W2 = W1  # W

        y2 = torch.empty((B, C_out2, H2, W2), device=x.device, dtype=torch.float32)

        # Launch conv2: conv2 input is y1_silu (B, C_in2=384, H, W), weight conv2_weight (C_out2=384, C_in2=384, 3, 3)
        stride_n2 = C_in2 * H2 * W2
        stride_cin2 = H2 * W2
        stride_h2 = W2
        stride_w2 = 1

        w_stride_cout2 = C_in2 * 3 * 3
        w_stride_cin2 = 3 * 3
        w_stride_kh2 = 3
        w_stride_kw2 = 1

        y_stride_n2 = C_out2 * H2 * W2
        y_stride_c2 = H2 * W2
        y_stride_h2 = W2
        y_stride_w2 = 1

        total_out2 = C_out2 * H2 * W2
        grid_conv2 = (B * total_out2,)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight.contiguous().to(torch.float32), y2,
            B, C_in2, C_out2, H2, W2,
            stride_n2, stride_cin2, stride_h2, stride_w2,
            w_stride_cout2, w_stride_cin2, w_stride_kh2, w_stride_kw2,
            y_stride_n2, y_stride_c2, y_stride_h2, y_stride_w2,
            BLOCK_CIN=self.conv_block_cin,
        )

        # GroupNorm2 + affine
        y2_gn = torch.empty_like(y2)
        grid_gn2 = (B * self.num_groups,)
        groupnorm_affine_nchw_fp32[grid_gn2](
            y2, y2_gn, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
            B, C_out2, H2, W2, self.num_groups, self.eps if eps is None else eps,
            BLOCK_HW=self.groupnorm_block_hw,
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_gn)
        total2 = B * C_out2 * H2 * W2
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_kernel[grid_silu2](y2_gn, y2_silu, total2, BLOCK=1024)

        # Residual add: y2_silu + x_fp32
        out = torch.empty_like(y2_silu)
        total_add = B * C_out2 * H2 * W2
        grid_add = (triton.cdiv(total_add, 1024),)
        add_residual_kernel[grid_add](y2_silu, x_fp32, out, total_add, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
