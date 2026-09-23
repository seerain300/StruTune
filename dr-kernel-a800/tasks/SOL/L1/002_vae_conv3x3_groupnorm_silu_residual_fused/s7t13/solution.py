import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_im2col_kernel_full(
    x_ptr,          # *float32, input (1, C_in, H, W)
    w_ptr,          # *float32, weights (C_out, C_in, 3, 3)
    y_ptr,          # *float32, output (1, C_out, H, W)
    C_in: tl.constexpr,   # int
    C_out: tl.constexpr,  # int
    H: tl.constexpr,      # int
    W: tl.constexpr,      # int
    BLOCK_OC: tl.constexpr,  # tile size for output channels
):
    # Single program handles batch n=0, tile of output channels
    oc_block_id = tl.program_id(0)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for output channels in this tile
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Flatten output positions: OHW = H * W
    OHW = H * W

    # For im2col, we iterate over input channels and 3x3 taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Compute input patch indices for all OHW positions
                # For each output (oh, ow): ih = oh + kh - 1, iw = ow + kw - 1 (zero padding via mask)
                # We'll compute vector indices over OHW and load using mask.
                for p in range(OHW):
                    oh = p // W
                    ow = p % W
                    ih = oh + kh - 1
                    iw = ow + kw - 1
                    valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                    # Input linear index: (((0 * C_in + cin) * H + ih) * W + iw)
                    # Note: batch n=0, so n*C_in term is 0
                    in_index = (((cin) * H + ih) * W + iw)
                    x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)

                    # Weight vector index for oc tile: (((oc * C_in + cin) * 9) + (kh*3 + kw))
                    for j in range(BLOCK_OC):
                        if oc_mask[oc_offsets[j]]:
                            w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                            w_val = tl.load(w_ptr + w_index)
                            acc[j] += x_val * w_val

    # Store results: y[n=0, oc, :, :] across all OHW positions
    for j in range(BLOCK_OC):
        if oc_mask[oc_offsets[j]]:
            for p in range(OHW):
                oh = p // W
                ow = p % W
                out_index = (oc_offsets[j] * OHW + p)
                tl.store(y_ptr + out_index, acc[j])


@triton.jit
def conv3x3_stride1_pad1_im2col_kernel_partial_launch(
    x_ptr,          # *float32, input (N, C_in, H, W)
    w_ptr,          # *float32, weights (C_out, C_in, 3, 3)
    y_ptr,          # *float32, output (N, C_out, H, W)
    N,              # int
    C_in: tl.constexpr,   # int
    C_out: tl.constexpr,  # int
    H: tl.constexpr,      # int
    W: tl.constexpr,      # int
    BLOCK_OC: tl.constexpr,
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    OHW = H * W

    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                for p in range(OHW):
                    oh = p // W
                    ow = p % W
                    ih = oh + kh - 1
                    iw = ow + kw - 1
                    valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                    in_index = (((n * C_in + cin) * H + ih) * W + iw)
                    x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)

                    for j in range(BLOCK_OC):
                        if oc_mask[oc_offsets[j]]:
                            w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                            w_val = tl.load(w_ptr + w_index)
                            acc[j] += x_val * w_val

    for j in range(BLOCK_OC):
        if oc_mask[oc_offsets[j]]:
            for p in range(OHW):
                oh = p // W
                ow = p % W
                out_index = (((n * C_out + oc_offsets[j]) * OHW) + p)
                tl.store(y_ptr + out_index, acc[j])


@triton.jit
def group_norm_affine_kernel(
    x_ptr,          # *float32 input tensor (N, C, H, W)
    scale_ptr,      # *float32 norm weight (C,)
    bias_ptr,       # *float32 norm bias (C,)
    y_ptr,          # *float32 output tensor (N, C, H, W)
    N,              # int
    C,              # int
    H,              # int
    W,              # int
    num_groups: tl.constexpr,  # int, e.g., 32
    eps: tl.constexpr,         # float
):
    # Grid: (N, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    sum_all = 0.0
    sum_sq_all = 0.0

    # First pass: compute sum and sum of squares across group channels and all spatial
    for cin in range(channels_per_group):
        c = group_start + cin
        for p in range(H * W):
            oh = p // W
            ow = p % W
            in_index = (((n * C + c) * H + oh) * W + ow)
            x_val = tl.load(x_ptr + in_index)
            sum_all += x_val
            sum_sq_all += x_val * x_val

    # Compute mean and inv_std
    elements = channels_per_group * (H * W)
    mean = sum_all / elements
    var = sum_sq_all / elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, write to output
    for cin in range(channels_per_group):
        c = group_start + cin
        scale_val = tl.load(scale_ptr + c)
        bias_val = tl.load(bias_ptr + c)
        for p in range(H * W):
            oh = p // W
            ow = p % W
            in_index = (((n * C + c) * H + oh) * W + ow)
            x_val = tl.load(x_ptr + in_index)
            y_val = (x_val - mean) * inv_std
            y_val = y_val * scale_val + bias_val
            out_index = (((n * C + c) * H + oh) * W + ow)
            tl.store(y_ptr + out_index, y_val)


@triton.jit
def silu_kernel_4d(
    x_ptr,          # *float32 input tensor (N, C, H, W)
    y_ptr,          # *float32 output tensor (N, C, H, W)
    N, C, H, W,
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    in_index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + in_index)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    out_index = (((n * C + c) * H + h) * W + w)
    tl.store(y_ptr + out_index, y_val)


@triton.jit
def add_residual_kernel_4d(
    y_ptr,          # *float32 input tensor (N, C, H, W)
    x_ptr,          # *float32 residual tensor (N, C, H, W)
    out_ptr,        # *float32 output tensor (N, C, H, W)
    N, C, H, W,
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    in_index = (((n * C + c) * H + h) * W + w)
    y_val = tl.load(y_ptr + in_index)
    x_val = tl.load(x_ptr + in_index)
    out_val = y_val + x_val
    out_index = (((n * C + c) * H + h) * W + w)
    tl.store(out_ptr + out_index, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Enforce GroupNorm constraint: channels must be divisible by 32
        assert x.dim() == 4, "Input x must be 4D (N, C, H, W)"
        B, C, H, W = x.shape
        assert C % 32 == 0, "num_groups=32 requires C to be divisible by 32"
        C1 = C
        C2 = C

        # 1) First Conv: 3x3 stride=1, padding=1, bias=None — Triton
        out1 = torch.empty((B, C1, H, W), dtype=torch.float32, device=x.device)
        if B == 1:
            conv3x3_stride1_pad1_im2col_kernel_full[(C1 // 32,)](
                x.contiguous().to(torch.float32),
                conv1_weight.contiguous().to(torch.float32),
                out1,
                C_in=C1, C_out=C1, H=H, W=W, BLOCK_OC=64,
                num_warps=4, num_stages=2
            )
        else:
            conv3x3_stride1_pad1_im2col_kernel_partial_launch[(B, (C1 + 31) // 32)](
                x.contiguous().to(torch.float32),
                conv1_weight.contiguous().to(torch.float32),
                out1,
                B, C_in=C1, C_out=C1, H=H, W=W, BLOCK_OC=64,
                num_warps=4, num_stages=2
            )

        # 2) GroupNorm1 (num_groups=32) — Triton
        out1_norm = torch.empty_like(out1, dtype=torch.float32, device=out1.device)
        group_norm_affine_kernel[(B, 32)](
            out1, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
            out1_norm, B, C1, H, W,
            num_groups=32, eps=eps,
            num_warps=4, num_stages=2
        )

        # 3) SiLU1 — Triton
        out1_silu = torch.empty_like(out1_norm, dtype=torch.float32, device=out1_norm.device)
        silu_kernel_4d[(B, C1, H, W)](
            out1_norm, out1_silu, B, C1, H, W,
            num_warps=4, num_stages=2
        )

        # 4) Second Conv: 3x3 stride=1, padding=1, bias=None — Triton
        out2 = torch.empty((B, C2, H, W), dtype=torch.float32, device=x.device)
        if B == 1:
            conv3x3_stride1_pad1_im2col_kernel_full[(C2 // 32,)](
                out1_silu, conv2_weight.contiguous().to(torch.float32),
                out2,
                C_in=C2, C_out=C2, H=H, W=W, BLOCK_OC=64,
                num_warps=4, num_stages=2
            )
        else:
            conv3x3_stride1_pad1_im2col_kernel_partial_launch[(B, (C2 + 31) // 32)](
                out1_silu, conv2_weight.contiguous().to(torch.float32),
                out2,
                B, C_in=C2, C_out=C2, H=H, W=W, BLOCK_OC=64,
                num_warps=4, num_stages=2
            )

        # 5) GroupNorm2 (num_groups=32) — Triton
        out2_norm = torch.empty_like(out2, dtype=torch.float32, device=out2.device)
        group_norm_affine_kernel[(B, 32)](
            out2, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
            out2_norm, B, C2, H, W,
            num_groups=32, eps=eps,
            num_warps=4, num_stages=2
        )

        # 6) SiLU2 — Triton
        out2_silu = torch.empty_like(out2_norm, dtype=torch.float32, device=out2_norm.device)
        silu_kernel_4d[(B, C2, H, W)](
            out2_norm, out2_silu, B, C2, H, W,
            num_warps=4, num_stages=2
        )

        # 7) Add residual x — Triton
        y_out = torch.empty_like(out2_silu, dtype=torch.float32, device=out2_silu.device)
        add_residual_kernel_4d[(B, C2, H, W)](
            out2_silu, x.contiguous().to(torch.float32), y_out,
            B, C2, H, W,
            num_warps=4, num_stages=2
        )

        return y_out


# Example usage (CUDA):
# model = ModelNew().cuda()
# x = torch.randn(1, 16, 112, 112, device='cuda', dtype=torch.float32)
# conv1_w = torch.randn(16, 16, 3, 3, device='cuda', dtype=torch.float32)
# norm1_w = torch.randn(16, device='cuda', dtype=torch.float32)
# norm1_b = torch.randn(16, device='cuda', dtype=torch.float32)
# conv2_w = torch.randn(16, 16, 3, 3, device='cuda', dtype=torch.float32)
# norm2_w = torch.randn(16, device='cuda', dtype=torch.float32)
# norm2_b = torch.randn(16, device='cuda', dtype=torch.float32)
# eps = 1e-5
# y = model(x, conv1_w, norm1_w, norm1_b, conv2_w, norm2_w, norm2_b, eps)
# print(y.shape)  # should be (1, 16, 112, 112)


def run(*args):
    return ModelNew()(*args)
