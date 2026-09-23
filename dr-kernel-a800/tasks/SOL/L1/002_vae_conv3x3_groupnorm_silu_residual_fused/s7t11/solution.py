import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel_full(
    x_ptr,          # *float32 input tensor (B, C_in, H, W)
    w_ptr,          # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,          # *float32 output tensor (B, C_out, H, W)
    B, C_in, H, W, C_out,
    BLOCK_OC: tl.constexpr,  # tile size for output channels per program
):
    # Each program handles one batch n and a tile of output channels
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for output channels in this tile
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Output height/width are the same as input due to stride=1, padding=1
                for oh in range(H):
                    for ow in range(W):
                        ih = oh + kh - 1
                        iw = ow + kw - 1
                        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                        # Load input x[n, cin, ih, iw]
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)

                        # Load weights for all oc in tile: w[oc, cin, kh, kw]
                        # weight linear index: ((oc * C_in + cin) * (3*3)) + (kh * 3 + kw)
                        for j in range(BLOCK_OC):
                            if oc_mask[oc_offsets[j]]:
                                w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                                w_val = tl.load(w_ptr + w_index)
                                acc[j] += x_val * w_val

    # Store results to y[n, oc, oh, ow] for all oh, ow
    # For each oc in tile, write acc[oc] across all positions
    for j in range(BLOCK_OC):
        if oc_mask[oc_offsets[j]]:
            for oh in range(H):
                for ow in range(W):
                    out_index = (((n * C_out + oc_offsets[j]) * H + oh) * W + ow)
                    # Store acc[j] to y at (n, oc, oh, ow)
                    tl.store(y_ptr + out_index, acc[j])


@triton.jit
def conv3x3_stride1_pad1_kernel_partial(
    x_ptr, w_ptr, y_ptr, B, C_in, H, W, C_out,
    BLOCK_OC: tl.constexpr,
):
    # Placeholder partial kernel (not used in forward for provided workloads).
    # It handles multiple batch samples per program. Kept for completeness.
    pass


@triton.jit
def group_norm_affine_kernel(
    x_ptr,       # *float32 input tensor (B, C, H, W)
    scale_ptr,   # *float32 per-channel scale (C,)
    bias_ptr,    # *float32 per-channel bias (C,)
    y_ptr,       # *float32 output tensor (B, C, H, W)
    N, C, H, W,
    num_groups: tl.constexpr,   # in our case 32
    eps: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group
    group_end = group_start + channels_per_group

    # Compute sum and sum of squares over group channels and all spatial positions
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for c in range(group_start, group_end):
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                sum_val += x_val
                sum_sq += x_val * x_val

    # Compute mean and variance
    numel = channels_per_group * H * W
    mean = sum_val / numel
    var = sum_sq / numel - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine: y = ((x - mean) * inv_std) * scale + bias
    for c in range(group_start, group_start + channels_per_group):
        scale = tl.load(scale_ptr + c)
        bias = tl.load(bias_ptr + c)
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                y_val = ((x_val - mean) * inv_std) * scale + bias
                y_index = x_index  # same linear indexing
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel_4d(
    x_ptr, y_ptr,
    B, C, H, W,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Grid is (B, C, H, W). Each program handles one element.
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    x_index = (((b * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + x_index)
    # SiLU: x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + x_index, y_val)


@triton.jit
def add_residual_kernel_4d(
    y_ptr, x_ptr, out_ptr,
    B, C, H, W,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Elementwise: out = y + x
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    index = (((b * C + c) * H + h) * W + w)
    y_val = tl.load(y_ptr + index)
    x_val = tl.load(x_ptr + index)
    out_val = y_val + x_val
    tl.store(out_ptr + index, out_val)


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
        Triton-only implementation of the fused residual block:
        Conv3x3 -> GroupNorm(num_groups=32) -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add residual
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        B, C, H, W = x.shape

        # 1) First Conv3x3 (stride=1, padding=1, bias=None) — Triton
        out1 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        BLOCK_OC = 32  # tile size for output channels
        grid_conv1 = (B, triton.cdiv(C, BLOCK_OC))
        conv3x3_stride1_pad1_kernel_full[grid_conv1](
            x, conv1_weight.to(torch.float32), out1,
            B, C, H, W, C,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 2) GroupNorm1 (num_groups=32) — Triton
        out1_norm = torch.empty_like(out1, dtype=torch.float32, device=out1.device)
        grid_gn1 = (B, 32)
        # Enforce GroupNorm constraint: C must be divisible by 32
        assert (C % 32) == 0, "Channels must be divisible by num_groups=32 for GroupNorm."
        group_norm_affine_kernel[grid_gn1](
            out1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out1_norm,
            B, C, H, W,
            num_groups=32,
            eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 3) SiLU1 — Triton (elementwise)
        out1_silu = torch.empty_like(out1_norm, dtype=torch.float32, device=out1_norm.device)
        grid_silu1 = (B, C, H, W)
        silu_kernel_4d[grid_silu1](
            out1_norm, out1_silu,
            B, C, H, W,
            num_warps=1,
            num_stages=1,
        )

        # Save residual x for later addition (compute in float32 for numerical stability)
        residual = x.to(torch.float32)

        # 4) Second Conv3x3 (stride=1, padding=1, bias=None) — Triton
        out2 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (B, triton.cdiv(C, BLOCK_OC))
        conv3x3_stride1_pad1_kernel_full[grid_conv2](
            out1_silu, conv2_weight.to(torch.float32), out2,
            B, C, H, W, C,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 5) GroupNorm2 (num_groups=32) — Triton
        out2_norm = torch.empty_like(out2, dtype=torch.float32, device=out2.device)
        group_norm_affine_kernel[grid_gn1](
            out2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), out2_norm,
            B, C, H, W,
            num_groups=32,
            eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 6) SiLU2 — Triton (elementwise)
        out2_silu = torch.empty_like(out2_norm, dtype=torch.float32, device=out2_norm.device)
        grid_silu2 = (B, C, H, W)
        silu_kernel_4d[grid_silu2](
            out2_norm, out2_silu,
            B, C, H, W,
            num_warps=1,
            num_stages=1,
        )

        # 7) Residual Add — Triton
        out_final = torch.empty_like(out2_silu, dtype=torch.float32, device=out2_silu.device)
        add_residual_kernel_4d[grid_silu2](
            out2_silu, residual, out_final,
            B, C, H, W,
            num_warps=1,
            num_stages=1,
        )

        # Ensure final output matches expected shape (B, C, H, W). out_final already has that shape.
        return out_final


# Optional local test
# if __name__ == "__main__":
#     B, C, H, W = 1, 64, 128, 128  # ensure C % 32 == 0
#     x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
#     conv1_w = torch.randn(C, C, 3, 3, device='cuda', dtype=torch.float32)
#     norm1_w = torch.randn(C, device='cuda', dtype=torch.float32)
#     norm1_b = torch.randn(C, device='cuda', dtype=torch.float32)
#     conv2_w = torch.randn(C, C, 3, 3, device='cuda', dtype=torch.float32)
#     norm2_w = torch.randn(C, device='cuda', dtype=torch.float32)
#     norm2_b = torch.randn(C, device='cuda', dtype=torch.float32)
#     eps = 1e-5
#     model = ModelNew().cuda()
#     y = model(x, conv1_w, norm1_w, norm1_b, conv2_w, norm2_w, norm2_b, eps)
#     print(y.shape)  # should be (B, C, H, W)


def run(*args):
    return ModelNew()(*args)
