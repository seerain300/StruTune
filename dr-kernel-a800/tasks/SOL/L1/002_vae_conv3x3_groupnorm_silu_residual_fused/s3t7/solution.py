import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, tiles over H*W). Each program handles one (b, oc) and one spatial tile.
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,  # e.g., 32
    BLOCK_SP: tl.constexpr,  # e.g., 1024
):
    b = tl.program_id(0)
    oc_base = tl.program_id(1)
    tile_id = tl.program_id(2)

    # tile over spatial dimension
    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)

    # map sp to (h, w)
    h = sp // W
    w = sp % W

    # initialize accumulator for [BLOCK_SP]
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    # loop over input channels in blocks
    ic_base = 0
    while ic_base < C_in:
        ic_vec = ic_base + tl.arange(0, BLOCK_OC)
        ic_mask = ic_vec < C_in  # [BLOCK_OC]

        # for each ic in the block, accumulate contributions
        for ic_i in range(BLOCK_OC):
            ic = ic_vec[ic_i]
            valid_ic = ic_mask[ic_i]

            # base x pointer for this (b, ic)
            x_base = x_ptr + b * x_stride_b + ic * x_stride_c

            # load 3x3 neighborhood (padding=1 => h_in = h - 1, w_in = w - 1)
            for kh in range(3):
                h_in = h - 1 + kh  # padded h indices
                for kw in range(3):
                    w_in = w - 1 + kw  # padded w indices

                    # compute pointers for this (h_in, w_in)
                    x_ptrs = x_base + h_in * x_stride_h + w_in * x_stride_w
                    # mask: valid spatial and valid ic
                    load_mask = sp_mask & valid_ic
                    x_vals = tl.load(x_ptrs, mask=load_mask, other=0.0)  # [BLOCK_SP]

                    # load weight for (ic, oc_base + ic_i, kh, kw)
                    oc = oc_base + ic_i
                    w_ptr_elem = w_ptr + ic * w_stride_cin + oc * w_stride_cout + kh * w_stride_kh + kw * w_stride_kw
                    w_val = tl.load(w_ptr_elem)  # scalar
                    acc += x_vals * w_val

        ic_base += BLOCK_OC

    # store results
    y_base = y_ptr + b * y_stride_b + oc_base * y_stride_c
    y_ptrs = y_base + h * y_stride_h + w * y_stride_w
    store_mask = sp_mask
    tl.store(y_ptrs, acc, mask=store_mask)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU
# x: input (B, C, H, W), weight: per-channel scale (C,), bias: per-channel (C,), y: output
# Grid: (B, num_groups). Assumes num_groups divides C and C % num_groups == 0. We treat group_elements = (C // num_groups) * H * W.
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, SiLU
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + ch, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        # SiLU: z * sigmoid(z)
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr + b * y_stride_b + ch * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    r = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y + r, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure CUDA and float32 compute
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be on CUDA."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA."

        B, C, H, W = x.shape
        C_in = C
        C_out = C
        tiles = triton.cdiv(H * W, 1024)

        # 1) conv1: y1 = conv3x3(x)
        y1 = torch.empty((B, C_out, H, W), device=x.device, dtype=x.dtype)
        conv3x3_triton[(B, C_out, tiles)](
            x, conv1_weight, y1,
            B, C_in, H, W, C_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=32, BLOCK_SP=1024,
            num_warps=4, num_stages=2
        )

        # 2) GroupNorm + affine + SiLU for y1
        y1_norm = torch.empty_like(y1)
        group_norm_affine_silu[(B, self.num_groups)](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C, H, W, self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            BLOCK=1024, num_warps=4, num_stages=2
        )

        # 3) conv2: y2 = conv3x3(y1_norm)
        y2 = torch.empty((B, C_out, H, W), device=x.device, dtype=x.dtype)
        conv3x3_triton[(B, C_out, tiles)](
            y1_norm, conv2_weight, y2,
            B, C_out, H, W, C_out,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=32, BLOCK_SP=1024,
            num_warps=4, num_stages=2
        )

        # 4) GroupNorm + affine + SiLU for y2
        y2_norm = torch.empty_like(y2)
        group_norm_affine_silu[(B, self.num_groups)](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C, H, W, self.num_groups, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            BLOCK=1024, num_warps=4, num_stages=2
        )

        # 5) Add residual: out = y2_norm + x
        out = torch.empty_like(y2_norm)
        N = out.numel()
        add_residual_kernel[(triton.cdiv(N, 1024),)](
            out, y2_norm, x, N, BLOCK=1024, num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
