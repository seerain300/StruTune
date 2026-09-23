import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,  # *const float32, shape [N, C_in, H, W]
    w_ptr,  # *const float32, shape [C_in, C_out, 3, 3]
    y_ptr,  # *float32,       shape [N, C_out, H, W]
    N, C_in, H, W, C_out,  # int32
    BLOCK_IN: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    # map pid to (n, c_out, h_out, w_out)
    HW = H * W
    n = pid // (C_out * HW)
    tmp = pid % (C_out * HW)
    c_out = tmp // HW
    tmp2 = tmp % HW
    h_out = tmp2 // W
    w_out = tmp2 % W

    # initialize accumulator
    acc = 0.0
    # loop over input channels in chunks
    for ic_start in range(0, C_in, BLOCK_IN):
        ic_offsets = ic_start + tl.arange(0, BLOCK_IN)
        mask_ic = ic_offsets < C_in
        # for each chunk, accumulate over 3x3 window
        # kh, kw loops are dynamic to support any H, W
        for kh in range(3):
            ih = h_out + kh - 1  # -1 because padding is applied via masks
            valid_h = (ih >= 0) & (ih < H)
            for kw in range(3):
                iw = w_out + kw - 1
                valid_w = (iw >= 0) & (iw < W)
                valid = valid_h & valid_w
                # base pointer for this (n, ic, ih, iw)
                # x[n, ic, ih, iw] with masks
                for j in range(BLOCK_IN):
                    ic = ic_offsets[j]
                    m = mask_ic[j]
                    # if valid, load; else 0
                    ptr_x = x_ptr + n * (C_in * H * W) + ic * (H * W) + ih * W + iw
                    x_val = tl.load(ptr_x, mask=valid & m, other=0.0)
                    # weight for this (ic, c_out, kh, kw)
                    ptr_w = w_ptr + ic * (C_out * 9) + c_out * 9 + kh * 3 + kw
                    w_val = tl.load(ptr_w)
                    acc += x_val * w_val
    # store result
    ptr_y = y_ptr + n * (C_out * H * W) + c_out * (H * W) + h_out * W + w_out
    tl.store(ptr_y, acc)


@triton.jit
def groupnorm_affine_fp32(
    x_ptr,  # *const float32, shape [N, C_in, H, W]
    scale_ptr,  # *const float32, shape [C_in]
    bias_ptr,   # *const float32, shape [C_in]
    y_ptr,      # *float32,       shape [N, C_in, H, W]
    N, C_in, H, W,  # int32
    group_id: tl.constexpr,  # which group
    group_size: tl.constexpr,  # channels per group
    num_groups: tl.constexpr,  # total groups (should be 32 in this task)
    eps,  # float32 epsilon
):
    # compute mean and variance over channels in group and all spatial positions
    sum_all = 0.0
    sum_sq_all = 0.0
    total_elems = H * W
    # channel range for this group
    # ci in [group_id * group_size, (group_id+1)*group_size)
    # loop over channels in the group
    ci_start = group_id * group_size
    for ci in range(group_size):
        c = ci_start + ci
        # loop over all spatial positions
        spatial_index = 0
        while spatial_index < total_elems:
            h = spatial_index // W
            w = spatial_index % W
            ptr_x = x_ptr + N * (C_in * H * W) + c * (H * W) + h * W + w
            x_val = tl.load(ptr_x)
            sum_all += x_val
            sum_sq_all += x_val * x_val
            spatial_index += 1
    mean = sum_all / (group_size * H * W)
    var = sum_sq_all / (group_size * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # normalize and apply affine, write back
    for ci in range(group_size):
        c = ci_start + ci
        for spatial_index in range(0, total_elems):
            h = spatial_index // W
            w = spatial_index % W
            ptr_x = x_ptr + N * (C_in * H * W) + c * (H * W) + h * W + w
            x_val = tl.load(ptr_x)
            norm = (x_val - mean) * inv_std
            scale = tl.load(scale_ptr + c)
            bias = tl.load(bias_ptr + c)
            y_val = norm * scale + bias
            ptr_y = y_ptr + N * (C_in * H * W) + c * (H * W) + h * W + w
            tl.store(ptr_y, y_val)


@triton.jit
def silu_kernel_fp32(x_ptr, y_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def add_residual_kernel_fp32(a_ptr, b_ptr, c_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups

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
        Triton-optimized fused residual block:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add(original input)
        """
        device = x.device
        dtype = torch.float32  # compute in float32
        N, C, H, W = x.shape
        C_in = C  # channels preserved by convs in the original pipeline

        # 1) First conv: conv1 -> [N, C_in, H, W]
        # Cast inputs and weights to float32 and contiguous
        x_in = x.to(torch.float32).contiguous()
        conv1_w = conv1_weight.to(torch.float32).contiguous()  # [C_in, C_in, 3, 3]
        y1 = torch.empty((N, C_in, H, W), device=device, dtype=torch.float32)

        grid = N * C_in * H * W
        conv3x3_nchw_fp32[(grid,)](
            x_in, conv1_w, y1,
            N, C_in, H, W, C_in,
            BLOCK_IN=16,
            num_warps=4,
        )

        # 2) GroupNorm1 (num_groups=32), affine, output -> [N, C_in, H, W]
        y1_norm = torch.empty_like(y1, device=device, dtype=torch.float32)
        # num_groups must be 32 here; assert channels divisible by 32
        assert C_in % self.num_groups == 0, "C_in must be divisible by num_groups (32) for GroupNorm."
        group_size1 = C_in // self.num_groups
        for group_id in range(self.num_groups):
            grid_g = (1,)
            groupnorm_affine_fp32[grid_g](
                y1, norm1_weight, norm1_bias, y1_norm,
                N, C_in, H, W,
                group_id, group_size1, self.num_groups, eps,
                num_warps=4,
            )

        # 3) SiLU1
        y1_silu = torch.empty_like(y1_norm, device=device, dtype=torch.float32)
        total_silu1 = y1_norm.numel()
        grid_silu1 = (triton.cdiv(total_silu1, 1024),)
        silu_kernel_fp32[grid_silu1](y1_norm, y1_silu, total_silu1, BLOCK=1024)

        # 4) Second conv: conv2 -> [N, C_in, H, W]
        conv2_w = conv2_weight.to(torch.float32).contiguous()  # [C_in, C_in, 3, 3]
        y2 = torch.empty((N, C_in, H, W), device=device, dtype=torch.float32)
        grid2 = N * C_in * H * W
        conv3x3_nchw_fp32[(grid2,)](
            y1_silu, conv2_w, y2,
            N, C_in, H, W, C_in,
            BLOCK_IN=16,
            num_warps=4,
        )

        # 5) GroupNorm2 (num_groups=32), affine, output -> [N, C_in, H, W]
        y2_norm = torch.empty_like(y2, device=device, dtype=torch.float32)
        group_size2 = C_in // self.num_groups
        for group_id in range(self.num_groups):
            grid_g2 = (1,)
            groupnorm_affine_fp32[grid_g2](
                y2, norm2_weight, norm2_bias, y2_norm,
                N, C_in, H, W,
                group_id, group_size2, self.num_groups, eps,
                num_warps=4,
            )

        # 6) SiLU2
        y2_silu = torch.empty_like(y2_norm, device=device, dtype=torch.float32)
        total_silu2 = y2_norm.numel()
        grid_silu2 = (triton.cdiv(total_silu2, 1024),)
        silu_kernel_fp32[grid_silu2](y2_norm, y2_silu, total_silu2, BLOCK=1024)

        # 7) Residual add: add original input x (cast to float32, contiguous)
        x_add = x.to(torch.float32).contiguous()  # shape [N, C_in, H, W]
        final_out = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        total_add = y2_silu.numel()
        grid_add = (triton.cdiv(total_add, 1024),)
        add_residual_kernel_fp32[grid_add](y2_silu, x_add, final_out, total_add, BLOCK=1024)

        return final_out


def run(*args):
    return ModelNew()(*args)
