import torch
import triton
import triton.language as tl


# Triton Conv3x3: out[n, c_out, h, w] = sum_{c_in} sum_{dh=-1..1} sum_{dw=-1..1} x[n, c_in, h+dh, w+dw] * w[c_out, c_in, 3+dh, 3+dw]
# Implementation uses im2col-like reduction per (n, c_out) and a tile of output pixels.
# Kernel 1: reduce_xkernel accumulates the 9 input elements for each (n, c_out, tile).
@triton.jit
def reduce_xkernel(
    x_ptr,           # *f32, input [B, C_in, H, W]
    out_ptr,         # *f32, output [B, C_out, N_TILES, 9]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, N_TILES: tl.constexpr,
    tile_start: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_t = tl.program_id(2)

    # tile_start = pid_t * BLOCK
    offs = tile_start + tl.arange(0, BLOCK)  # vector of output pixel positions in this tile
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    # Accumulate the 9 input elements for each output pixel in the tile
    acc = tl.zeros([BLOCK, 9], dtype=tl.float32)
    # Loop over input channels (compile-time constant for Triton)
    for cin in range(C_in):
        # 3x3 neighborhood
        for dh in range(-1, 2):
            rh = h + dh  # row in input
            for dw in range(-1, 2):
                rw = w + dw  # col in input
                valid = (rh >= 0) & (rh < H) & (rw >= 0) & (rw < W)
                idx = (((pid_n * C_in) + cin) * H + rh) * W + rw
                val = tl.load(x_ptr + idx, mask=mask & valid, other=0.0)
                p = dh + 1  # 0..2
                q = dw + 1  # 0..2
                pos = p * 3 + q
                acc[:, pos] = val

    # Store accumulated inputs; we'll weight-accumulate in next kernel
    # out_ptr layout: ((n * C_out) + c_out) * (N_TILES * 9) + tile * 9 + pos
    base = ((pid_n * C_out) + pid_co) * (N_TILES * 9)
    tile_idx = pid_t * 9
    out_idx = base + tile_idx + tl.arange(0, 9)
    # Write acc[:, :] into out_ptr with broadcasting along the second dim
    for i in range(BLOCK):
        tl.store(out_ptr + out_idx + i * 9, acc[i, :], mask=True)


# Kernel 2: conv_accumulatekernel multiplies accumulated inputs by corresponding weights and accumulates into output.
@triton.jit
def conv_accumulatekernel(
    acc_ptr,         # *f32, [B, C_out, N_TILES, 9]
    w_ptr,           # *f32, [C_out, C_in, 3, 3]
    out_ptr,         # *f32, [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, N_TILES: tl.constexpr,
    tile_start: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_t = tl.program_id(2)

    offs = tile_start + tl.arange(0, BLOCK)
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    acc = tl.load(acc_ptr + ((pid_n * C_out) + pid_co) * (N_TILES * 9) + pid_t * 9 + tl.arange(0, 9))

    # Accumulate into output y[n, co, h, w]
    y_val = tl.zeros([BLOCK], dtype=tl.float32)
    for cin in range(C_in):
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                p = dh + 1
                q = dw + 1
                pos = p * 3 + q
                val = acc[pos]
                w_val = tl.load(w_ptr + (pid_co * C_in + cin) * 9 + pos)
                y_val += val * w_val

    base_out = ((pid_n * C_out) + pid_co) * (H * W)
    y_idx = base_out + offs
    tl.store(out_ptr + y_idx, y_val, mask=mask)


# Triton GroupNorm reduction: per (n, group, channel-in-group), compute sum and sumsq over all H*W.
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,          # *f32, input [B, C, H, W]
    mean_ptr,       # *f32, [C]
    rstd_ptr,       # *f32, [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)  # group id
    pid_c = tl.program_id(2)  # channel id within group
    c = pid_c + pid_g * channels_per_group  # output channel

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for t in range(N_TILES):
        tile_start = t * BLOCK_HW
        offs = tile_start + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)
        idx = (((pid_n * C) + c) * H + (offs // W)) * W + (offs % W)
        x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x_vec, axis=0)
        sum_sq += tl.sum(x_vec * x_vec, axis=0)

    hw = H * W
    mean = sum_val / hw
    var = sum_sq / hw - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-12)  # numerical stability

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# Triton GroupNorm apply + SiLU: per (n, group, channel-in-group, tile), normalize and apply affine+SiLU.
@triton.jit
def group_norm_apply_silu_kernel(
    x_ptr,          # *f32, input [B, C, H, W]
    mean_ptr,       # *f32, [C]
    rstd_ptr,       # *f32, [C]
    gamma_ptr,      # *f32, [C]
    beta_ptr,       # *f32, [C]
    y_ptr,          # *f32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_c = tl.program_id(2)
    pid_t = tl.program_id(3)

    c = pid_c + pid_g * channels_per_group

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    tile_start = pid_t * BLOCK_HW
    offs = tile_start + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)

    base = (((pid_n * C) + c) * H) * W
    idx = base + offs

    x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
    norm = (x_vec - mean) * rstd
    norm = norm * gamma + beta

    # SiLU: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-norm))
    y_vec = norm * sig

    tl.store(y_ptr + idx, y_vec, mask=mask)


# Triton residual add: y = a + b, elementwise, a, b, y are [B, C, H, W]
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    HW = H * W
    hw = pid_hw
    if hw >= HW:
        return
    h = hw // W
    w = hw % W

    idx = (((pid_n * C) + pid_c) * H + h) * W + w
    a_val = tl.load(a_ptr + idx)
    b_val = tl.load(b_ptr + idx)
    tl.store(out_ptr + idx, a_val + b_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C, H, W) input
        conv1_weight: (C1, C, 3, 3)
        norm1_weight, norm1_bias: (C1,)
        conv2_weight: (C2, C1, 3, 3)
        norm2_weight, norm2_bias: (C2,)
        Returns: (B, C2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure tensors are float32 contiguous
        x_f32 = x.contiguous().to(torch.float32)

        # Prepare weights as float32
        conv1_w = conv1_weight.contiguous().to(torch.float32)  # (C1, C_in, 3, 3)
        conv2_w = conv2_weight.contiguous().to(torch.float32)  # (C2, C1, 3, 3)

        # First conv: y1 = conv3x3(x)
        C1 = conv1_w.shape[0]
        assert (C1 % self.num_groups) == 0, "C1 must be divisible by num_groups for GroupNorm"
        # Output y1: (B, C1, H, W)
        y1 = torch.empty((B, C1, H, W), device=x.device, dtype=torch.float32)

        # Launch conv1: we implement conv via Triton kernels (reduce_xkernel + conv_accumulatekernel)
        N_TILES = triton.cdiv(H * W, 1024)  # tile count for reduction, 1024 is a good block size
        BLOCK = 128  # number of output pixels per tile for the first kernel

        # Reduce inputs for conv1
        # grid: (B, C1, N_TILES)
        reduce_xkernel[(B, C1, N_TILES)](
            x_f32, y1,  # acc_ptr is y1, but we’ll allocate an intermediate acc tensor per kernel; instead, we will implement
            # Note: Triton doesn't support writing into y1 directly from this kernel; we need an intermediate
            # So we'll use a separate buffer for accumulated inputs per tile. To keep code compact, we'll
            # allocate an intermediate buffer 'acc' with shape (B, C1, N_TILES, 9) in Python, then call
            # conv_accumulatekernel to fill y1.
        )

        # We need an intermediate accumulation buffer 'acc' to avoid writing directly into y1 from reduce_xkernel.
        # Allocate acc: [B, C1, N_TILES, 9]
        acc1 = torch.empty((B, C1, N_TILES, 9), device=x.device, dtype=torch.float32)

        # Run reduce_xkernel to fill acc1
        reduce_xkernel[(B, C1, N_TILES)](
            x_f32, acc1,
            B=B, C_in=C_in, H=H, W=W,
            C_out=C1, N_TILES=N_TILES, tile_start=0, BLOCK=BLOCK
        )

        # Now accumulate into y1 using conv2d-like weights
        # conv_accumulatekernel takes acc1 and conv1_w to produce y1
        conv_accumulatekernel[(B, C1, N_TILES)](
            acc1, conv1_w, y1,
            B=B, C_in=C_in, H=H, W=W,
            C_out=C1, N_TILES=N_TILES, tile_start=0, BLOCK=BLOCK
        )

        # GroupNorm + SiLU for first block
        mean1 = torch.empty(C1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C1, device=x.device, dtype=torch.float32)
        group_norm_reduce_kernel[(B, self.num_groups, C1 // self.num_groups)](
            y1, mean1, rstd1,
            B=B, C=C1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=(C1 // self.num_groups),
            N_TILES=N_TILES, BLOCK_HW=1024
        )
        # Apply + SiLU
        y1_norm = torch.empty_like(y1)
        group_norm_apply_silu_kernel[(B, self.num_groups, C1 // self.num_groups, N_TILES)](
            y1, mean1, rstd1, norm1_weight, norm1_bias, y1_norm,
            B=B, C=C1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=(C1 // self.num_groups),
            N_TILES=N_TILES, BLOCK_HW=1024
        )

        # Second conv: y2 = conv3x3(y1_norm)
        C2 = conv2_w.shape[0]
        assert (C2 % self.num_groups) == 0, "C2 must be divisible by num_groups for GroupNorm"

        # Allocate intermediate acc2 for conv2
        y2 = torch.empty((B, C2, H, W), device=x.device, dtype=torch.float32)
        acc2 = torch.empty((B, C2, N_TILES, 9), device=x.device, dtype=torch.float32)

        # Reduce acc2 from y1_norm
        # First we need conv1_w to know the mapping; actually, reduce_xkernel doesn't need weights, it loads x.
        reduce_xkernel[(B, C2, N_TILES)](
            y1_norm, acc2,
            B=B, C_in=C1, H=H, W=W,
            C_out=C2, N_TILES=N_TILES, tile_start=0, BLOCK=BLOCK
        )
        # Accumulate into y2 using conv2_w
        conv_accumulatekernel[(B, C2, N_TILES)](
            acc2, conv2_w, y2,
            B=B, C_in=C1, H=H, W=W,
            C_out=C2, N_TILES=N_TILES, tile_start=0, BLOCK=BLOCK
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty(C2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C2, device=x.device, dtype=torch.float32)
        group_norm_reduce_kernel[(B, self.num_groups, C2 // self.num_groups)](
            y2, mean2, rstd2,
            B=B, C=C2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=(C2 // self.num_groups),
            N_TILES=N_TILES, BLOCK_HW=1024
        )
        y2_norm = torch.empty_like(y2)
        group_norm_apply_silu_kernel[(B, self.num_groups, C2 // self.num_groups, N_TILES)](
            y2, mean2, rstd2, norm2_weight, norm2_bias, y2_norm,
            B=B, C=C2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=(C2 // self.num_groups),
            N_TILES=N_TILES, BLOCK_HW=1024
        )

        # Residual add: y2_norm + x
        out = torch.empty_like(y2_norm)
        residual_add_kernel[(B, C2, H * W)](
            y2_norm, x_f32, out
        )

        return out


def run(*args):
    return ModelNew()(*args)
