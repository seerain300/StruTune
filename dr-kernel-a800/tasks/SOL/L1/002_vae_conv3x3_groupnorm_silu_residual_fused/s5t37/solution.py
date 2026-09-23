import torch
import triton
import triton.language as tl


# Triton kernel: Convert input x (B, C_in, H, W) into "columns" for a given output channel tile (BLOCK_CO).
# It outputs X_cols of shape [M, N], where M = C_in*9 and N = B * H_out * W_out, flattened.
@triton.jit
def conv3x3_im2col_to_col_kernel(
    x_ptr,            # *f32, input [B, C_in, H, W]
    X_ptr,            # *f32, output [M, N]
    B: tl.constexpr,  C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # program ids: tile over output channels (c_out_idx) and spatial positions (pos_idx)
    # We will flatten pos over (B, H_out, W_out).
    # Here, each program handles one output channel index in the tile and one pos index.
    # Grid is (C_out, B*H_out*W_out)
    c_out = tl.program_id(0)
    pos = tl.program_id(1)

    # For pos, compute (n, h_out, w_out)
    # Note: pos ranges from 0 to B*H_out*W_out - 1
    hw = H_out * W_out
    n = pos // hw
    rem = pos % hw
    h_out = rem // W_out
    w_out = rem % W_out

    # Corresponding input top-left (due to padding=1)
    h0 = h_out - 1
    w0 = w_out - 1

    # For each input channel, iterate over 3x3 neighborhood
    # Each row in X_cols corresponds to (c_in, dh, dw) flattened as (c_in*9 + idx)
    M = C_in * 9

    # We'll write one row per (c_in, dh, dw) for this (c_out, pos)
    # For each c_in
    for cin in range(0, C_in):
        # compute contributions for 3x3
        # idx = cin * 9 + ((dh + 3) * 3 + dw + 3)
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                idx = cin * 9 + ((dh + 1) * 3 + (dw + 1))  # map to 0..8

                h_in = h0 + dh
                w_in = w0 + dw

                # valid if within input bounds
                if (h_in >= 0 and h_in < H) and (w_in >= 0 and w_in < W):
                    value = tl.load(x_ptr + ((n * C_in + cin) * H + h_in) * W + w_in)
                else:
                    # zero for out-of-bounds (padding)
                    value = 0.0

                # write to X_cols at row idx, col pos
                col = pos  # pos is the flattened spatial position for this batch
                tl.store(X_ptr + idx * (B * H_out * W_out) + col, value)


# Triton kernel: Given X_cols [M, N] and W_flat [C_out_tile, M], compute Y_out_tile [C_out_tile, N].
@triton.jit
def conv3x3_col2out_kernel(
    X_ptr,            # *f32, [M, N]
    W_ptr,            # *f32, [C_out_tile, M]
    Y_ptr,            # *f32, [C_out_tile, N]
    M: tl.constexpr,  N: tl.constexpr,
    C_out_tile: tl.constexpr,
):
    # Grid: (C_out_tile, N)
    co = tl.program_id(0)
    pos = tl.program_id(1)

    # Accumulate over M rows
    acc = 0.0  # scalar accumulation
    for m in range(0, M):
        x_val = tl.load(X_ptr + m * N + pos)
        w_val = tl.load(W_ptr + co * M + m)
        acc += x_val * w_val

    # store result
    tl.store(Y_ptr + co * N + pos, acc)


# Triton kernel: GroupNorm + SiLU per channel (num_groups assumed provided; here we use per-channel, i.e., num_groups=1)
# Reduction kernel to compute mean and rstd for each channel
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,            # *f32, input tensor to normalize, shape [B, C, H, W]
    mean_ptr,         # *f32, per-channel mean
    rstd_ptr,         # *f32, per-channel rstd
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    BLOCK_HW: tl.constexpr, N_TILES: tl.constexpr,
):
    n = tl.program_id(0)  # batch index
    g = tl.program_id(1)  # group index
    c = tl.program_id(2)  # channel index within this group
    # Accumulate sum and sumsq over all HW elements
    total_sum = 0.0
    total_sumsq = 0.0
    for t in range(0, N_TILES):
        start = t * BLOCK_HW
        idx = start + tl.arange(0, BLOCK_HW)
        mask = idx < (H * W)
        # linear index in x_ptr: ((n*C + c)*H + (idx // W)) * W + (idx % W)
        h = idx // W
        w = idx % W
        ptr = ((n * C + c) * H + h) * W + w
        x_vec = tl.load(x_ptr + ptr, mask=mask, other=0.0)
        total_sum += tl.sum(x_vec, axis=0)
        total_sumsq += tl.sum(x_vec * x_vec, axis=0)
    mean = total_sum / (H * W)
    var = total_sumsq / (H * W) - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(mean_ptr + (g * channels_per_group + c), mean)
    tl.store(rstd_ptr + (g * channels_per_group + c), rstd)


# Triton kernel: Apply GroupNorm + affine + SiLU for each channel
@triton.jit
def group_norm_apply_kernel(
    x_ptr,            # *f32, input tensor [B, C, H, W]
    y_ptr,            # *f32, output tensor [B, C, H, W]
    mean_ptr,         # *f32, per-channel mean
    rstd_ptr,         # *f32, per-channel rstd
    scale_ptr,        # *f32, per-channel scale (gamma)
    bias_ptr,         # *f32, per-channel bias (beta)
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    BLOCK_HW: tl.constexpr, N_TILES: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)  # within group
    t = tl.program_id(3)  # tile over HW

    start = t * BLOCK_HW
    idx = start + tl.arange(0, BLOCK_HW)
    mask = idx < (H * W)

    h = idx // W
    w = idx % W

    # per-channel mean/rstd/scale/bias
    mean = tl.load(mean_ptr + (g * channels_per_group + c))
    rstd = tl.load(rstd_ptr + (g * channels_per_group + c))
    scale = tl.load(scale_ptr + c)
    beta = tl.load(bias_ptr + c)

    base = ((n * C + c) * H + h) * W + w
    x_vec = tl.load(x_ptr + base, mask=mask, other=0.0)

    # GroupNorm: (x - mean) * rstd
    y_vec = (x_vec - mean) * rstd
    # affine
    y_vec = y_vec * scale + beta
    # SiLU: y * sigmoid(y)
    y_vec = y_vec * (1.0 / (1.0 + tl.exp(-y_vec)))
    tl.store(y_ptr + base, y_vec, mask=mask)


# Triton elementwise kernel: y = out + x
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)

    h = hw // W
    w = hw % W

    idx = (((n * C) + c) * H + h) * W + w
    a_val = tl.load(a_ptr + idx)
    b_val = tl.load(b_ptr + idx)
    tl.store(out_ptr + idx, a_val + b_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C, H, W)
        conv weights: (C_out, C_in, 3, 3)
        norm scales/bias: (C,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure float32 and contiguous
        x_f32 = x.contiguous().to(torch.float32)

        # First conv3x3 via Triton im2col + GEMM
        C_out1 = int(conv1_weight.shape[0])
        C_in1 = int(conv1_weight.shape[1])

        # Output after first conv: (B, C_out1, H, W)
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        # Prepare im2col buffer: shape [C_in1*9, B*H*W]
        N = B * H * W
        M1 = C_in1 * 9
        X1_cols = torch.empty((M1, N), device=x.device, dtype=torch.float32)

        # Launch im2col kernel: grid over (C_out1, B*H*W)
        grid_im2col = (C_out1, N)
        conv3x3_im2col_to_col_kernel[grid_im2col](
            x_f32, X1_cols,
            B=B, C_in=C_in, H=H, W=W,
            C_out=C_out1, H_out=H, W_out=W,
            BLOCK_CO=1,  # single output channel at a time
            num_warps=4, num_stages=2,
        )

        # Flatten weights for each output channel: (C_out1, M1)
        W1_flat = conv1_weight.to(torch.float32).reshape(C_out1, C_in1 * 9)

        # Prepare output cols buffer: (C_out1, N)
        Y1_cols = torch.empty((C_out1, N), device=x.device, dtype=torch.float32)

        # Launch col2out kernel: grid over (C_out1, N)
        conv3x3_col2out_kernel[(C_out1, N)](
            X1_cols, W1_flat, Y1_cols,
            M=M1, N=N, C_out_tile=C_out1,
            num_warps=4, num_stages=2,
        )

        # Convert Y1_cols back to (B, C_out1, H, W)
        # For each output channel co, copy Y1_cols[co, :] into out1[:, co, :, :]
        for co in range(C_out1):
            out1[:, co, :, :] = Y1_cols[co, :].view(B, H, W)

        # First GroupNorm + SiLU (per-channel, i.e., num_groups=1)
        # But the original uses num_groups=32. We assert divisibility or fall back to per-channel.
        # To match original, we set num_groups=32 and require C_out1 % 32 == 0.
        assert C_out1 % self.num_groups == 0, "Channels for first GroupNorm must be divisible by num_groups (32)"
        channels_per_group = C_out1 // self.num_groups

        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)

        BLOCK_HW = 1024  # tile over H*W
        N_TILES1 = (H * W + BLOCK_HW - 1) // BLOCK_HW

        grid_reduce1 = (B, self.num_groups, channels_per_group)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES1,
            num_warps=4, num_stages=2,
        )

        out1_norm = torch.empty_like(out1)

        grid_apply1 = (B, self.num_groups, channels_per_group, N_TILES1)
        group_norm_apply_kernel[grid_apply1](
            out1, out1_norm, mean1, rstd1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32),
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES1,
            num_warps=4, num_stages=2,
        )

        # Second conv3x3 via Triton im2col + GEMM
        C_out2 = int(conv2_weight.shape[0])
        C_in2 = int(conv2_weight.shape[1])  # C_in2 == C_out1
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)

        # Prepare im2col buffer: (C_in2*9, B*H*W)
        N2 = B * H * W
        M2 = C_in2 * 9
        X2_cols = torch.empty((M2, N2), device=x.device, dtype=torch.float32)

        # Launch im2col kernel: grid over (C_out2, B*H*W)
        grid_im2col2 = (C_out2, N2)
        conv3x3_im2col_to_col_kernel[grid_im2col2](
            out1_norm, X2_cols,
            B=B, C_in=C_in2, H=H, W=W,
            C_out=C_out2, H_out=H, W_out=W,
            BLOCK_CO=1,
            num_warps=4, num_stages=2,
        )

        # Flatten weights for second conv: (C_out2, M2)
        W2_flat = conv2_weight.to(torch.float32).reshape(C_out2, C_in2 * 9)

        # Output cols buffer: (C_out2, N2)
        Y2_cols = torch.empty((C_out2, N2), device=x.device, dtype=torch.float32)

        # Launch col2out kernel: grid over (C_out2, N2)
        conv3x3_col2out_kernel[(C_out2, N2)](
            X2_cols, W2_flat, Y2_cols,
            M=M2, N=N2, C_out_tile=C_out2,
            num_warps=4, num_stages=2,
        )

        # Convert Y2_cols back to (B, C_out2, H, W)
        for co in range(C_out2):
            out2[:, co, :, :] = Y2_cols[co, :].view(B, H, W)

        # Second GroupNorm + SiLU (per-channel, i.e., num_groups=1), but we must match original num_groups=32.
        assert C_out2 % self.num_groups == 0, "Channels for second GroupNorm must be divisible by num_groups (32)"
        channels_per_group2 = C_out2 // self.num_groups

        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)

        N_TILES2 = (H * W + BLOCK_HW - 1) // BLOCK_HW

        grid_reduce2 = (B, self.num_groups, channels_per_group2)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES2,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)

        grid_apply2 = (B, self.num_groups, channels_per_group2, N_TILES2)
        group_norm_apply_kernel[grid_apply2](
            out2, out2_norm, mean2, rstd2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32),
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES2,
            num_warps=4, num_stages=2,
        )

        # Residual add in Triton: out2_norm + x_f32
        out = torch.empty_like(out2_norm)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
