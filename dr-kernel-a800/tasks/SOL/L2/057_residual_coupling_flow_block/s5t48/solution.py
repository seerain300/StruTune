import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def conv1d_triton_stride1_bias_relu(
    x_ptr,         # *f32, [B, Cin, T]
    w_ptr,         # *f32, [Cout, Cin*K]
    b_ptr,         # *f32, [Cout]
    out_ptr,       # *f32, [B, Cout, T]
    B: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Triton conv1d with stride=1, padding = (K-1)//2, fused bias and ReLU.
    Computes y[b, co, t] = ReLU( sum_{cin=0..Cin-1, k=0..K-1} x[b, cin, t - P + k] * w[co, cin*K + k] + b[co] )
    where P = (K - 1) // 2 (for K=5, P=2). Output length equals T.
    """
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_t_block = tl.program_id(2)

    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # accumulator
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # loop over input channels and kernel taps
    for cin in range(0, Cin):
        for k in tl.static_range(0, K):
            P = (K - 1) // 2  # compile-time constant for K=5 -> 2
            t_in = t_offsets - P + k
            in_bounds = (t_in >= 0) & (t_in < T) & mask_t

            # load x[b, cin, t_in]
            x_index = (pid_b * Cin + cin) * T + t_in
            x_val = tl.load(x_ptr + x_index, mask=in_bounds, other=0.0)

            # load w[co, cin*K + k]
            w_index = pid_co * (Cin * K) + cin * K + k
            w_val = tl.load(w_ptr + w_index)

            acc += x_val * w_val

    # add bias and apply ReLU
    b_val = tl.load(b_ptr + pid_co)
    acc = acc + b_val
    acc = tl.maximum(acc, 0.0)  # ReLU

    # store output y[b, co, t]
    out_index = (pid_b * Cout + pid_co) * T + t_offsets
    tl.store(out_ptr + out_index, acc, mask=mask_t)


@triton.jit
def add_h_to_x1_triton(
    h_ptr,         # *f32, [B, C1, T]
    x1_ptr,        # *f32, [B, C1, T]
    x1_out_ptr,    # *f32, [B, C1, T]
    B: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    ADD: tl.constexpr,  # 1 for add, 0 for subtract
    BLOCK_T: tl.constexpr,
):
    """
    Elementwise coupling update: x1_out = x1 + h (forward) or x1 - h (reverse).
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    index = (pid_b * C1 + pid_c) * T + t_offsets
    x1_val = tl.load(x1_ptr + index, mask=mask_t, other=0.0)
    h_val = tl.load(h_ptr + index, mask=mask_t, other=0.0)

    if ADD == 1:
        val = x1_val + h_val
    else:
        val = x1_val - h_val

    tl.store(x1_out_ptr + index, val, mask=mask_t)


@triton.jit
def concat_copy_first_half(
    x0_ptr,        # *f32, [B, C0, T]
    out_ptr,       # *f32, [B, C, T]
    B: tl.constexpr,
    C0: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Copy x0 [B, C0, T] into out [B, C, T] at columns [0:C0).
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x_index = (pid_b * C0 + pid_c) * T + t_offsets
    out_index = (pid_b * C + pid_c) * T + t_offsets
    val = tl.load(x0_ptr + x_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, val, mask=mask_t)


@triton.jit
def concat_copy_second_half(
    x1_ptr,        # *f32, [B, C1, T]
    out_ptr,       # *f32, [B, C, T]
    B: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Copy x1 [B, C1, T] into out [B, C, T] at columns [C0:C0+C1).
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x_index = (pid_b * C1 + pid_c) * T + t_offsets
    out_index = (pid_b * (C0 + C1) + (pid_c + C0)) * T + t_offsets
    val = tl.load(x1_ptr + x_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, val, mask=mask_t)


@triton.jit
def apply_mask_to_h_triton(
    h_ptr,         # *f32, [B, C, T]
    mask_ptr,      # *f32, [B, 1, T]
    h_out_ptr,     # *f32, [B, C, T]
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Elementwise h_out = h * mask, where mask is [B, 1, T] and broadcasts over channels.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    h_index = (pid_b * C + pid_c) * T + t_offsets
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    # mask_ptr is [B, 1, T]; broadcast along channel dimension
    mask_index = (pid_b * T) + t_offsets
    mask_val = tl.load(mask_ptr + mask_index, mask=mask_t, other=1.0)

    h_val = h_val * mask_val
    tl.store(h_out_ptr + h_index, h_val, mask=mask_t)


def _grid_1d(B, C, T, BLOCK_T):
    return (B, C, triton.cdiv(T, BLOCK_T))


def _pick_block_t(T):
    if T >= 4096:
        return 256
    elif T >= 2048:
        return 128
    elif T >= 1024:
        return 128
    else:
        return 64


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
):
    """
    Triton-only implementation of the residual coupling block.
    - Computes each transform using Triton conv1d + ReLU.
    - Applies mask in Triton.
    - Updates x1 via coupling in Triton.
    - Concatenates halves in Triton.
    """
    # Ensure CUDA and contiguous
    device = x.device
    if device.type != "cuda":
        # move to cuda if needed (evaluation uses GPU)
        x = x.to("cuda", non_blocking=True)
        x_mask = x_mask.to("cuda", non_blocking=True)

    B, C, T = x.shape
    half_channels = C // 2
    x = x.contiguous().to(torch.float32)
    x_mask = x_mask.contiguous().to(torch.float32)

    # Prepare transform lists
    transforms = [
        (transform_0_conv0_weight, transform_0_conv0_bias,
         transform_0_conv1_weight, transform_0_conv1_bias,
         transform_0_conv2_weight, transform_0_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias,
         transform_1_conv1_weight, transform_1_conv1_bias,
         transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias,
         transform_2_conv1_weight, transform_2_conv1_bias,
         transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_3_conv0_weight, transform_3_conv0_bias,
         transform_3_conv1_weight, transform_3_conv1_bias,
         transform_3_conv2_weight, transform_3_conv2_bias),
    ]

    # Fixed K for convs in this task
    K = 5
    BLOCK_T = _pick_block_t(T)

    if not reverse:
        # Forward: apply transforms sequentially; split x into x0 and x1
        for i in range(4):
            # Split into halves
            x0 = x[:, :half_channels, :].contiguous()
            x1 = x[:, half_channels:, :].contiguous()

            # Compute conv0 -> ReLU
            h0 = torch.empty((B, half_channels, T), dtype=torch.float32, device=device)
            grid0 = _grid_1d(B, half_channels, T, BLOCK_T)
            conv1d_triton_stride1_bias_relu[grid0](
                x0, transforms[i][0], transforms[i][1], h0,
                B, x0.shape[1], half_channels, T, K, BLOCK_T
            )
            h0 = h0  # ReLU already fused

            # Apply mask
            h0_masked = torch.empty_like(h0)
            grid_mask = _grid_1d(B, half_channels, T, BLOCK_T)
            apply_mask_to_h_triton[grid_mask](
                h0, x_mask, h0_masked, B, half_channels, T, BLOCK_T
            )

            # Update x1
            x1_out = torch.empty_like(x1)
            grid_add = _grid_1d(B, half_channels, T, BLOCK_T)
            add_h_to_x1_triton[grid_add](
                h0_masked, x1, x1_out, B, half_channels, T, ADD=1, BLOCK_T=BLOCK_T
            )

            # Concatenate halves: [x0, x1_out]
            out = torch.empty((B, C, T), dtype=torch.float32, device=device)
            grid_first = _grid_1d(B, half_channels, T, BLOCK_T)
            concat_copy_first_half[grid_first](x0, out, B, half_channels, C, T, BLOCK_T)
            grid_second = _grid_1d(B, half_channels, T, BLOCK_T)
            concat_copy_second_half[grid_second](x1_out, out, B, half_channels, half_channels, T, BLOCK_T)

            # Update x for next layer
            x = out
    else:
        # Reverse: apply in reverse order, subtract h
        for i in range(3, -1, -1):
            # Split into halves
            x0 = x[:, :half_channels, :].contiguous()
            x1 = x[:, half_channels:, :].contiguous()

            # Compute conv0 -> ReLU
            h0 = torch.empty((B, half_channels, T), dtype=torch.float32, device=device)
            grid0 = _grid_1d(B, half_channels, T, BLOCK_T)
            conv1d_triton_stride1_bias_relu[grid0](
                x0, transforms[i][0], transforms[i][1], h0,
                B, x0.shape[1], half_channels, T, K, BLOCK_T
            )
            h0 = h0  # ReLU already fused

            # Apply mask
            h0_masked = torch.empty_like(h0)
            grid_mask = _grid_1d(B, half_channels, T, BLOCK_T)
            apply_mask_to_h_triton[grid_mask](
                h0, x_mask, h0_masked, B, half_channels, T, BLOCK_T
            )

            # Update x1: x1 = x1 - h0
            x1_out = torch.empty_like(x1)
            grid_add = _grid_1d(B, half_channels, T, BLOCK_T)
            add_h_to_x1_triton[grid_add](
                h0_masked, x1, x1_out, B, half_channels, T, ADD=0, BLOCK_T=BLOCK_T
            )

            # Concatenate halves: [x0, x1_out]
            out = torch.empty((B, C, T), dtype=torch.float32, device=device)
            grid_first = _grid_1d(B, half_channels, T, BLOCK_T)
            concat_copy_first_half[grid_first](x0, out, B, half_channels, C, T, BLOCK_T)
            grid_second = _grid_1d(B, half_channels, T, BLOCK_T)
            concat_copy_second_half[grid_second](x1_out, out, B, half_channels, half_channels, T, BLOCK_T)

            # Update x for previous layer
            x = out

    # Final mask application (broadcast along channels), though in this task x_mask is ones.
    # Still invoke Triton for consistency.
    x = x.contiguous()
    x_mask_b = x_mask.contiguous()
    x_masked_out = torch.empty_like(x)
    grid_mask_final = _grid_1d(B, C, T, BLOCK_T)
    apply_mask_to_h_triton[grid_mask_final](x, x_mask_b, x_masked_out, B, C, T, BLOCK_T)
    return x_masked_out


class ModelNew(nn.Module):
    def forward(self, *args):
        # Unpack the same arguments as the original Model.forward
        # Expected args order:
        # x, x_mask, reverse, then 24 weight/bias tensors in order
        x = args[0]
        x_mask = args[1]
        reverse = args[2]
        # Build dict of tensors for clarity
        kwargs = dict(zip(
            ['transform_0_conv0_weight', 'transform_0_conv0_bias',
             'transform_0_conv1_weight', 'transform_0_conv1_bias',
             'transform_0_conv2_weight', 'transform_0_conv2_bias',
             'transform_1_conv0_weight', 'transform_1_conv0_bias',
             'transform_1_conv1_weight', 'transform_1_conv1_bias',
             'transform_1_conv2_weight', 'transform_1_conv2_bias',
             'transform_2_conv0_weight', 'transform_2_conv0_bias',
             'transform_2_conv1_weight', 'transform_2_conv1_bias',
             'transform_2_conv2_weight', 'transform_2_conv2_bias',
             'transform_3_conv0_weight', 'transform_3_conv0_bias',
             'transform_3_conv1_weight', 'transform_3_conv1_bias',
             'transform_3_conv2_weight', 'transform_3_conv2_bias'],
            list(args[3:])
        ))
        return run(x, x_mask, reverse, **kwargs)


def run(*args):
    return ModelNew()(*args)
