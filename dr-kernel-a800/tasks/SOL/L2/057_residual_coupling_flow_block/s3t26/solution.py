import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv1d_stride1_pad0_k5_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                                  N, Cin, L_in, Cout, L_out,
                                  x_sN, x_sC, x_sL,
                                  w_sO, w_sI, w_sK,
                                  b_sO,
                                  y_sN, y_sO, y_sL,
                                  rev_add_bias: tl.constexpr,  # 0 for forward, 1 for add b; we always pass 0 here, bias handled separately
                                  BLOCK_T: tl.constexpr):
    # 2D grid: (N*Cout, tiles along L_out)
    pid_oc = tl.program_id(0)
    n = pid_oc // Cout
    oc = pid_oc % Cout

    pid_tile = tl.program_id(1)
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L_out

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps
    # Note: Cin and K are runtime, loop in Python (Triton supports such loops)
    # For each input channel c_in and each tap k in [0..4), accumulate x[n, c_in, t_out - k] * w[oc, c_in, k]
    # We use masks to avoid out-of-range reads when padding=0.
    for c_in in range(Cin):
        # w[oc, c_in, :] vector
        # For k in 0..4:
        for k in range(5):
            # t_in = t_offsets - k
            t_in = t_offsets - k
            valid = (t_in >= 0) & (t_in < L_in) & mask_t
            # Compute x pointers: x[n, c_in, t_in]
            x_ptrs = x_ptr + n * x_sN + c_in * x_sC + t_in * x_sL
            x_vals = tl.load(x_ptrs, mask=valid, other=0.0)
            # w scalar: w[oc, c_in, k]
            w_val = tl.load(w_ptr + oc * w_sO + c_in * w_sI + k * w_sK)
            acc += x_vals * w_val

    # Add bias if requested
    if rev_add_bias:
        b_val = tl.load(b_ptr + oc * b_sO)
        acc += b_val

    # Store to y: y[n, oc, t_offsets]
    y_ptrs = y_ptr + n * y_sN + oc * y_sO + t_offsets * y_sL
    tl.store(y_ptrs, acc, mask=mask_t)


@triton.jit
def relu_kernel(x_ptr, y_ptr, N, C, L, sN, sC, sL, BLOCK_T: tl.constexpr):
    pid_nc = tl.program_id(0)  # over N*C
    pid_tile = tl.program_id(1)  # over tiles of L
    n = pid_nc // C
    c = pid_nc % C
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_ptrs = x_ptr + n * sN + c * sC + t_offsets * sL
    x_vals = tl.load(x_ptrs, mask=mask_t, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)
    y_ptrs = y_ptr + n * sN + c * sC + t_offsets * sL
    tl.store(y_ptrs, y_vals, mask=mask_t)


@triton.jit
def mask_mul_kernel(x_ptr, mask_ptr, y_ptr, N, C, L,
                    x_sN, x_sC, x_sL,
                    mask_sN, mask_sC, mask_sL,  # mask has shape [N, 1, L], so mask_sC should correspond to the channel dim of 1; but we only need L
                    y_sN, y_sC, y_sL,
                    BLOCK_T: tl.constexpr):
    pid_nc = tl.program_id(0)  # over N*C
    pid_tile = tl.program_id(1)  # over tiles of L
    n = pid_nc // C
    c = pid_nc % C
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_ptrs = x_ptr + n * x_sN + c * x_sC + t_offsets * x_sL
    mask_ptrs = mask_ptr + n * mask_sN + 0 * mask_sC + t_offsets * mask_sL  # channel dim is 1
    x_vals = tl.load(x_ptrs, mask=mask_t, other=0.0)
    mask_vals = tl.load(mask_ptrs, mask=mask_t, other=1.0)
    y_vals = x_vals * mask_vals
    y_ptrs = y_ptr + n * y_sN + c * y_sC + t_offsets * y_sL
    tl.store(y_ptrs, y_vals, mask=mask_t)


@triton.jit
def addsub_masked_kernel(x_ptr, h_ptr, y_ptr, N, C, L,
                         x_sN, x_sC, x_sL,
                         h_sN, h_sC, h_sL,
                         rev: tl.constexpr,  # 0 for add, 1 for subtract
                         BLOCK_T: tl.constexpr):
    pid_nc = tl.program_id(0)  # over N*C
    pid_tile = tl.program_id(1)  # over tiles of L
    n = pid_nc // C
    c = pid_nc % C
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_ptrs = x_ptr + n * x_sN + c * x_sC + t_offsets * x_sL
    h_ptrs = h_ptr + n * h_sN + c * h_sC + t_offsets * h_sL
    x_vals = tl.load(x_ptrs, mask=mask_t, other=0.0)
    h_vals = tl.load(h_ptrs, mask=mask_t, other=0.0)
    if rev:
        y_vals = x_vals - h_vals
    else:
        y_vals = x_vals + h_vals
    y_ptrs = y_ptr + n * y_sN + c * y_sC + t_offsets * y_sL
    tl.store(y_ptrs, y_vals, mask=mask_t)


@triton.jit
def channel_concat_kernel(x0_ptr, x1_ptr, y_ptr, N, C0, C1, L,
                          x0_sN, x0_sC, x0_sL,
                          x1_sN, x1_sC, x1_sL,
                          y_sN, y_sC, y_sL,
                          BLOCK_T: tl.constexpr):
    # y has channels C0 + C1
    pid_nc = tl.program_id(0)  # over N*(C0+C1)
    pid_tile = tl.program_id(1)  # over tiles of L
    totalC = C0 + C1
    n = pid_nc // totalC
    c = pid_nc % totalC
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    # Determine source tensor
    is_x0 = c < C0
    if is_x0:
        srcC = c
        src_ptr = x0_ptr + n * x0_sN + srcC * x0_sC + t_offsets * x0_sL
    else:
        srcC = c - C0
        src_ptr = x1_ptr + n * x1_sN + srcC * x1_sC + t_offsets * x1_sL

    y_ptrs = y_ptr + n * y_sN + c * y_sC + t_offsets * y_sL
    vals = tl.load(src_ptr, mask=mask_t, other=0.0)
    tl.store(y_ptrs, vals, mask=mask_t)


def _conv1d_stride1_pad0_k5(x, w, b=None):
    # x: [N, Cin, L_in], w: [Cout, Cin, 5], b: [Cout] or None
    assert x.is_cuda and w.is_cuda, "Triton conv requires CUDA tensors"
    assert x.dtype == torch.float32 and w.dtype == torch.float32, "Use float32 for Triton kernels"
    N, Cin, L_in = x.shape
    Cout, Cin_w, K = w.shape
    assert Cin == Cin_w, "Mismatch in input channels"
    assert K == 5, "Kernel size must be 5"
    L_out = L_in - 4  # padding=0
    y = torch.empty((N, Cout, L_out), device=x.device, dtype=torch.float32)

    # Choose tile size
    BLOCK_T = 128  # safe for most L_out; we will mask out-of-range
    grid = (N * Cout, triton.cdiv(L_out, BLOCK_T))
    # Launch kernel: we do not add bias here; conv1d in PyTorch default add bias; we add it separately.
    # We pass rev_add_bias=0 since bias is added in a separate step (we can pass b_ptr as None if no bias).
    conv1d_stride1_pad0_k5_kernel[grid](
        x, w, (b if b is not None else torch.empty(1, device=x.device, dtype=x.dtype)),  # dummy b_ptr when no bias
        y,
        N, Cin, L_in, Cout, L_out,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        (b.stride(0) if b is not None else 0),
        y.stride(0), y.stride(1), y.stride(2),
        rev_add_bias=0,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    if b is not None:
        # Add bias: y += b[oc] for each oc
        # We implement it as a simple elementwise add using PyTorch to keep robustness:
        y = y + b.unsqueeze(1).expand(N, Cout, L_out)
        # Note: To fully Triton, we could write another kernel adding vector b per oc, but this is simple and fast.
    return y


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
    Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    N, C, L = x.shape
    assert C % 2 == 0, "Number of channels must be even"
    half = C // 2

    # Kernel launch parameters
    BLOCK_T = 128  # safe default for tiles along time

    # Process 4 transforms sequentially
    for _ in range(4):
        # Split x into halves along channels
        x0 = x[:, :half, :]
        x1 = x[:, half:, :]

        # conv0: Cin=half, Cout=hidden(192), K=5, padding=0, bias=True
        h0 = _conv1d_stride1_pad0_k5(x0, transform_0_conv0_weight, transform_0_conv0_bias)
        # ReLU
        h0_relu = torch.empty_like(h0)
        grid_relu = (N * h0.shape[1], triton.cdiv(h0.shape[2], BLOCK_T))
        relu_kernel[grid_relu](
            h0, h0_relu,
            N, h0.shape[1], h0.shape[2],
            h0.stride(0), h0.stride(1), h0.stride(2),
            BLOCK_T=BLOCK_T, num_warps=4
        )

        # conv1: Cin=hidden(192), Cout=hidden(192), K=5, padding=0, bias=True
        h1 = _conv1d_stride1_pad0_k5(h0_relu, transform_0_conv1_weight, transform_0_conv1_bias)
        # ReLU
        h1_relu = torch.empty_like(h1)
        grid_relu2 = (N * h1.shape[1], triton.cdiv(h1.shape[2], BLOCK_T))
        relu_kernel[grid_relu2](
            h1, h1_relu,
            N, h1.shape[1], h1.shape[2],
            h1.stride(0), h1.stride(1), h1.stride(2),
            BLOCK_T=BLOCK_T, num_warps=4
        )

        # conv2: Cin=hidden(192), Cout=half, K=5, padding=0, bias=True (provided) or None
        h2 = _conv1d_stride1_pad0_k5(h1_relu, transform_0_conv2_weight, transform_0_conv2_bias)

        # Multiply by mask
        h2_masked = torch.empty_like(h2)
        grid_mask = (N * h2.shape[1], triton.cdiv(h2.shape[2], BLOCK_T))
        mask_mul_kernel[grid_mask](
            h2, x_mask,
            h2_masked,
            N, h2.shape[1], h2.shape[2],
            h2.stride(0), h2.stride(1), h2.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),  # mask is [N,1,L] -> treat channel stride as ignored (always 1)
            h2_masked.stride(0), h2_masked.stride(1), h2_masked.stride(2),
            BLOCK_T=BLOCK_T, num_warps=4
        )

        # Affine coupling: x1 = x1 + h2_masked (forward) or x1 = x1 - h2_masked (reverse)
        x1_new = torch.empty_like(x1)
        grid_addsub = (N * x1.shape[1], triton.cdiv(x1.shape[2], BLOCK_T))
        addsub_masked_kernel[grid_addsub](
            x1, h2_masked,
            x1_new,
            N, x1.shape[1], x1.shape[2],
            x1.stride(0), x1.stride(1), x1.stride(2),
            h2_masked.stride(0), h2_masked.stride(1), h2_masked.stride(2),
            rev=1 if reverse else 0,
            BLOCK_T=BLOCK_T, num_warps=4
        )

        # Concatenate [x0, x1_new] along channels
        y_half = torch.empty((N, C, x0.shape[2]), device=x.device, dtype=torch.float32)
        grid_concat = (N * (half + half), triton.cdiv(x0.shape[2], BLOCK_T))
        channel_concat_kernel[grid_concat](
            x0, x1_new,
            y_half,
            N, half, half, x0.shape[2],
            x0.stride(0), x0.stride(1), x0.stride(2),
            x1_new.stride(0), x1_new.stride(1), x1_new.stride(2),
            y_half.stride(0), y_half.stride(1), y_half.stride(2),
            BLOCK_T=BLOCK_T, num_warps=4
        )

        # Update x for next iteration
        x = y_half

    return x


class ModelNew(nn.Module):
    def forward(self, *args):
        # The run function expects the same signature as the original forward.
        # It will apply Triton kernels for convs, ReLU, mask multiply, add/sub, and concatenation.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
