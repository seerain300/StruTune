import math
import torch
import torch.nn.functional as F

# Triton imports
import triton
import triton.language as tl


@triton.jit
def conv1d_stride1_pad0_k5_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Cout, L_in, L_out,
    # Strides (in elements)
    x_sN, x_sC, x_sL,
    w_sCo, w_sCi, w_sK,
    y_sN, y_sC, y_sL,
    # Grid meta
    BLOCK_T: tl.constexpr,
):
    # program ids
    pid_nc = tl.program_id(0)  # over N*Cout
    pid_tile = tl.program_id(1)  # over tiles of L_out

    # compute n and oc from pid_nc
    n = pid_nc // Cout
    oc = pid_nc % Cout

    # time offsets for this tile
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L_out

    # accumulator vector for this oc
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # loop over input channels and kernel taps
    for ci in range(0, Cin):
        for k in range(0, 5):  # kernel_size=5
            # compute input time index
            t_in = t_offsets - k  # valid for k in [0, 4], with t_offsets in [0, L_out-1]
            # mask to avoid out-of-bounds when t_in < 0 (padding=0 semantics: zeros)
            valid = (t_in >= 0) & (t_in < L_in) & mask_t

            # pointers to x[n, ci, t_in]
            x_ptrs = x_ptr + n * x_sN + ci * x_sC + t_in * x_sL
            x_vals = tl.load(x_ptrs, mask=valid, other=0.0)

            # weight scalar for (oc, ci, k)
            w_ptr_val = w_ptr + oc * w_sCo + ci * w_sCi + k * w_sK
            w_val = tl.load(w_ptr_val)  # scalar

            # accumulate
            acc += x_vals * w_val

    # add bias if provided
    if b_ptr != 0:
        b_val = tl.load(b_ptr + oc)
        acc += b_val

    # store to y[n, oc, t_offsets]
    y_ptrs = y_ptr + n * y_sN + oc * y_sC + t_offsets * y_sL
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
def mul_mask_kernel(x_ptr, mask_ptr, y_ptr, N, C, L, sN, sC, sL, msN, msC, msL, BLOCK_T: tl.constexpr):
    pid_nc = tl.program_id(0)  # over N*C
    pid_tile = tl.program_id(1)  # over tiles of L
    n = pid_nc // C
    c = pid_nc % C
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L
    x_ptrs = x_ptr + n * sN + c * sC + t_offsets * sL
    m_ptrs = mask_ptr + n * msN + 0 * msC + t_offsets * msL  # mask has shape [N,1,L]
    x_vals = tl.load(x_ptrs, mask=mask_t, other=0.0)
    m_vals = tl.load(m_ptrs, mask=mask_t, other=1.0)
    y_vals = x_vals * m_vals
    y_ptrs = y_ptr + n * sN + c * sC + t_offsets * sL
    tl.store(y_ptrs, y_vals, mask=mask_t)


@triton.jit
def addsub_affine_kernel(x_ptr, h_ptr, y_ptr, N, C, L, sN, sC, sL, rev: tl.constexpr, BLOCK_T: tl.constexpr):
    # x is "other half", h is transformed half. y = x + h if rev==0 else y = x - h
    pid_nc = tl.program_id(0)  # over N*C
    pid_tile = tl.program_id(1)  # over tiles of L
    n = pid_nc // C
    c = pid_nc % C
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L
    x_ptrs = x_ptr + n * sN + c * sC + t_offsets * sL
    h_ptrs = h_ptr + n * sN + c * sC + t_offsets * sL
    x_vals = tl.load(x_ptrs, mask=mask_t, other=0.0)
    h_vals = tl.load(h_ptrs, mask=mask_t, other=0.0)
    if rev:
        y_vals = x_vals - h_vals
    else:
        y_vals = x_vals + h_vals
    y_ptrs = y_ptr + n * sN + c * sC + t_offsets * sL
    tl.store(y_ptrs, y_vals, mask=mask_t)


@triton.jit
def channel_concat_kernel(x0_ptr, x1_ptr, y_ptr, N, C0, C1, L, x0_sN, x0_sC, x0_sL, x1_sN, x1_sC, x1_sL, y_sN, y_sC, y_sL, BLOCK_T: tl.constexpr):
    # y has channels C0+C1; we copy x0[:, :, :] into y[:, :C0, :], and x1[:, :, :] into y[:, C0:, :]
    # 1D grid over N*(C0+C1) and tiles over L
    pid_nc = tl.program_id(0)  # over N*(C0+C1)
    pid_tile = tl.program_id(1)  # over tiles of L
    totalC = C0 + C1
    n = pid_nc // totalC
    c = pid_nc % totalC
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    # determine source tensor based on c
    # c in [0, C0-1] comes from x0, otherwise from x1
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


def conv1d_stride1_pad0_k5_triton(x, w, b=None):
    """
    x: [N, Cin, L_in] float32 tensor on CUDA
    w: [Cout, Cin, 5] float32 tensor on CUDA
    b: [Cout] or None
    returns y: [N, Cout, L_out], L_out = L_in - 4
    """
    assert x.is_cuda and w.is_cuda, "Triton conv requires CUDA tensors"
    assert x.dtype == torch.float32 and w.dtype == torch.float32, "Use float32 for Triton kernels"
    N, Cin, L_in = x.shape
    Cout, Cin_w, K = w.shape
    assert Cin == Cin_w and K == 5, "Kernel size must be 5"
    L_out = L_in - 4
    y = torch.empty((N, Cout, L_out), device=x.device, dtype=torch.float32)

    x_sN, x_sC, x_sL = x.stride()
    w_sCo, w_sCi, w_sK = w.stride()
    y_sN, y_sC, y_sL = y.stride()

    BLOCK_T = 128
    grid = (N * Cout, triton.cdiv(L_out, BLOCK_T))
    conv1d_stride1_pad0_k5_kernel[grid](
        x, w, (b if b is not None else 0), y,
        N, Cin, Cout, L_in, L_out,
        x_sN, x_sC, x_sL,
        w_sCo, w_sCi, w_sK,
        y_sN, y_sC, y_sL,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return y


def relu_triton(x):
    N, C, L = x.shape
    y = torch.empty_like(x)
    sN, sC, sL = x.stride()
    BLOCK_T = 128
    grid = (N * C, triton.cdiv(L, BLOCK_T))
    relu_kernel[grid](
        x, y, N, C, L, sN, sC, sL,
        BLOCK_T=BLOCK_T, num_warps=4
    )
    return y


def mul_mask_triton(x, mask):
    """
    x: [N, C, L], mask: [N, 1, L]
    returns y = x * mask
    """
    assert x.is_cuda and mask.is_cuda, "Triton requires CUDA tensors"
    assert x.dtype == torch.float32 and mask.dtype == torch.float32
    N, C, L = x.shape
    msN, msC, msL = mask.shape
    assert msN == N and msC == 1 and msL == L, "mask must be [N, 1, L]"
    y = torch.empty_like(x)
    sN, sC, sL = x.stride()
    msN_s, msC_s, msL_s = mask.stride()
    BLOCK_T = 128
    grid = (N * C, triton.cdiv(L, BLOCK_T))
    mul_mask_kernel[grid](
        x, mask, y, N, C, L, sN, sC, sL, msN_s, msC_s, msL_s,
        BLOCK_T=BLOCK_T, num_warps=4
    )
    return y


def addsub_affine_triton(x, h, reverse=False):
    """
    x: [N, C, L], h: [N, C, L]
    returns y = x + h if reverse=False else y = x - h
    """
    assert x.is_cuda and h.is_cuda, "Triton requires CUDA tensors"
    assert x.dtype == torch.float32 and h.dtype == torch.float32
    N, C, L = x.shape
    y = torch.empty_like(x)
    sN, sC, sL = x.stride()
    BLOCK_T = 128
    grid = (N * C, triton.cdiv(L, BLOCK_T))
    addsub_affine_kernel[grid](
        x, h, y, N, C, L, sN, sC, sL, reverse,
        BLOCK_T=BLOCK_T, num_warps=4
    )
    return y


def channel_concat_triton(x0, x1):
    """
    x0: [N, C0, L], x1: [N, C1, L]
    returns y: [N, C0+C1, L]
    """
    assert x0.is_cuda and x1.is_cuda
    assert x0.dtype == torch.float32 and x1.dtype == torch.float32
    N, C0, L = x0.shape
    N1, C1, L1 = x1.shape
    assert N == N1 and L == L1, "x0 and x1 must have same N and L"
    totalC = C0 + C1
    y = torch.empty((N, totalC, L), device=x0.device, dtype=torch.float32)
    x0_sN, x0_sC, x0_sL = x0.stride()
    x1_sN, x1_sC, x1_sL = x1.stride()
    y_sN, y_sC, y_sL = y.stride()
    BLOCK_T = 128
    grid = (N * totalC, triton.cdiv(L, BLOCK_T))
    channel_concat_kernel[grid](
        x0, x1, y, N, C0, C1, L, x0_sN, x0_sC, x0_sL, x1_sN, x1_sC, x1_sL, y_sN, y_sC, y_sL,
        BLOCK_T=BLOCK_T, num_warps=4
    )
    return y


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
    Triton-optimized forward that mimics the original behavior:
    - Apply 4 transforms sequentially.
    - Each transform: split x into halves, conv->ReLU->conv->ReLU->conv, mask, affine on x1, concat, mask.
    - All heavy ops are implemented via Triton kernels.
    """
    # Number of channels and half
    C = x.shape[1]
    half_channels = C // 2

    # Perform 4 transforms
    for t in range(4):
        # Split current x into x0 and x1
        x0 = x[:, :half_channels, :]
        x1 = x[:, half_channels:, :]

        # conv0: Cin=half_channels (96), Cout=hidden (192), K=5
        h = conv1d_stride1_pad0_k5_triton(x0, transform_0_conv0_weight, transform_0_conv0_bias)
        # ReLU
        h = relu_triton(h)
        # conv1: Cin=hidden (192), Cout=hidden (192), K=5
        h = conv1d_stride1_pad0_k5_triton(h, transform_0_conv1_weight, transform_0_conv1_bias)
        # ReLU
        h = relu_triton(h)
        # conv2: Cin=hidden (192), Cout=half_channels (96), K=5
        # conv2 has no bias in the original code; pass zeros
        if transform_0_conv2_bias is None:
            b2 = torch.zeros(half_channels, device=h.device, dtype=h.dtype)
        else:
            b2 = transform_0_conv2_bias
        h = conv1d_stride1_pad0_k5_triton(h, transform_0_conv2_weight, b2)

        # Multiply by mask
        h = mul_mask_triton(h, x_mask)

        # Affine coupling
        if not reverse:
            x1 = addsub_affine_triton(x1, h, reverse=False)
        else:
            x1 = addsub_affine_triton(x1, h, reverse=True)

        # Concatenate back
        x = channel_concat_triton(x0, x1)

        # Apply mask to output (broadcast [N,1,L] over channels)
        x = mul_mask_triton(x, x_mask)

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The original run signature expects:
        # (x, x_mask, reverse, conv weights/biases for 4 transforms).
        # We will invoke the Triton-enabled run function to perform all computations.
        # Note: This function mirrors the original forward logic, using Triton for all heavy ops.
        # The entry point must be ModelNew.forward. The harness will provide all required args.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
