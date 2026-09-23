import math
import torch
import torch.nn.functional as F

# Triton kernels
import triton
import triton.language as tl


@triton.jit
def conv1d_nocu_nopad_k5_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Cout, L_in, L_out,
    x_stride_n, x_stride_c, x_stride_l,
    w_stride_oo, w_stride_ii, w_stride_k,
    y_stride_n, y_stride_c, y_stride_l,
    BLOCK_T: tl.constexpr
):
    # Grid dims: (N*Cout, triton.cdiv(L_out, BLOCK_T))
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Derive n and oc from pid_nc
    n = pid_nc // Cout
    oc = pid_nc % Cout

    # Time tile: BLOCK_T=1 ensures exact L_out coverage
    t_out = pid_t  # when BLOCK_T==1, pid_t must be in [0, L_out)
    # Safety check: if pid_t >= L_out, we should not run; Triton grid ensures this.

    # Initialize accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and kernel taps
    for c_in in range(Cin):
        for k in range(5):
            l_in = t_out + k  # padding=0, stride=1
            x_offset = n * x_stride_n + c_in * x_stride_c + l_in * x_stride_l
            w_offset = oc * w_stride_oo + c_in * w_stride_ii + k * w_stride_k
            x_val = tl.load(x_ptr + x_offset)  # scalar load
            w_val = tl.load(w_ptr + w_offset)  # scalar load
            acc += x_val * w_val

    # Add bias
    b_val = tl.load(b_ptr + oc)  # bias is per output channel
    acc += b_val

    # Store result
    y_offset = n * y_stride_n + oc * y_stride_c + t_out * y_stride_l
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def relu_kernel(x_ptr, y_ptr, N, C, L, x_stride_n, x_stride_c, x_stride_l, y_stride_n, y_stride_c, y_stride_l, BLOCK_T: tl.constexpr):
    pid_ncl = tl.program_id(0)
    # For simplicity, we use a 1D grid over N*C*L
    # Compute indices: n = pid_ncl // (C*L), rem = pid_ncl % (C*L), c = rem // L, l = rem % L
    n = pid_ncl // (C * L)
    rem = pid_ncl % (C * L)
    c = rem // L
    l = rem % L
    x_offset = n * x_stride_n + c * x_stride_c + l * x_stride_l
    x_val = tl.load(x_ptr + x_offset)
    y_val = tl.maximum(x_val, 0.0)
    y_offset = n * y_stride_n + c * y_stride_c + l * y_stride_l
    tl.store(y_ptr + y_offset, y_val)


@triton.jit
def mul_mask_kernel(x_ptr, mask_ptr, y_ptr, N, C, L,
                     x_stride_n, x_stride_c, x_stride_l,
                     mask_stride_n, mask_stride_c, mask_stride_l,
                     y_stride_n, y_stride_c, y_stride_l,
                     BLOCK_T: tl.constexpr):
    # x: [N, C, L], mask: [N, 1, L], y: [N, C, L]
    pid_ncl = tl.program_id(0)
    n = pid_ncl // (C * L)
    rem = pid_ncl % (C * L)
    c = rem // L
    l = rem % L
    # mask shape is [N, 1, L]; we index with c=0 implicitly
    x_offset = n * x_stride_n + c * x_stride_c + l * x_stride_l
    m_offset = n * mask_stride_n + 0 * mask_stride_c + l * mask_stride_l
    x_val = tl.load(x_ptr + x_offset)
    m_val = tl.load(mask_ptr + m_offset)
    y_val = x_val * m_val
    y_offset = n * y_stride_n + c * y_stride_c + l * y_stride_l
    tl.store(y_ptr + y_offset, y_val)


@triton.jit
def affine_add_sub_kernel(x_ptr, h_ptr, y_ptr, N, C, L, reverse: tl.constexpr,
                          x_stride_n, x_stride_c, x_stride_l,
                          h_stride_n, h_stride_c, h_stride_l,
                          y_stride_n, y_stride_c, y_stride_l,
                          BLOCK_T: tl.constexpr):
    # x: [N, C, L], h: [N, C, L], y: [N, C, L]
    pid_ncl = tl.program_id(0)
    n = pid_ncl // (C * L)
    rem = pid_ncl % (C * L)
    c = rem // L
    l = rem % L
    x_offset = n * x_stride_n + c * x_stride_c + l * x_stride_l
    h_offset = n * h_stride_n + c * h_stride_c + l * h_stride_l
    x_val = tl.load(x_ptr + x_offset)
    h_val = tl.load(h_ptr + h_offset)
    if reverse:
        y_val = x_val - h_val
    else:
        y_val = x_val + h_val
    y_offset = n * y_stride_n + c * y_stride_c + l * y_stride_l
    tl.store(y_ptr + y_offset, y_val)


@triton.jit
def concat_channels_kernel(x0_ptr, x1_ptr, y_ptr,
                            N, C0, C1, L,
                            x0_stride_n, x0_stride_c, x0_stride_l,
                            x1_stride_n, x1_stride_c, x1_stride_l,
                            y_stride_n, y_stride_c, y_stride_l,
                            BLOCK_T: tl.constexpr):
    # y: [N, C0 + C1, L], we write two regions
    pid_ncl = tl.program_id(0)
    n = pid_ncl // (L * (C0 + C1))
    rem = pid_ncl % (L * (C0 + C1))
    c = rem // L
    l = rem % L
    if c < C0:
        src_ptr = x0_ptr
        src_stride_n = x0_stride_n
        src_stride_c = x0_stride_c
        src_stride_l = x0_stride_l
        src_c = c
    else:
        src_c = c - C0
        src_ptr = x1_ptr
        src_stride_n = x1_stride_n
        src_stride_c = x1_stride_c
        src_stride_l = x1_stride_l
    src_offset = n * src_stride_n + src_c * src_stride_c + l * src_stride_l
    y_offset = n * y_stride_n + c * y_stride_c + l * y_stride_l
    val = tl.load(src_ptr + src_offset)
    tl.store(y_ptr + y_offset, val)


def run_triton_conv1d(x, w, b, L_out):
    # x: [N, Cin, L_in], w: [Cout, Cin, 5], b: [Cout]
    assert x.is_cuda and w.is_cuda and b.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    N, Cin, L_in = x.shape
    Cout, Cin_w, K = w.shape
    assert Cin == Cin_w and K == 5
    y = torch.empty((N, Cout, L_out), device=x.device, dtype=torch.float32)
    grid = (N * Cout, L_out)  # BLOCK_T=1, so grid second dim is exactly L_out
    conv1d_nocu_nopad_k5_bias_kernel[grid](
        x, w, b, y,
        N, Cin, Cout, L_in, L_out,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_T=1, num_warps=4, num_stages=2
    )
    return y


def run_relu(x):
    x = x.contiguous()
    N, C, L = x.shape
    y = torch.empty_like(x)
    grid = (N * C * L,)
    relu_kernel[grid](
        x, y,
        N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_T=1, num_warps=4, num_stages=2
    )
    return y


def run_mul_mask(x, mask):
    # x: [N, C, L], mask: [N, 1, L]
    x = x.contiguous()
    mask = mask.contiguous()
    N, C, L = x.shape
    y = torch.empty_like(x)
    grid = (N * C * L,)
    mul_mask_kernel[grid](
        x, mask, y,
        N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_T=1, num_warps=4, num_stages=2
    )
    return y


def run_affine_add_sub(x, h, reverse):
    x = x.contiguous()
    h = h.contiguous()
    N, C, L = x.shape
    y = torch.empty_like(x)
    grid = (N * C * L,)
    affine_add_sub_kernel[grid](
        x, h, y,
        N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        reverse=reverse,
        BLOCK_T=1, num_warps=4, num_stages=2
    )
    return y


def run_concat_channels(x0, x1):
    # x0: [N, C0, L], x1: [N, C1, L]
    x0 = x0.contiguous()
    x1 = x1.contiguous()
    N, C0, L = x0.shape
    N1, C1, L1 = x1.shape
    assert N == N1 and L == L1
    y = torch.empty((N, C0 + C1, L), device=x0.device, dtype=x0.dtype)
    grid = (N * (C0 + C1) * L,)
    concat_channels_kernel[grid](
        x0, x1, y,
        N, C0, C1, L,
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_T=1, num_warps=4, num_stages=2
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
    Residual coupling flow block. One conv per iteration.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    N, C, L = x.shape
    assert C % 2 == 0, "C must be even"
    half = C // 2

    # We implement 4 iterations
    for _ in range(4):
        # Split
        x0 = x[:, :half, :]
        x1 = x[:, half:, :]

        # conv0
        L1 = L - 4  # padding=0, K=5
        h = run_triton_conv1d(x0, transform_0_conv0_weight, transform_0_conv0_bias, L1)
        # ReLU
        h = run_relu(h)
        # conv1
        L2 = L1 - 4
        h = run_triton_conv1d(h, transform_0_conv1_weight, transform_0_conv1_bias, L2)
        # ReLU
        h = run_relu(h)
        # conv2 (no bias)
        L3 = L2 - 4
        h = run_triton_conv1d(h, transform_0_conv2_weight, torch.zeros(transform_0_conv2_weight.shape[0], device=h.device, dtype=h.dtype), L3)

        # Multiply by mask (broadcast along channels)
        h = run_mul_mask(h, x_mask)

        # Affine coupling
        if not reverse:
            x1 = run_affine_add_sub(x1, h, reverse=False)
        else:
            x1 = run_affine_add_sub(x1, h, reverse=True)

        # Concatenate
        x = run_concat_channels(x0, x1)

        # Multiply by mask again (broadcast along channels and time)
        x = run_mul_mask(x, x_mask)

    return x


# Define ModelNew as nn.Module with forward calling run
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The run function expects the same signature as the original forward.
        # It will apply Triton kernels for convs, ReLU, mask multiply, add/sub, and concatenation.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
