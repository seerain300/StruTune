import math
import triton
import triton.language as tl


@triton.jit
def conv1d_stride1_pad0_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Cout, Lin, Lout,
    x_stride_n, x_stride_c, x_stride_t,
    w_stride_oc, w_stride_ic, w_stride_k,
    y_stride_n, y_stride_oc, y_stride_t,
    apply_relu: tl.constexpr, has_bias: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # Each program computes a slice for one (n, oc) over a tile of Lout
    pid = tl.program_id(0)
    n = pid // Cout
    oc = pid % Cout

    t_offsets = tl.arange(0, BLOCK_T)
    mask_t = t_offsets < Lout

    # Accumulator for the tile
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps (kernel_size=5, stride=1, padding=0)
    for ic in range(0, Cin):
        for k in range(0, 5):
            t_in = t_offsets - k  # valid since t_offsets in [0, Lout-1], k in [0,4]
            x_addr = x_ptr + n * x_stride_n + ic * x_stride_c + t_in * x_stride_t
            x_val = tl.load(x_addr, mask=mask_t, other=0.0)
            w_addr = w_ptr + oc * w_stride_oc + ic * w_stride_ic + k * w_stride_k
            w_val = tl.load(w_addr)  # scalar float32
            acc += x_val * w_val

    if has_bias:
        b_val = tl.load(b_ptr + oc)
        acc += b_val

    if apply_relu:
        acc = tl.maximum(acc, 0.0)

    y_addr = y_ptr + n * y_stride_n + oc * y_stride_oc + t_offsets * y_stride_t
    tl.store(y_addr, acc, mask=mask_t)


@triton.jit
def relu_kernel(x_ptr, y_ptr, N, C, L,
                x_stride_n, x_stride_c, x_stride_t,
                y_stride_n, y_stride_c, y_stride_t,
                BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)  # iterate over N*C
    n = pid // C
    c = pid % C

    t_offsets = tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_addr = x_ptr + n * x_stride_n + c * x_stride_c + t_offsets * x_stride_t
    y_addr = y_ptr + n * y_stride_n + c * y_stride_c + t_offsets * y_stride_t

    x_val = tl.load(x_addr, mask=mask_t, other=0.0)
    y_val = tl.maximum(x_val, 0.0)
    tl.store(y_addr, y_val, mask=mask_t)


@triton.jit
def mul_mask_kernel(x_ptr, mask_ptr, y_ptr,
                     N, C, L,
                     x_stride_n, x_stride_c, x_stride_t,
                     mask_stride_n, mask_stride_c, mask_stride_t,
                     y_stride_n, y_stride_c, y_stride_t,
                     BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)  # iterate over N*C
    n = pid // C
    c = pid % C

    t_offsets = tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_addr = x_ptr + n * x_stride_n + c * x_stride_c + t_offsets * x_stride_t
    m_addr = mask_ptr + n * mask_stride_n + 0 * mask_stride_c + t_offsets * mask_stride_t
    y_addr = y_ptr + n * y_stride_n + c * y_stride_c + t_offsets * y_stride_t

    x_val = tl.load(x_addr, mask=mask_t, other=0.0)
    m_val = tl.load(m_addr, mask=mask_t, other=1.0)
    y_val = x_val * m_val
    tl.store(y_addr, y_val, mask=mask_t)


@triton.jit
def add_sub_affine_kernel(x_ptr, h_ptr, y_ptr, N, C, L, add_flag: tl.constexpr,
                          x_stride_n, x_stride_c, x_stride_t,
                          h_stride_n, h_stride_c, h_stride_t,
                          y_stride_n, y_stride_c, y_stride_t,
                          BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)  # iterate over N*C
    n = pid // C
    c = pid % C

    t_offsets = tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_addr = x_ptr + n * x_stride_n + c * x_stride_c + t_offsets * x_stride_t
    h_addr = h_ptr + n * h_stride_n + c * h_stride_c + t_offsets * h_stride_t
    y_addr = y_ptr + n * y_stride_n + c * y_stride_c + t_offsets * y_stride_t

    x_val = tl.load(x_addr, mask=mask_t, other=0.0)
    h_val = tl.load(h_addr, mask=mask_t, other=0.0)
    res = x_val + h_val if add_flag else x_val - h_val
    tl.store(y_addr, res, mask=mask_t)


@triton.jit
def concat_channels_kernel(x0_ptr, x1_ptr, y_ptr,
                            N, C0, C1, L,
                            x0_stride_n, x0_stride_c, x0_stride_t,
                            x1_stride_n, x1_stride_c, x1_stride_t,
                            y_stride_n, y_stride_c, y_stride_t,
                            BLOCK_T: tl.constexpr):
    # Writes [N, C0 + C1, L] concatenating along channel dim
    pid = tl.program_id(0)  # iterate over N*(C0+C1)
    nc = pid // 1  # invalid; use separate launch per n
    # We'll launch grid over (N, C0+C1), pid = n*(C0+C1) + c_total
    n = pid // (C0 + C1)
    c_total = pid % (C0 + C1)

    t_offsets = tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    # Determine source
    if c_total < C0:
        src_ptr = x0_ptr
        src_stride_n = x0_stride_n
        src_stride_c = x0_stride_c
        src_stride_t = x0_stride_t
    else:
        src_ptr = x1_ptr
        src_stride_n = x1_stride_n
        src_stride_c = x1_stride_c
        src_stride_t = x1_stride_t
        c_src = c_total - C0

    src_addr = src_ptr + n * src_stride_n + c_src * (src_stride_c) + t_offsets * src_stride_t
    dst_addr = y_ptr + n * y_stride_n + c_total * y_stride_c + t_offsets * y_stride_t

    vals = tl.load(src_addr, mask=mask_t, other=0.0)
    tl.store(dst_addr, vals, mask=mask_t)


def triton_conv1d_stride1_pad0(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, apply_relu: bool = False) -> torch.Tensor:
    """
    x: [N, Cin, L_in], w: [Cout, Cin, 5], b: [Cout] or None
    returns y: [N, Cout, L_out] with L_out = L_in - 4
    All tensors should be CUDA and float32.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float32 and w.dtype == torch.float32, "Use float32 for computation"
    N, Cin, L_in = x.shape
    Cout, Cin_w, K = w.shape
    assert Cin == Cin_w and K == 5, "Weight shape must be [Cout, Cin, 5]"
    L_out = L_in - 4
    if L_out <= 0:
        # Handle degenerate case: no output time
        return torch.empty((N, Cout, 0), device=x.device, dtype=x.dtype)
    y = torch.empty((N, Cout, L_out), device=x.device, dtype=x.dtype)

    # Strides
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    w_stride_oc, w_stride_ic, w_stride_k = w.stride()
    y_stride_n, y_stride_oc, y_stride_t = y.stride()

    # Choose tile along time; ensure grid is exact
    BLOCK_T = 128
    grid = (N * Cout,)

    has_bias = b is not None
    b_ptr = b if has_bias else torch.empty(1, device=x.device, dtype=x.dtype)

    conv1d_stride1_pad0_kernel[grid](
        x, w, b_ptr, y,
        N, Cin, Cout, L_in, L_out,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_oc, w_stride_ic, w_stride_k,
        y_stride_n, y_stride_oc, y_stride_t,
        apply_relu=apply_relu, has_bias=has_bias,
        BLOCK_T=BLOCK_T, num_warps=4, num_stages=2
    )
    return y


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    N, C, L = x.shape
    y = torch.empty_like(x)
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128
    grid = (N * C,)
    relu_kernel[grid](x, y, N, C, L, x_stride_n, x_stride_c, x_stride_t, y_stride_n, y_stride_c, y_stride_t, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2)
    return y


def triton_mul_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [N, C, L], mask: [N, 1, L] (float32), broadcast across channels
    """
    assert x.is_cuda and mask.is_cuda, "Inputs must be CUDA tensors"
    N, C, L = x.shape
    y = torch.empty_like(x)
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    m_stride_n, m_stride_c, m_stride_t = mask.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128
    grid = (N * C,)
    mul_mask_kernel[grid](x, mask, y, N, C, L, x_stride_n, x_stride_c, x_stride_t, m_stride_n, m_stride_c, m_stride_t, y_stride_n, y_stride_c, y_stride_t, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2)
    return y


def triton_add_sub_affine(x: torch.Tensor, h: torch.Tensor, add_flag: bool) -> torch.Tensor:
    N, C, L = x.shape
    y = torch.empty_like(x)
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    h_stride_n, h_stride_c, h_stride_t = h.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128
    grid = (N * C,)
    add_sub_affine_kernel[grid](x, h, y, N, C, L, add_flag, x_stride_n, x_stride_c, x_stride_t, h_stride_n, h_stride_c, h_stride_t, y_stride_n, y_stride_c, y_stride_t, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2)
    return y


def triton_concat_channels(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    x0: [N, C0, L], x1: [N, C1, L], returns y: [N, C0+C1, L]
    """
    assert x0.is_cuda and x1.is_cuda, "Inputs must be CUDA tensors"
    N, C0, L = x0.shape
    N1, C1, L1 = x1.shape
    assert N == N1 and L == L1, "Shapes must align along N and L"
    y = torch.empty((N, C0 + C1, L), device=x0.device, dtype=x0.dtype)
    x0_stride_n, x0_stride_c, x0_stride_t = x0.stride()
    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    grid = (N * (C0 + C1),)
    # We need c_src for x1: c_src = c_total - C0
    concat_channels_kernel[grid](x0, x1, y, N, C0, C1, L, x0_stride_n, x0_stride_c, x0_stride_t, x1_stride_n, x1_stride_c, x1_stride_t, y_stride_n, y_stride_c, y_stride_t, BLOCK_T=128, num_warps=4, num_stages=2)
    return y


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                reverse: bool,
                # 4 transforms
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
                transform_3_conv2_bias: torch.Tensor):
        """
        Triton-optimized forward that applies 4 sequential transforms:
        For each transform:
        - Split x into x0 (first half channels) and x1 (second half channels).
        - conv0 -> ReLU -> conv1 -> ReLU -> conv2
        - Multiply by x_mask
        - Affine coupling on x1: +h (forward) or -h (reverse)
        - Concatenate [x0, x1] and multiply by x_mask again.
        """
        assert x.is_cuda and x.dtype == torch.float32, "x must be CUDA float32"
        N, C, L = x.shape
        half = C // 2
        assert C == 192 and half == 96, "This implementation assumes C=192, half=96 as in the original code"

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

        # Iterate transforms
        for i, (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b) in enumerate(transforms):
            # Split halves
            x0 = x[:, :half, :]
            x1 = x[:, half:, :]

            # conv0: [N, 96, L] -> [N, 192, L-4]
            h0 = triton_conv1d_stride1_pad0(x0, conv0_w, conv0_b, apply_relu=False)
            # ReLU
            h0 = triton_relu(h0)

            # conv1: [N, 192, L-4] -> [N, 192, (L-4)-4]
            h1 = triton_conv1d_stride1_pad0(h0, conv1_w, conv1_b, apply_relu=False)
            h1 = triton_relu(h1)

            # conv2: [N, 192, (L-8)] -> [N, 96, (L-8)-4]
            h2 = triton_conv1d_stride1_pad0(h1, conv2_w, conv2_b, apply_relu=False)

            # Multiply by mask (broadcast over channels)
            h2 = triton_mul_mask(h2, x_mask)

            # Affine coupling: x1 = x1 + h2 (forward) or x1 = x1 - h2 (reverse)
            if reverse:
                x1 = triton_add_sub_affine(x1, h2, add_flag=False)
            else:
                x1 = triton_add_sub_affine(x1, h2, add_flag=True)

            # Concatenate [x0, x1] along channels
            x = triton_concat_channels(x0, x1)

            # Multiply by mask (broadcast)
            x = triton_mul_mask(x, x_mask)

        return x


def run(*args):
    return ModelNew()(*args)
