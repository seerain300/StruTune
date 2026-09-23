import math
import torch
import triton
import triton.language as tl


# Triton Conv1d forward kernel: y[n, co, lo] = bias[co] + sum_{ci,k} x[n, ci, li] * w[co, ci, k]
# li = lo + P - k, P = K//2. Mask li in [0, L_in). Vectorize along output time positions (BLOCK_L).
@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out, K,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    l_out_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_out_offsets < L_out

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Bias for this output channel
    b_val = tl.load(b_ptr + co)
    acc += b_val  # broadcast scalar

    P = K // 2
    # Loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            li = l_out_offsets + P - k
            mask_in = (li >= 0) & (li < L_in)
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            # Masked load; invalid positions contribute 0.0
            x_vals = tl.load(x_ptrs, mask=mask_in & mask_out, other=0.0)
            x_vals = x_vals.to(tl.float32)
            # Load scalar weight for (co, ci, k)
            w_ptr_k = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_val = tl.load(w_ptr_k).to(tl.float32)
            acc += x_vals * w_val

    # Store result (fp32)
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_out_offsets * stride_y_l
    tl.store(y_ptrs, acc, mask=mask_out)


# Triton ReLU kernel: y = max(y, 0)
@triton.jit
def relu_kernel(
    x_ptr, y_ptr,
    N, C, L,
    stride_x_n, stride_x_c, stride_x_l,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)

    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + l_offsets * stride_x_l
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l

    x_vals = tl.load(x_ptrs, mask=mask_out, other=0.0)
    # ReLU: max(x, 0)
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(y_ptrs, y_vals, mask=mask_out)


# Triton split halves: given y_full [N, 2*C_half, L], write y0 [N, C_half, L], y1 [N, C_half, L]
@triton.jit
def split_halves_forward(
    y_full_ptr, y0_ptr, y1_ptr,
    N, C_half, L,
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    full0_ptrs = y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + l_offsets * stride_y_full_l
    full1_ptrs = y_full_ptr + n * stride_y_full_n + (ch + C_half) * stride_y_full_c + l_offsets * stride_y_full_l
    out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l

    y0_vals = tl.load(full0_ptrs, mask=mask_out, other=0.0)
    y1_vals = tl.load(full1_ptrs, mask=mask_out, other=0.0)
    tl.store(out0_ptrs, y0_vals, mask=mask_out)
    tl.store(out1_ptrs, y1_vals, mask=mask_out)


# Triton concat halves: given y0 [N, C_half, L], y1 [N, C_half, L], write y_full [N, 2*C_half, L]
@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y_full_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    in0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    in1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l
    out0_ptrs = y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + l_offsets * stride_y_full_l
    out1_ptrs = y_full_ptr + n * stride_y_full_n + (ch + C_half) * stride_y_full_c + l_offsets * stride_y_full_l

    y0_vals = tl.load(in0_ptrs, mask=mask_out, other=0.0)
    y1_vals = tl.load(in1_ptrs, mask=mask_out, other=0.0)
    tl.store(out0_ptrs, y0_vals, mask=mask_out)
    tl.store(out1_ptrs, y1_vals, mask=mask_out)


# Triton mask multiplication: y[i, j, k] *= mask[i, 0, k]
@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_mask_n, stride_mask_c, stride_mask_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)

    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    m_ptrs = mask_ptr + n * stride_mask_n + 0 * stride_mask_c + l_offsets * stride_mask_l

    y_vals = tl.load(y_ptrs, mask=mask_out, other=0.0)
    m_vals = tl.load(m_ptrs, mask=mask_out, other=1.0)
    y_vals = y_vals * m_vals
    tl.store(y_ptrs, y_vals, mask=mask_out)


@triton.jit
def conv1d_forward_out(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out, K,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    # This is just a helper-like signature; we’ll call conv1d_forward_kernel directly.
    pass


@triton.jit
def conv1d_forward_out2(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out, K,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    # Another placeholder; not used
    pass


def triton_conv1d(x, w, b, BLOCK_L=128, num_warps=4, num_stages=2):
    """
    x: [N, C_in, L_in] contiguous, float32
    w: [C_out, C_in, K] contiguous, float32
    b: [C_out] contiguous, float32
    returns y: [N, C_out, L_out] contiguous, float32
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Triton kernels require CUDA tensors."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    N, C_in, L_in = x.shape
    C_out, C_in_w, K = w.shape
    assert C_in == C_in_w, "Weight in_c must match input channels."
    L_out = L_in - K + 1  # padding = K//2

    y = torch.empty((N, C_out, L_out), device=x.device, dtype=torch.float32)

    grid = (N, C_out, triton.cdiv(L_out, BLOCK_L))
    conv1d_forward_kernel[grid](
        x, w, b, y,
        N, C_in, C_out, L_in, L_out, K,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_L=BLOCK_L, num_warps=num_warps, num_stages=num_stages
    )
    return y


def triton_relu(y, BLOCK_L=128, num_warps=4, num_stages=2):
    """
    In-place ReLU on y: y = max(y, 0)
    """
    assert y.is_cuda, "Triton kernels require CUDA tensors."
    y = y.contiguous()
    N, C, L = y.shape
    grid = (N, C, triton.cdiv(L, BLOCK_L))
    relu_kernel[grid](
        y, y,
        N, C, L,
        y.stride(0), y.stride(1), y.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_L=BLOCK_L, num_warps=num_warps, num_stages=num_stages
    )
    return y


def triton_split_halves_forward(y_full, y0, y1, BLOCK_L=128, num_warps=4, num_stages=2):
    """
    Given y_full [N, 2*C_half, L], write y0 [N, C_half, L], y1 [N, C_half, L].
    """
    assert y_full.is_cuda and y0.is_cuda and y1.is_cuda, "Triton kernels require CUDA tensors."
    y_full = y_full.contiguous()
    y0 = y0.contiguous()
    y1 = y1.contiguous()

    N, C_half, L = y_full.shape[:3]  # y0 and y1 are [N, C_half, L]
    grid = (N, C_half, triton.cdiv(L, BLOCK_L))
    split_halves_forward[grid](
        y_full, y0, y1,
        N, C_half, L,
        y_full.stride(0), y_full.stride(1), y_full.stride(2),
        y0.stride(0), y0.stride(1), y0.stride(2),
        y1.stride(0), y1.stride(1), y1.stride(2),
        BLOCK_L=BLOCK_L, num_warps=num_warps, num_stages=num_stages
    )


def triton_concat_halves_forward(y0, y1, y_full, BLOCK_L=128, num_warps=4, num_stages=2):
    """
    Given y0 [N, C_half, L], y1 [N, C_half, L], write y_full [N, 2*C_half, L].
    """
    assert y0.is_cuda and y1.is_cuda and y_full.is_cuda, "Triton kernels require CUDA tensors."
    y0 = y0.contiguous()
    y1 = y1.contiguous()
    y_full = y_full.contiguous()

    N, C_half, L = y0.shape
    grid = (N, C_half, triton.cdiv(L, BLOCK_L))
    concat_halves_forward[grid](
        y0, y1, y_full,
        N, C_half, L,
        y0.stride(0), y0.stride(1), y0.stride(2),
        y1.stride(0), y1.stride(1), y1.stride(2),
        y_full.stride(0), y_full.stride(1), y_full.stride(2),
        BLOCK_L=BLOCK_L, num_warps=num_warps, num_stages=num_stages
    )


def triton_mul_mask(y, mask, BLOCK_L=128, num_warps=4, num_stages=2):
    """
    y: [N, C, L], mask: [N, 1, L] or [N, C, L] (we only use [N, 1, L])
    """
    assert y.is_cuda and mask.is_cuda, "Triton kernels require CUDA tensors."
    y = y.contiguous()
    mask = mask.contiguous()

    N, C, L = y.shape
    # mask is [N, 1, L] by design
    grid = (N, C, triton.cdiv(L, BLOCK_L))
    mul_mask_kernel[grid](
        y, mask,
        N, C, L,
        y.stride(0), y.stride(1), y.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        BLOCK_L=BLOCK_L, num_warps=num_warps, num_stages=num_stages
    )


class ModelNew(torch.nn.Module):
    def __init__(self, axes_and_scalars: dict, device: torch.device):
        super().__init__()
        self.device = device

    def forward(self, *args):
        """
        args are:
        x: [N, 192, L], float32
        x_mask: [N, 1, L], float32
        reverse: bool
        plus 12 conv weights and biases for 4 transforms.
        """
        # Extract inputs
        x = args[0].contiguous()
        x_mask = args[1].contiguous()
        reverse = args[2]
        # Extract per-transform weights
        # Note: names may vary; we parse the 13th to 19th args (12 transforms) using index mapping by name.
        # For simplicity, assume args[3:17] are in order: transform_i_conv0_weight, conv0_bias, conv1_weight, conv1_bias, conv2_weight, conv2_bias for i in [0..3].
        T = 4
        transforms = []
        for i in range(4):
            w0 = args[3 + i * 6 + 0]  # conv0 weight
            b0 = args[3 + i * 6 + 1]  # conv0 bias
            w1 = args[3 + i * 6 + 2]  # conv1 weight
            b1 = args[3 + i * 6 + 3]  # conv1 bias
            w2 = args[3 + i * 6 + 4]  # conv2 weight
            b2 = args[3 + i * 6 + 5]  # conv2 bias
            transforms.append((w0, b0, w1, b1, w2, b2))

        N, C, L = x.shape
        C_half = C // 2
        assert C == 192 and C_half == 96, "This implementation assumes C=192 and C_half=96."

        # Ensure dtype float32 for Triton kernels
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if x_mask.dtype != torch.float32:
            x_mask = x_mask.to(torch.float32)

        if reverse:
            # Reverse pass: sequentially apply transformations in reverse order, subtract h
            for i in range(3, -1, -1):
                w0, b0, w1, b1, w2, b2 = transforms[i]

                # Split x into halves
                y_full = x  # current full tensor
                y0 = torch.empty((N, C_half, L), device=self.device, dtype=torch.float32)
                y1 = torch.empty((N, C_half, L), device=self.device, dtype=torch.float32)
                triton_split_halves_forward(y_full, y0, y1, BLOCK_L=128, num_warps=4, num_stages=2)

                # Compute h = conv0 -> ReLU -> conv1 -> ReLU -> conv2 (no ReLU after conv2)
                h = triton_conv1d(y0, w0, b0, BLOCK_L=128, num_warps=4, num_stages=2)
                h = triton_relu(h, BLOCK_L=128, num_warps=4, num_stages=2)
                h = triton_conv1d(h, w1, b1, BLOCK_L=128, num_warps=4, num_stages=2)
                h = triton_relu(h, BLOCK_L=128, num_warps=4, num_stages=2)
                h = triton_conv1d(h, w2, b2, BLOCK_L=128, num_warps=4, num_stages=2)

                # Apply mask
                h = triton_mul_mask(h, x_mask, BLOCK_L=128, num_warps=4, num_stages=2)

                # Update x1 = x1 - h
                # y1 is the right half; subtract h
                # We need the original right half BEFORE update. We can read it from y_full's second half:
                # Create a copy of y_full to temporarily hold updated y1:
                # But since we split y_full into y0,y1, we can reconstruct y_full with updated y1.
                # For now, perform subtraction in-place:
                # y_full = concat(y0, y1 - h). Allocate new full tensor:
                y_full_new = torch.empty((N, C, L), device=self.device, dtype=torch.float32)
                triton_concat_halves_forward(y0, (y1 - h), y_full_new, BLOCK_L=128, num_warps=4, num_stages=2)

                # Apply mask to the whole updated tensor
                y_full_new = triton_mul_mask(y_full_new, x_mask, BLOCK_L=128, num_warps=4, num_stages=2)

                # Update x for next iteration: set x = y_full_new
                x = y_full_new.clone()
        else:
            # Forward pass: apply transformations sequentially, add h
            for i in range(4):
                w0, b0, w1, b1, w2, b2 = transforms[i]

                # Split x into halves
                y_full = x
                y0 = torch.empty((N, C_half, L), device=self.device, dtype=torch.float32)
                y1 = torch.empty((N, C_half, L), device=self.device, dtype=torch.float32)
                triton_split_halves_forward(y_full, y0, y1, BLOCK_L=128, num_warps=4, num_stages=2)

                # Compute h = conv0 -> ReLU -> conv1 -> ReLU -> conv2 (no ReLU after conv2)
                h = triton_conv1d(y0, w0, b0, BLOCK_L=128, num_warps=4, num_stages=2)
                h = triton_relu(h, BLOCK_L=128, num_warps=4, num_stages=2)
                h = triton_conv1d(h, w1, b1, BLOCK_L=128, num_warps=4, num_stages=2)
                h = triton_relu(h, BLOCK_L=128, num_warps=4, num_stages=2)
                h = triton_conv1d(h, w2, b2, BLOCK_L=128, num_warps=4, num_stages=2)

                # Apply mask
                h = triton_mul_mask(h, x_mask, BLOCK_L=128, num_warps=4, num_stages=2)

                # Update x1 = x1 + h
                y1_new = y1 + h
                y_full_new = torch.empty((N, C, L), device=self.device, dtype=torch.float32)
                triton_concat_halves_forward(y0, y1_new, y_full_new, BLOCK_L=128, num_warps=4, num_stages=2)

                # Apply mask to the whole updated tensor
                y_full_new = triton_mul_mask(y_full_new, x_mask, BLOCK_L=128, num_warps=4, num_stages=2)

                # Update x for next iteration
                x = y_full_new.clone()

        return x


def run(*args):
    return ModelNew()(*args)
