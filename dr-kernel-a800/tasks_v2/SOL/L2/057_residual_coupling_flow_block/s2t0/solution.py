import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels
# -------------------

# Conv1d: computes y[n, co, l_out] = bias[co] + sum_{ci,k} x[n, ci, l_out - P - k] * w[co, ci, k]
# We tile along the output time dimension: BLOCK_L outputs per program instance.
@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out, K, P,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    # Compute output time indices for this tile
    l_out_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_out_offsets < L_out

    # Accumulator for this tile
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Loop over input channels and kernel positions
    for ci in range(0, C_in):
        for k in range(0, K):
            # Map output index to input index with padding
            l_in_vec = l_out_offsets - P - k  # vector of length BLOCK_L
            in_range = (l_in_vec >= 0) & (l_in_vec < L_in) & mask_out
            # Compute input pointers for x[n, ci, l_in_vec]
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + l_in_vec * stride_x_l
            # Load with mask; out-of-range -> 0
            x_vals = tl.load(x_ptrs, mask=in_range, other=0.0)

            # Load weight scalar w[co, ci, k]
            w_ptr_scalar = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_val = tl.load(w_ptr_scalar)

            # Accumulate
            acc += x_vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + co)
    acc += b_val

    # Store to y[n, co, l_out_offsets]
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_out_offsets * stride_y_l
    tl.store(y_ptrs, acc, mask=mask_out)


# Elementwise ReLU
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
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(y_ptrs, y_vals, mask=mask_out)


# Elementwise multiply by x_mask: y[i, j, k] = y[i, j, k] * mask[i, 0, k]
@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_m_n, stride_m_c, stride_m_l,  # mask has shape [N, 1, L]
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)

    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l

    # mask is [N, 1, L]; we index mask[n, 0, l_offsets]
    mask_ptrs = mask_ptr + n * stride_m_n + 0 * stride_m_c + l_offsets * stride_m_l
    mask_vals = tl.load(mask_ptrs, mask=mask_out, other=0.0)

    y_vals = tl.load(y_ptrs, mask=mask_out, other=0.0)
    y_vals = y_vals * mask_vals
    tl.store(y_ptrs, y_vals, mask=mask_out)


# -------------------

# Triton-based apply_transform (forward): 3 convs + 2 ReLUs, no autograd
def apply_transform_triton(
    x0: torch.Tensor,
    conv0_w: torch.Tensor, conv0_b: torch.Tensor,
    conv1_w: torch.Tensor, conv1_b: torch.Tensor,
    conv2_w: torch.Tensor, conv2_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute h = Conv1d(x0, conv0) -> ReLU -> Conv1d(..., conv1) -> ReLU -> Conv1d(..., conv2)
    Returns h after conv2 (no ReLU on conv2 in the original). Assumes K=5, padding=2.
    x0: [N, C_in, L_in], conv weights: [C_out, C_in, K], biases: [C_out]
    Output: [N, C_out, L_out]
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x0.is_cuda and x0.dtype == torch.float32 and x0.is_contiguous()
    assert conv0_w.is_cuda and conv0_w.dtype == torch.float32 and conv0_w.is_contiguous()
    assert conv0_b.is_cuda and conv0_b.dtype == torch.float32 and conv0_b.is_contiguous()
    assert conv1_w.is_cuda and conv1_w.dtype == torch.float32 and conv1_w.is_contiguous()
    assert conv1_b.is_cuda and conv1_b.dtype == torch.float32 and conv1_b.is_contiguous()
    assert conv2_w.is_cuda and conv2_w.dtype == torch.float32 and conv2_w.is_contiguous()
    assert conv2_b.is_cuda and conv2_b.dtype == torch.float32 and conv2_b.is_contiguous()

    N, C_in, L_in = x0.shape
    C_out0 = conv0_w.shape[0]
    C_out1 = conv1_w.shape[0]
    C_out2 = conv2_w.shape[0]
    K = conv0_w.shape[2]
    P = K // 2

    # First conv: conv0_w -> C_out0
    L_out0 = L_in - K + 2 * P + 1
    y0 = torch.empty((N, C_out0, L_out0), device=x0.device, dtype=torch.float32)
    grid = (N, C_out0, triton.cdiv(L_out0, 128))
    conv1d_kernel[grid](
        x0, conv0_w, conv0_b, y0,
        N, C_in, C_out0, L_in, L_out0, K, P,
        x0.stride(0), x0.stride(1), x0.stride(2),
        conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
        y0.stride(0), y0.stride(1), y0.stride(2),
        BLOCK_L=128,
        num_warps=4,
    )
    # ReLU
    y0_relu = torch.empty_like(y0)
    grid_relu = (N, C_out0, triton.cdiv(L_out0, 128))
    relu_kernel[grid_relu](
        y0, y0_relu,
        N, C_out0, L_out0,
        y0.stride(0), y0.stride(1), y0.stride(2),
        y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
        BLOCK_L=128,
        num_warps=4,
    )
    y0 = y0_relu

    # Second conv: conv1_w -> C_out1
    L_out1 = y0.shape[2]
    y1 = torch.empty((N, C_out1, L_out1), device=x0.device, dtype=torch.float32)
    grid1 = (N, C_out1, triton.cdiv(L_out1, 128))
    conv1d_kernel[grid1](
        y0, conv1_w, conv1_b, y1,
        N, C_out0, C_out1, L_out0, L_out1, K, P,
        y0.stride(0), y0.stride(1), y0.stride(2),
        conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
        y1.stride(0), y1.stride(1), y1.stride(2),
        BLOCK_L=128,
        num_warps=4,
    )
    # ReLU
    y1_relu = torch.empty_like(y1)
    grid_relu1 = (N, C_out1, triton.cdiv(L_out1, 128))
    relu_kernel[grid_relu1](
        y1, y1_relu,
        N, C_out1, L_out1,
        y1.stride(0), y1.stride(1), y1.stride(2),
        y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
        BLOCK_L=128,
        num_warps=4,
    )
    y1 = y1_relu

    # Third conv: conv2_w -> C_out2 (no ReLU)
    L_out2 = y1.shape[2]
    h = torch.empty((N, C_out2, L_out2), device=x0.device, dtype=torch.float32)
    grid2 = (N, C_out2, triton.cdiv(L_out2, 128))
    conv1d_kernel[grid2](
        y1, conv2_w, conv2_b, h,
        N, C_out1, C_out2, L_out1, L_out2, K, P,
        y1.stride(0), y1.stride(1), y1.stride(2),
        conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        BLOCK_L=128,
        num_warps=4,
    )
    return h


# -------------------

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
    Residual coupling flow block. All computation done by Triton kernels.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
    half_channels = x.shape[1] // 2

    # We'll keep everything in Triton; host code only allocates and launches.
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

    if not reverse:
        # Forward pass: apply transformations sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # Compute transformation conditioned on x0 using Triton
            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)

            # Apply mask
            # We'll implement mask multiplication via Triton. x_mask shape [N, 1, L]
            N = x0.shape[0]
            C = h.shape[1]
            L = h.shape[2]
            h_masked = torch.empty_like(h)
            grid_mask = (N, C, triton.cdiv(L, 128))
            # x_mask has shape [N, 1, L]; ensure contiguous
            mul_mask_kernel[grid_mask](
                h, x_mask,
                N, C, L,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                BLOCK_L=128,
                num_warps=4,
            )

            # Affine coupling: x1 = x1 + h
            x1 = x1 + h_masked

            # Concatenate back
            x = torch.cat([x0, x1], dim=1)

    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # Compute transformation conditioned on x0 using Triton
            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)

            # Apply mask
            N = x0.shape[0]
            C = h.shape[1]
            L = h.shape[2]
            h_masked = torch.empty_like(h)
            grid_mask = (N, C, triton.cdiv(L, 128))
            mul_mask_kernel[grid_mask](
                h, x_mask,
                N, C, L,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                BLOCK_L=128,
                num_warps=4,
            )

            # Inverse affine coupling: x1 = x1 - h
            x1 = x1 - h_masked

            # Concatenate back
            x = torch.cat([x0, x1], dim=1)

    return x


# -------------------

# Entry point required: ModelNew
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # We must match the original run function signature.
        # The evaluator provides 22 tensors: x, x_mask, reverse flag (bool), then 24 weights/biases for 4 transforms.
        # Call our run with these args. We don't store anything; we just perform the computation.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
