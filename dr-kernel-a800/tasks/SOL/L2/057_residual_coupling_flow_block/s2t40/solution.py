import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    P: tl.constexpr, K: tl.constexpr,
):
    # program ids: (batch, output_channel, output_time)
    n = tl.program_id(0)
    co = tl.program_id(1)
    lo = tl.program_id(2)

    # Accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Add bias
    b_val = tl.load(b_ptr + co)  # scalar
    acc += b_val

    # Sum over input channels and kernel taps
    for ci in range(C_in):
        for k in range(K):
            li = lo + P - k  # since stride=1, dilation=1
            # Guard against out-of-bounds (shouldn't happen when P=K//2 and lo in [0, L_out))
            # Triton doesn't support Python if on scalars; we use a mask based on li
            in_range = (li >= 0) & (li < L_in)
            # Compute pointers for x[n, ci, li] and w[co, ci, k]
            x_idx = n * stride_x_n + ci * stride_x_c + li * stride_x_l
            w_idx = co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            # Load with mask: if in_range is False, use 0.0
            x_val = tl.load(x_ptr + x_idx, mask=in_range, other=0.0)
            w_val = tl.load(w_ptr + w_idx)  # scalar
            # Accumulate in fp32
            acc += tl.cast(x_val, tl.float32) * tl.cast(w_val, tl.float32)

    # Store result to y[n, co, lo]
    y_idx = n * stride_y_n + co * stride_y_c + lo * stride_y_l
    # Note: we assume output dtype is float32 (matches inputs from get_inputs)
    tl.store(y_ptr + y_idx, acc)


@triton.jit
def relu_kernel(
    x_ptr, y_ptr,
    N, C, L,
    stride_x_n, stride_x_c, stride_x_l,
    stride_y_n, stride_y_c, stride_y_l,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    l = tl.program_id(2)
    x_val = tl.load(x_ptr + n * stride_x_n + c * stride_x_c + l * stride_x_l)
    y_val = tl.maximum(x_val, 0.0)
    tl.store(y_ptr + n * stride_y_n + c * stride_y_c + l * stride_y_l, y_val)


@triton.jit
def split_halves_forward(
    y_full_ptr, y0_ptr, y1_ptr,
    N, C_half, L,
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)
    l = tl.program_id(2)
    # Read from first half into y0
    val0 = tl.load(y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + l * stride_y_full_l)
    tl.store(y0_ptr + n * stride_y0_n + ch * stride_y0_c + l * stride_y0_l, val0)
    # Read from second half into y1
    val1 = tl.load(y_full_ptr + n * stride_y_full_n + (ch + C_half) * stride_y_full_c + l * stride_y_full_l)
    tl.store(y1_ptr + n * stride_y1_n + ch * stride_y1_c + l * stride_y1_l, val1)


@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y_full_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)
    l = tl.program_id(2)
    val0 = tl.load(y0_ptr + n * stride_y0_n + ch * stride_y0_c + l * stride_y0_l)
    tl.store(y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + l * stride_y_full_l, val0)
    val1 = tl.load(y1_ptr + n * stride_y1_n + ch * stride_y1_c + l * stride_y1_l)
    tl.store(y_full_ptr + n * stride_y_full_n + (ch + C_half) * stride_y_full_c + l * stride_y_full_l, val1)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_mask_n, stride_mask_c, stride_mask_l,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    l = tl.program_id(2)
    y_val = tl.load(y_ptr + n * stride_y_n + c * stride_y_c + l * stride_y_l)
    m_val = tl.load(mask_ptr + n * stride_mask_n + 0 * stride_mask_c + l * stride_mask_l)  # mask is [N, 1, L]
    # Multiply: m_val is 1.0 for valid positions (mask is ones in provided inputs)
    y_val = y_val * m_val
    tl.store(y_ptr + n * stride_y_n + c * stride_y_c + l * stride_y_l, y_val)


@triton.jit
def split_halves_backward(
    y_full_ptr, y0_ptr, y1_ptr,
    N, C_half, L,
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)
    l = tl.program_id(2)
    val0 = tl.load(y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + l * stride_y_full_l)
    tl.store(y0_ptr + n * stride_y0_n + ch * stride_y0_c + l * stride_y0_l, val0)
    val1 = tl.load(y_full_ptr + n * stride_y_full_n + (ch + C_half) * stride_y_full_c + l * stride_y_full_l)
    tl.store(y1_ptr + n * stride_y1_n + ch * stride_y1_c + l * stride_y1_l, val1)


@triton.jit
def concat_halves_backward(
    y0_ptr, y1_ptr, y_full_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)
    l = tl.program_id(2)
    val0 = tl.load(y0_ptr + n * stride_y0_n + ch * stride_y0_c + l * stride_y0_l)
    tl.store(y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + l * stride_y_full_l, val0)
    val1 = tl.load(y1_ptr + n * stride_y1_n + ch * stride_y1_c + l * stride_y1_l)
    tl.store(y_full_ptr + n * stride_y_full_n + (ch + C_half) * stride_y_full_c + l * stride_y_full_l, val1)


# The main forward function used by ModelNew
@torch.no_grad()
def run_triton(
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
    assert x.is_cuda, "Tensors must be on CUDA for Triton kernels"
    N, C, L = x.shape
    C_half = C // 2

    # Helper: split x into x0 and x1 along channels
    x0 = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)
    x1 = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)
    split_halves_forward[(N, C_half, L)](
        x, x0, x1,
        N, C_half, L,
        x.stride(0), x.stride(1), x.stride(2),
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        num_warps=1,
    )

    half_channels = C_half
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
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Compute transformations on x0: conv0 -> relu -> conv1 -> relu -> conv2
            y0_0 = torch.empty((N, conv0_w.shape[0], L), device=x.device, dtype=x.dtype)
            conv1d_forward_kernel[(N, conv0_w.shape[0], L)](
                x0, conv0_w, conv0_b, y0_0,
                N, conv0_w.shape[1], conv0_w.shape[0], x0.shape[2], L,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                y0_0.stride(0), y0_0.stride(1), y0_0.stride(2),
                P=2, K=5,
                num_warps=4,
            )
            # ReLU
            y0_1 = torch.empty_like(y0_0)
            relu_kernel[(N, y0_0.shape[1], L)](
                y0_0, y0_1,
                N, y0_0.shape[1], L,
                y0_0.stride(0), y0_0.stride(1), y0_0.stride(2),
                y0_1.stride(0), y0_1.stride(1), y0_1.stride(2),
                num_warps=1,
            )

            y1_0 = torch.empty((N, conv1_w.shape[0], L), device=x.device, dtype=x.dtype)
            conv1d_forward_kernel[(N, conv1_w.shape[0], L)](
                y0_1, conv1_w, conv1_b, y1_0,
                N, conv1_w.shape[1], conv1_w.shape[0], y0_1.shape[2], L,
                y0_1.stride(0), y0_1.stride(1), y0_1.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                y1_0.stride(0), y1_0.stride(1), y1_0.stride(2),
                P=2, K=5,
                num_warps=4,
            )
            # ReLU
            y1_1 = torch.empty_like(y1_0)
            relu_kernel[(N, y1_0.shape[1], L)](
                y1_0, y1_1,
                N, y1_0.shape[1], L,
                y1_0.stride(0), y1_0.stride(1), y1_0.stride(2),
                y1_1.stride(0), y1_1.stride(1), y1_1.stride(2),
                num_warps=1,
            )

            y2_0 = torch.empty((N, conv2_w.shape[0], L), device=x.device, dtype=x.dtype)
            conv1d_forward_kernel[(N, conv2_w.shape[0], L)](
                y1_1, conv2_w, conv2_b, y2_0,
                N, conv2_w.shape[1], conv2_w.shape[0], y1_1.shape[2], L,
                y1_1.stride(0), y1_1.stride(1), y1_1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                y2_0.stride(0), y2_0.stride(1), y2_0.stride(2),
                P=2, K=5,
                num_warps=4,
            )

            # Affine coupling: x1 = x1 + h
            x1 = x1 + y2_0

            # Concatenate halves back
            x_full = torch.empty((N, C, L), device=x.device, dtype=x.dtype)
            concat_halves_forward[(N, C_half, L)](
                x0, x1, x_full,
                N, C_half, L,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                x_full.stride(0), x_full.stride(1), x_full.stride(2),
                num_warps=1,
            )

            # Apply mask
            x_full = x_full * x_mask

            # Update x for next iteration
            x = x_full

    else:
        # Reverse pass
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split current x into x0 and x1
            x0 = x[:, :C_half, :]
            x1 = x[:, C_half:, :]

            # Compute transformation on x0: conv0 -> relu -> conv1 -> relu -> conv2
            y0_0 = torch.empty((N, conv0_w.shape[0], L), device=x.device, dtype=x.dtype)
            conv1d_forward_kernel[(N, conv0_w.shape[0], L)](
                x0, conv0_w, conv0_b, y0_0,
                N, conv0_w.shape[1], conv0_w.shape[0], x0.shape[2], L,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                y0_0.stride(0), y0_0.stride(1), y0_0.stride(2),
                P=2, K=5,
                num_warps=4,
            )
            # ReLU
            y0_1 = torch.empty_like(y0_0)
            relu_kernel[(N, y0_0.shape[1], L)](
                y0_0, y0_1,
                N, y0_0.shape[1], L,
                y0_0.stride(0), y0_0.stride(1), y0_0.stride(2),
                y0_1.stride(0), y0_1.stride(1), y0_1.stride(2),
                num_warps=1,
            )

            y1_0 = torch.empty((N, conv1_w.shape[0], L), device=x.device, dtype=x.dtype)
            conv1d_forward_kernel[(N, conv1_w.shape[0], L)](
                y0_1, conv1_w, conv1_b, y1_0,
                N, conv1_w.shape[1], conv1_w.shape[0], y0_1.shape[2], L,
                y0_1.stride(0), y0_1.stride(1), y0_1.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                y1_0.stride(0), y1_0.stride(1), y1_0.stride(2),
                P=2, K=5,
                num_warps=4,
            )
            # ReLU
            y1_1 = torch.empty_like(y1_0)
            relu_kernel[(N, y1_0.shape[1], L)](
                y1_0, y1_1,
                N, y1_0.shape[1], L,
                y1_0.stride(0), y1_0.stride(1), y1_0.stride(2),
                y1_1.stride(0), y1_1.stride(1), y1_1.stride(2),
                num_warps=1,
            )

            y2_0 = torch.empty((N, conv2_w.shape[0], L), device=x.device, dtype=x.dtype)
            conv1d_forward_kernel[(N, conv2_w.shape[0], L)](
                y1_1, conv2_w, conv2_b, y2_0,
                N, conv2_w.shape[1], conv2_w.shape[0], y1_1.shape[2], L,
                y1_1.stride(0), y1_1.stride(1), y1_1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                y2_0.stride(0), y2_0.stride(1), y2_0.stride(2),
                P=2, K=5,
                num_warps=4,
            )

            # Inverse coupling: x1 = x1 - h
            x1 = x1 - y2_0

            # Concatenate halves back
            x_full = torch.empty((N, C, L), device=x.device, dtype=x.dtype)
            concat_halves_forward[(N, C_half, L)](
                x0, x1, x_full,
                N, C_half, L,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                x_full.stride(0), x_full.stride(1), x_full.stride(2),
                num_warps=1,
            )

            # Apply mask
            x_full = x_full * x_mask

            # Update x for previous iteration
            x = x_full

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect the same signature as Model.forward
        # x: [N, C, L], x_mask: [N, 1, L], and 4 sets of weights/biases for each transform
        # We'll reconstruct the args into the expected order from get_inputs
        # get_inputs returns: x, x_mask, reverse, then 4*3 weights and biases.
        # We'll manually extract the first 4 groups of 3 conv weights/bias each.
        # The rest are ignored (the original run function accepts many params, but we only use needed).

        # args are: x, x_mask, reverse, transform_0..., transform_1..., transform_2..., transform_3...
        # Count arguments
        num_args = len(args)
        # x and x_mask and reverse are the first three
        x = args[0]
        x_mask = args[1]
        reverse = args[2] if len(args) > 2 else False

        # Collect transforms in groups of 3: conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b
        # The reference get_inputs provides 4 transforms with 3 convs each.
        # We'll read them sequentially. If fewer than needed, we raise.
        # Ensure we have at least 6 tensors for transform 0
        if num_args < 6:
            raise RuntimeError("Not enough arguments provided to ModelNew.forward")
        transform_0_conv0_weight = args[3]
        transform_0_conv0_bias = args[4]
        transform_0_conv1_weight = args[5]
        transform_0_conv1_bias = args[6]
        transform_0_conv2_weight = args[7]
        transform_0_conv2_bias = args[8]

        if num_args < 13:
            raise RuntimeError("Not enough arguments provided to ModelNew.forward")

        transform_1_conv0_weight = args[9]
        transform_1_conv0_bias = args[10]
        transform_1_conv1_weight = args[11]
        transform_1_conv1_bias = args[12]
        transform_1_conv2_weight = args[13]
        transform_1_conv2_bias = args[14]

        if num_args < 22:
            raise RuntimeError("Not enough arguments provided to ModelNew.forward")

        transform_2_conv0_weight = args[15]
        transform_2_conv0_bias = args[16]
        transform_2_conv1_weight = args[17]
        transform_2_conv1_bias = args[18]
        transform_2_conv2_weight = args[19]
        transform_2_conv2_bias = args[20]

        if num_args < 31:
            raise RuntimeError("Not enough arguments provided to ModelNew.forward")

        transform_3_conv0_weight = args[21]
        transform_3_conv0_bias = args[22]
        transform_3_conv1_weight = args[23]
        transform_3_conv1_bias = args[24]
        transform_3_conv2_weight = args[25]
        transform_3_conv2_bias = args[26]

        # Call Triton-based forward
        return run_triton(
            x, x_mask, reverse,
            transform_0_conv0_weight, transform_0_conv0_bias,
            transform_0_conv1_weight, transform_0_conv1_bias,
            transform_0_conv2_weight, transform_0_conv2_bias,
            transform_1_conv0_weight, transform_1_conv0_bias,
            transform_1_conv1_weight, transform_1_conv1_bias,
            transform_1_conv2_weight, transform_1_conv2_bias,
            transform_2_conv0_weight, transform_2_conv0_bias,
            transform_2_conv1_weight, transform_2_conv1_bias,
            transform_2_conv2_weight, transform_2_conv2_bias,
            transform_3_conv0_weight, transform_3_conv0_bias,
            transform_3_conv1_weight, transform_3_conv1_bias,
            transform_3_conv2_weight, transform_3_conv2_bias,
        )


def run(*args):
    return ModelNew()(*args)
