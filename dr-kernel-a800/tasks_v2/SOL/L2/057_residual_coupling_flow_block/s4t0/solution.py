import math
import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def conv1d_relu_fwd_kernel(
        x_ptr,          # *const float, shape [N, C_in, T_in]
        w_ptr,          # *const float, shape [C_out, C_in, K]
        b_ptr,          # *const float, shape [C_out]
        y_ptr,          # *float,       shape [N, C_out, T_out]
        N: tl.int32,    # batch size
        C_in: tl.int32, # input channels
        T_in: tl.int32, # input time length
        C_out: tl.int32,# output channels
        T_out: tl.int32,# output time length (T_in - K + 1 + 2*padding)
        K: tl.int32,    # kernel size (fixed 5)
        PAD: tl.int32,  # padding (fixed 2)
        BLOCK_T_OUT: tl.constexpr,  # tile size along time output
    ):
        # program ids: batch, output channel, time tile
        n = tl.program_id(0)
        co = tl.program_id(1)
        tile_id = tl.program_id(2)

        # time indices this program handles
        t_offsets = tile_id * BLOCK_T_OUT + tl.arange(0, BLOCK_T_OUT)
        t_mask = t_offsets < T_out

        # initialize accumulator
        acc = tl.zeros([BLOCK_T_OUT], dtype=tl.float32)

        # loop over input channels and kernel positions
        # k is fixed 5: we unroll explicitly for simplicity and speed
        # conv output index t_out corresponds to input t_in = t_out + k - (PAD+1)
        # Since PAD=2, input t_in = t_out + k - 2
        # For padding, valid input indices are 0 <= t_in < T_in
        # We will load with mask to handle padding boundaries
        for ci in range(0, C_in):
            # for k=0..4
            # We'll compute input t_in for each k
            for k_val in (0, 1, 2, 3, 4):
                # compute input t_in for this k
                t_in = t_offsets + (k_val - PAD)
                valid = (t_in >= 0) & (t_in < T_in) & t_mask
                # compute linear index for x[n, ci, t_in]
                x_index = (((n * C_in) + ci) * T_in) + t_in
                x_vals = tl.load(x_ptr + x_index, mask=valid, other=0.0)
                # weight scalar w[co, ci, k_val]
                w_index = (((co * C_in) + ci) * K) + k_val
                w_val = tl.load(w_ptr + w_index)
                acc += x_vals * w_val

        # add bias
        b_val = tl.load(b_ptr + co)
        acc += b_val

        # ReLU
        acc = tl.maximum(acc, 0.0)

        # store to y[n, co, t_offsets]
        y_index = (((n * C_out) + co) * T_out) + t_offsets
        tl.store(y_ptr + y_index, acc, mask=t_mask)


def conv1d_relu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """
    Compute conv1d(x, weight, bias, padding=padding) followed by ReLU using Triton.
    x: [N, C_in, T_in], weight: [C_out, C_in, K], bias: [C_out]
    Returns: y: [N, C_out, T_out], where T_out = T_in - K + 1 + 2*padding
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype in (torch.float32, torch.float16), "Expected float tensor"
    assert weight.dtype == x.dtype and bias.dtype == x.dtype

    N, C_in, T_in = x.shape
    C_out, C_in_w, K = weight.shape
    assert C_in == C_in_w, "Weight in_channels must match x channels"
    assert K == 5, "This Triton kernel expects K=5"
    T_out = T_in - K + 1 + 2 * padding  # for k=5, padding=2 => T_out = T_in - 1

    # ensure contiguous tensors
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()

    # allocate output
    y = torch.empty((N, C_out, T_out), device=x.device, dtype=x.dtype)

    # grid: (N, C_out, num_tiles along T_out)
    # Choose BLOCK_T_OUT tile. 128 is a reasonable default for these sizes.
    BLOCK_T_OUT = 128
    grid = (N, C_out, triton.cdiv(T_out, BLOCK_T_OUT))

    # launch kernel
    conv1d_relu_fwd_kernel[grid](
        x_c, w_c, b_c, y,
        N, C_in, T_in, C_out, T_out, K, padding,
        BLOCK_T_OUT=BLOCK_T_OUT,
        num_warps=4,  # modest parallelism; tune as needed
        num_stages=2,
    )
    return y


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    # weights/biases for 4 transforms
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
    half_channels = x.shape[1] // 2

    # Collect all transform weights
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

    # Forward only (reverse not used in the provided harness)
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
        x0 = x[:, :half_channels, :]
        x1 = x[:, half_channels:, :]

        # Conv1d with padding (k=5 -> padding=2)
        # Triton conv1d+ReLU
        h = conv1d_relu_triton(x0, conv0_w, conv0_b, padding=2)
        h = conv1d_relu_triton(h, conv1_w, conv1_b, padding=2)
        h = conv1d_relu_triton(h, conv2_w, conv2_b, padding=2)

        # Apply mask (zeros whole time axis per batch item)
        h = h * x_mask

        # Affine coupling: x1 = x1 + h
        x1 = x1 + h

        # Concatenate back
        x = torch.cat([x0, x1], dim=1)

        # Apply mask to output
        x = x * x_mask

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Same signature as the original Model.forward
        return run(*args)


def run(*args):
    return ModelNew()(*args)
