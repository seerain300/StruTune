import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv1d_ncl_bias_kernel(
    weight_ptr,   # *ptr to weights [C_out, C_in, K]
    input_ptr,    # *ptr to input [N, C_in, L_in]
    output_ptr,   # *ptr to output [N, C_out, L_in]
    bias_ptr,     # *ptr to bias [C_out]
    N: tl.constexpr,  # batch size (not used in indexing, but kept for type)
    C_in,          # input channels
    C_out,         # output channels
    L_in,          # input length
    stride_n_in, stride_ci_in, stride_t_in,  # input strides
    stride_n_w, stride_ci_w, stride_k_w,     # weight strides (we use contiguous, so simple)
    stride_n_out, stride_co_out, stride_t_out,  # output strides
    K: tl.constexpr,  # kernel size (must be 5)
):
    # program ids: over (N*C_out) and tiles of L_in
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n = pid0 // C_out
    c_out = pid0 % C_out

    BLOCK_T = 128
    t_offsets = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L_in  # since L_out == L_in for padding=0, stride=1

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Accumulate over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            # input[n, ci, t_offsets + k]
            in_ptrs = input_ptr + n * stride_n_in + ci * stride_ci_in + (t_offsets + k) * stride_t_in
            w_ptr = weight_ptr + c_out * stride_ci_w + ci * stride_ci_w + k * stride_k_w  # weight[co, ci, k]
            w_val = tl.load(w_ptr)
            in_vec = tl.load(in_ptrs, mask=mask_t, other=0.0)
            acc += in_vec * w_val

    # add bias
    b_val = tl.load(bias_ptr + c_out)
    acc += b_val

    # store results
    out_ptrs = output_ptr + n * stride_n_out + c_out * stride_co_out + t_offsets * stride_t_out
    tl.store(out_ptrs, acc, mask=mask_t)


def triton_conv1d_bias(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized 1D convolution (stride=1, padding=0, kernel_size=K=5) + bias.
    - input: [N, C_in, L_in], float32, CUDA, contiguous
    - weight: [C_out, C_in, 5], float32, CUDA, contiguous
    - bias: [C_out], float32, CUDA, contiguous
    Returns: output [N, C_out, L_in]
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be CUDA for Triton."
    assert input.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32, "Use float32."
    N, C_in, L_in = input.shape
    C_out, C_in_w, K = weight.shape
    assert C_in == C_in_w, "Input channels must match weight's input channels."
    assert K == 5, "This Triton kernel is specialized for kernel_size=5."

    input_c = input.contiguous()
    weight_c = weight.contiguous()
    bias_c = bias.contiguous()

    output = torch.empty((N, C_out, L_in), device=input.device, dtype=torch.float32)

    # Get strides
    stride_n_in, stride_ci_in, stride_t_in = input_c.stride()
    stride_n_out, stride_co_out, stride_t_out = output.stride()
    # For weights, since it's [C_out, C_in, K] contiguous, strides are (C_in*K, K, 1)
    stride_ci_w = weight_c.stride(1)
    stride_k_w = weight_c.stride(2)
    stride_n_w = weight_c.stride(0)  # not used in address computation

    # Grid: (N * C_out, ceil_div(L_in, BLOCK_T))
    BLOCK_T = 128
    grid = (N * C_out, triton.cdiv(L_in, BLOCK_T))

    conv1d_ncl_bias_kernel[grid](
        weight_c, input_c, output, bias_c,
        N, C_in, C_out, L_in,
        stride_n_in, stride_ci_in, stride_t_in,
        stride_n_w, stride_ci_w, stride_k_w,
        stride_n_out, stride_co_out, stride_t_out,
        K=5,
    )

    return output


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Triton-optimized version of the original run function.
        - Uses Triton for all Conv1d computations (with bias).
        - Applies ReLU in PyTorch after each conv.
        - Keeps mask and coupling in PyTorch.
        - Supports forward (add) and reverse (subtract) passes.
        """
        # Args order: x, x_mask, reverse, then 48 arguments for 4 transforms
        x = args[0]  # [N, channels, time]
        x_mask = args[1]  # [N, 1, time]
        reverse = args[2]  # bool

        # Extract the 4 sets of weights/biases
        # Each transform has 3 weights and 3 biases:
        # conv0: [C_out=hidden, C_in=half, K=5], conv1: [C_out=hidden, C_in=hidden, K=5], conv2: [C_out=half, C_in=hidden, K=5]
        # There are 4 transforms, so 4 * 3 = 12 sets of weights + biases = 24 tensors total (48 arguments listed, but we take 24 in practice).
        # From the original function signature, there are actually 48 arguments after x_mask: 4 * (3 weights + 3 biases) = 24 sets total (likely per transform).
        # Here we take them sequentially: t0, t1, t2, t3 each with conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b.

        # First transform
        t0_w0, t0_b0, t0_w1, t0_b1, t0_w2, t0_b2 = args[3], args[4], args[5], args[6], args[7], args[8]
        # Second transform
        t1_w0, t1_b0, t1_w1, t1_b1, t1_w2, t1_b2 = args[9], args[10], args[11], args[12], args[13], args[14]
        # Third transform
        t2_w0, t2_b0, t2_w1, t2_b1, t2_w2, t2_b2 = args[15], args[16], args[17], args[18], args[19], args[20]
        # Fourth transform
        t3_w0, t3_b0, t3_w1, t3_b1, t3_w2, t3_b2 = args[21], args[22], args[23], args[24], args[25], args[26]

        # Ensure tensors are on CUDA and contiguous
        # Make all weights/biases contiguous
        for i in range(0, 27, 6):
            w0, b0, w1, b1, w2, b2 = args[i:i + 6]
            w0 = w0.contiguous()
            b0 = b0.contiguous()
            w1 = w1.contiguous()
            b1 = b1.contiguous()
            w2 = w2.contiguous()
            b2 = b2.contiguous()

        N, C, T = x.shape
        half_channels = C // 2
        x = x.contiguous()
        x_mask = x_mask.contiguous()

        # We reuse the same transform 4 times (consistent with the original loop logic).
        for _ in range(4):
            # Split into halves
            x0 = x[:, :half_channels, :]  # [N, half, T]
            x1 = x[:, half_channels:, :]  # [N, half, T]

            # conv0 + bias
            h0 = triton_conv1d_bias(x0, t0_w0, t0_b0)  # [N, hidden, T]
            # ReLU
            h0 = F.relu(h0)

            # conv1 + bias
            h1 = triton_conv1d_bias(h0, t0_w1, t0_b1)  # [N, hidden, T]
            # ReLU
            h1 = F.relu(h1)

            # conv2 + bias
            h2 = triton_conv1d_bias(h1, t0_w2, t0_b2)  # [N, half, T]
            # ReLU
            h2 = F.relu(h2)

            # Apply mask
            h2 = h2 * x_mask  # x_mask is [N, 1, T], broadcasting over channel dim

            # Affine coupling: x1 = x1 + h2 (forward) or x1 = x1 - h2 (reverse)
            if not reverse:
                x1 = x1 + h2
            else:
                x1 = x1 - h2

            # Concatenate back
            x = torch.cat([x0, x1], dim=1)  # [N, channels, T]
            # Apply mask to output (same as original)
            x = x * x_mask

        return x


def run(*args):
    return ModelNew()(*args)
