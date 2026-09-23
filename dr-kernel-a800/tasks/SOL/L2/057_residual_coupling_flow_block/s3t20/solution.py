import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: Conv1d (stride=1, padding=0, kernel_size=5, bias=True)
# x: [N, Cin, L_in], w: [Cout, Cin, K], b: [Cout], y: [N, Cout, L_out], L_out = L_in - 4
@triton.jit
def conv1d_bias_stride1_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Cout, L_in, L_out,
    x_sN, x_sC, x_sL,
    w_sOC, w_sIC, w_sK,
    y_sN, y_sC, y_sL,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # program ids
    pid_nc = tl.program_id(0)  # over N*Cout
    pid_t = tl.program_id(1)   # over tiles of L_out

    # compute n and out_channel from pid_nc
    n = pid_nc // Cout
    oc = pid_nc % Cout

    # tile offsets along time
    offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_out = offsets < L_out  # valid output positions

    # accumulate in float32
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # loop over kernel taps
    for k in range(K):
        # input index for each output position
        l_in_vec = offsets - k  # valid when 0 <= l_in_vec < L_in
        # validity mask for input loads
        mask_in = (l_in_vec >= 0) & (l_in_vec < L_in) & mask_out

        # compute base pointer for x[n, :, l_in_vec]
        # we need to load over input channels Cin
        # Note: Cin is dynamic; we loop k is constexpr; we recompute for each k
        # For each input channel c_in, load x[n, c_in, l_in_vec]
        # Weight w[oc, c_in, k] is scalar for this k, oc, c_in.
        # We compute x pointer for each c_in: x_ptr + n*x_sN + c_in*x_sC + l_in_vec*x_sL
        # Loop over Cin implicitly by computing contribution for each oc, c_in, k:
        # y[oc, offsets] += sum_{c_in} w[oc, c_in, k] * x[n, c_in, offsets - k]
        # We can do this by iterating c_in and loading x and multiplying by w scalar.
        # However, Triton prefers vectorized loads. Since Cin is not constexpr, we perform
        # the accumulation with a loop over c_in:
        # We need to load w scalar and multiply by vector x loads across Cin.
        # Better: precompute x contributions for each c_in and accumulate.
        # Since Cin is dynamic, we iterate c_in:
        # For each c_in, compute x_vec = tl.load(...), w_scalar = tl.load(w_ptr + oc*w_sOC + c_in*w_sIC + k*w_sK),
        # acc += x_vec * w_scalar
        # Note: In Triton, we can compute pointer and load inside the loop. We'll do it by indexing.
        # We'll use nested loop approach: compute acc += w[oc, c_in, k] * x[n, c_in, offsets - k].
        # But we need to loop over c_in and inside we load vector x and multiply by scalar w, then acc += sum across c_in.
        # However, Triton will broadcast scalar w to vector x_vec when we multiply, but acc should be scalar per oc, so:
        # We'll accumulate per oc. We can do this by setting w_scalar as scalar and x_vec as vector, multiply, and acc +=.

        # We need to loop over c_in to get contribution:
        # For each c_in, load scalar w and vector x, multiply, and add to acc.

        # Prepare w_scalar as a scalar float32 by loading from w_ptr
        # We'll do a loop over Cin:
        # Initialize acc to zero per oc per tile offsets (already done above)
        for c_in in range(0, Cin):
            # Load weight scalar w[oc, c_in, k]
            w_off = oc * w_sOC + c_in * w_sIC + k * w_sK
            w_scalar = tl.load(w_ptr + w_off)
            # Convert to float32 for accumulation
            w_scalar = w_scalar.to(tl.float32)
            # Compute x pointers for this c_in and offsets
            x_off = n * x_sN + c_in * x_sC + l_in_vec * x_sL
            x_vec = tl.load(x_ptr + x_off, mask=mask_in, other=0.0)
            x_vec = x_vec.to(tl.float32)
            acc += x_vec * w_scalar

    # add bias
    b_val = tl.load(b_ptr + oc)
    acc = acc + b_val.to(tl.float32)

    # store to y[n, oc, offsets]
    y_off = n * y_sN + oc * y_sC + offsets * y_sL
    tl.store(y_ptr + y_off, acc, mask=mask_out)


# Triton ReLU
@triton.jit
def relu_kernel(x_ptr, y_ptr, N, C, L, x_sN, x_sC, x_sL, y_sN, y_sC, y_sL, BLOCK: tl.constexpr):
    # Grid: (N*C, tiles of L)
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offsets = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < L

    x_off = n * x_sN + c * x_sC + offsets * x_sL
    y_off = n * y_sN + c * y_sC + offsets * y_sL

    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    x = x.to(tl.float32)
    x = tl.maximum(x, 0.0)
    tl.store(y_ptr + y_off, x, mask=mask)


# Triton Concatenate channels: y[n, c_total, l] where c_total = C0 + C1, y = [x0_channels, x1_channels]
@triton.jit
def concatenate_channels_kernel(x0_ptr, x1_ptr, y_ptr,
                                 N, C0, C1, L,
                                 x0_sN, x0_sC, x0_sL,
                                 x1_sN, x1_sC, x1_sL,
                                 y_sN, y_sC, y_sL,
                                 BLOCK: tl.constexpr):
    # Grid: (N*C_total, tiles of L)
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    c_total = C0 + C1
    n = pid_nc // c_total
    c = pid_nc % c_total

    offsets = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < L

    # Decide source: first C0 channels come from x0, rest from x1
    is_x0 = c < C0

    if is_x0:
        src_sC = x0_sC
        src_ptr = x0_ptr
    else:
        src_sC = x1_sC
        src_ptr = x1_ptr

    if is_x0:
        src_n = n
        src_c = c
    else:
        src_n = n
        src_c = c - C0

    src_off = src_n * (src_sN if src_ptr == x0_ptr else x1_sN) + src_c * src_sC + offsets * (x0_sL if src_ptr == x0_ptr else x1_sL)
    y_off = n * y_sN + c * y_sC + offsets * y_sL

    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    val = val.to(tl.float32)
    tl.store(y_ptr + y_off, val, mask=mask)


# Triton Multiply by mask: x: [N, C, L], mask: [N, 1, L], y = x * mask
@triton.jit
def multiply_mask_kernel(x_ptr, mask_ptr, y_ptr,
                          N, C, L,
                          x_sN, x_sC, x_sL,
                          mask_sN, mask_sL,  # mask_sC is 1, ignored
                          y_sN, y_sC, y_sL,
                          BLOCK: tl.constexpr):
    # Grid: (N*C, tiles of L)
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offsets = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < L

    x_off = n * x_sN + c * x_sC + offsets * x_sL
    # mask is [N, 1, L], take channel 0
    mask_off = n * mask_sN + offsets * mask_sL
    m = tl.load(mask_ptr + mask_off, mask=mask, other=1.0)
    m = m.to(tl.float32)

    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    x = x.to(tl.float32)
    y = x * m
    y_off = n * y_sN + c * y_sC + offsets * y_sL
    tl.store(y_ptr + y_off, y, mask=mask)


# Triton Add/Subtract masked: y = x + h if add_flag else y = x - h
# x, h, y have shape [N, C, L], y = x + h or y = x - h
@triton.jit
def add_masked_kernel(x_ptr, h_ptr, y_ptr,
                       N, C, L,
                       x_sN, x_sC, x_sL,
                       h_sN, h_sC, h_sL,
                       y_sN, y_sC, y_sL,
                       add_flag: tl.constexpr,
                       BLOCK: tl.constexpr):
    # Grid: (N*C, tiles of L)
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offsets = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < L

    x_off = n * x_sN + c * x_sC + offsets * x_sL
    h_off = n * h_sN + c * h_sC + offsets * h_sL
    y_off = n * y_sN + c * y_sC + offsets * y_sL

    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(h_ptr + h_off, mask=mask, other=0.0).to(tl.float32)
    if add_flag:
        y = x + h
    else:
        y = x - h

    tl.store(y_ptr + y_off, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool,
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
                transform_3_conv2_weight, transform_3_conv2_bias):
        """
        ModelNew.forward must invoke Triton kernels for all numeric operations.
        It performs the same as the original run function: 4 sequential transforms,
        forward or reverse depending on the 'reverse' flag. All convs, ReLU, mask,
        and concatenation are done in Triton.
        """

        # Ensure inputs are on CUDA and Triton is available
        assert x.is_cuda and TRITON_AVAILABLE, "Triton kernels require CUDA tensors and Triton installed."

        N, C, L = x.shape
        half_channels = C // 2

        # Helper to apply a single transform sequentially using Triton
        # We will process convs, ReLU, add/subtract, multiply, and concatenate in Triton.
        # Note: x_mask shape is [N, 1, L], keep it as float32.

        # The run function applies transforms in a for loop; here we pass all weights/biases,
        # but we will only use the current ones per iteration. For clarity, we’ll implement a
        # single-transform step and call it 4 times. To avoid passing too many args, we can
        # extract the needed ones in the loop. However, evaluation expects a single forward.
        # So we will perform the 4 transforms by indexing the provided tensors in sequence.

        # Define a list of transforms (tuples of (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b))
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

        # If reverse, we process in reverse order
        if reverse:
            transforms = list(reversed(transforms))

        # Process 4 transforms
        for i, (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b) in enumerate(transforms):
            # Split x into two halves along channels
            x0 = x[:, :half_channels, :]   # [N, half_channels, L]
            x1 = x[:, half_channels:, :]   # [N, half_channels, L]

            # conv0: [Cin=half_channels, Cout=hidden=192, K=5], bias
            w0 = conv0_w.contiguous()
            b0 = conv0_b.contiguous()
            L_in0 = L
            L_out0 = L_in0 - 4  # padding=0, K=5 => output length shrinks by 4
            h0 = torch.empty((N, w0.shape[0], L_out0), device=x.device, dtype=torch.float32)
            grid0 = (N * w0.shape[0], triton.cdiv(L_out0, 128))
            conv1d_bias_stride1_kernel[grid0](
                x0, w0, b0, h0,
                N, x0.shape[1], w0.shape[0], L_in0, L_out0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                w0.stride(0), w0.stride(1), w0.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                K=5, BLOCK_T=128, num_warps=4
            )

            # ReLU on h0
            h0 = h0.to(torch.float32)  # ensure float32
            h0 = triton_relu(h0)

            # conv1: [Cin=hidden=192, Cout=hidden=192, K=5], bias
            w1 = conv1_w.contiguous()
            b1 = conv1_b.contiguous()
            L_in1 = h0.shape[2]
            L_out1 = L_in1 - 4
            h1 = torch.empty((h0.shape[0], w1.shape[0], L_out1), device=x.device, dtype=torch.float32)
            grid1 = (h0.shape[0] * w1.shape[0], triton.cdiv(L_out1, 128))
            conv1d_bias_stride1_kernel[grid1](
                h0, w1, b1, h1,
                h0.shape[0], h0.shape[1], w1.shape[0], h0.shape[2], L_out1,
                h0.stride(0), h0.stride(1), h0.stride(2),
                w1.stride(0), w1.stride(1), w1.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                K=5, BLOCK_T=128, num_warps=4
            )

            # ReLU on h1
            h1 = triton_relu(h1)

            # conv2: [Cin=hidden=192, Cout=half_channels=96, K=5], bias None
            # Original code passes bias for conv0 and conv1 but not for conv2. Here we do not pass bias.
            w2 = conv2_w.contiguous()
            L_in2 = h1.shape[2]
            L_out2 = L_in2 - 4
            h2 = torch.empty((h1.shape[0], w2.shape[0], L_out2), device=x.device, dtype=torch.float32)
            grid2 = (h1.shape[0] * w2.shape[0], triton.cdiv(L_out2, 128))
            # We need a bias tensor; but original code applies conv2 without bias in this structure.
            # Create a zero bias tensor to match signature.
            b2 = torch.zeros(w2.shape[0], device=x.device, dtype=torch.float32)
            conv1d_bias_stride1_kernel[grid2](
                h1, w2, b2, h2,
                h1.shape[0], h1.shape[1], w2.shape[0], h1.shape[2], L_out2,
                h1.stride(0), h1.stride(1), h1.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                K=5, BLOCK_T=128, num_warps=4
            )

            # Apply mask: h2 = h2 * x_mask
            h2 = triton_multiply_mask(h2, x_mask)

            # Affine coupling on second half: x1 = x1 + h2 if not reverse else x1 = x1 - h2
            x1 = triton_add_masked(x1, h2, reverse)  # launch Triton kernel

            # Concatenate [x0, x1] along channels
            x_concat = torch.empty((N, 2 * half_channels, h2.shape[2]), device=x.device, dtype=torch.float32)
            # x0 shape [N, half_channels, L_out2], x1 shape [N, half_channels, L_out2], y shape [N, 2*half_channels, L_out2]
            # Launch concatenate kernel
            grid_concat = (N * (2 * half_channels), triton.cdiv(h2.shape[2], 128))
            concatenate_channels_kernel[grid_concat](
                x0, x1, x_concat,
                N, x0.shape[1], x1.shape[1], h2.shape[2],
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                x_concat.stride(0), x_concat.stride(1), x_concat.stride(2),
                BLOCK=128, num_warps=4
            )

            # Multiply concatenated output by mask
            x = triton_multiply_mask(x_concat, x_mask)

        return x


# Helper functions to invoke Triton kernels (must be used by ModelNew.forward)
def triton_relu(x: torch.Tensor) -> torch.Tensor:
    # x: [N, C, L]
    y = torch.empty_like(x, dtype=torch.float32, device=x.device)
    grid = (x.shape[0] * x.shape[1], triton.cdiv(x.shape[2], 128))
    relu_kernel[grid](
        x, y, x.shape[0], x.shape[1], x.shape[2],
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK=128, num_warps=4
    )
    return y


def triton_multiply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: [N, C, L], mask: [N, 1, L]
    y = torch.empty_like(x, dtype=torch.float32, device=x.device)
    grid = (x.shape[0] * x.shape[1], triton.cdiv(x.shape[2], 128))
    multiply_mask_kernel[grid](
        x, mask, y,
        x.shape[0], x.shape[1], x.shape[2],
        x.stride(0), x.stride(1), x.stride(2),
        mask.stride(0), mask.stride(2),  # mask channel is 1
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK=128, num_warps=4
    )
    return y


def triton_add_masked(x: torch.Tensor, h: torch.Tensor, reverse: bool) -> torch.Tensor:
    # x, h: [N, C, L]
    y = torch.empty_like(x, dtype=torch.float32, device=x.device)
    grid = (x.shape[0] * x.shape[1], triton.cdiv(x.shape[2], 128))
    add_masked_kernel[grid](
        x, h, y,
        x.shape[0], x.shape[1], x.shape[2],
        x.stride(0), x.stride(1), x.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        add_flag=(not reverse),  # if reverse, subtract; else add
        BLOCK=128, num_warps=4
    )
    return y


# The original Model.run expects x, x_mask, reverse, plus 24 conv weights/biases.
# ModelNew.forward mirrors the same signature and uses Triton kernels for all numeric ops.
# Ensure tensors are CUDA for Triton; the evaluation harness sets device appropriately.


def run(*args):
    return ModelNew()(*args)
