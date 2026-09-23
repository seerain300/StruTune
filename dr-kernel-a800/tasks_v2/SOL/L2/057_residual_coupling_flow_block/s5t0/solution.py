import math
import torch
import torch.nn.functional as F

# Triton imports
import triton
import triton.language as tl


@triton.jit
def conv1d_triton_fused_relu(
    x_ptr,         # *const float, input [B, Cin, T]
    w_ptr,         # *const float, weights_flat [Cout, Cin*K]
    b_ptr,         # *const float, bias [Cout]
    y_ptr,         # *float, output [B, Cout, T_out]
    B: tl.int32, Cin: tl.int32, Cout: tl.int32, T: tl.int32, K: tl.int32,
    stride_b: tl.int32, stride_cin: tl.int32, stride_t: tl.int32,
    w_stride_co: tl.int32, w_stride_k: tl.int32,  # strides for W: [Cout, Cin*K]
    pad: tl.int32,
    T_out: tl.int32,
    # Triton tile sizes
    BLOCK_POS: tl.constexpr,  # tile over output positions
    BLOCK_CO: tl.constexpr,   # tile over output channels
):
    # program ids
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_pos = tl.program_id(2)

    # Compute channel and position offsets for this program
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)  # [BLOCK_CO]
    pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)  # [BLOCK_POS]

    # Masks for valid tiles
    co_mask = co_offsets < Cout
    pos_mask = pos_offsets < T_out

    # Initialize output accumulator [BLOCK_CO, BLOCK_POS]
    # We accumulate in fp32
    acc = tl.zeros((BLOCK_CO, BLOCK_POS), dtype=tl.float32)

    # Loop over input channels and kernel elements
    # K = kernel_size, Cin is input channels
    # For each (ci, kk), compute contributions to all pos_offsets and co_offsets
    # Note: out_pos = pos_offsets; input_pos = out_pos - pad + kk
    # We'll compute for each kk and ci
    # We use x[b, ci, input_pos] for valid input_pos.
    # We don't materialize im2col; we index directly and multiply with w[co, ci*K + kk].
    # We will cast loads to fp32 and do accumulation in fp32.
    for ci in range(0, Cin):
        # inner loop over kernel elements
        for kk in range(0, K):
            # compute input positions for this kk
            input_pos = pos_offsets - pad + kk  # [BLOCK_POS]
            # mask for valid input positions (within [0, T))
            in_bounds = (input_pos >= 0) & (input_pos < T) & pos_mask

            # Load x[b, ci, input_pos] as vector over BLOCK_POS
            # Use other=0.0 for out-of-bounds
            # Address: x_ptr + pid_b*stride_b + ci*stride_cin + input_pos*stride_t
            x_vec = tl.load(
                x_ptr + pid_b * stride_b + ci * stride_cin + input_pos * stride_t,
                mask=in_bounds,
                other=0.0
            )  # shape [BLOCK_POS], dtype inferred (we cast to fp32)
            x_vec = x_vec.to(tl.float32)  # ensure fp32 accumulation

            # Load weights for this (ci, kk) over co_offsets: W_flat[co, ci*K + kk]
            w_idx = co_offsets * w_stride_co + (ci * K + kk) * w_stride_k  # [BLOCK_CO]
            w_vec = tl.load(w_ptr + w_idx, mask=co_mask, other=0.0).to(tl.float32)  # [BLOCK_CO]

            # Outer product accumulate: acc += w_vec[:, None] * x_vec[None, :]
            # Note: broadcasting
            acc += w_vec[:, None] * x_vec[None, :]

    # Add bias
    b_vec = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)  # [BLOCK_CO]
    acc += b_vec[:, None]  # broadcast across pos

    # Apply ReLU (fused)
    acc = tl.maximum(acc, 0.0)

    # Store results to y[b, co, pos]
    # y strides: assume y is contiguous [B, Cout, T_out]
    # We store as fp32
    # Address: y_ptr + pid_b * B_stride + co * C_stride + pos * P_stride
    # But Triton expects strides; we can assume y is contiguous: y_stride_b = Cout*T_out, y_stride_c = T_out, y_stride_p = 1
    # We pass y_ptr as a pointer; Triton uses tensor strides implicitly via pointer arithmetic. Here we use the tensor's strides.
    # However, since we allocate y with torch.empty and pass its strides, we can compute address directly.
    # For simplicity, we assume y is contiguous in (co, pos) for each batch element. So:
    y_offsets = pid_b * (Cout * T_out) + co_offsets[:, None] * T_out + pos_offsets[None, :]
    store_mask = co_mask[:, None] & pos_mask[None, :]
    tl.store(y_ptr + y_offsets, acc, mask=store_mask)


def triton_conv1d_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, padding: int):
    """
    x: [B, Cin, T], weight: [Cout, Cin, K], bias: [Cout]
    Returns y: [B, Cout, T_out], where T_out = T - K + 1 + 2*padding, but since stride=1, pad already accounted.
    For this implementation, padding only affects indexing; output length is T - K + 1 (no additional elements).
    """
    assert x.is_cuda, "x must be on CUDA for Triton kernel"
    assert weight.is_cuda, "weight must be on CUDA for Triton kernel"
    assert bias.is_cuda, "bias must be on CUDA for Triton kernel"
    B, Cin, T = x.shape
    Cout, Cin_w, K = weight.shape
    assert Cin == Cin_w, "Input channels of x must match weight's Cin"
    # Compute output length: stride=1, dilation=1
    T_out = T - K + 1 + 2 * padding  # but with no stride, correct formula is T - K + 1; padding is for indexing
    # We'll implement correct T_out = T - K + 1 with padding for indexing by pos - pad + kk
    T_out = T - K + 1 + 2 * padding  # actually equals T - K + 1 when pad is used properly

    # Make x contiguous for simpler stride handling (we still pass strides explicitly)
    x = x.contiguous()
    weight_flat = weight.contiguous().view(Cout, Cin * K)
    bias = bias.contiguous()

    # Allocate output in fp32 for accumulation, cast later if needed
    y = torch.empty((B, Cout, T_out), device=x.device, dtype=torch.float32)

    # Strides
    stride_b = x.stride(0)
    stride_cin = x.stride(1)
    stride_t = x.stride(2)
    # W_flat strides
    w_stride_co = weight_flat.stride(0)  # typically 1
    w_stride_k = weight_flat.stride(1)   # typically Cin*K

    # Grid: (B, tiles over Cout, tiles over T_out)
    BLOCK_POS = 64
    BLOCK_CO = 32
    grid = (B, triton.cdiv(Cout, BLOCK_CO), triton.cdiv(T_out, BLOCK_POS))

    conv1d_triton_fused_relu[grid](
        x, weight_flat, bias, y,
        B, Cin, Cout, T, K,
        stride_b, stride_cin, stride_t,
        w_stride_co, w_stride_k,
        padding,
        T_out,
        BLOCK_POS=BLOCK_POS,
        BLOCK_CO=BLOCK_CO,
        num_warps=4,
        num_stages=2
    )
    return y


def apply_transform_triton(x0: torch.Tensor, conv0_w: torch.Tensor, conv0_b: torch.Tensor,
                           conv1_w: torch.Tensor, conv1_b: torch.Tensor,
                           conv2_w: torch.Tensor, conv2_b: torch.Tensor,
                           x_mask: torch.Tensor):
    """
    Apply a single transform: Conv1d -> ReLU -> Conv1d -> ReLU -> Conv1d using Triton.
    x0: [B, Cin0, T] where Cin0 = half_channels (96 in provided setup)
    conv* params: [Cout, Cin, K]
    """
    assert x0.is_cuda and conv0_w.is_cuda and conv0_b.is_cuda and x_mask.is_cuda, "All tensors must be on CUDA"
    # conv0: Cin0 -> Cout=192, kernel K=5
    h = triton_conv1d_relu(x0, conv0_w, conv0_b, padding=2)  # pad = (K-1)//2 = 2
    h = F.relu(h)  # ReLU already fused in Triton conv, but keep torch relu for clarity; or omit since fused. We already fused. Use h directly.
    # For safety, we can call torch relu to match original semantics; since we fused ReLU, this is redundant. We skip.

    # conv1: Cout -> Cout, kernel K=5
    h = triton_conv1d_relu(h, conv1_w, conv1_b, padding=2)
    h = F.relu(h)

    # conv2: Cout -> Cout_half = 96, kernel K=5
    h = triton_conv1d_relu(h, conv2_w, conv2_b, padding=2)
    return h


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
    """
    Triton-optimized residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    assert x.is_cuda and x_mask.is_cuda, "All tensors must be on CUDA device for Triton"
    B, C, T = x.shape
    half_channels = C // 2

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

    if not reverse:
        # Forward pass: apply transformations sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # Compute transformation conditioned on x0
            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, x_mask)

            # Apply mask
            h = h * x_mask

            # Affine coupling: x1 = x1 + h
            x1 = x1 + h

            # Concatenate back
            x = torch.cat([x0, x1], dim=1)

            # Apply mask to output
            x = x * x_mask
    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # Compute transformation conditioned on x0
            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, x_mask)

            # Apply mask
            h = h * x_mask

            # Inverse affine coupling: x1 = x1 - h
            x1 = x1 - h

            # Concatenate back
            x = torch.cat([x0, x1], dim=1)

            # Apply mask to output
            x = x * x_mask

    return x


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # The original 'run' function signature is known. We unpack args accordingly.
        # Expect: x, x_mask, reverse, then 4*3 conv weights and biases.
        # The number of args is fixed as per original function.
        # We assert the number of arguments for clarity.
        if len(args) < 3:
            raise RuntimeError("ModelNew.forward expects at least (x, x_mask, reverse)")
        x = args[0]
        x_mask = args[1]
        reverse = bool(args[2])

        # Ensure all tensors are on CUDA (required by Triton)
        if not x.is_cuda:
            x = x.cuda()
        if not x_mask.is_cuda:
            x_mask = x_mask.cuda()

        # The rest are conv weights and biases. Ensure they are on CUDA as well.
        # We'll move them if needed.
        def _ensure_cuda(t: torch.Tensor):
            if not t.is_cuda:
                t = t.cuda()
            return t

        # Unpack args: indices follow the original order
        # Args after the third are 4 sets of (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
        # We need to collect them into a list of 4 tuples.
        transforms = []
        # Each transform has 6 tensors: conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b
        # Total tensors = 4 * 6 = 24
        if len(args) != 27:
            # The original example has 24 convs and biases, so 24 tensors after reverse flag
            # If len(args) != 24 + 3, we raise
            raise RuntimeError("Expected exactly 24 conv weights/biases after (x, x_mask, reverse)")

        # Iterate and group into 4 transforms
        for i in range(4):
            conv0_w = _ensure_cuda(args[3 + i * 6 + 0])
            conv0_b = _ensure_cuda(args[3 + i * 6 + 1])
            conv1_w = _ensure_cuda(args[3 + i * 6 + 2])
            conv1_b = _ensure_cuda(args[3 + i * 6 + 3])
            conv2_w = _ensure_cuda(args[3 + i * 6 + 4])
            conv2_b = _ensure_cuda(args[3 + i * 6 + 5])
            transforms.append((conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b))

        # Now call the Triton-optimized run
        return run_triton(x, x_mask, reverse, *transforms[0], *transforms[1], *transforms[2], *transforms[3])


def run(*args):
    return ModelNew()(*args)
