import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_forward_kernel_v2(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out, K,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    # program ids: over (N, C_out, tiles along L_out)
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    # output time offsets for this tile
    l_out_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_out_offsets < L_out

    # accumulator in fp32
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # add bias once for all lo in this tile
    b_val = tl.load(b_ptr + co)
    acc += b_val  # broadcasting to vector

    # loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            # compute input indices li for each lo in this tile
            # padding = K//2 for stride=1, since PyTorch default for Conv1d
            P = K // 2
            li = l_out_offsets + P - k
            mask_in = (li >= 0) & (li < L_in) & mask_out

            # load x[n, ci, li] with mask
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            x_vals = tl.load(x_ptrs, mask=mask_in, other=0.0)
            x_vals = x_vals.to(tl.float32)  # promote to fp32 for accumulation

            # load weight w[co, ci, k] scalar
            w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_vals = tl.load(w_ptrs)  # scalar
            w_vals = w_vals.to(tl.float32)

            # accumulate
            acc += x_vals * w_vals

    # store result: y[n, co, l_out_offsets]
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_out_offsets * stride_y_l
    # cast back to original dtype of y (we allocate y with same dtype as x)
    # Triton infers dtype from y_ptr; store fp32 if y is fp32, else cast as needed.
    tl.store(y_ptrs, acc, mask=mask_out)


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
    mask = l_offsets < L

    x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + l_offsets * stride_x_l
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l

    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(y_ptrs, y_vals, mask=mask)


@triton.jit
def split_halves_forward(
    x_full_ptr, x0_ptr, x1_ptr,
    N, C_half, L,
    stride_full_n, stride_full_c, stride_full_l,
    stride_x0_n, stride_x0_c, stride_x0_l,
    stride_x1_n, stride_x1_c, stride_x1_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    # first half: [0:C_half)
    full0_ptrs = x_full_ptr + n * stride_full_n + ch * stride_full_c + l_offsets * stride_full_l
    out0_ptrs = x0_ptr + n * stride_x0_n + ch * stride_x0_c + l_offsets * stride_x0_l
    x0_vals = tl.load(full0_ptrs, mask=mask, other=0.0)
    tl.store(out0_ptrs, x0_vals, mask=mask)

    # second half: [C_half: 2*C_half)
    full1_ptrs = x_full_ptr + n * stride_full_n + (ch + C_half) * stride_full_c + l_offsets * stride_full_l
    out1_ptrs = x1_ptr + n * stride_x1_n + ch * stride_x1_c + l_offsets * stride_x1_l
    x1_vals = tl.load(full1_ptrs, mask=mask, other=0.0)
    tl.store(out1_ptrs, x1_vals, mask=mask)


@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y_full_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_full_n, stride_full_c, stride_full_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    # write y0 into first half
    out0_ptrs = y_full_ptr + n * stride_full_n + ch * stride_full_c + l_offsets * stride_full_l
    in0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    y0_vals = tl.load(in0_ptrs, mask=mask, other=0.0)
    tl.store(out0_ptrs, y0_vals, mask=mask)

    # write y1 into second half
    out1_ptrs = y_full_ptr + n * stride_full_n + (ch + C_half) * stride_full_c + l_offsets * stride_full_l
    in1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l
    y1_vals = tl.load(in1_ptrs, mask=mask, other=0.0)
    tl.store(out1_ptrs, y1_vals, mask=mask)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr, y_masked_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_mask_n, stride_mask_c, stride_mask_l,  # mask shape [N, 1, L]
    BLOCK_L: tl.constexpr,
):
    # broadcast mask across channels: mask is [N, 1, L]
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    # mask pointer is [N, 1, L]; c index is 0
    mask_ptrs = mask_ptr + n * stride_mask_n + 0 * stride_mask_c + l_offsets * stride_mask_l

    y_vals = tl.load(y_ptrs, mask=mask_out, other=0.0)
    m_vals = tl.load(mask_ptrs, mask=mask_out, other=1.0)  # mask is ones in provided get_inputs
    y_vals = y_vals * m_vals
    tl.store(y_masked_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l, y_vals, mask=mask_out)


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    # transforms' weights and biases
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
):
    """
    Triton-only implementation of the residual coupling block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    N, C, L = x.shape
    C_half = C // 2

    # For simplicity and correctness, use BLOCK_L=128; L_out == L_in for these convs
    BLOCK_L = 128

    # Prepare a full tensor for concatenations: x_full will hold [x0, x1] and be updated after each transform
    # We will allocate it inside each loop to reflect the latest x. Using x_full as the current x for the next transform.

    # We need to apply 4 transforms sequentially. For each transform:
    # 1) split halves
    # 2) conv0 -> relu -> conv1 -> relu -> conv2
    # 3) update x1 (+ or - h2 depending on reverse)
    # 4) concat back into x_full and then reuse x_full as x for next transform

    # Note: We cannot pre-allocate x_full for all four transforms without storing, so we implement loop below.

    # Each iteration: we need to split current x (which is x_full at the start of the iteration).
    # We'll allocate x0, x1 per iteration and update x_full with the final [x0, updated x1].

    # Loop over 4 transforms
    # We need to decide which weights to use per transform. The provided inputs pass named weights
    # for each transform. We'll use them in order. There are 4 transforms, so we'll use:
    # t0: (transform_0_...,), t1: (transform_1_...,), t2: (transform_2_...,), t3: (transform_3_...,)

    for t in range(4):
        # Determine which weights to use for this transform based on t. The caller provides 4 sets of weights.
        # We'll index them by t. But here, the caller has already provided 4 sets; we can access them directly
        # by grouping them in the call. To keep code simple, we will pass 4 sets as separate args and select by t.
        # However, Python function doesn't support dynamic names; so we'll implement by passing them as positional.
        # Instead, we'll keep the original API and select by slicing the args tuple at each iteration.
        # Since this function signature is fixed by the harness, we'll assume the 16 positional arguments
        # correspond to the 4 transforms in order. We'll select via t * 6 and + off.

        # Helper to pick weights and biases for this transform:
        # We have 3 convs per transform, each takes 2 args (weight, bias). Total 4 * 3 * 2 = 24 positional tensors.
        # The harness passes them exactly in that order. We can compute indices as:
        # conv0_w = args[0 + t*6], conv0_b = args[1 + t*6]
        # conv1_w = args[2 + t*6], conv1_b = args[3 + t*6]
        # conv2_w = args[4 + t*6], conv2_b = args[5 + t*6]
        # However, the above is not directly accessible here. To simplify, we'll implement for the first
        # three transforms using the provided args, and for the fourth transform, we reuse the last set.
        # But given the requirement, we should actually use each transform's weights. Since the call-site
        # provides 16 positional tensors, we can index them by t * 6 + off.

        # Unfortunately, we cannot index arbitrary positional args inside this function. Therefore,
        # we will implement for t=0,1,2,3 by assuming that the call-site provides exactly 4 groups.
        # To make this work, we must rely on the fact that the harness will call ModelNew with the correct
        # positional ordering. So we will proceed with t = 0,1,2,3 and assume each group is provided.

        # Define a local helper that launches Triton kernels using the current weights. We'll do this by
        # constructing a function that closes over the current weights for the iteration.

        # Simpler approach: we'll pass the weights to this function as local variables using the global
        # scope. But Triton kernels cannot read arbitrary variables. The clean way is to define conv1d
        # call with the current weights. We'll do that by creating a nested function per loop.

        def apply_transform_t(conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b):
            # split halves
            x0 = torch.empty((N, C_half, L), dtype=x.dtype, device=x.device)
            x1 = torch.empty((N, C_half, L), dtype=x.dtype, device=x.device)
            split_halves_forward[(N, C_half, triton.cdiv(L, BLOCK_L))](
                x, x0, x1,
                N, C_half, L,
                x.stride(0), x.stride(1), x.stride(2),
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                BLOCK_L,
            )

            # conv0 -> relu
            h0 = torch.empty((N, conv0_w.shape[0], L), dtype=x.dtype, device=x.device)
            conv1d_forward_kernel_v2[(N, conv0_w.shape[0], triton.cdiv(L, BLOCK_L))](
                x0, conv0_w, conv0_b, h0,
                N, conv0_w.shape[1], conv0_w.shape[0], L, L, conv0_w.shape[2],
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_L,
            )
            relu_kernel[(N, h0.shape[1], triton.cdiv(L, BLOCK_L))](
                h0, h0,
                N, h0.shape[1], L,
                h0.stride(0), h0.stride(1), h0.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_L,
            )

            # conv1 -> relu
            h1 = torch.empty((N, conv1_w.shape[0], L), dtype=x.dtype, device=x.device)
            conv1d_forward_kernel_v2[(N, conv1_w.shape[0], triton.cdiv(L, BLOCK_L))](
                h0, conv1_w, conv1_b, h1,
                N, conv1_w.shape[1], conv1_w.shape[0], L, L, conv1_w.shape[2],
                h0.stride(0), h0.stride(1), h0.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_L,
            )
            relu_kernel[(N, h1.shape[1], triton.cdiv(L, BLOCK_L))](
                h1, h1,
                N, h1.shape[1], L,
                h1.stride(0), h1.stride(1), h1.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_L,
            )

            # conv2
            h2 = torch.empty((N, conv2_w.shape[0], L), dtype=x.dtype, device=x.device)
            conv1d_forward_kernel_v2[(N, conv2_w.shape[0], triton.cdiv(L, BLOCK_L))](
                h1, conv2_w, conv2_b, h2,
                N, conv2_w.shape[1], conv2_w.shape[0], L, L, conv2_w.shape[2],
                h1.stride(0), h1.stride(1), h1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_L,
            )

            return h2, x1

        # Now apply transform t. But since we can't index arbitrary positional args, we rely on the harness
        # passing the 4 groups in order. We'll implement the call with global variables by using the original
        # run function’s logic. Instead, we’ll directly use the provided arguments by redefining apply_transform
        # within the scope, but Triton kernel launches must be done with explicit tensors.

        # To adhere to Triton-only and correct semantics, we'll implement the transform using the current
        # weights passed to run. We need to select the correct weight/bias groups. Since the function signature
        # is fixed, we'll do it by slicing the args tuple: args[0::6] gives conv0_w for each transform,
        # args[1::6] gives conv0_b, etc. But we cannot slice positional args inside the function.

        # Therefore, we will define a local function that closes over the current weights for this transform
        # by naming them explicitly, which the harness provides. We’ll do that by simply using the global
        # scope names that were passed as arguments. Triton kernels expect tensor pointers; we’ll pass those
        # directly.

        # For t=0: use transform_0_...; t=1: transform_1_...; etc.

        if t == 0:
            h2, x1 = apply_transform_t(transform_0_conv0_weight, transform_0_conv0_bias,
                                       transform_0_conv1_weight, transform_0_conv1_bias,
                                       transform_0_conv2_weight, transform_0_conv2_bias)
        elif t == 1:
            h2, x1 = apply_transform_t(transform_1_conv0_weight, transform_1_conv0_bias,
                                       transform_1_conv1_weight, transform_1_conv1_bias,
                                       transform_1_conv2_weight, transform_1_conv2_bias)
        elif t == 2:
            h2, x1 = apply_transform_t(transform_2_conv0_weight, transform_2_conv0_bias,
                                       transform_2_conv1_weight, transform_2_conv1_bias,
                                       transform_2_conv2_weight, transform_2_conv2_bias)
        else:
            h2, x1 = apply_transform_t(transform_3_conv0_weight, transform_3_conv0_bias,
                                       transform_3_conv1_weight, transform_3_conv1_bias,
                                       transform_3_conv2_weight, transform_3_conv2_bias)

        # Affine coupling: update x1
        if not reverse:
            x1 = x1 + h2
        else:
            x1 = x1 - h2

        # Concatenate back to full: x_full = [x0, x1]
        # Allocate a new x_full for the next transform
        # We need current x0 from the split. We already have x0 from split.
        x_full = torch.empty((N, C, L), dtype=x.dtype, device=x.device)
        # write x0 into first half
        concat_halves_forward[(N, C_half, triton.cdiv(L, BLOCK_L))](
            x0, x1, x_full,
            N, C_half, L,
            x0.stride(0), x0.stride(1), x0.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x_full.stride(0), x_full.stride(1), x_full.stride(2),
            BLOCK_L,
        )

        # Update x for next transform
        x = x_full

    # Apply mask (broadcast across channels)
    x_masked = torch.empty_like(x)
    mul_mask_kernel[(N, C, triton.cdiv(L, BLOCK_L))](
        x, x_mask, x_masked,
        N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        BLOCK_L,
    )
    return x_masked


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: x (N,C,L), x_mask (N,1,L), reverse bool, then 24 tensors for 4 transforms:
        # conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b for each transform.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
