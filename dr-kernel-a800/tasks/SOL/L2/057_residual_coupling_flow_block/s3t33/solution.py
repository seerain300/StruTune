import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels

@triton.jit
def conv1d_strided_nopad_kernel(
    x_ptr,          # *f32, input [N, Cin, L_in]
    w_ptr,          # *f32, weight [Cout, Cin, K], K=5
    b_ptr,          # *f32, bias [Cout] (zeros if conv2)
    y_ptr,          # *f32, output [N, Cout, L_out], L_out = L_in - 4
    N, Cin, L_in, Cout, L_out, K,
    stride_xn, stride_xc, stride_xt,
    stride_woc, stride_wic, stride_wk,
    stride_yn, stride_yc, stride_yt,
    BLOCK_T: tl.constexpr,
):
    # program ids: pid0 -> (n, oc), pid1 -> tile along time
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n = pid0 // Cout
    oc = pid0 % Cout

    t_offsets = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L_out

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    c_in = 0
    while c_in < Cin:
        k = 0
        while k < K:
            l_in = t_offsets + k  # valid because L_out = L_in - K
            x_off = n * stride_xn + c_in * stride_xc + l_in * stride_xt
            # load x values for this k across the tile; mask ensures l_in < L_in
            x_vals = tl.load(x_ptr + x_off, mask=mask_t, other=0.0)
            # load weight scalar w[oc, c_in, k]
            w_off = oc * stride_woc + c_in * stride_wic + k * stride_wk
            w_val = tl.load(w_ptr + w_off)
            acc += x_vals * w_val
            k += 1
        c_in += 1

    # add bias if provided
    if b_ptr != 0:
        b_val = tl.load(b_ptr + oc)
        acc += b_val

    # store to y[n, oc, t_offsets]
    y_off = n * stride_yn + oc * stride_yc + t_offsets * stride_yt
    tl.store(y_ptr + y_off, acc, mask=mask_t)


@triton.jit
def relu_kernel(x_ptr, y_ptr, N, C, L, stride_xn, stride_xc, stride_xt, stride_yn, stride_yc, stride_yt, BLOCK_T: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    n = pid0 // C
    c = pid0 % C
    t_offsets = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L
    x_off = n * stride_xn + c * stride_xc + t_offsets * stride_xt
    x_vals = tl.load(x_ptr + x_off, mask=mask_t, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)
    y_off = n * stride_yn + c * stride_yc + t_offsets * stride_yt
    tl.store(y_ptr + y_off, y_vals, mask=mask_t)


@triton.jit
def mul_mask_kernel(x_ptr, mask_ptr, y_ptr, N, C, Lx, mask_L, stride_xn, stride_xc, stride_xt, stride_yn, stride_yc, stride_yt,
                     stride_mn, stride_mc, stride_mt, BLOCK_T: tl.constexpr):
    # mask shape [N, 1, Lmask], here Lmask == Lx, Cmask == 1
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    n = pid0 // C
    c = pid0 % C
    t_offsets = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < Lx

    x_off = n * stride_xn + c * stride_xc + t_offsets * stride_xt
    x_vals = tl.load(x_ptr + x_off, mask=mask_t, other=0.0)

    # mask is [N, 1, Lx], so load with c=0
    m_off = n * stride_mn + 0 * stride_mc + t_offsets * stride_mt
    m_vals = tl.load(mask_ptr + m_off, mask=mask_t, other=1.0)

    y_vals = x_vals * m_vals
    y_off = n * stride_yn + c * stride_yc + t_offsets * stride_yt
    tl.store(y_ptr + y_off, y_vals, mask=mask_t)


@triton.jit
def add_affine_kernel(x1_ptr, h_ptr, y1_ptr, N, C, L, stride_x1n, stride_x1c, stride_x1t,
                      stride_hn, stride_hc, stride_ht, stride_y1n, stride_y1c, stride_y1t, reverse: tl.int1, BLOCK_T: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    n = pid0 // C
    c = pid0 % C
    t_offsets = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x1_off = n * stride_x1n + c * stride_x1c + t_offsets * stride_x1t
    h_off = n * stride_hn + c * stride_hc + t_offsets * stride_ht
    x1_vals = tl.load(x1_ptr + x1_off, mask=mask_t, other=0.0)
    h_vals = tl.load(h_ptr + h_off, mask=mask_t, other=0.0)

    if reverse:
        y1_vals = x1_vals - h_vals
    else:
        y1_vals = x1_vals + h_vals

    y1_off = n * stride_y1n + c * stride_y1c + t_offsets * stride_y1t
    tl.store(y1_ptr + y1_off, y1_vals, mask=mask_t)


@triton.jit
def concat_channel_kernel(y0_ptr, y1_ptr, y_ptr, N, C0, C1, L, stride_y0n, stride_y0c, stride_y0t,
                           stride_y1n, stride_y1c, stride_y1t, stride_yn, stride_yc, stride_yt, BLOCK_T: tl.constexpr):
    # This kernel concatenates two channel tensors y0 [N, C0, L] and y1 [N, C1, L] into y [N, C0+C1, L].
    total_c = C0 + C1
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    n = pid0 // total_c
    c_total = pid0 % total_c

    t_offsets = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    if c_total < C0:
        src_ptr = y0_ptr
        c_src = c_total
        dst_c = c_total  # since c_total < C0, no offset needed
    else:
        src_ptr = y1_ptr
        c_src = c_total - C0
        dst_c = c_total

    src_off = n * stride_y0n if c_src < C0 else n * stride_y1n + c_src * (stride_y1c - stride_y0c)  # wait, better compute separately

    # The above comment line is not ideal; let's correct by separate handling:
    if c_total < C0:
        src_off = n * stride_y0n + c_src * stride_y0c + t_offsets * stride_y0t
        dst_off = n * stride_yn + dst_c * stride_yc + t_offsets * stride_yt
        vals = tl.load(src_ptr + src_off, mask=mask_t, other=0.0)
        tl.store(y_ptr + dst_off, vals, mask=mask_t)
    else:
        src_off = n * stride_y1n + (c_src) * stride_y1c + t_offsets * stride_y1t
        dst_off = n * stride_yn + dst_c * stride_yc + t_offsets * stride_yt
        vals = tl.load(src_ptr + src_off, mask=mask_t, other=0.0)
        tl.store(y_ptr + dst_off, vals, mask=mask_t)


# Function that runs a single transform using Triton kernels.
def run_single_transform(x: torch.Tensor, conv0_w: torch.Tensor, conv0_b: torch.Tensor,
                         conv1_w: torch.Tensor, conv1_b: torch.Tensor,
                         conv2_w: torch.Tensor, conv2_b: torch.Tensor,
                         x_mask: torch.Tensor, reverse: bool, device: torch.device):
    N, C, L = x.shape
    half = C // 2

    x0 = x[:, :half, :]
    x1 = x[:, half:, :]

    # conv0: Cin=half, Cout=192, K=5
    Cin0 = half
    Cout0 = 192
    L0 = L - 4  # output time length
    y0 = torch.empty((N, Cout0, L0), dtype=x.dtype, device=device)
    grid0 = (N * Cout0, triton.cdiv(L0, 128))
    conv1d_strided_nopad_kernel[grid0](
        x0, conv0_w, conv0_b if conv0_b is not None else torch.zeros(192, dtype=x.dtype, device=device),
        y0, N, Cin0, L, Cout0, L0, 5,
        x0.stride(0), x0.stride(1), x0.stride(2),
        conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
        y0.stride(0), y0.stride(1), y0.stride(2),
        BLOCK_T=128, num_warps=4
    )
    # ReLU
    y0_relu = torch.empty_like(y0)
    grid_relu = (N * Cout0, triton.cdiv(L0, 128))
    relu_kernel[grid_relu](
        y0, y0_relu, N, Cout0, L0,
        y0.stride(0), y0.stride(1), y0.stride(2),
        y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
        BLOCK_T=128, num_warps=4
    )
    # conv1: Cin=192, Cout=192, K=5
    Cin1 = 192
    Cout1 = 192
    L1 = L0 - 4
    y1 = torch.empty((N, Cout1, L1), dtype=x.dtype, device=device)
    grid1 = (N * Cout1, triton.cdiv(L1, 128))
    conv1d_strided_nopad_kernel[grid1](
        y0_relu, conv1_w, conv1_b if conv1_b is not None else torch.zeros(192, dtype=x.dtype, device=device),
        y1, N, Cin1, L0, Cout1, L1, 5,
        y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
        conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
        y1.stride(0), y1.stride(1), y1.stride(2),
        BLOCK_T=128, num_warps=4
    )
    # ReLU
    y1_relu = torch.empty_like(y1)
    grid_relu1 = (N * Cout1, triton.cdiv(L1, 128))
    relu_kernel[grid_relu1](
        y1, y1_relu, N, Cout1, L1,
        y1.stride(0), y1.stride(1), y1.stride(2),
        y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
        BLOCK_T=128, num_warps=4
    )
    # conv2: Cin=192, Cout=96, K=5
    Cin2 = 192
    Cout2 = 96
    L2 = L1 - 4
    h = torch.empty((N, Cout2, L2), dtype=x.dtype, device=device)
    grid2 = (N * Cout2, triton.cdiv(L2, 128))
    conv1d_strided_nopad_kernel[grid2](
        y1_relu, conv2_w, conv2_b if conv2_b is not None else torch.zeros(96, dtype=x.dtype, device=device),
        h, N, Cin2, L1, Cout2, L2, 5,
        y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
        conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        BLOCK_T=128, num_warps=4
    )
    # Multiply by mask [N,1,L], broadcast across channels
    h_masked = torch.empty_like(h)
    grid_mul = (N * Cout2, triton.cdiv(L2, 128))
    mul_mask_kernel[grid_mul](
        h, x_mask, h_masked, N, Cout2, L2, x_mask.shape[2],
        h.stride(0), h.stride(1), h.stride(2),
        h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        BLOCK_T=128, num_warps=4
    )
    # Affine coupling on x1
    x1_out = torch.empty_like(x1)
    grid_affine = (N * half, triton.cdiv(L, 128))
    add_affine_kernel[grid_affine](
        x1, h_masked, x1_out, N, half, L,
        x1.stride(0), x1.stride(1), x1.stride(2),
        h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
        x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
        reverse, BLOCK_T=128, num_warps=4
    )
    # Concatenate x0 and updated x1 along channels to form new x
    new_x = torch.empty((N, C, L), dtype=x.dtype, device=device)
    grid_concat = (N * (half + half), triton.cdiv(L, 128))
    concat_channel_kernel[grid_concat](
        x0, x1_out, new_x, N, half, half, L,
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
        new_x.stride(0), new_x.stride(1), new_x.stride(2),
        BLOCK_T=128, num_warps=4
    )
    # Multiply final output by x_mask (broadcast across channels)
    masked_new_x = torch.empty_like(new_x)
    grid_final_mul = (N * C, triton.cdiv(L, 128))
    mul_mask_kernel[grid_final_mul](
        new_x, x_mask, masked_new_x, N, C, L, x_mask.shape[2],
        new_x.stride(0), new_x.stride(1), new_x.stride(2),
        masked_new_x.stride(0), masked_new_x.stride(1), masked_new_x.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        BLOCK_T=128, num_warps=4
    )
    return masked_new_x


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool,
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
        Triton-optimized forward. All numerical ops are executed in Triton kernels.
        The computation replicates the original behavior:
        - Split x into x0 and x1 halves along channels.
        - Apply 3 Conv1d (padding=0, K=5), ReLU after each.
        - Multiply final h by x_mask.
        - Affine coupling: x1 = x1 + h (forward) or x1 = x1 - h (reverse).
        - Concatenate x0 and updated x1, then multiply by x_mask.
        This is repeated 4 times.
        """
        N, C, L = x.shape
        device = x.device

        # Ensure masks are float32 for Triton compute
        x_mask = x_mask.to(dtype=torch.float32, device=device)

        # Perform 4 transforms sequentially
        x = run_single_transform(
            x, transform_0_conv0_weight, transform_0_conv0_bias,
            transform_0_conv1_weight, transform_0_conv1_bias,
            transform_0_conv2_weight, transform_0_conv2_bias,
            x_mask, reverse, device
        )

        x = run_single_transform(
            x, transform_1_conv0_weight, transform_1_conv0_bias,
            transform_1_conv1_weight, transform_1_conv1_bias,
            transform_1_conv2_weight, transform_1_conv2_bias,
            x_mask, reverse, device
        )

        x = run_single_transform(
            x, transform_2_conv0_weight, transform_2_conv0_bias,
            transform_2_conv1_weight, transform_2_conv1_bias,
            transform_2_conv2_weight, transform_2_conv2_bias,
            x_mask, reverse, device
        )

        x = run_single_transform(
            x, transform_3_conv0_weight, transform_3_conv0_bias,
            transform_3_conv1_weight, transform_3_conv1_bias,
            transform_3_conv2_weight, transform_3_conv2_bias,
            x_mask, reverse, device
        )

        return x


def run(*args):
    return ModelNew()(*args)
