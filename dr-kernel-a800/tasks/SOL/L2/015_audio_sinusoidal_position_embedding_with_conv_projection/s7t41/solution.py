import math
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_pad1_bias_gelu_kernel(
    x_ptr,          # *const bfloat16, input [B, C_in, H, W]
    w_ptr,          # *const bfloat16, weights [C_out, C_in, 3, 3]
    b_ptr,          # *const bfloat16, biases [C_out]
    y_ptr,          # *bfloat16, output [B, C_out, H_out, W_out]
    B, C_in, H, W,  # int32 runtime sizes
    C_out, H_out, W_out,  # int32 output sizes
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,  # int32 strides for x
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,  # int32 strides for w
    stride_y_b, stride_y_c, stride_y_h, stride_y_w,  # int32 strides for y
    scale,  # float32 embed_scale (unused here; kept for signature symmetry)
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # tile coordinates
    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # masks for valid output region
    oh_mask = oh < H_out
    ow_mask = ow < W_out
    # build 2D mask
    mask_out = oh_mask[:, None] & ow_mask[None, :]

    # initialize output tile
    # we'll accumulate in float32 for numerical stability
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # loop over input channels and 3x3 kernel
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # input coordinates with padding=1, stride=2
                ih = 2 * oh - 1 + kh  # (oh - kh) * 2 + (2*pad - kh) -> use 2*oh - 1 + kh
                iw = 2 * ow - 1 + kw  # similar for width
                # valid input mask (also respects oh, ow validity)
                valid_ih = (ih >= 0) & (ih < H)
                valid_iw = (iw >= 0) & (iw < W)
                valid_mask = valid_ih[:, None] & valid_iw[None, :] & mask_out

                # input pointers for the tile
                x_ptrs = x_ptr + pid_b * stride_x_b + ci * stride_x_c + ih[:, None] * stride_x_h + iw[None, :] * stride_x_w
                # load input tile; cast to float32
                x_tile = tl.load(x_ptrs, mask=valid_mask, other=0.0)
                x_tile = x_tile.to(tl.float32)

                # load corresponding weight scalar (bias-free for now; we'll add bias later)
                # weight is [C_out, C_in, 3, 3]
                w_scalar = tl.load(w_ptr + pid_co * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw)
                w_scalar = w_scalar.to(tl.float32)

                # elementwise multiply and accumulate
                acc += x_tile * w_scalar

    # add bias
    b_val = tl.load(b_ptr + pid_co)
    acc = acc + b_val.to(tl.float32)

    # GELU: tanh approximation
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # constants
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + c1 * x3)))

    # store to output (cast back to bfloat16)
    y_ptrs = y_ptr + pid_b * stride_y_b + pid_co * stride_y_c + oh[:, None] * stride_y_h + ow[None, :] * stride_y_w
    tl.store(y_ptrs, gelu.to(tl.bfloat16), mask=mask_out)


@triton.jit
def linear_gemm_scale_add_pos_kernel(
    x_ptr,          # *const bfloat16, input [B*S, K] flattened
    w_ptr,          # *const bfloat16, weights [N, K] (we access as W[N,K] and multiply rows)
    y_ptr,          # *bfloat16, output [B*S, N] flattened
    pos_ptr,        # *const bfloat16, positional embedding [S, N] flattened
    B, S, N, K,     # int32 sizes
    scale,          # float32 embed_scale
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program handles one "row" of x (i.e., one (batch, time) pair)
    row = tl.program_id(0)

    # total number of rows
    total_rows = B * S
    # if row >= total_rows, we can early return; Triton masks handle this via row < total_rows
    # compute b and t indices from row
    b = row // S
    t = row % S

    # pointer to this row in x
    x_row_ptr = x_ptr + row * K

    # accumulate in float32
    acc = tl.zeros((N,), dtype=tl.float32)

    # loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_ids < K

        # load x_row[k_ids]
        x_k = tl.load(x_row_ptr + k_ids, mask=k_mask, other=0.0).to(tl.float32)

        # load w_chunk [BLOCK_N] corresponding to k_ids
        # w is [N, K]; we access columns k_ids
        w_ptrs = w_ptr + tl.arange(0, BLOCK_N) * K + k_ids
        w_chunk = tl.load(w_ptrs, mask=(tl.arange(0, BLOCK_N) < N) & k_mask, other=0.0).to(tl.float32)

        # partial dot product: sum over BLOCK_K
        # acc[n] += sum(w_chunk[n, :] * x_k[:])
        # Broadcast multiply and reduce
        acc += tl.sum(w_chunk * x_k[None, :], axis=1)

    # add positional embedding for this (b, t)
    # pos layout is [S, N] flattened as [S*N], index = t * N + n
    # We add it to acc and then scale by embed_scale
    # Note: pos_ptr is 1D flattened
    # Triton does not support pythonic loops over N easily here; we handle one n at a time.
    # To keep code simple, we add pos per channel in a small loop.
    # This loop is limited to N=1024; acceptable for performance.
    for n in range(0, N):
        pos_val = tl.load(pos_ptr + t * N + n).to(tl.float32)
        acc[n] += pos_val
    # scale
    acc = acc * scale

    # store to y at row
    y_row_ptr = y_ptr + row * N
    tl.store(y_row_ptr, acc.to(tl.bfloat16))


# Elementwise kernels (not used in the main path but defined for completeness if needed)
@triton.jit
def scale_elementwise_kernel(x_ptr, scale, N_elems: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x * scale
    tl.store(x_ptr + offs, x.to(tl.bfloat16), mask=mask)


@triton.jit
def add_pos_emb_elementwise_kernel(y_ptr, pos_ptr, N_elems: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    # y_ptr is [B*S*N] flattened; derive (b, t) and add pos[n]
    # For simplicity, we assume that this kernel is used only for adding pos to y after scaling.
    # Each element corresponds to y[row*N + n] where row = batch*S + time index is known via mask;
    # but Triton doesn't have row decoding here. We'll instead use the linear_gemm_scale_add_pos_kernel
    # to perform both operations in one go.

# Main ModelNew class
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        Triton-only implementation:
        - Three conv2d (stride=2, padding=1) + GELU in Triton.
        - Linear projection using Triton GEMM in one kernel (plus scale and add positional embedding).
        """
        device = input_features.device
        dtype_x = input_features.dtype  # torch.bfloat16

        # Conv1: (B, 1, 80, T) -> (B, 384, 40, T//2)
        B = input_features.shape[0]
        C_in1 = 1
        H1 = input_features.shape[2]
        W1 = input_features.shape[3]
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H1 + 2*1 - 3)//2 + 1
        W_out1 = (W1 + 2*1 - 3)//2 + 1
        x1 = torch.empty((B, C_out1, H_out1, W_out1), device=device, dtype=torch.bfloat16)

        grid1 = (B, C_out1, triton.cdiv(H_out1, 8), triton.cdiv(W_out1, 8))
        conv2d_stride2_pad1_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, C_in1, H1, W1, C_out1, H_out1, W_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            float(embed_scale),
            BLOCK_H=8, BLOCK_W=8, num_warps=4, num_stages=2
        )

        # Conv2: (B, 384, 40, T//2) -> (B, 384, 20, T//4)
        C_in2 = C_out1
        H2 = x1.shape[2]
        W2 = x1.shape[3]
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = (H2 + 2*1 - 3)//2 + 1
        W_out2 = (W2 + 2*1 - 3)//2 + 1
        x2 = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=torch.bfloat16)

        grid2 = (B, C_out2, triton.cdiv(H_out2, 8), triton.cdiv(W_out2, 8))
        conv2d_stride2_pad1_bias_gelu_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            float(embed_scale),
            BLOCK_H=8, BLOCK_W=8, num_warps=4, num_stages=2
        )

        # Conv3: (B, 384, 20, T//4) -> (B, 384, 10, T//8)
        C_in3 = C_out2
        H3 = x2.shape[2]
        W3 = x2.shape[3]
        C_out3 = conv2d3_weight.shape[0]
        H_out3 = (H3 + 2*1 - 3)//2 + 1
        W_out3 = (W3 + 2*1 - 3)//2 + 1
        x3 = torch.empty((B, C_out3, H_out3, W_out3), device=device, dtype=torch.bfloat16)

        grid3 = (B, C_out3, triton.cdiv(H_out3, 8), triton.cdiv(W_out3, 8))
        conv2d_stride2_pad1_bias_gelu_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, C_in3, H3, W3, C_out3, H_out3, W_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            float(embed_scale),
            BLOCK_H=8, BLOCK_W=8, num_warps=4, num_stages=2
        )

        # Reshape: (B, T//8, 384*10)
        S = x3.shape[3]  # time_after_conv
        C_F = C_out3 * x3.shape[2]  # channels*freq = 384*10
        x_flat = x3.permute(0, 3, 1, 2).contiguous().view(B, S, C_F)

        # Linear projection: (B, S, K=3840) @ (N=1024, K)^T -> (B, S, N)
        # Flatten x to [B*S, K]
        B2 = B
        S2 = S
        K = C_F  # 3840
        N = conv_out_weight.shape[0]  # 1024
        x_row = x_flat.view(B2 * S2, K).contiguous()  # [B*S, K] bfloat16
        w = conv_out_weight  # [N, K] bfloat16
        y = torch.empty((B2 * S2, N), device=device, dtype=torch.bfloat16)

        # Launch Triton GEMM + scale + add positional embedding in one kernel
        BLOCK_N = 128
        BLOCK_K = 64
        grid_linear = (B2 * S2,)
        linear_gemm_scale_add_pos_kernel[grid_linear](
            x_row, w, y,
            positional_embedding.to(torch.bfloat16).contiguous().view(-1),  # flatten [S*N]
            B2, S2, N, K,
            float(embed_scale),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape back to [B, S, N]
        y = y.view(B, S, N)

        return y


def run(*args):
    return ModelNew()(*args)
