import math
import torch


# Triton kernels

@triton.jit
def conv2d_stride2_3x3(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B: tl.int32, C_out: tl.int32, C_in: tl.int32,
    IH: tl.int32, IW: tl.int32,  # input height/width
    OH: tl.int32, OW: tl.int32,  # output height/width
    x_stride_b: tl.int32, x_stride_c: tl.int32, x_stride_h: tl.int32, x_stride_w: tl.int32,
    w_stride_oc: tl.int32, w_stride_ic: tl.int32, w_stride_h: tl.int32, w_stride_w: tl.int32,
    y_stride_b: tl.int32, y_stride_c: tl.int32, y_stride_h: tl.int32, y_stride_w: tl.int32,
):
    # program ids
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # accumulate in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # iterate over input channels and 3x3 taps
    for ic in range(0, C_in):
        for kh in range(0, 3):
            ih = 2 * oh + kh - 1
            in_h_ok = (ih >= 0) & (ih < IH)
            for kw in range(0, 3):
                iw = 2 * ow + kw - 1
                in_w_ok = (iw >= 0) & (iw < IW)
                in_bounds = in_h_ok & in_w_ok
                if in_bounds:
                    x_ptr_elem = X_ptr + b * x_stride_b + ic * x_stride_c + ih * x_stride_h + iw * x_stride_w
                    x_val = tl.load(x_ptr_elem).to(tl.float32)
                else:
                    x_val = tl.zeros((), dtype=tl.float32)
                # load weight for (oc, ic, kh, kw)
                w_ptr_elem = W_ptr + oc * w_stride_oc + ic * w_stride_ic + kh * w_stride_h + kw * w_stride_w
                w_val = tl.load(w_ptr_elem).to(tl.float32)
                acc += x_val * w_val

    # add bias
    bias_val = tl.load(BIAS_ptr + oc).to(tl.float32)
    acc += bias_val

    # store to Y
    y_ptr_elem = Y_ptr + b * y_stride_b + oc * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr_elem, acc)


@triton.jit
def gelu_1d_kernel(X_ptr, Y_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    # tanh approximation of GELU
    # gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c0 = 0.5
    c1 = 0.7978845608028654  # sqrt(2/pi)
    c2 = 0.044715
    x3 = x * x * x
    t = c1 * (x + c2 * x3)
    y = c0 * x * (1.0 + tl.tanh(t))
    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def linear_project_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B: tl.int32, T: tl.int32, N: tl.int32, M: tl.int32,
    scale: tl.float32,
):
    # Grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)

    # We process N in tiles; for each m tile, accumulate across N
    BLOCK_M = 128
    BLOCK_N = 256

    for m0 in range(0, M, BLOCK_M):
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < M
        acc = tl.zeros([BLOCK_M], dtype=tl.float32)

        for n0 in range(0, N, BLOCK_N):
            n_offsets = n0 + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < N
            # Load X[b, t, n_offsets]
            x_ptrs = X_ptr + b * (T * N) + t * N + n_offsets
            x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)  # [BLOCK_N]
            # Load W[m_offsets, n_offsets] as [BLOCK_M, BLOCK_N]
            w_ptrs = W_ptr + m_offsets[:, None] * N + n_offsets[None, :]
            w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        # scale and add positional embedding
        acc = acc * scale
        pos_ptrs = pos_ptr + t * M + m_offsets
        pos_vals = tl.load(pos_ptrs, mask=mask_m, other=0.0)
        acc = acc + pos_vals
        # store to Y[b, t, m_offsets]
        y_ptrs = Y_ptr + b * (T * M) + t * M + m_offsets
        tl.store(y_ptrs, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T), dtype bfloat16 (we cast to float32 for computation)
        conv2d* weights: (C_out, C_in, 3, 3), bias: (C_out)
        conv_out_weight: (1024, 384*10) = (1024, 3840)
        positional_embedding: (1500, 1024)
        """
        device = input_features.device
        # Cast inputs for computation
        x = input_features.to(torch.float32)

        # Conv1: (B, 1, 80, T) -> (B, 384, 40, (T+1)//2)
        B, C_in1, IH, IW = x.shape
        C_out1 = 384
        OH1 = (IH - 1) // 2 + 1  # = 40
        T1 = (IW + 1) // 2
        x1 = torch.empty((B, C_out1, OH1, T1), device=device, dtype=torch.float32)

        grid1 = (B, C_out1, OH1, T1)
        conv2d_stride2_3x3[grid1](
            x, conv2d1_weight, conv2d1_bias, x1,
            B, C_out1, C_in1, IH, IW, OH1, T1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # GELU after conv1
        N1 = x1.numel()
        x1_flat = x1.view(-1).contiguous()
        y1_flat = torch.empty(N1, device=device, dtype=torch.float32)
        BLOCK1 = 1024
        grid_g1 = (triton.cdiv(N1, BLOCK1),)
        gelu_1d_kernel[grid_g1](x1_flat, y1_flat, N1, BLOCK1)
        x1 = y1_flat.view_as(x1)

        # Conv2: (B, 384, 40, T1) -> (B, 384, 20, (T1+1)//2)
        C_in2 = C_out1
        C_out2 = C_in2  # 384
        OH2 = (OH1 - 1) // 2 + 1  # = 20
        T2 = (T1 + 1) // 2
        x2 = torch.empty((B, C_out2, OH2, T2), device=device, dtype=torch.float32)

        grid2 = (B, C_out2, OH2, T2)
        conv2d_stride2_3x3[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_out2, C_in2, OH1, T1, OH2, T2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # GELU after conv2
        N2 = x2.numel()
        x2_flat = x2.view(-1).contiguous()
        y2_flat = torch.empty(N2, device=device, dtype=torch.float32)
        grid_g2 = (triton.cdiv(N2, BLOCK1),)
        gelu_1d_kernel[grid_g2](x2_flat, y2_flat, N2, BLOCK1)
        x2 = y2_flat.view_as(x2)

        # Conv3: (B, 384, 20, T2) -> (B, 384, 10, (T2+1)//2) == (B, 384, 10, time_after_conv)
        C_in3 = C_out2
        C_out3 = C_in3  # 384
        OH3 = (OH2 - 1) // 2 + 1  # = 10
        T3 = (T2 + 1) // 2  # equals time_after_conv in harness
        x3 = torch.empty((B, C_out3, OH3, T3), device=device, dtype=torch.float32)

        grid3 = (B, C_out3, OH3, T3)
        conv2d_stride2_3x3[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, C_out3, C_in3, OH2, T2, OH3, T3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # GELU after conv3
        N3 = x3.numel()
        x3_flat = x3.view(-1).contiguous()
        y3_flat = torch.empty(N3, device=device, dtype=torch.float32)
        grid_g3 = (triton.cdiv(N3, BLOCK1),)
        gelu_1d_kernel[grid_g3](x3_flat, y3_flat, N3, BLOCK1)
        x3 = y3_flat.view_as(x3)

        # Final linear projection and positional embedding addition
        # N = 384 * 10 = 3840
        B_T = B * T3
        N = C_out3 * 10  # 3840
        M = conv_out_weight.shape[0]  # 1024
        # Reshape X3 to (B, T3, N)
        X = x3.view(B, T3, N).contiguous()
        X_flat = X.view(B_T * N).contiguous()

        # Allocate output (B, T3, M)
        Y_flat = torch.empty(B_T * M, device=device, dtype=torch.float32)

        grid_lin = (B, T3)
        linear_project_pos_kernel[grid_lin](
            X_flat, conv_out_weight, positional_embedding, Y_flat,
            B, T3, N, M, float(embed_scale),
        )

        output = Y_flat.view(B, T3, M)
        return output


def run(*args):
    return ModelNew()(*args)
