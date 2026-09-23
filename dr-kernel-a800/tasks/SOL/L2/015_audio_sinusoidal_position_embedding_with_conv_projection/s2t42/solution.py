import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: 2D conv with stride=2, padding=1, 3x3
# Computes one output element y[b, oc, oh, ow] for a given (b, oc, oh, ow)
@triton.jit
def conv2d_stride2_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H_in, W_in,
    OC, H_out, W_out,
    X_stride_b, X_stride_c, X_stride_h, X_stride_w,
    W_stride_oc, W_stride_ic, W_stride_kh, W_stride_kw,
    BIAS_stride,
    Y_stride_b, Y_stride_oc, Y_stride_h, Y_stride_w,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator for output
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for ic in range(0, C_in):
        for kh in range(0, 3):
            ih = 2 * oh + kh - 1
            for kw in range(0, 3):
                iw = 2 * ow + kw - 1
                # in-bounds check
                in_bounds = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)
                # Load input value (X[b, ic, ih, iw])
                x_ptr = X_ptr + b * X_stride_b + ic * X_stride_c + ih * X_stride_h + iw * X_stride_w
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)
                # Load weight for this (oc, ic, kh, kw) (W[oc, ic, kh, kw])
                w_ptr = W_ptr + oc * W_stride_oc + ic * W_stride_ic + kh * W_stride_kh + kw * W_stride_kw
                w_val = tl.load(w_ptr)
                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + oc * BIAS_stride)
    acc += bias_val

    # Store to Y[b, oc, oh, ow]
    y_ptr = Y_ptr + b * Y_stride_b + oc * Y_stride_oc + oh * Y_stride_h + ow * Y_stride_w
    tl.store(y_ptr, acc)


# Triton kernel: GELU (tanh approximation) over a flattened tensor
@triton.jit
def gelu_tanh_inplace_kernel(X_ptr, NEL):
    idx = tl.program_id(0)
    if idx < NEL:
        x = tl.load(X_ptr + idx)
        # GELU tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        x3 = x * x * x
        gelu = 0.5 * x * (1.0 + tl.tanh(c0 * (x + c1 * x3)))
        tl.store(X_ptr + idx, gelu)


# Triton kernel: final linear projection (X @ W^T), scale, add pos_emb
# X: [B, T_after, N], W: [M=1024, N], pos_emb: [T_after, M], Y: [B, T_after, M]
@triton.jit
def linear_project_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B, T_after, N, M,
    scale: tl.float32,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    m_block = tl.program_id(2)

    m_offsets = m_block * 128 + tl.arange(0, 128)
    mask_m = m_offsets < M

    acc = tl.zeros([128], dtype=tl.float32)

    # Loop over N in tiles of 256
    for n0 in range(0, N, 256):
        n_offsets = n0 + tl.arange(0, 256)
        mask_n = n_offsets < N
        # Load X[b, t, n_offsets] -> [256]
        x_ptrs = X_ptr + b * (T_after * N) + t * N + n_offsets
        x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        # Load W[m_offsets, n_offsets] -> [128, 256]
        w_ptrs = W_ptr + m_offsets[:, None] * N + n_offsets[None, :]
        w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        # Accumulate per m: sum over n of x * w
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Apply scale and add pos_emb[t, :]
    acc = acc * scale
    pos_vec = tl.load(pos_ptr + t * M + m_offsets, mask=mask_m, other=0.0).to(tl.float32)
    acc = acc + pos_vec

    # Store to Y[b, t, m_offsets]
    y_ptrs = Y_ptr + b * (T_after * M) + t * M + m_offsets
    tl.store(y_ptrs, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T), bfloat16
        conv2d1_weight: (384, 1, 3, 3), bfloat16
        conv2d1_bias: (384), bfloat16
        conv2d2_weight, conv2d3_weight: (384, 384, 3, 3), bfloat16
        conv2d2_bias, conv2d3_bias: (384), bfloat16
        conv_out_weight: (1024, 3840), bfloat16
        positional_embedding: (1500, 1024), bfloat16
        embed_scale: float, e.g., 32.0
        """
        B = input_features.shape[0]
        H_in = 80
        W_in = input_features.shape[3]  # original T
        # Stage 1 conv: C_in=1, C_out=384
        OC1 = 384
        H_out1 = (H_in + 1) // 2  # 40
        W_out1 = (W_in + 1) // 2  # (T+1)//2
        x1 = torch.empty((B, OC1, H_out1, W_out1), device=input_features.device, dtype=torch.float32)
        # Launch conv kernel for conv1
        grid1 = (B, OC1, H_out1, W_out1)
        conv2d_stride2_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, 1, H_in, W_in, OC1, H_out1, W_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            conv2d1_bias.stride(0),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            num_warps=4, num_stages=2
        )
        # GELU conv1
        x1_flat = x1.reshape(-1)
        NEL1 = x1_flat.numel()
        grid_g1 = (NEL1,)
        gelu_tanh_inplace_kernel[grid_g1](x1_flat, NEL1, num_warps=4, num_stages=2)
        x1 = x1.reshape(B, OC1, H_out1, W_out1)

        # Stage 2 conv: C_in=384, C_out=384
        OC2 = 384
        H_out2 = (H_out1 + 1) // 2  # 20
        W_out2 = (W_out1 + 1) // 2  # (T+3)//4
        x2 = torch.empty((B, OC2, H_out2, W_out2), device=input_features.device, dtype=torch.float32)
        grid2 = (B, OC2, H_out2, W_out2)
        conv2d_stride2_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, OC1, H_out1, W_out1, OC2, H_out2, W_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            conv2d2_bias.stride(0),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            num_warps=4, num_stages=2
        )
        # GELU conv2
        x2_flat = x2.reshape(-1)
        NEL2 = x2_flat.numel()
        grid_g2 = (NEL2,)
        gelu_tanh_inplace_kernel[grid_g2](x2_flat, NEL2, num_warps=4, num_stages=2)
        x2 = x2.reshape(B, OC2, H_out2, W_out2)

        # Stage 3 conv: C_in=384, C_out=384
        OC3 = 384
        H_out3 = (H_out2 + 1) // 2  # 10
        W_out3 = (W_out2 + 1) // 2  # (T+3)//8 == time_after_conv
        x3 = torch.empty((B, OC3, H_out3, W_out3), device=input_features.device, dtype=torch.float32)
        grid3 = (B, OC3, H_out3, W_out3)
        conv2d_stride2_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, OC2, H_out2, W_out2, OC3, H_out3, W_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            conv2d3_bias.stride(0),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            num_warps=4, num_stages=2
        )
        # GELU conv3
        x3_flat = x3.reshape(-1)
        NEL3 = x3_flat.numel()
        grid_g3 = (NEL3,)
        gelu_tanh_inplace_kernel[grid_g3](x3_flat, NEL3, num_warps=4, num_stages=2)
        x3 = x3.reshape(B, OC3, H_out3, W_out3)

        # Compute T_after = time_after_conv from axes; here we use W_out3 as per stride formula
        T_after = W_out3  # equals (T_in + 3)//8
        # Reshape for final projection: (B, T_after, 384*10) where N = 3840
        N = OC3 * 10  # 384 * 10
        X_proj = x3.reshape(B, T_after, N).contiguous().to(torch.float32)

        # Final linear projection and add positional embedding
        M = conv_out_weight.shape[0]  # 1024
        # Ensure W is (M, N)
        W = conv_out_weight  # shape (1024, 3840)
        # pos_emb: (1500, 1024), we only need first B*T_after rows: rows 0..(B*T_after-1)
        # Embed scale
        scale = float(embed_scale)
        Y = torch.empty((B, T_after, M), device=input_features.device, dtype=torch.float32)
        grid_linear = (B, T_after, triton.cdiv(M, 128))
        linear_project_pos_kernel[grid_linear](
            X_proj, W, positional_embedding, Y,
            B, T_after, N, M, scale,
            num_warps=4, num_stages=2
        )
        return Y


def run(*args):
    return ModelNew()(*args)
