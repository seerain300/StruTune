import math
import torch

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    # Conv2D stride=2, padding=1, 3x3, no bias. Grid: (B, OC, OH, OW)
    @triton.jit
    def conv2d_stride2_3x3_kernel(
        X_ptr, W_ptr, BIAS_ptr, Y_ptr,
        B, OC, C_in, IH, IW, OH, OW,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        w_stride_oc, w_stride_c, w_stride_kh, w_stride_kw,
        y_stride_b, y_stride_c, y_stride_h, y_stride_w,
        BLOCK_OC: tl.constexpr,
    ):
        b = tl.program_id(0)
        oc = tl.program_id(1)
        oh = tl.program_id(2)
        ow = tl.program_id(3)

        # Initialize accumulator
        acc = tl.zeros((), dtype=tl.float32)

        # Iterate over input channels and 3x3 taps
        for ic in range(0, C_in):
            for kh in range(0, 3):
                for kw in range(0, 3):
                    ih = 2 * oh + kh - 1
                    iw = 2 * ow + kw - 1

                    # Bounds check
                    in_bounds = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                    if not in_bounds:
                        continue

                    x_ptr_elem = X_ptr + b * x_stride_b + ic * x_stride_c + ih * x_stride_h + iw * x_stride_w
                    x_val = tl.load(x_ptr_elem, mask=in_bounds, other=0.0)  # scalar

                    w_ptr_elem = W_ptr + oc * w_stride_oc + ic * w_stride_c + kh * w_stride_kh + kw * w_stride_kw
                    w_val = tl.load(w_ptr_elem)  # scalar

                    acc += x_val * w_val

        # Add bias if provided
        bias_val = 0.0
        # BIAS_ptr shape: (OC,) so can load bias_val if BIAS_ptr is provided; here we always have bias
        bias_ptr_elem = BIAS_ptr + oc
        bias_val = tl.load(bias_ptr_elem)

        acc += bias_val

        # Store result
        y_ptr_elem = Y_ptr + b * y_stride_b + oc * y_stride_c + oh * y_stride_h + ow * y_stride_w
        tl.store(y_ptr_elem, acc)


    # GELU (tanh approximation) over flattened 1D tensor
    @triton.jit
    def gelu_1d_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
        # tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c = 0.7978845608028654  # sqrt(2/pi)
        x3 = x * x * x
        inner = c * (x + 0.044715 * x3)
        t = tl.tanh(inner)
        y = 0.5 * x * (1.0 + t)
        tl.store(Y_ptr + offsets, y, mask=mask)


    # Final linear projection + positional embedding addition
    # Input X_flat: (B*T*N), W: (M, N), pos: (T, M), Output Y_flat: (B*T*M)
    @triton.jit
    def linear_pos_kernel(
        X_ptr, W_ptr, pos_ptr, Y_ptr,
        B, T, N, M, scale: tl.float32, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        b = tl.program_id(0)
        t = tl.program_id(1)
        base = b * T
        # Loop over M in tiles
        for m0 in range(0, M, BLOCK_M):
            m_offsets = m0 + tl.arange(0, BLOCK_M)
            mask_m = m_offsets < M
            acc = tl.zeros([BLOCK_M], dtype=tl.float32)
            # Loop over N in tiles
            for n0 in range(0, N, BLOCK_N):
                n_offsets = n0 + tl.arange(0, BLOCK_N)
                mask_n = n_offsets < N
                # Load X[b, t, n_offsets] as vector
                x_ptrs = X_ptr + base * N + t * N + n_offsets
                x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)  # [BLOCK_N]
                # Load W[m_offsets, n_offsets] as [BLOCK_M, BLOCK_N]
                w_ptrs = W_ptr + m_offsets[:, None] * N + n_offsets[None, :]
                w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_M, BLOCK_N]
                acc += tl.sum(w_vals * x_vals[None, :], axis=1)
            # Scale and add pos_emb[t, :]
            acc = acc * scale
            pos_vec = tl.load(pos_ptr + t * M + m_offsets, mask=mask_m, other=0.0)
            acc = acc + pos_vec
            # Store to Y[b, t, m_offsets]
            y_ptrs = Y_ptr + base * M + t * M + m_offsets
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
        conv_out_weight: (1024, 384*10) i.e., (1024, 3840), bfloat16
        positional_embedding: (1500, 1024), bfloat16
        embed_scale: float
        """
        # Ensure device is CUDA for Triton
        device = input_features.device
        B, _, IH, IW = input_features.shape  # IH=80, IW=T

        # Stage 1: Conv2d (1 -> 384) stride=2, padding=1, 3x3
        OC1 = 384
        x1 = torch.empty((B, OC1, (IH - 1) // 2 + 1, (IW + 1) // 2), device=device, dtype=torch.float32)
        # Strides
        x1_strides = x1.stride()
        w1_strides = conv2d1_weight.stride()

        grid1 = (B, OC1, x1.shape[2], x1.shape[3])
        conv2d_stride2_3x3_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, OC1, 1, IH, IW, x1.shape[2], x1.shape[3],
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            w1_strides[0], w1_strides[1], w1_strides[2], w1_strides[3],
            x1_strides[0], x1_strides[1], x1_strides[2], x1_strides[3],
            BLOCK_OC=1,
        )

        # GELU after conv1
        N1 = x1.numel()
        x1_flat = x1.view(-1).contiguous()
        y1_flat = torch.empty(N1, device=device, dtype=torch.float32)
        grid_g1 = (triton.cdiv(N1, 1024),)
        gelu_1d_kernel[grid_g1](x1_flat, y1_flat, N1, BLOCK=1024)
        x1 = y1_flat.view_as(x1)

        # Stage 2: Conv2d (384 -> 384) stride=2, padding=1, 3x3
        OC2 = 384
        IH2 = x1.shape[2]
        IW2 = x1.shape[3]
        T1 = IW2  # time after conv1
        x2 = torch.empty((B, OC2, (IH2 - 1) // 2 + 1, (T1 + 1) // 2), device=device, dtype=torch.float32)
        x2_strides = x2.stride()
        w2_strides = conv2d2_weight.stride()

        grid2 = (B, OC2, x2.shape[2], x2.shape[3])
        conv2d_stride2_3x3_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, OC2, OC1, IH2, T1, x2.shape[2], x2.shape[3],
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            w2_strides[0], w2_strides[1], w2_strides[2], w2_strides[3],
            x2_strides[0], x2_strides[1], x2_strides[2], x2_strides[3],
            BLOCK_OC=1,
        )

        # GELU after conv2
        N2 = x2.numel()
        x2_flat = x2.view(-1).contiguous()
        y2_flat = torch.empty(N2, device=device, dtype=torch.float32)
        grid_g2 = (triton.cdiv(N2, 1024),)
        gelu_1d_kernel[grid_g2](x2_flat, y2_flat, N2, BLOCK=1024)
        x2 = y2_flat.view_as(x2)

        # Stage 3: Conv2d (384 -> 384) stride=2, padding=1, 3x3
        OC3 = 384
        IH3 = x2.shape[2]
        IW3 = x2.shape[3]
        T2 = IW3  # time after conv2 (this equals time_after_conv in harness)
        x3 = torch.empty((B, OC3, (IH3 - 1) // 2 + 1, (T2 + 1) // 2), device=device, dtype=torch.float32)
        x3_strides = x3.stride()
        w3_strides = conv2d3_weight.stride()

        grid3 = (B, OC3, x3.shape[2], x3.shape[3])
        conv2d_stride2_3x3_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, OC3, OC2, IH3, T2, x3.shape[2], x3.shape[3],
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            w3_strides[0], w3_strides[1], w3_strides[2], w3_strides[3],
            x3_strides[0], x3_strides[1], x3_strides[2], x3_strides[3],
            BLOCK_OC=1,
        )

        # GELU after conv3
        N3 = x3.numel()
        x3_flat = x3.view(-1).contiguous()
        y3_flat = torch.empty(N3, device=device, dtype=torch.float32)
        grid_g3 = (triton.cdiv(N3, 1024),)
        gelu_1d_kernel[grid_g3](x3_flat, y3_flat, N3, BLOCK=1024)
        x3 = y3_flat.view_as(x3)

        # Final: linear projection to 1024 and add positional embedding
        # We need N = 384 * 10 = 3840
        N = 384 * 10
        M = conv_out_weight.shape[0]  # 1024
        T = x3.shape[3]  # time_after_conv from harness
        B_T = B * T
        # Flatten X to (B*T*N)
        X_flat = x3.view(B_T, -1)      # (B*T, N)
        X_flat = X_flat.contiguous().view(-1)  # (B_T*N)

        # Output buffer (B_T*M)
        Y_flat = torch.empty(B_T * M, device=device, dtype=torch.float32)

        # Launch linear + pos kernel
        grid_lin = (B_T,)  # one program per (b, t)
        # Note: pos_embedding is (1500, 1024), we will slice by t in kernel; here we pass as-is.
        # Triton expects contiguous pointers; cast weights to float32 for computation:
        W = conv_out_weight.to(torch.float32).contiguous()
        pos = positional_embedding.to(torch.float32).contiguous()
        linear_pos_kernel[grid_lin](
            X_flat, W, pos, Y_flat,
            B, T, N, M, float(embed_scale),
            BLOCK_M=128, BLOCK_N=256,
        )

        # Reshape to (B, T, M)
        output = Y_flat.view(B, T, M)
        return output


def run(*args):
    return ModelNew()(*args)
