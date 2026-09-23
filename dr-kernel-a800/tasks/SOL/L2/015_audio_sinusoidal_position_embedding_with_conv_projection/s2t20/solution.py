import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: stride=2, padding=1, 3x3
@triton.jit
def conv2d_stride2_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    IH: tl.constexpr, IW: tl.constexpr,  # input height/width
    OH: tl.constexpr, OW: tl.constexpr,  # output height/width
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,  # 3x3
    # Strides (in elements)
    X_bs, X_cs, X_hs, X_ws,  # input strides
    W_ocs, W_cis, W_khs, W_kws,  # weight strides
    Y_bs, Y_cs, Y_hs, Y_ws,  # output strides
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over input channels and 3x3 taps
    for ic in range(0, C_in):
        for kh in range(0, KERNEL_H):
            ih = 2 * oh + kh - 1  # stride=2, padding=1
            in_h_valid = (ih >= 0) & (ih < IH)
            for kw in range(0, KERNEL_W):
                iw = 2 * ow + kw - 1
                in_w_valid = (iw >= 0) & (iw < IW)
                if in_h_valid & in_w_valid:
                    x_ptr = X_ptr + b * X_bs + ic * X_cs + ih * X_hs + iw * X_ws
                    x_val = tl.load(x_ptr)  # assume fp32 input for accumulation
                    w_ptr = W_ptr + oc * W_ocs + ic * W_cis + kh * W_khs + kw * W_kws
                    w_val = tl.load(w_ptr)
                    acc += x_val * w_val
    # Add bias
    bias_val = tl.load(BIAS_ptr + oc)
    acc = acc + bias_val
    # Store to Y
    y_ptr = Y_ptr + b * Y_bs + oc * Y_cs + oh * Y_hs + ow * Y_ws
    tl.store(y_ptr, acc)


# Triton GELU (tanh approximation) kernel over 1D tensor
@triton.jit
def gelu_kernel_1d(
    X_ptr, Y_ptr, N,
    alpha: tl.float32, beta: tl.float32,  # alpha = sqrt(2/pi), beta = 0.044715
):
    pid = tl.program_id(0)
    x = tl.load(X_ptr + pid)
    x3 = x * x * x
    inner = alpha * (x + beta * x3)
    t = tl.math.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    tl.store(Y_ptr + pid, y)


# Triton linear projection and positional embedding add
@triton.jit
def linear_project_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B, T, N, M,
    scale: tl.float32,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # Accumulate per m across N in tiles of 256
    for m0 in range(0, M, 128):
        m_offsets = m0 + tl.arange(0, 128)
        mask_m = m_offsets < M
        acc = tl.zeros([128], dtype=tl.float32)
        for n0 in range(0, N, 256):
            n_offsets = n0 + tl.arange(0, 256)
            mask_n = n_offsets < N
            # Load X[b, t, n_offsets]
            x_vals = tl.load(X_ptr + b * (T * N) + t * N + n_offsets, mask=mask_n, other=0.0)
            # Load W[m_offsets, n_offsets]
            w_ptrs = W_ptr + m_offsets[:, None] * N + n_offsets[None, :]
            w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
        # Apply scale and add pos_emb[t, :]
        acc = acc * scale
        pos_vec = tl.load(pos_ptr + t * M + m_offsets, mask=mask_m, other=0.0)
        acc = acc + pos_vec
        # Store to Y[b, t, m_offsets]
        tl.store(Y_ptr + b * (T * M) + t * M + m_offsets, acc, mask=mask_m)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T), bfloat16
        conv weights and biases: bfloat16
        conv_out_weight: (1024, 384*10), bfloat16 (for given shapes, 3840)
        positional_embedding: (1500, 1024), bfloat16
        embed_scale: float (sqrt(1024) = 32.0)
        """
        # Ensure tensors are on CUDA and contiguous; move if necessary
        def to_cuda_contig(t):
            if not t.is_cuda:
                return t.to('cuda')
            return t.contiguous()

        input_features = to_cuda_contig(input_features)
        conv2d1_weight = to_cuda_contig(conv2d1_weight)
        conv2d1_bias = to_cuda_contig(conv2d1_bias)
        conv2d2_weight = to_cuda_contig(conv2d2_weight)
        conv2d2_bias = to_cuda_contig(conv2d2_bias)
        conv2d3_weight = to_cuda_contig(conv2d3_weight)
        conv2d3_bias = to_cuda_contig(conv2d3_bias)
        conv_out_weight = to_cuda_contig(conv_out_weight)
        positional_embedding = to_cuda_contig(positional_embedding)

        B, C_in, IH, IW = input_features.shape  # C_in = 1

        # Stage 1 conv: (B, 1, 80, T) -> (B, 384, 40, OW1) with OW1 = (T + 1) // 2
        C_out1 = 384
        OH1 = (IH - 1) // 2 + 1  # stride=2, padding=1 => OH = (IH - 1)//2 + 1
        OW1 = (IW - 1) // 2 + 1
        x1 = torch.empty((B, C_out1, OH1, OW1), dtype=torch.float32, device=input_features.device)
        conv2d_stride2_kernel[(B, C_out1, OH1, OW1)](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, 1, C_out1, IH, IW, OH1, OW1, 3, 3,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            num_warps=4, num_stages=2
        )
        # GELU stage 1
        x1_flat = x1.reshape(-1).to(torch.float32)
        x1_out = torch.empty_like(x1_flat, dtype=torch.float32, device=input_features.device)
        gelu_kernel_1d[(x1_flat.numel(),)](
            x1_flat, x1_out, alpha=0.7978845608028654, beta=0.044715
        )
        x1 = x1_out.reshape(x1.shape)

        # Stage 2 conv: (B, 384, 40, OW1) -> (B, 384, 20, OW2) with OW2 = (OW1 + 1)//2
        C_out2 = 384
        OH2 = (OH1 - 1) // 2 + 1
        OW2 = (OW1 - 1) // 2 + 1
        x2 = torch.empty((B, C_out2, OH2, OW2), dtype=torch.float32, device=input_features.device)
        conv2d_stride2_kernel[(B, C_out2, OH2, OW2)](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_out1, C_out2, OH1, OW1, OH2, OW2, 3, 3,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            num_warps=4, num_stages=2
        )
        # GELU stage 2
        x2_flat = x2.reshape(-1).to(torch.float32)
        x2_out = torch.empty_like(x2_flat, dtype=torch.float32, device=input_features.device)
        gelu_kernel_1d[(x2_flat.numel(),)](
            x2_flat, x2_out, alpha=0.7978845608028654, beta=0.044715
        )
        x2 = x2_out.reshape(x2.shape)

        # Stage 3 conv: (B, 384, 20, OW2) -> (B, 384, 10, OW3) with OW3 = (OW2 + 1)//2
        C_out3 = 384
        OH3 = (OH2 - 1) // 2 + 1
        OW3 = (OW2 - 1) // 2 + 1
        x3 = torch.empty((B, C_out3, OH3, OW3), dtype=torch.float32, device=input_features.device)
        conv2d_stride2_kernel[(B, C_out3, OH3, OW3)](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, C_out2, C_out3, OH2, OW2, OH3, OW3, 3, 3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            num_warps=4, num_stages=2
        )
        # GELU stage 3
        x3_flat = x3.reshape(-1).to(torch.float32)
        x3_out = torch.empty_like(x3_flat, dtype=torch.float32, device=input_features.device)
        gelu_kernel_1d[(x3_flat.numel(),)](
            x3_flat, x3_out, alpha=0.7978845608028654, beta=0.044715
        )
        x3 = x3_out.reshape(x3.shape)

        # Reshape to (B, T_final, 384*10), where T_final = OW3
        T_final = OW3
        N = C_out3 * 10  # 384 * 10 = 3840
        X_linear = x3.permute(0, 3, 1, 2).contiguous().view(B, T_final, N)

        # Final linear projection (no bias) and positional embedding add
        M = conv_out_weight.shape[0]  # 1024
        Y = torch.empty((B, T_final, M), dtype=torch.float32, device=input_features.device)
        # Triton expects bf16 or fp32 weights; here we pass fp32 conv_out_weight
        linear_project_pos_kernel[(B, T_final)](
            X_linear, conv_out_weight, positional_embedding, Y,
            B, T_final, N, M,
            scale=embed_scale
        )

        return Y


def run(*args):
    return ModelNew()(*args)
