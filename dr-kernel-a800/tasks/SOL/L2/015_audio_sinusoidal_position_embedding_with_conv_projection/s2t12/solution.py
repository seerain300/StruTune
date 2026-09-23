import math
import torch
import torch.nn.functional as F

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: conv2d with stride=2, padding=1, 3x3 kernels
# X: [B, C_in, IH, IW] (float32), W: [OC, C_in, 3, 3] (float32), BIAS: [OC] (float32), Y: [B, OC, OH, OW] (float32)
@triton.jit
def conv2d_stride2_bias_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, OC, IH, IW, OH, OW,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros([1], dtype=tl.float32)

    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                ih = 2 * oh + kh - 1
                iw = 2 * ow + kw - 1
                valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                x_index = b * (C_in * IH * IW) + ci * (IH * IW) + ih * IW + iw
                x_val = tl.load(X_ptr + x_index, mask=valid, other=0.0)
                w_index = oc * (C_in * 9) + ci * 9 + kh * 3 + kw
                w_val = tl.load(W_ptr + w_index)
                acc += x_val * w_val

    bias_val = tl.load(BIAS_ptr + oc)
    acc += bias_val

    y_index = b * (OC * OH * OW) + oc * (OH * OW) + oh * OW + ow
    tl.store(Y_ptr + y_index, acc[0])


# Triton kernel: GELU (tanh approximation) over 1D
@triton.jit
def gelu_tanh_kernel(X_ptr, Y_ptr, N, scale: tl.float32):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(t))
    y = y * scale
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton kernel: linear projection and add positional embedding
# X: [B, T, N], W: [M, N], POS: [T, M], Y: [B, T, M]
@triton.jit
def linear_project_pos_kernel(
    X_ptr, W_ptr, POS_ptr, Y_ptr,
    B, T, N, M, scale: tl.float32,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    for m0 in range(0, M, 128):
        m_offsets = m0 + tl.arange(0, 128)
        mask_m = m_offsets < M
        acc = tl.zeros([128], dtype=tl.float32)
        for n0 in range(0, N, 256):
            n_offsets = n0 + tl.arange(0, 256)
            mask_n = n_offsets < N
            x_ptrs = X_ptr + b * (T * N) + t * N + n_offsets
            x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)  # [256]
            w_ptrs = W_ptr + m_offsets[:, None] * N + n_offsets[None, :]
            w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)  # [128, 256]
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
        acc = acc * scale
        pos_vec = tl.load(POS_ptr + t * M + m_offsets, mask=mask_m, other=0.0)
        acc = acc + pos_vec
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
        input_features: (B, 1, 80, T) bfloat16
        conv2d1_weight: (384, 1, 3, 3) bfloat16
        conv2d1_bias: (384) bfloat16
        conv2d2_weight, conv2d3_weight: (384, 384, 3, 3) bfloat16
        conv2d2_bias, conv2d3_bias: (384) bfloat16
        conv_out_weight: (1024, 3840) float32
        positional_embedding: (1500, 1024) bfloat16
        embed_scale: float
        Returns: (B, time_after_conv, 1024) bfloat16
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # Compute conv1: (B,1,80,T) -> (B,384,40,(T+1)//2)
        B, _, IH, IW = input_features.shape
        OC1 = conv2d1_weight.shape[0]
        # Prepare output buffer for conv1 (fp32)
        OW1 = (IW - 1) // 2 + 1
        OH1 = (IH - 1) // 2 + 1
        X1 = torch.empty((B, OC1, OH1, OW1), dtype=torch.float32, device=input_features.device)
        # Cast to float32 for kernel
        X1_in = input_features.to(torch.float32)
        W1 = conv2d1_weight.to(torch.float32)
        B1 = conv2d1_bias.to(torch.float32)
        grid1 = (B, OC1, OH1, OW1)
        conv2d_stride2_bias_kernel[grid1](
            X1_in, W1, B1, X1,
            B, 1, OC1, IH, IW, OH1, OW1,
        )
        # GELU conv1
        X1_flat = X1.reshape(-1)
        Y1_flat = torch.empty_like(X1_flat, dtype=torch.float32, device=input_features.device)
        N1 = X1_flat.numel()
        grid_gelu1 = (triton.cdiv(N1, 1024),)
        gelu_tanh_kernel[grid_gelu1](X1_flat, Y1_flat, N1, 1.0)
        X1_gelu = Y1_flat.reshape(X1.shape)

        # Conv 2: (B,384,40,(T+1)//2) -> (B,384,20,((T+1)//2)//2)
        B2, C2, IH2, IW2 = X1_gelu.shape
        OC2 = conv2d2_weight.shape[0]
        OW2 = (IW2 - 1) // 2 + 1
        OH2 = (IH2 - 1) // 2 + 1
        X2 = torch.empty((B2, OC2, OH2, OW2), dtype=torch.float32, device=input_features.device)
        W2 = conv2d2_weight.to(torch.float32)
        B2_bias = conv2d2_bias.to(torch.float32)
        grid2 = (B2, OC2, OH2, OW2)
        conv2d_stride2_bias_kernel[grid2](
            X1_gelu, W2, B2_bias, X2,
            B2, C2, OC2, IH2, IW2, OH2, OW2,
        )
        # GELU conv2
        X2_flat = X2.reshape(-1)
        Y2_flat = torch.empty_like(X2_flat, dtype=torch.float32, device=input_features.device)
        N2 = X2_flat.numel()
        grid_gelu2 = (triton.cdiv(N2, 1024),)
        gelu_tanh_kernel[grid_gelu2](X2_flat, Y2_flat, N2, 1.0)
        X2_gelu = Y2_flat.reshape(X2.shape)

        # Conv 3: (B,384,20,((T+1)//2)//2) -> (B,384,10,((T+1)//2)//4)
        B3, C3, IH3, IW3 = X2_gelu.shape
        OC3 = conv2d3_weight.shape[0]
        OW3 = (IW3 - 1) // 2 + 1
        OH3 = (IH3 - 1) // 2 + 1
        X3 = torch.empty((B3, OC3, OH3, OW3), dtype=torch.float32, device=input_features.device)
        W3 = conv2d3_weight.to(torch.float32)
        B3_bias = conv2d3_bias.to(torch.float32)
        grid3 = (B3, OC3, OH3, OW3)
        conv2d_stride2_bias_kernel[grid3](
            X2_gelu, W3, B3_bias, X3,
            B3, C3, OC3, IH3, IW3, OH3, OW3,
        )

        # Final stage: reshape to (B, T_final, 384*10) and linear to (B, T_final, 1024), add pos emb
        B4, C4, IH4, IW4 = X3.shape
        T_final = IW4  # time_after_conv
        N_dim = C4 * IH4 * T_final  # should be 384 * 10 * T_final
        X3_flat = X3.reshape(B4, T_final, N_dim)

        # Linear projection: (B*T_final, N_dim) @ conv_out_weight (1024, N_dim)^T -> (B*T_final, 1024)
        M = conv_out_weight.shape[0]  # 1024
        W = conv_out_weight.to(torch.float32)  # (M, N_dim)
        assert W.shape[1] == N_dim, "Weight dim mismatch for final linear"
        X_flat = X3_flat.reshape(B4 * T_final, N_dim)
        Y = torch.empty((B4, T_final, M), dtype=torch.float32, device=input_features.device)
        POS = positional_embedding.to(torch.float32)  # (1500, 1024)
        scale = float(embed_scale)  # 32.0
        grid_linear = (B4, T_final)
        linear_project_pos_kernel[grid_linear](
            X_flat, W, POS, Y,
            B4, T_final, N_dim, M, scale,
        )

        # Return bfloat16 as in original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
