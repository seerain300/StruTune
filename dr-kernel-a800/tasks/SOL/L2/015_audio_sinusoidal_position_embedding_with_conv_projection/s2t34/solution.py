import math
import torch
import triton
import triton.language as tl


# Triton kernel: 2D conv stride=2, padding=1, 3x3
# X: [B, C_in, H, W] flattened
# W: [C_out, C_in, 3, 3]
# BIAS: [C_out]
# Y: [B, C_out, H_out, W_out]
@triton.jit
def conv2d_stride2_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    BLOCK_OC: tl.constexpr,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over input channels and 3x3 taps
    for ic in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                ih = 2 * oh + kh - 1
                iw = 2 * ow + kw - 1
                # bounds check
                if (ih >= 0 and ih < H) and (iw >= 0 and iw < W):
                    x_idx = ((b * C_in + ic) * H + ih) * W + iw
                    x_val = tl.load(X_ptr + x_idx).to(tl.float32)
                    w_idx = ((oc * C_in + ic) * 9) + (kh * 3 + kw)
                    w_val = tl.load(W_ptr + w_idx).to(tl.float32)
                    acc += x_val * w_val

    # add bias
    bias_val = tl.load(BIAS_ptr + oc).to(tl.float32)
    acc += bias_val

    # store to Y
    y_idx = ((b * C_out + oc) * H_out + oh) * W_out + ow
    tl.store(Y_ptr + y_idx, acc)


# Triton GELU kernel (tanh approximation) over 1D tensor
@triton.jit
def gelu_kernel_1d(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    # tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton kernel: final linear projection and add positional embedding
# X: [B, T, N] flattened as 1D of length B*T*N
# W: [M=1024, N] flattened as 1D of length M*N
# POS: [T*M] flattened as 1D
# Y: [B, T, M] flattened as 1D
@triton.jit
def final_proj_pos_kernel(X_ptr, W_ptr, POS_ptr, Y_ptr, B, T, N, M, scale: tl.float32):
    b = tl.program_id(0)
    t = tl.program_id(1)
    base = b * (T * N) + t * N
    # For each m in [0, M), compute sum_n X[b,t,n]*W[m,n], multiply by scale, add pos[t,m]
    for m in range(0, M):
        acc = tl.zeros((), dtype=tl.float32)
        # tile over N
        for n0 in range(0, N, 128):
            n = n0 + tl.arange(0, 128)
            mask_n = n < N
            x_vals = tl.load(X_ptr + base + n, mask=mask_n, other=0.0)
            w_vals = tl.load(W_ptr + m * N + n, mask=mask_n, other=0.0)
            acc += tl.sum(w_vals * x_vals, axis=0)
        y_val = acc * scale
        pos_val = tl.load(POS_ptr + t * M + m)
        y_val = y_val + pos_val
        tl.store(Y_ptr + b * (T * M) + t * M + m, y_val)


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
        conv_out_weight: (1024, 3840), bfloat16 (will be treated as float32 in kernel)
        positional_embedding: (1500, 1024), bfloat16
        embed_scale: float (e.g., 32.0)
        Returns: (B, T, 1024) float32 tensor
        """
        B, C_in, H, W = input_features.shape  # (B, 1, 80, T)
        device = input_features.device

        # Conv1: (B, 1, 80, T) -> (B, 384, 40, (T+1)//2)
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H - 1) // 2 + 1  # 40
        W_out1 = (W - 1) // 2 + 1  # (T+1)//2
        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.float32, device=device)
        grid1 = (B, C_out1, H_out1, W_out1)
        conv2d_stride2_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, C_in, H, W, C_out1, H_out1, W_out1,
            BLOCK_OC=1,
        )
        # GELU
        y1_flat = y1.reshape(-1)
        n_elements = y1_flat.shape[0]
        gelu_kernel_1d[(n_elements + 1023) // 1024,](y1_flat, y1_flat, n_elements, BLOCK=1024)
        y1 = y1_flat.reshape(B, C_out1, H_out1, W_out1)

        # Conv2: (B, 384, 40, (T+1)//2) -> (B, 384, 40, ((T+1)//2 + 1)//2)
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = H_out1  # still 40
        W_out2 = (W_out1 - 1) // 2 + 1
        y2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.float32, device=device)
        grid2 = (B, C_out2, H_out2, W_out2)
        conv2d_stride2_kernel[grid2](
            y1, conv2d2_weight, conv2d2_bias, y2,
            B, C_out1, H_out1, W_out1, C_out2, H_out2, W_out2,
            BLOCK_OC=1,
        )
        # GELU
        y2_flat = y2.reshape(-1)
        n_elements2 = y2_flat.shape[0]
        gelu_kernel_1d[(n_elements2 + 1023) // 1024,](y2_flat, y2_flat, n_elements2, BLOCK=1024)
        y2 = y2_flat.reshape(B, C_out2, H_out2, W_out2)

        # Conv3: (B, 384, 40, (W_out2)) -> (B, 384, 40, ((W_out2 + 1)//2))
        C_out3 = conv2d3_weight.shape[0]
        H_out3 = H_out2  # still 40
        W_out3 = (W_out2 - 1) // 2 + 1
        y3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.float32, device=device)
        grid3 = (B, C_out3, H_out3, W_out3)
        conv2d_stride2_kernel[grid3](
            y2, conv2d3_weight, conv2d3_bias, y3,
            B, C_out2, H_out2, W_out2, C_out3, H_out3, W_out3,
            BLOCK_OC=1,
        )
        # GELU
        y3_flat = y3.reshape(-1)
        n_elements3 = y3_flat.shape[0]
        gelu_kernel_1d[(n_elements3 + 1023) // 1024,](y3_flat, y3_flat, n_elements3, BLOCK=1024)
        y3 = y3_flat.reshape(B, C_out3, H_out3, W_out3)

        # Final: reshape to (B, T, N), N = 384*10 = 3840
        B, C_out3, H_out3, W_out3 = y3.shape  # H_out3 = 40, W_out3 = time_after_conv
        T = W_out3
        N = C_out3 * 10  # 3840

        # We need Y of shape (B, T, M=1024). Launch Triton kernel to compute it.
        M = 1024
        # Prepare X as flattened: (B, T, N)
        X = y3.view(B, T, N).contiguous()
        X_flat = X.view(-1).to(torch.float32)

        # Prepare W as flattened: (M, N)
        W = conv_out_weight.to(torch.float32).contiguous()
        W_flat = W.view(M * N).contiguous()

        # Prepare POS: positional_embedding[:T, :] flattened
        pos = positional_embedding[:T, :].to(torch.float32).contiguous()
        POS_flat = pos.view(-1).contiguous()  # length T*M

        Y_flat = torch.empty(B * T * M, dtype=torch.float32, device=device)
        grid_final = (B, T)
        final_proj_pos_kernel[grid_final](
            X_flat, W_flat, POS_flat, Y_flat, B, T, N, M, embed_scale,
        )
        # Reshape back to (B, T, M)
        Y = Y_flat.view(B, T, M)

        return Y


def run(*args):
    return ModelNew()(*args)
