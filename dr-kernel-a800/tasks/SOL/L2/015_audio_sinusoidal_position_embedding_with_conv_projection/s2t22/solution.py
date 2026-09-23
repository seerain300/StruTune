import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1
# Computes y[b, oc, oh, ow] = sum_{c, kh,kw} W[oc, c, kh, kw] * X[b, c, 2*oh+kh-1, 2*ow+kw-1] + bias[oc]
@triton.jit
def conv2d_stride2_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, IC, IH, IW,
    OC, OH, OW,
    X_sN, X_sC, X_sH, X_sW,
    W_sOC, W_sIC, W_sKH, W_sKW,
    Y_sN, Y_sOC, Y_sH, Y_sW,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for c in range(0, IC):
        for kh in range(0, 3):
            ih = 2 * oh + kh - 1
            in_bounds_h = (ih >= 0) & (ih < IH)
            for kw in range(0, 3):
                iw = 2 * ow + kw - 1
                in_bounds_w = (iw >= 0) & (iw < IW)
                in_bounds = in_bounds_h & in_bounds_w

                x_ptr = X_ptr + b * X_sN + c * X_sC + ih * X_sH + iw * X_sW
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)

                w_ptr = W_ptr + oc * W_sOC + c * W_sIC + kh * W_sKH + kw * W_sKW
                w_val = tl.load(w_ptr)  # scalar
                acc += x_val * w_val

    # Add bias
    bias_ptr = BIAS_ptr + oc
    bias_val = tl.load(bias_ptr)
    acc += bias_val

    # Store
    y_ptr = Y_ptr + b * Y_sN + oc * Y_sOC + oh * Y_sH + ow * Y_sW
    tl.store(y_ptr, acc)


# Triton GELU (tanh approximation) kernel: 1D over elements
@triton.jit
def gelu_approx_kernel(X_ptr, Y_ptr, N, scale: tl.float32):
    # scale is not used here; GELU is applied in-place to X_ptr into Y_ptr
    idx = tl.program_id(0)
    if idx < N:
        x = tl.load(X_ptr + idx)
        # tanh approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        x3 = x * x * x
        u = c0 * (x + c1 * x3)
        y = 0.5 * x * (1.0 + tl.tanh(u))
        tl.store(Y_ptr + idx, y)


# Triton linear projection + positional embedding
# X: [B, T, N], W: [M, N], pos: [T, M], Y: [B, T, M]
@triton.jit
def linear_project_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B, T, N, M,
    scale: tl.float32,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # Accumulate per m across N in tiles
    for m0 in range(0, M, 128):
        m_offsets = m0 + tl.arange(0, 128)
        mask_m = m_offsets < M
        acc = tl.zeros([128], dtype=tl.float32)
        # Loop over N in tiles of 256
        for n0 in range(0, N, 256):
            n_offsets = n0 + tl.arange(0, 256)
            mask_n = n_offsets < N
            x_ptr_elem = X_ptr + b * (T * N) + t * N + n_offsets
            x_vals = tl.load(x_ptr_elem, mask=mask_n, other=0.0)  # [256]
            w_ptrs = W_ptr + m_offsets[:, None] * N + n_offsets[None, :]
            w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)  # [128, 256]
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
        # Apply scale and add pos_emb[t, :]
        acc = acc * scale
        pos_vec = tl.load(pos_ptr + t * M + m_offsets, mask=mask_m, other=0.0)
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
        All tensors must be on CUDA. This forward only launches Triton kernels.
        input_features: (B, 1, 80, T), bfloat16
        conv2d1_weight: (384, 1, 3, 3), bfloat16
        conv2d1_bias: (384), bfloat16
        conv2d2_weight, conv2d3_weight: (384, 384, 3, 3), bfloat16
        conv2d2_bias, conv2d3_bias: (384), bfloat16
        conv_out_weight: (1024, 384*10) = (1024, 3840), bfloat16
        positional_embedding: (1500, 1024), bfloat16
        embed_scale: float32 (sqrt(1024)=32.0)
        """
        # Ensure CUDA tensors
        if not (input_features.is_cuda and conv2d1_weight.is_cuda and conv_out_weight.is_cuda and positional_embedding.is_cuda):
            raise RuntimeError("All tensors must be on CUDA for Triton kernels.")

        B, C, IH, IW = input_features.shape
        assert C == 1, "input_features must have 1 channel."

        # Stage 1 conv: (B, 1, 80, T) -> (B, 384, 40, OW1), OW1 = (T+1)//2
        OC1 = 384
        OH1 = (IH - 1) // 2 + 1  # 40
        OW1 = (IW + 1) // 2
        X1 = torch.empty((B, OC1, OH1, OW1), device=input_features.device, dtype=torch.float32)
        conv2d_stride2_kernel[(B, OC1, OH1, OW1)](
            input_features.float(), conv2d1_weight.float(), conv2d1_bias.float(),
            X1,
            B, 1, IH, IW,
            OC1, OH1, OW1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            X1.stride(0), X1.stride(1), X1.stride(2), X1.stride(3),
        )
        # GELU stage 1 (approx tanh)
        X1_g = torch.empty_like(X1)
        N1 = X1.numel()
        gelu_approx_kernel[(N1,)](X1, X1_g, N1, 1.0)
        X1 = X1_g  # update to GELU-ed tensor

        # Stage 2 conv: (B, 384, 40, OW1) -> (B, 384, 20, OW2), OW2 = (OW1+1)//2
        OC2 = 384
        OH2 = (OH1 - 1) // 2 + 1  # 20
        OW2 = (OW1 + 1) // 2
        X2 = torch.empty((B, OC2, OH2, OW2), device=input_features.device, dtype=torch.float32)
        conv2d_stride2_kernel[(B, OC2, OH2, OW2)](
            X1, conv2d2_weight.float(), conv2d2_bias.float(),
            X2,
            B, OC1, OH1, OW1,
            OC2, OH2, OW2,
            X1.stride(0), X1.stride(1), X1.stride(2), X1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            X2.stride(0), X2.stride(1), X2.stride(2), X2.stride(3),
        )
        # GELU stage 2 (approx tanh)
        X2_g = torch.empty_like(X2)
        N2 = X2.numel()
        gelu_approx_kernel[(N2,)](X2, X2_g, N2, 1.0)
        X2 = X2_g

        # Stage 3 conv: (B, 384, 20, OW2) -> (B, 384, 10, OW3), OW3 = (OW2+1)//2
        OC3 = 384
        OH3 = (OH2 - 1) // 2 + 1  # 10
        OW3 = (OW2 + 1) // 2
        X3 = torch.empty((B, OC3, OH3, OW3), device=input_features.device, dtype=torch.float32)
        conv2d_stride2_kernel[(B, OC3, OH3, OW3)](
            X2, conv2d3_weight.float(), conv2d3_bias.float(),
            X3,
            B, OC2, OH2, OW2,
            OC3, OH3, OW3,
            X2.stride(0), X2.stride(1), X2.stride(2), X2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            X3.stride(0), X3.stride(1), X3.stride(2), X3.stride(3),
        )
        # GELU stage 3 (approx tanh)
        X3_g = torch.empty_like(X3)
        N3 = X3.numel()
        gelu_approx_kernel[(N3,)](X3, X3_g, N3, 1.0)
        X3 = X3_g

        # Final linear projection: (B, T_final, 384*10) -> (B, T_final, 1024)
        # T_final = OW3 == time_after_conv from the caller
        T_final = OW3
        # Reshape X3 to (B, T_final, N) where N=384*10
        X_flat = X3.permute(0, 3, 1, 2).contiguous().view(B, T_final, 384 * 10)

        # Linear + positional embedding
        M = conv_out_weight.shape[0]  # 1024
        N = X_flat.shape[2]            # 3840
        Y = torch.empty((B, T_final, M), device=input_features.device, dtype=torch.float32)
        linear_project_pos_kernel[(B, T_final)](
            X_flat, conv_out_weight.float(), positional_embedding.float(), Y,
            B, T_final, N, M,
            embed_scale
        )

        return Y


def run(*args):
    return ModelNew()(*args)
