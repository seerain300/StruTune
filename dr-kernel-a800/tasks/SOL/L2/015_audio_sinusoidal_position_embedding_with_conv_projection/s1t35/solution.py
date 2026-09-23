import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1
# Input: X(B, Cin, H, T) bfloat16, shape: (B, Cin, H, T)
# Weight: W(Cout, Cin, 3, 3) bfloat16, shape: (Cout, Cin, 3, 3)
# Bias: Bias(Cout) bfloat16, shape: (Cout,)
# Output: Y(B, Cout, H_out, T_out) bfloat16, where H_out=H//2, T_out=(T-3)//2+1
@triton.jit
def conv2d_3x3_stride2_padding1_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, Cin, H, T, Cout, T_out,
    BLOCK_C: tl.constexpr,
):
    # Grid dims: (B*H_out, tiles over Cout, T_out)
    pid_bh = tl.program_id(0)
    pid_ct = tl.program_id(1)
    pid_to = tl.program_id(2)

    # Derive b and output row index
    H_out = H // 2
    b = pid_bh // H_out
    oh = pid_bh % H_out

    t_out_idx = pid_to

    # Output channel tile
    c_offsets = pid_ct * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel positions
    for cin in range(0, Cin):
        for kh in range(3):
            for kt in range(3):
                # Input indices with padding=1, stride=2
                ih = oh + kh - 1
                it = t_out_idx + kt - 1

                # Bounds check
                valid_h = (ih >= 0) & (ih < H)
                valid_t = (it >= 0) & (it < T)

                # Load X[b, cin, ih, it] (masked if out-of-bounds)
                x_ptr = X_ptr + b * (Cin * H * T) + cin * (H * T) + ih * T + it
                x_val = tl.load(x_ptr, mask=valid_h & valid_t, other=0.0).to(tl.float32)

                # Load W[c_offsets, cin, kh, kt] for all c in tile
                for c_idx in range(BLOCK_C):
                    c = c_offsets[c_idx]
                    if c < Cout:
                        w_ptr = W_ptr + c * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt
                        w_val = tl.load(w_ptr, mask=True, other=0.0).to(tl.float32)
                        acc[c_idx] += x_val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias

    # Apply GELU tanh approximation in-kernel:
    # gelu(x) = 0.5*x*(1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
    c1 = 0.7978845608028654  # sqrt(2/pi)
    x = acc
    x3 = x * x * x
    inner = c1 * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))

    # Store to Y[b, c_offsets, oh, t_out_idx] as bfloat16
    y_ptr = Y_ptr + b * (Cout * H_out * T_out) + (c_offsets * (H_out * T_out)) + (oh * T_out) + t_out_idx
    tl.store(y_ptr, gelu.to(tl.bfloat16), mask=mask_c)


# Triton GEMM + add scaled positional embedding
# X: (B, T, K) bfloat16 (we pass as float32 for compute)
# WT: (K, N) bfloat16 conv_out_weight.T
# POS: (T, N) bfloat16 positional embedding slice
# Y: (B, T, N) bfloat16
@triton.jit
def gemm_add_pos_kernel(
    X_ptr, WT_ptr, POS_ptr, Y_ptr,
    B, T, K, N,
    scale: tl.float32,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B*T, tiles over N)
    pid_bt = tl.program_id(0)
    pid_nt = tl.program_id(1)

    b = pid_bt // T
    t = pid_bt % T

    n_offsets = pid_nt * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load X[b, t, k_offsets]
        x_ptr = X_ptr + b * (T * K) + t * K + k_offsets
        x_vec = tl.load(x_ptr, mask=mask_k, other=0.0).to(tl.float32)  # length BLOCK_K

        # Load WT[k_offsets, n_offsets] -> (BLOCK_K, BLOCK_N)
        wt_ptrs = WT_ptr + k_offsets[:, None] * N + n_offsets[None, :]
        wt_mat = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc += sum_k x_vec[k] * wt_mat[k, :]
        acc += tl.sum(wt_mat * x_vec[:, None], axis=0)

    # Add scaled positional embedding: POS[t, n_offsets]
    pos_ptrs = POS_ptr + t * N + n_offsets
    pos_vec = tl.load(pos_ptrs, mask=mask_n, other=0.0).to(tl.float32)
    acc += scale * pos_vec

    # Store Y[b, t, n_offsets] as bfloat16
    y_ptrs = Y_ptr + b * (T * N) + t * N + n_offsets
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args correspond to:
        # input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        (
            input_features, conv2d1_weight, conv2d1_bias,
            conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
            conv_out_weight, positional_embedding, embed_scale,
        ) = args

        # Ensure tensors are on CUDA and dtype bfloat16
        assert input_features.is_cuda and conv2d1_weight.is_cuda and conv2d3_weight.is_cuda and conv_out_weight.is_cuda and positional_embedding.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."

        # Conv 1: input (B, 1, H, T)
        B, Cin1, H1, T1 = input_features.shape
        Cout1, Cin2, kH, kT = conv2d1_weight.shape
        assert kH == 3 and kT == 3 and Cin2 == 1, "conv2d1_weight must be (Cout, 1, 3, 3)."
        H_out1 = H1 // 2
        T_out1 = (T1 - 3) // 2 + 1

        Y1 = torch.empty((B, Cout1, H_out1, T_out1), dtype=torch.bfloat16, device=input_features.device)

        BLOCK_C1 = 32
        grid_conv1 = (B * H_out1, triton.cdiv(Cout1, BLOCK_C1), T_out1)
        conv2d_3x3_stride2_padding1_kernel[grid_conv1](
            input_features, conv2d1_weight, conv2d1_bias, Y1,
            B, 1, H1, T1, Cout1, T_out1,
            BLOCK_C=BLOCK_C1,
        )

        # Conv 2: input Y1 (B, 384, H_out1, T_out1)
        Cout2, Cin3, kH2, kT2 = conv2d2_weight.shape
        assert kH2 == 3 and kT2 == 3, "conv2d2_weight must be (Cout, Cin, 3, 3)."
        H_in2 = H_out1
        T_in2 = T_out1
        H_out2 = H_in2 // 2
        T_out2 = (T_in2 - 3) // 2 + 1

        Y2 = torch.empty((B, Cout2, H_out2, T_out2), dtype=torch.bfloat16, device=input_features.device)

        BLOCK_C2 = 64
        grid_conv2 = (B * H_out2, triton.cdiv(Cout2, BLOCK_C2), T_out2)
        conv2d_3x3_stride2_padding1_kernel[grid_conv2](
            Y1, conv2d2_weight, conv2d2_bias, Y2,
            B, Cout1, H_out2, T_out2, Cout2, T_out2,
            BLOCK_C=BLOCK_C2,
        )

        # Conv 3: input Y2 (B, 384, H_out2, T_out2)
        Cout3, Cin4, kH3, kT3 = conv2d3_weight.shape
        assert kH3 == 3 and kT3 == 3, "conv2d3_weight must be (Cout, Cin, 3, 3)."
        H_in3 = H_out2
        T_in3 = T_out2
        H_out3 = H_in3 // 2
        T_out3 = (T_in3 - 3) // 2 + 1

        Y3 = torch.empty((B, Cout3, H_out3, T_out3), dtype=torch.bfloat16, device=input_features.device)

        BLOCK_C3 = 64
        grid_conv3 = (B * H_out3, triton.cdiv(Cout3, BLOCK_C3), T_out3)
        conv2d_3x3_stride2_padding1_kernel[grid_conv3](
            Y2, conv2d3_weight, conv2d3_bias, Y3,
            B, Cout2, H_out3, T_out3, Cout3, T_out3,
            BLOCK_C=BLOCK_C3,
        )

        # Reshape: (B, Cout3, H_out3, T_out3) -> (B, T_out3, Cout3*H_out3)
        B, Cout3, H_out3, T_out3 = Y3.shape
        C3


def run(*args):
    return ModelNew()(*args)
