import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton conv2d kernel: 3x3, stride=2, padding=1, with bias and fused GELU (tanh approximation).
# Input: x [B, C_in, H, W], w [C_out, C_in, 3, 3], bias [C_out]
# Output: y [B, C_out, H_out, W_out]
if TRITON_AVAILABLE:
    @triton.jit
    def conv2d_stride2_pad1_bias_gelu_kernel(
        X_ptr, W_ptr, Bias_ptr, Y_ptr,
        B, C_in, H, W,
        C_out, H_out, W_out,
        stride_x_b, stride_x_ci, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_y_b, stride_y_co, stride_y_h, stride_y_w,
        scale,  # embed_scale (unused here, kept for signature consistency)
        BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        # program ids
        pid_b = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_h = tl.program_id(2)
        pid_w = tl.program_id(3)

        # output tile offsets
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

        # masks for output tile
        mask_h = offs_h < H_out
        mask_w = offs_w < W_out
        mask_hw = mask_h[:, None] & mask_w[None, :]

        # accumulator for this output tile
        acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.bfloat16)

        # loop over input channels and 3x3 kernel
        for ci in range(0, C_in):
            for kh in range(0, 3):
                ih = 2 * offs_h[None, :] + (1 - kh)  # [1, BLOCK_H]
                ih_valid = (ih >= 0) & (ih < H)
                for kw in range(0, 3):
                    iw = 2 * offs_w[:, None] + (1 - kw)  # [BLOCK_W, 1]
                    iw_valid = (iw >= 0) & (iw < W)

                    x_ptrs = X_ptr + pid_b * stride_x_b + ci * stride_x_ci + ih * stride_x_h + iw * stride_x_w
                    x_mask = mask_hw & ih_valid & iw_valid
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                    # load weight scalar w[pid_co, ci, kh, kw]
                    w_scalar = tl.load(W_ptr + pid_co * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw)

                    # outer product accumulation
                    acc += x_vals[:, None] * w_scalar

        # add bias
        b = tl.load(Bias_ptr + pid_co)
        acc = acc + b

        # fused GELU (tanh approximation)
        c0 = 0.5
        c1 = 0.7978845608028654  # sqrt(2/pi)
        x3 = acc * acc * acc
        gelu_inner = c1 * (acc + 0.044715 * x3)
        gelu_approx = c0 * acc * (1.0 + tl.math.tanh(gelu_inner))
        acc = gelu_approx

        # store result
        y_ptrs = Y_ptr + pid_b * stride_y_b + pid_co * stride_y_co + offs_h[:, None] * stride_y_h + offs_w[None, :] * stride_y_w
        tl.store(y_ptrs, acc, mask=mask_hw)


# Triton GEMM: X_rowwise [M, K] dot WT [K, N] -> Y [M, N]
if TRITON_AVAILABLE:
    @triton.jit
    def matmul_linear_kernel(
        X_ptr, WT_ptr, Y_ptr,
        M, K, N,
        stride_xm, stride_xk,
        stride_wtk, stride_wtn,
        stride_ym, stride_yn,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_N
        offs_n = pid_n * BLOCK_K

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.bfloat16)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            wt_ptrs = WT_ptr + offs_k[:, None] * stride_wtk + offs_n[None, :] * stride_wtn
            wt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)
            acc += tl.dot(x, wt)

        y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, acc, mask=y_mask)


# Triton kernel: fuse scaling and positional embedding addition
if TRITON_AVAILABLE:
    @triton.jit
    def scale_add_pos_emb_kernel(
        Y_ptr, POS_ptr, Y_out_ptr,
        N, S,
        scale,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = N * S
        mask = offs < total

        s = offs // N
        n = offs % N
        y_vals = tl.load(Y_ptr + offs, mask=mask, other=0.0)
        pe_vals = tl.load(POS_ptr + s * N + n, mask=mask, other=0.0)
        y_vals = y_vals * scale + pe_vals
        tl.store(Y_out_ptr + offs, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Unpack inputs exactly as provided by get_inputs
        # args: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale = args

        # Ensure all tensors are on CUDA for Triton kernels
        # The provided inputs are bfloat16; Triton kernels operate on bfloat16.
        B, C_in, H, W = input_features.shape

        # Stage 1: Conv2d (1 -> 384 channels) + GELU (fused in kernel)
        x = input_features
        w1 = conv2d1_weight
        b1 = conv2d1_bias

        C_out1 = w1.shape[0]
        H_out1 = (H + 2 * 1 - 3) // 2 + 1
        W_out1 = (W + 2 * 1 - 3) // 2 + 1
        y1 = torch.empty((B, C_out1, H_out1, W_out1), device=x.device, dtype=torch.bfloat16)

        BLOCK_H = 8
        BLOCK_W = 8
        grid1 = (B, C_out1, triton.cdiv(H_out1, BLOCK_H), triton.cdiv(W_out1, BLOCK_W))
        conv2d_stride2_pad1_bias_gelu_kernel[grid1](
            x, w1, b1, y1,
            B, C_in, H, W,
            C_out1, H_out1, W_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            float(embed_scale),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )

        # Stage 2: Conv2d (384 -> 384 channels) + GELU (fused in kernel)
        x = y1
        w2 = conv2d2_weight
        b2 = conv2d2_bias

        C_out2 = w2.shape[0]
        H_out2 = (x.shape[2] + 2 * 1 - 3) // 2 + 1
        W_out2 = (x.shape[3] + 2 * 1 - 3) // 2 + 1
        y2 = torch.empty((B, C_out2, H_out2, W_out2), device=x.device, dtype=torch.bfloat16)

        BLOCK_H = 8
        BLOCK_W = 8
        grid2 = (B, C_out2, triton.cdiv(H_out2, BLOCK_H), triton.cdiv(W_out2, BLOCK_W))
        conv2d_stride2_pad1_bias_gelu_kernel[grid2](
            x, w2, b2, y2,
            B, x.shape[1], x.shape[2], x.shape[3],
            C_out2, H_out2, W_out2,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            float(embed_scale),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )

        # Stage 3: Conv2d (384 -> 384 channels) + GELU (fused in kernel)
        x = y2
        w3 = conv2d3_weight
        b3 = conv2d3_bias

        C_out3 = w3.shape[0]
        H_out3 = (x.shape[2] + 2 * 1 - 3) // 2 + 1
        W_out3 = (x.shape[3] + 2 * 1 - 3) // 2 + 1
        y3 = torch.empty((B, C_out3, H_out3, W_out3), device=x.device, dtype=torch.bfloat16)

        BLOCK_H = 8
        BLOCK_W = 8
        grid3 = (B, C_out3, triton.cdiv(H_out3, BLOCK_H), triton.cdiv(W_out3, BLOCK_W))
        conv2d_stride2_pad1_bias_gelu_kernel[grid3](
            x, w3, b3, y3,
            B, x.shape[1], x.shape[2], x.shape[3],
            C_out3, H_out3, W_out3,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            float(embed_scale),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = y3.shape
        x_proj = y3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # [B, time_after_conv, 3840]

        # Linear projection to d_model=1024: x_proj [B, S, K] @ conv_out_weight [N, K]^T -> [B, S, N]
        B, S, K = x_proj.shape
        N = conv_out_weight.shape[0]
        x_rowwise = x_proj.reshape(B * S, K).contiguous()  # [B*S, K]
        WT = conv_out_weight.transpose(0, 1).contiguous()  # [K, N]
        y_matmul = torch.empty((B * S, N), device=x_rowwise.device, dtype=torch.bfloat16)

        BLOCK_N = 128
        BLOCK_K = 64
        grid_gemm = (B * S, triton.cdiv(N, BLOCK_N))
        matmul_linear_kernel[grid_gemm](
            x_rowwise, WT, y_matmul,
            B * S, K, N,
            x_rowwise.stride(0), x_rowwise.stride(1),
            WT.stride(0), WT.stride(1),
            y_matmul.stride(0), y_matmul.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        y = y_matmul.view(B, S, N)  # [B, S, N]

        # Scale by embed_scale
        y_flat = y.view(-1)  # [B*S*N]
        N_elems = y_flat.numel()
        BLOCK_SCALE = 1024
        grid_scale = (triton.cdiv(N_elems, BLOCK_SCALE),)
        # First pass: scale in-place
        scale_add_pos_emb_kernel[grid_scale](
            y_flat, y_flat, y_flat,
            N, S, float(embed_scale),
            BLOCK_SIZE=BLOCK_SCALE
        )

        # Add positional embedding [S, N], broadcast over batch
        # Second pass: add pos


def run(*args):
    return ModelNew()(*args)
