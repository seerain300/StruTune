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
        scale,  # embed_scale (used after GELU)
        BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
    ):
        pid_b = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_h = tl.program_id(2)
        pid_w = tl.program_id(3)

        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

        mask_h = offs_h < H_out
        mask_w = offs_w < W_out
        mask_hw = mask_h[:, None] & mask_w[None, :]

        # Accumulator for the tile
        acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.bfloat16)

        # Loop over input channels
        for ci in range(0, C_in):
            # For each 3x3 kernel, compute contributions
            # We unroll the kh/kw loops explicitly
            for kh in range(3):
                for kw in range(3):
                    # ih = 2*oh + 1 - kh, iw = 2*ow + 1 - kw (since stride=2, padding=1)
                    ih = offs_h[:, None] * 2 + (1 - kh)  # [BH, 1]
                    iw = offs_w[None, :] * 2 + (1 - kw)  # [1, BW]
                    # Compute pointers into input tensor
                    x_ptrs = X_ptr + pid_b * stride_x_b + ci * stride_x_ci + ih * stride_x_h + iw * stride_x_w
                    # Validity masks due to padding
                    mask_x = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                    # Load values; if out of bounds, use 0
                    x_vals = tl.load(x_ptrs, mask=(mask_hw & mask_x), other=0.0)

                    # Load corresponding weight scalar w[co, ci, kh, kw]
                    w_val = tl.load(W_ptr + pid_co * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw)
                    # FMA: acc += x_vals * w_val
                    acc += x_vals * w_val  # broadcasting over the tile

        # Add bias
        bias_val = tl.load(Bias_ptr + pid_co)
        acc += bias_val  # broadcast over tile

        # Fused GELU (tanh approximation)
        # gelu(x) ≈ 0.5*x*(1 + tanh(√(2/π) * (x + 0.044715 x^3)))
        x3 = acc * acc * acc
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        gelu_inner = c0 * (acc + c1 * x3)
        gelu_approx = 0.5 * acc * (1.0 + tl.math.tanh(gelu_inner))

        # Store with mask
        y_ptrs = Y_ptr + pid_b * stride_y_b + pid_co * stride_y_co + offs_h[:, None] * stride_y_h + offs_w[None, :] * stride_y_w
        tl.store(y_ptrs, gelu_approx, mask=mask_hw)

        # Optional: apply output scale (even though scale was passed, we keep the kernel independent of it)
        # y = gelu_approx * scale
        # We do scaling outside the kernel for simplicity in the forward, to avoid reusing parameters mistakenly.


# Triton GEMM: X_rowwise [M, K] dot WT [K, N] -> Y [M, N] (no bias in the original)
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
        pid_m = tl.program_id(0)  # over rows
        pid_n = tl.program_id(1)  # over output channels
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


# Triton elementwise kernel: scale tensor by a scalar (embed_scale), in-place
if TRITON_AVAILABLE:
    @triton.jit
    def scale_elementwise_kernel(Y_ptr, N_elems, scale, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_elems
        y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
        y = y * scale
        tl.store(Y_ptr + offs, y, mask=mask)


# Triton elementwise kernel: add positional embedding [S, N] to [B, S, N], broadcast over batch
# We implement it to operate on a flat buffer, mapping index -> (s, n) and adding pos[s, n]
if TRITON_AVAILABLE:
    @triton.jit
    def add_pos_emb_kernel(Y_ptr, POS_ptr, N_elems, S, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_elems

        # Map linear index to (s, n): s = offs // N, n = offs % N
        s = offs // N
        n = offs % N

        y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
        pos_val = tl.load(POS_ptr + s * N + n, mask=mask, other=0.0)
        y = y + pos_val
        tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args order per get_inputs: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale = args

        device = input_features.device
        dtype = torch.bfloat16

        # Convert to bfloat16 tensors on device
        x = input_features.to(device=device, dtype=dtype).contiguous()
        w1 = conv2d1_weight.to(device=device, dtype=dtype).contiguous()
        b1 = conv2d1_bias.to(device=device, dtype=dtype).contiguous()
        w2 = conv2d2_weight.to(device=device, dtype=dtype).contiguous()
        b2 = conv2d2_bias.to(device=device, dtype=dtype).contiguous()
        w3 = conv2d3_weight.to(device=device, dtype=dtype).contiguous()
        b3 = conv2d3_bias.to(device=device, dtype=dtype).contiguous()
        conv_out_weight = conv_out_weight.to(device=device, dtype=dtype).contiguous()
        pos_emb = positional_embedding.to(device=device, dtype=dtype).contiguous()

        # Constants for conv tiling
        BLOCK_H = 8
        BLOCK_W = 32
        NUM_CO_TILES = 1  # C_out loop is implicit per grid pid, not needed as constexpr here

        # Conv1: (1, 80, T) -> (384, 40, T//2)
        B, C_in1, H1, W1 = x.shape
        C_out1 = w1.shape[0]
        H_out1 = (H1 + 2 * 1 - 3) // 2 + 1
        W_out1 = (W1 + 2 * 1 - 3) // 2 + 1
        y1 = torch.empty((B, C_out1, H_out1, W_out1), device=device, dtype=dtype)

        grid1 = (B, C_out1, triton.cdiv(H_out1, BLOCK_H), triton.cdiv(W_out1, BLOCK_W))
        conv2d_stride2_pad1_bias_gelu_kernel[grid1](
            x, w1, b1, y1,
            B, C_in1, H1, W1,
            C_out1, H_out1, W_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            float(embed_scale),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )

        # Conv2: (384, 40, T//2) -> (384, 20, T//4)
        B, C_in2, H2, W2 = y1.shape
        C_out2 = w2.shape[0]
        H_out2 = (H2 + 2 * 1 - 3) // 2 + 1
        W_out2 = (W2 + 2 * 1 - 3) // 2 + 1
        y2 = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=dtype)

        grid2 = (B, C_out2, triton.cdiv(H_out2, BLOCK_H), triton.cdiv(W_out2, BLOCK_W))
        conv2d_stride2_pad1_bias_gelu_kernel[grid2](
            y1, w2, b2, y2,
            B, C_in2, H2, W2,
            C_out2, H_out2, W_out2,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            float(embed_scale),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )

        # Conv3: (384, 20, T//4) -> (384, 10, T//8)
        B, C_in3, H3, W3 = y2.shape
        C_out3 = w3.shape[0]
        H_out3 = (H3 + 2 * 1 - 3) // 2 + 1
        W_out3 = (W3 + 2 * 1 - 3) // 2 + 1
        y3 = torch.empty((B, C_out3, H_out3, W_out3), device=device, dtype=dtype)

        grid3 = (B, C_out3, triton.cdiv(H_out3, BLOCK_H), triton.cdiv(W_out3, BLOCK_W))
        conv2d_stride2_pad1_bias_gelu_kernel[grid3](
            y2, w3, b3, y3,
            B, C_in3, H3, W3,
            C_out3, H_out3, W_out3,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            float(embed_scale),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )

        # Reshape: (B, 10, T//8, 384) -> (B, T//8, 384*10)
        b, c, f, t = y3.shape
        x_proj = y3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # [B, time_after_conv, 3840]

        # Linear projection: [B*S, K] @ [K, N] -> [B*S, N]
        B, S, K = x_proj.shape
        N = conv_out_weight.shape[0]
        x_rowwise = x_proj.reshape(B * S, K).contiguous()  # [B*S, K]
        WT = conv_out_weight.transpose(0, 1).contiguous()  # [K, N]
        y_matmul = torch.empty((B * S, N), device=device, dtype=dtype)  # [B*S, N]

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

        # Scale by embed_scale: y *= embed_scale
        N_elems = y.numel()
        BLOCK_SCALE = 1024
        grid_scale = (triton.cdiv(N_elems, BLOCK_SCALE),)
        scale_elementwise_kernel[grid_scale](
            y, N_elems, float(embed_scale), BLOCK_SIZE=BLOCK_SCALE
        )

        # Add positional embedding [S, N] broadcast over batch: y += pos_emb
        # Operate in-place
        N_elems_add = y.numel()
        grid_add = (triton.cdiv(N_elems_add, BLOCK_SCALE),)
        add_pos_emb_kernel[grid_add](
            y, pos_emb.view(-1), N_elems_add, S, N, BLOCK_SIZE=BLOCK_SCALE
        )

        return y


def run(*args):
    return ModelNew()(*args)
