import math
import torch

# Conditional Triton import; guard to avoid ImportError in environments without Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# --------- Triton kernels --------- #
if TRITON_AVAILABLE:
    @triton.jit
    def conv2d_stride2_pad1_bias_gelu(x: tl.pointer, w: tl.pointer, bias: tl.pointer, y: tl.pointer,
                                       B: tl.int32, C_in: tl.int32, H: tl.int32, W: tl.int32,
                                       C_out: tl.int32, H_out: tl.int32, W_out: tl.int32,
                                       stride_x_b: tl.int32, stride_x_ci: tl.int32, stride_x_h: tl.int32, stride_x_w: tl.int32,
                                       stride_w_co: tl.int32, stride_w_ci: tl.int32, stride_w_kh: tl.int32, stride_w_kw: tl.int32,
                                       stride_y_b: tl.int32, stride_y_co: tl.int32, stride_y_h: tl.int32, stride_y_w: tl.int32,
                                       scale: tl.float32,  # embed_scale, used after GELU
                                       BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
        """
        Conv2D: x [B, C_in, H, W] * w [C_out, C_in, 3, 3] -> y [B, C_out, H_out, W_out]
        stride=2, padding=1. Fused bias add and GELU.
        """
        b = tl.program_id(0)
        co = tl.program_id(1)
        oh_block = tl.program_id(2)
        ow_block = tl.program_id(3)

        oh_start = oh_block * BLOCK_H
        ow_start = ow_block * BLOCK_W

        oh_offsets = oh_start + tl.arange(0, BLOCK_H)
        ow_offsets = ow_start + tl.arange(0, BLOCK_W)

        # Initialize accumulator for this output tile
        acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.bfloat16)

        # Loop over input channels and 3x3 kernel
        for ci in range(0, C_in):
            for kh in range(0, 3):
                for kw in range(0, 3):
                    # Compute input coordinates for this (oh, ow)
                    ih = oh_offsets * 2 + (1 - kh)  # stride=2, padding=1
                    iw = ow_offsets * 2 + (1 - kw)
                    # Valid mask for padding
                    valid = (ih[:, None] >= 0) & (ih[:, None] < H) & (iw[None, :] >= 0) & (iw[None, :] < W)
                    # Load input tile
                    x_ptrs = x + b * stride_x_b + ci * stride_x_ci + ih[:, None] * stride_x_h + iw[None, :] * stride_x_w
                    x_vals = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.bfloat16)  # [BLOCK_H, BLOCK_W]

                    # Load corresponding weight scalar w[co, ci, kh, kw]
                    w_val = tl.load(w + co * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw).to(tl.bfloat16)

                    # Accumulate
                    acc += x_vals * w_val

        # Add bias
        b_val = tl.load(bias + co).to(tl.bfloat16)
        acc = acc + b_val

        # Fused GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        # Compute in fp32 for stability, then cast back
        acc_fp32 = acc.to(tl.float32)
        c0 = 0.044715
        c1 = 0.7978845608028654  # sqrt(2/pi)
        acc_cubed = acc_fp32 * acc_fp32 * acc_fp32
        gelu_inner = c1 * (acc_fp32 + c0 * acc_cubed)
        gelu_approx = 0.5 * acc_fp32 * (1.0 + tl.tanh(gelu_inner))
        acc = gelu_approx.to(tl.bfloat16)

        # Scale (post-GELU)
        acc = acc * scale

        # Store to output
        y_ptrs = y + b * stride_y_b + co * stride_y_co + oh_offsets[:, None] * stride_y_h + ow_offsets[None, :] * stride_y_w
        out_mask = (oh_offsets[:, None] < H_out) & (ow_offsets[None, :] < W_out)
        tl.store(y_ptrs, acc, mask=out_mask)

    @triton.jit
    def triton_linear_proj(x_rowwise: tl.pointer, w_t: tl.pointer, y_flat: tl.pointer,
                            M: tl.int32, K: tl.int32, N: tl.int32,
                            stride_x_m: tl.int32, stride_x_k: tl.int32,
                            stride_wk: tl.int32, stride_wn: tl.int32,
                            stride_y_m: tl.int32, stride_y_n: tl.int32,
                            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """
        y_flat = x_rowwise @ w_t, where:
          x_rowwise: [M, K]
          w_t: [K, N]  (conv_out_weight.T)
          y_flat: [M, N]
        """
        m = tl.program_id(0)
        n_block = tl.program_id(1)
        n_start = n_block * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_N,), dtype=tl.bfloat16)

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + k_offsets
            x_vals = tl.load(x_rowwise + m * stride_x_m + k_idx * stride_x_k, mask=k_idx < K, other=0.0).to(tl.bfloat16)  # [BLOCK_K]
            w_ptrs = w_t + k_idx[:, None] * stride_wk + n_offsets[None, :] * stride_wn
            w_mask = (k_idx[:, None] < K) & (n_offsets[None, :] < N)
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.bfloat16)  # [BLOCK_K, BLOCK_N]
            acc += tl.sum(w_vals * x_vals[:, None], axis=0)

        y_ptrs = y_flat + m * stride_y_m + n_offsets * stride_y_n
        n_mask = n_offsets < N
        tl.store(y_ptrs, acc, mask=n_mask)

    @triton.jit
    def scale_elementwise(y_flat: tl.pointer, N_elems: tl.int32, scale: tl.float32):
        idx = tl.program_id(0) * 1024 + tl.arange(0, 1024)
        mask = idx < N_elems
        y = tl.load(y_flat + idx, mask=mask, other=0.0).to(tl.bfloat16)
        y = y * scale
        tl.store(y_flat + idx, y, mask=mask)

    @triton.jit
    def add_pos_emb(y_flat: tl.pointer, pos_flat: tl.pointer, N_elems: tl.int32):
        idx = tl.program_id(0) * 1024 + tl.arange(0, 1024)
        mask = idx < N_elems
        y = tl.load(y_flat + idx, mask=mask, other=0.0).to(tl.bfloat16)
        pos = tl.load(pos_flat + idx, mask=mask, other=0.0).to(tl.bfloat16)
        y = y + pos
        tl.store(y_flat + idx, y, mask=mask)


# --------- ModelNew forward --------- #
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args order matches get_inputs(...):
        # input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale = args

        device = input_features.device
        dtype = input_features.dtype  # bfloat16

        # Fallback: if Triton not available, use original PyTorch ops for correctness
        if not TRITON_AVAILABLE:
            x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
            x = F.gelu(x)
            x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
            x = F.gelu(x)
            x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
            x = F.gelu(x)

            b, c, f, t = x.size()
            x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
            x = F.linear(x, conv_out_weight)  # conv_out_weight: [1024, 3840]
            x = x * float(embed_scale)

            # Broadcast positional embedding across batch
            pos_emb = positional_embedding[:t, :].unsqueeze(0).to(dtype)  # [1, t, 1024]
            # Since x is [B, t, 1024], we add pos_emb[:, :, :] across batch
            # Align shapes: [B, t, 1024]
            # Broadcast along batch dimension
            # Note: we can simply expand pos_emb to [B, t, 1024] without copying
            pos_emb = pos_emb.expand(x.shape[0], -1, -1)
            x = x + pos_emb
            return x

        # Triton path: implement all ops in Triton
        B, Cin, H, W = input_features.shape
        C_out1 = conv2d1_weight.shape[0]
        C_out2 = conv2d2_weight.shape[0]
        C_out3 = conv2d3_weight.shape[0]
        H_out1 = (H + 2 * 1 - 3) // 2 + 1
        W_out1 = (W + 2 * 1 - 3) // 2 + 1
        H_out2 = (H_out1 + 2 * 1 - 3) // 2 + 1
        W_out2 = (W_out1 + 2 * 1 - 3) // 2 + 1
        H_out3 = (H_out2 + 2 * 1 - 3) // 2 + 1
        W_out3 = (W_out2 + 2 * 1 - 3) // 2 + 1

        # Allocate outputs for convs
        y1 = torch.empty((B, C_out1, H_out1, W_out1), device=device, dtype=torch.bfloat16)
        y2 = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=torch.bfloat16)
        y3 = torch.empty((B, C_out3, H_out3, W_out3), device=device, dtype=torch.bfloat16)

        # Launch conv1 + GELU
        BLOCK_H = 8
        BLOCK_W = 32
        grid1 = (B, C_out1, triton.cdiv(H_out1, BLOCK_H), triton.cdiv(W_out1, BLOCK_W))
        conv2d_stride2_pad1_bias_gelu[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, Cin, H, W, C_out1, H_out1, W_out1,
            *input_features.stride(),
            *conv2d1_weight.stride(),
            *y1.stride(),
            float(embed_scale),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )

        # conv2 + GELU
        grid2 = (B, C_out2, triton.cdiv(H_out2, BLOCK_H), triton.cdiv(W_out2, BLOCK_W))
        conv2d_stride2_pad1_bias_gelu[grid2](
            y1, conv2d2_weight, conv2d2_bias, y2,
            B, C_out1, H_out1, W_out1, C_out2, H_out2, W_out2,
            *y1.stride(),
            *conv2d2_weight.stride(),
            *y2.stride(),
            float(embed_scale),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )

        # conv3 + GELU
        grid3 = (B, C_out3, triton.cdiv(H_out3, BLOCK_H), triton.cdiv(W_out3, BLOCK_W))
        conv2d_stride2_pad1_bias_gelu[grid3](
            y2, conv2d3_weight, conv2d3_bias, y3,
            B, C_out2, H_out2, W_out2, C_out3, H_out3, W_out3,
            *y2.stride(),
            *conv2d3_weight.stride(),
            *y3.stride(),
            float(embed_scale),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )

        # Reshape to [B, t, K] where K = C_out3 * F (F=10)
        b, c, f, t = y3.shape
        assert f == 10, "Final channels must be 384 and final frequency must be 10 per code, adjust if inputs change."
        K = c * f
        x_2d = y3.permute(0, 3, 1, 2).contiguous().view(b, t, K)  # [B, t, K]

        # Linear projection: x @ conv_out_weight.T, conv_out_weight: [N=1024, K=3840]
        N = conv_out_weight.shape[0]  # 1024
        w_t = conv_out_weight.T.contiguous()  # [K, N]
        M = b * t
        x_rowwise = x_2d.contiguous().view(M, K)
        y_flat = torch.empty((M, N), device=device, dtype=torch.bfloat16)

        BLOCK_N = 128
        BLOCK_K = 64
        grid_gemm = (M, triton.cdiv(N, BLOCK_N))
        triton_linear_proj[grid_gemm](
            x_rowwise, w_t, y_flat,
            M, K, N,
            x_rowwise.stride(0), x_rowwise.stride(1),
            w_t.stride(0), w_t.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape to [B, t, N]
        y = y_flat.view(b, t, N)

        # Scale by embed_scale
        y_flat_scaled = y.view(-1)
        N_elems_scaled = y_flat_scaled.numel()
        grid_scale = (triton.cdiv(N_elems_scaled, 1024),)
        scale_elementwise[grid_scale](y_flat_scaled, scale=float(embed_scale), N_elems=N_elems_scaled, num_warps=4, num_stages=2)

        y = y_flat_scaled.view(b, t, N)

        # Add positional embedding [t, N], broadcast over batch
        pos_emb = positional_embedding[:t, :].contiguous().to(torch.bfloat16)  # [t, N]
        y_flat_add = y.view(-1)  # [b*t*N]
        N_elems_add = y_flat_add.numel()
        grid_add = (triton.cdiv(N_elems_add, 1024),)
        add_pos_emb[grid_add](y_flat_add, pos_emb.view(-1), N_elems_add, num_warps=4, num_stages=2)

        return y


def run(*args):
    return ModelNew()(*args)
