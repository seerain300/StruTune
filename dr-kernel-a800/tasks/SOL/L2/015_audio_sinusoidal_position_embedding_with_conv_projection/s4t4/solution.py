import math
import triton
import triton.language as tl


# Conv2d 3x3 stride=2, padding=1
# X: (B, C_in, H, W), W: (C_out, C_in, 3, 3), BIAS: (C_out)
# Y: (B, C_out, H_out, W_out)
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)         # batch
    pid_hw = tl.program_id(1)        # flattened (h_out * w_out)
    pid_co_blk = tl.program_id(2)    # output channel block id

    h_out_idx = pid_hw // W_out
    w_out_idx = pid_hw % W_out

    co_start = pid_co_blk * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # accumulator
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # reduce over input channels and 3x3 taps
    for ci in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                in_h = h_out_idx * 2 + kh
                in_w = w_out_idx * 2 + kw
                # guard for input bounds
                in_bounds = (0 <= in_h < H) & (0 <= in_w < W)
                # load X[b, ci, in_h, in_w]
                x_ptr = X_ptr + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0).to(tl.float32)
                # load weights W[co, ci, kh, kw] for this output channel tile
                w_base = W_ptr + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_base, mask=co_mask, other=0.0).to(tl.float32)
                acc += x_val * w_vec

    # add bias
    bias = tl.load(BIAS_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # store to Y
    y_ptr = Y_ptr + pid_b * stride_yb + co_offsets * stride_yc + h_out_idx * stride_yh + w_out_idx * stride_yw
    tl.store(y_ptr, acc.to(tl.bfloat16), mask=co_mask)


# GELU using tanh approximation (elementwise)
@triton.jit
def gelu_tanh_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y_ptr + offsets, gelu.to(tl.bfloat16), mask=mask)


# Linear projection (no bias): X (B, T, K), W (M, K) -> Y (B, T, M)
@triton.jit
def linear_no_bias_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,  # e.g., 64
    BLOCK_K: tl.constexpr,  # e.g., 128
):
    # grid: (B, T, ceil(M / BLOCK_M))
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_mb = tl.program_id(2)

    m_start = pid_mb * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # reduce over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        # X[b, t, k] vector of length BLOCK_K
        x_ptrs = X_ptr + pid_b * stride_xb + pid_t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)
        # W[m, k] matrix (BLOCK_M, BLOCK_K)
        w_ptrs = W_ptr + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_K)
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # store Y[b, t, m]
    y_ptrs = Y_ptr + pid_b * stride_yb + pid_t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=m_mask)


# Add scaled positional embedding: Y (B, T, M), POS (1, T, M) from positional_embedding
@triton.jit
def add_scaled_pos_emb_kernel(
    Y_ptr, POS_ptr, SCALE, B, T, M,
    stride_yb, stride_yt, stride_ym,
    stride_psb, stride_pst, stride_psm,
):
    # grid: (B, T, ceil(M / 128))
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_mb = tl.program_id(2)

    m_start = pid_mb * 128
    m_offsets = m_start + tl.arange(0, 128)
    m_mask = m_offsets < M

    # load Y[b, t, m] tile
    y_ptrs = Y_ptr + pid_b * stride_yb + pid_t * stride_yt + m_offsets * stride_ym
    y_vals = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    # load POS[0, t, m] tile
    pos_ptrs = POS_ptr + 0 * stride_psb + pid_t * stride_pst + m_offsets * stride_psm
    pos_vals = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    y_vals += pos_vals * SCALE

    tl.store(y_ptrs, y_vals.to(tl.bfloat16), mask=m_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size: int, time_dim: int, device: torch.device):
        super().__init__()
        self.batch_size = batch_size
        self.time_dim = time_dim
        self.device = device

    def forward(self, input_features: torch.Tensor,
                conv2d1_weight: torch.Tensor, conv2d1_bias: torch.Tensor,
                conv2d2_weight: torch.Tensor, conv2d2_bias: torch.Tensor,
                conv2d3_weight: torch.Tensor, conv2d3_bias: torch.Tensor,
                conv_out_weight: torch.Tensor,
                positional_embedding: torch.Tensor,
                embed_scale: float):
        # input: (B, 1, 80, T)
        B = self.batch_size
        T = self.time_dim
        dtype = torch.bfloat16
        device = self.device

        # Stage 1: Conv1 (1 -> 384)
        C_in1 = 1
        C_out1 = 384
        H1 = 80
        W1 = T
        H_out1 = (H1 - 3) // 2 + 1  # 40
        W_out1 = (W1 - 3) // 2 + 1  # T // 2

        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=dtype, device=device)
        x1 = input_features.contiguous()

        x_strides1 = x1.stride()
        w_strides1 = conv2d1_weight.stride()
        y1_strides = y1.stride()
        grid1 = (B, H_out1 * W_out1, triton.cdiv(C_out1, 64))
        conv2d_3x3_stride2_pad1_kernel[grid1](
            x1, conv2d1_weight, conv2d1_bias, y1,
            B, C_in1, H1, W1, C_out1, H_out1, W_out1,
            x_strides1[0], x_strides1[1], x_strides1[2], x_strides1[3],
            w_strides1[0], w_strides1[1], w_strides1[2], w_strides1[3],
            y1_strides[0], y1_strides[1], y1_strides[2], y1_strides[3],
            BLOCK_CO=64,
        )

        # GELU1
        y1_gelu = torch.empty_like(y1)
        N1 = y1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](y1, y1_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2 (384 -> 384)
        C_in2 = C_out1
        C_out2 = C_out1
        H2 = H_out1
        W2 = W_out1
        H_out2 = (H2 - 3) // 2 + 1  # 20
        W_out2 = (W2 - 3) // 2 + 1  # T // 4

        y2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=dtype, device=device)
        x2 = y1_gelu.contiguous()

        x_strides2 = x2.stride()
        w_strides2 = conv2d2_weight.stride()
        y2_strides = y2.stride()
        grid2 = (B, H_out2 * W_out2, triton.cdiv(C_out2, 64))
        conv2d_3x3_stride2_pad1_kernel[grid2](
            x2, conv2d2_weight, conv2d2_bias, y2,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x_strides2[0], x_strides2[1], x_strides2[2], x_strides2[3],
            w_strides2[0], w_strides2[1], w_strides2[2], w_strides2[3],
            y2_strides[0], y2_strides[1], y2_strides[2], y2_strides[3],
            BLOCK_CO=64,
        )

        # GELU2
        y2_gelu = torch.empty_like(y2)
        N2 = y2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2, y2_gelu, N2, BLOCK=1024)

        # Stage 3: Conv3 (384 -> 384)
        C_in3 = C_out2
        C_out3 = C_out2
        H3 = H_out2
        W3 = W_out2
        H_out3 = (H3 - 3) // 2 + 1  # 10
        W_out3 = (W3 - 3) // 2 + 1  # T // 8

        y3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=dtype, device=device)
        x3 = y2_gelu.contiguous()

        x_strides3 = x3.stride()
        w_strides3 = conv2d3_weight.stride()
        y3_strides = y3.stride()
        grid3 = (B, H_out3 * W_out3, triton.cdiv(C_out3, 64))
        conv2d_3x3_stride2_pad1_kernel[grid3](
            x3, conv2d3_weight, conv2d3_bias, y3,
            B, C_in3, H3, W3, C_out3, H_out3, W_out3,
            x_strides3[0], x_strides3[1], x_strides3[2], x_strides3[3],
            w_strides3[0], w_strides3[1], w_strides3[2], w_strides3[3],
            y3_strides[0], y3_strides[1], y3_strides[2], y3_strides[3],
            BLOCK_CO=64,
        )

        # GELU3
        y3_gelu = torch.empty_like(y3)
        N3 = y3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](y3, y3_gelu, N3, BLOCK=1024)

        # Linear projection: (B, 10, 384, T//8) -> (B, T//8, 1024)
        B, C_out3, H_out3, W_out3 = y3_gelu.shape  # B, 384, 10, T//8
        T_after = W_out3
        K = C_out3 * H_out3 * 1  # 384 * 10 * 1 = 3840
        M = 1024

        # Reshape X to (B, T_after, K)
        x_proj = y3_gelu.contiguous().view(B, T_after, K)

        # Allocate output (B, T_after, M)
        y_linear = torch.empty((B, T_after, M), dtype=dtype, device=device)

        # Strides
        x_proj_strides = x_proj.stride()  # (K, 1, K)
        w_strides = conv_out_weight.stride()  # (M, K)
        y_linear_strides = y_linear.stride()

        # Launch linear kernel: grid over (B, T_after, ceil(M/64))
        grid_lin = (B, T_after, triton.cdiv(M, 64))
        linear_no_bias_kernel[grid_lin](
            x_proj, conv_out_weight, y_linear,
            B, T_after, K, M,
            x_proj_strides[0], x_proj_strides[1], x_proj_strides[2],
            w_strides[0], w_strides[1],
            y_linear_strides[0], y_linear_strides[1], y_linear_strides[2],
            BLOCK_M=64, BLOCK_K=128,
        )

        # Add scaled positional embedding: positional_embedding shape (1500, 1024), cast to bf16 and slice [:T_after, :]
        pos_emb = positional_embedding.to(dtype)  # (1500, 1024)
        # Broadcast to (1, T_after, 1024)
        pos_emb_b = pos_emb[:T_after, :].unsqueeze(0).contiguous()
        SCALE = embed_scale  # float

        # Launch add kernel: grid over (B, T_after, ceil(M/128))
        grid_pos = (B, T_after, triton.cdiv(M, 128))
        add_scaled_pos_emb_kernel[grid_pos](
            y_linear, pos_emb_b, SCALE, B, T_after, M,
            y_linear.stride(0), y_linear.stride(1), y_linear.stride(2),
            pos_emb_b.stride(0), pos_emb_b.stride(1), pos_emb_b.stride(2),
        )

        return y_linear


def run(*args):
    return ModelNew()(*args)
