import torch
import triton
import triton.language as tl


# 1) Triton Linear Kernel: compute OUT[b, m, n] = sum_k X[b, n, k] * W[m, k] + bias[m]
# In our usage: X is (B, S, H) -> n=S, k=H; W is (M, H) where M in {H, H, H}; OUT is (B, S, M)
@triton.jit
def TritonLinearKernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    Bsz, N, M, H,   # N = S, M = out channels (here H), H = hidden
    BLOCK_N: tl.constexpr,   # tile size over S
    BLOCK_M: tl.constexpr,   # tile size over M (out_channels)
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along S
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along M (here H)

    mask_n = n_offsets < N
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    # Loop over hidden dimension to accumulate
    for k in range(0, H):
        x_ptrs = X_ptr + pid_b * (N * H) + n_offsets[:, None] * H + k  # (BLOCK_N, 1)
        w_ptrs = W_ptr + m_offsets[None, :] * H + k                    # (1, BLOCK_M)

        x_vals = tl.load(x_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float32)

        acc += x_vals * w_vals  # (BLOCK_N, BLOCK_M)

    bias_vals = tl.load(BIAS_ptr + m_offsets, mask=mask_m, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    out_ptrs = OUT_ptr + pid_b * (N * M) + n_offsets[:, None] * M + m_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_n[:, None] & mask_m[None, :])


# 2) Element-wise gating: OUT[b, s, h] = B[b, s, h] * X[b, s, h]
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    b_ptrs = B_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    x_ptrs = X_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# 3) Left-pad along sequence by PAD=3 to create Bx_pad[b, h, s_out] where s_out = s + PAD
# INPUT: Bx [B, H, S], OUTPUT: out_pad [B, H, S + PAD]
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, OUT_ptr,
    B, H, S, PAD,  # PAD=3
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sp = tl.program_id(2)  # tiling over S_out = S + PAD

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_out_start = pid_sp * BLOCK_S

    # Write zeros at the first PAD columns
    for i in range(0, PAD):
        out_ptr_pos = OUT_ptr + b * (H * S_out) + h * S_out + i
        tl.store(out_ptr_pos, 0.0)

    # Copy Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            val = tl.load(Bx_ptr + b * (H * S) + h * S + s_in)
            tl.store(OUT_ptr + b * (H * S_out) + h * S_out + (s_in + PAD), val)


# 4) Grouped causal 1D convolution (groups=H): OUT[b, h, s] = sum_{k=0..3} Bx_pad[b, h, s + k] * conv_weight[h, 0, k] + conv_bias[h]
@triton.jit
def TritonCausalConvKernel(
    Bx_pad_ptr, CONV_W_ptr, CONV_BIAS_ptr, OUT_ptr,
    B, H, S, PAD,  # PAD=3
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_start = pid_s * BLOCK_S
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    s_offsets = s_start + tl.arange(0, BLOCK_S)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Sum over kernel k in [0..3]
    for k in range(0, 4):
        s_k = s_offsets + k
        in_bounds = s_k < (S + PAD)
        vals = tl.load(Bx_pad_ptr + pid_b * (H * (S + PAD)) + h_offsets[None, :] * (S + PAD) + s_k[:, None],
                       mask=mask_s[:, None] & in_bounds[:, None], other=0.0)
        w_val = tl.load(CONV_W_ptr + h_offsets[None, :] * 4 + k, mask=mask_h[None, :], other=0.0)
        acc += vals * w_val  # broadcast w_val over columns

    bias_vals = tl.load(CONV_BIAS_ptr + h_offsets, mask=mask_h, other=0.0)
    acc += bias_vals[None, :]

    out_ptrs = OUT_ptr + pid_b * (H * S) + h_offsets[None, :] * S + s_offsets[:, None]
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


# 5) Element-wise gating: OUT[b, h, s] = C[b, h, s] * CONV[b, h, s]
@triton.jit
def TritonGateOutKernel(
    C_ptr, CONV_ptr, OUT_ptr,
    B, H, S,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    c_ptrs = C_ptr + pid_b * (H * S) + h_offsets[None, :] * S + s_offsets[:, None]
    conv_ptrs = CONV_ptr + pid_b * (H * S) + h_offsets[None, :] * S + s_offsets[:, None]
    out_ptrs = OUT_ptr + pid_b * (H * S) + h_offsets[None, :] * S + s_offsets[:, None]

    c_vals = tl.load(c_ptrs, mask=mask_h[None, :] & mask_s[:, None], other=0.0).to(tl.float32)
    conv_vals = tl.load(conv_ptrs, mask=mask_h[None, :] & mask_s[:, None], other=0.0).to(tl.float32)

    out_vals = c_vals * conv_vals

    tl.store(out_ptrs, out_vals, mask=mask_h[None, :] & mask_s[:, None])


# 6) Final linear projection: OUT[b, s, h] = sum_k IN[b, h, k] * W[h, k] + bias[h]
# IN is (B, H, S), W is (H, H), OUT is (B, S, H)
@triton.jit
def TritonFinalLinearKernel(
    IN_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    Bsz, S, H,  # S and H used for indexing
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Loop over hidden dimension H for accumulation
    for k in range(0, H):
        in_ptrs = IN_ptr + pid_b * (H * S) + h_offsets[None, :] * S + k  # (BLOCK_H, 1)
        w_ptrs = W_ptr + h_offsets[None, :] * H + k                      # (1, BLOCK_H)

        in_vals = tl.load(in_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)

        acc += in_vals * w_vals  # (BLOCK_S, BLOCK_H)

    bias_vals = tl.load(BIAS_ptr + h_offsets, mask=mask_h, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        # Ensure contiguity for safe indexing
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        Bsz, S, H = x.shape

        # 1) Three linear projections: (B, S, H)
        # First group: W0 = in_proj_weight[:H, :]
        W0 = in_proj_weight[:H, :].contiguous()
        b0 = in_proj_bias[:H].contiguous()
        B = torch.empty((Bsz, S, H), device=x.device, dtype=x.dtype)
        TritonLinearKernel[(Bsz, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            x, W0, b0, B, Bsz, S, H, H, 128, 64
        )

        # Second group: W1 = in_proj_weight[H:2*H, :]
        W1 = in_proj_weight[H:2 * H, :].contiguous()
        b1 = in_proj_bias[H:2 * H].contiguous()
        C = torch.empty((Bsz, S, H), device=x.device, dtype=x.dtype)
        TritonLinearKernel[(Bsz, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            x, W1, b1, C, Bsz, S, H, H, 128, 64
        )

        # Third group: W2 = in_proj_weight[2*H:3*H, :]
        W2 = in_proj_weight[2 * H:3 * H, :].contiguous()
        b2 = in_proj_bias[2 * H:3 * H].contiguous()
        x_proj = torch.empty((Bsz, S, H), device=x.device, dtype=x.dtype)
        TritonLinearKernel[(Bsz, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            x, W2, b2, x_proj, Bsz, S, H, H, 128, 64
        )

        # 2) Element-wise gating: Bx = B * x_proj, shape (B, S, H)
        Bx = torch.empty((Bsz, S, H), device=x.device, dtype=x.dtype)
        TritonGateKernel[(Bsz, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            B, x_proj, Bx, Bsz, S, H, 128, 64
        )

        # 3) Left-pad along S by PAD=3 to produce Bx_pad shape (B, H, S+3)
        S_pad = S + 3
        Bx_pad = torch.empty((Bsz, H, S_pad), device=x.device, dtype=x.dtype)
        TritonPadLeftKernel[(Bsz, triton.cdiv(H, 64), triton.cdiv(S_pad, 128))](  # we can choose large to cover, but 128 is fine
            Bx, Bx_pad, Bsz, H, S, 3, 128
        )  # Note: TritonPadLeftKernel expects inputs with contiguous layout; we'll ensure by passing Bx contiguous

        # 4) Grouped causal conv: conv_out (B, H, S)
        conv_out = torch.empty((Bsz, H, S), device=x.device, dtype=x.dtype)
        TritonCausalConvKernel[(Bsz, triton.cdiv(H, 64), triton.cdiv(S, 128))](
            Bx_pad, conv_weight, conv_bias, conv_out, Bsz, H, S, 3, 128, 64
        )

        # 5) Output gating: y = C * conv_out, shape (B, H, S)
        y = torch.empty((Bsz, H, S), device=x.device, dtype=x.dtype)
        TritonGateOutKernel[(Bsz, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            C, conv_out, y, Bsz, H, S, 128, 64
        )

        # 6) Final linear projection: output (B, S, H)
        output = torch.empty((Bsz, S, H), device=x.device, dtype=x.dtype)
        TritonFinalLinearKernel[(Bsz, triton.cdiv(S, 128), triton.cdiv(H, 64))](
            y, out_proj_weight, out_proj_bias, output, Bsz, S, H, 128, 64
        )

        return output


def run(*args):
    return ModelNew()(*args)
