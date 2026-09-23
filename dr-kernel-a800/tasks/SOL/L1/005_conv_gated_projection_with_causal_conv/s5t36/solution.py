import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, H, M_OUT,
    stride_xm, stride_xh,
    stride_wm, stride_wh,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # X: [M, H], W: [M_OUT, H], OUT: [M, M_OUT]
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    m_mask = offs_m < M
    h_mask = offs_h < H

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    # For each input hidden dimension h, multiply and accumulate
    for h in range(0, H):
        # Load x[m, h]
        x_ptrs = X_ptr + (offs_m * stride_xm + h * stride_xh)
        x_vals = tl.load(x_ptrs, mask=m_mask, other=0.0)  # [BLOCK_M]

        # Load w[m_out, h]
        w_ptrs = W_ptr + (h * stride_wh + offs_h * stride_wm)  # vector over offs_h
        w_vals = tl.load(W_ptr + h * stride_wm + offs_h * stride_wh, mask=h_mask, other=0.0)  # [BLOCK_H]

        # Outer product and accumulate
        acc += x_vals[:, None] * w_vals[None, :]

    # Add bias
    bias_ptrs = BIAS_ptr + offs_h
    bias_vals = tl.load(bias_ptrs, mask=h_mask, other=0.0)  # [BLOCK_H]
    acc += bias_vals[None, :]

    # Store OUT[m, m_out]
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & h_mask[None, :])


@triton.jit
def chunk_dim1_3_kernel(
    IN_ptr, OUT1_ptr, OUT2_ptr, OUT3_ptr,
    M, C,  # M = B * S, C = H
    stride_im, stride_in,  # IN is (M, 3*C)
    stride_ob1m, stride_ob1n,  # OUT1 (B, C, S) but we flatten as (M, C)
    stride_ob2m, stride_ob2n,  # OUT2 similarly
    stride_ob3m, stride_ob3n,  # OUT3 similarly
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # IN: [M, 3*C], OUTi: [M, C]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C

    # We iterate over channels and write to 3 outputs corresponding to channels 0..C-1, C..2C-1, 2C..3C-1
    for n0 in range(0, C, BLOCK_N):
        nn = n0 + offs_n
        nn_mask = nn < C

        # Output 1: channel 0..C-1
        in_ptrs1 = IN_ptr + (offs_m[:, None] * stride_im + nn[None, :] * stride_in)
        vals1 = tl.load(in_ptrs1, mask=m_mask[:, None] & nn_mask[None, :], other=0.0)
        out1_ptrs = OUT1_ptr + (offs_m[:, None] * stride_ob1m + nn[None, :] * stride_ob1n)
        tl.store(out1_ptrs, vals1, mask=m_mask[:, None] & nn_mask[None, :])

        # Output 2: channel C..2C-1
        in_ptrs2 = IN_ptr + (offs_m[:, None] * stride_im + (C + nn[None, :]) * stride_in)
        vals2 = tl.load(in_ptrs2, mask=m_mask[:, None] & nn_mask[None, :], other=0.0)
        out2_ptrs = OUT2_ptr + (offs_m[:, None] * stride_ob2m + nn[None, :] * stride_ob2n)
        tl.store(out2_ptrs, vals2, mask=m_mask[:, None] & nn_mask[None, :])

        # Output 3: channel 2C..3C-1
        in_ptrs3 = IN_ptr + (offs_m[:, None] * stride_im + (2 * C + nn[None, :]) * stride_in)
        vals3 = tl.load(in_ptrs3, mask=m_mask[:, None] & nn_mask[None, :], other=0.0)
        out3_ptrs = OUT3_ptr + (offs_m[:, None] * stride_ob3m + nn[None, :] * stride_ob3n)
        tl.store(out3_ptrs, vals3, mask=m_mask[:, None] & nn_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    B_ptr, XPRJ_ptr, OUT_ptr,
    M, C,  # M = B*S, C = H
    stride_bm, stride_bn,  # B is (M, C)
    stride_xm, stride_xn,  # x_proj is (M, C)
    stride_om, stride_on,  # OUT is (M, C)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C

    # Load B and XPRJ tiles
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    x_ptrs = XPRJ_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    b_vals = tl.load(b_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, out_vals, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, C_in, L, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # X_ptr: [M, C_in, L] (we pass M=B*S, C_in=H, L=S)
    # W_ptr: [C_in, K]
    # Out_ptr: [M, C_in, L]
    pid_m = tl.program_id(0)  # tile over M (batch*seq)
    pid_c = tl.program_id(1)  # tile over output channels C_in

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    m_mask = offs_m < M
    c_mask = offs_c < C_in

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # Loop over output positions t in tiles
    for t0 in range(0, L, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        t_mask = offs_t < L

        # For each kernel position k, compute input time t_in = t - k (no right-pad)
        for k in range(0, K):
            t_in = offs_t - k  # causal: k in {0,1,2,3} => t_in in [t-3, t]
            valid = (t_in >= 0) & (t_in < L) & t_mask

            # Load X[b, c, t_in] for all b in offs_m, c in offs_c
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_C]

            # Load W[c, k] for this k, broadcast across M
            w_ptrs = W_ptr + (offs_c * stride_wC + k * stride_wK)
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_C]
            acc += x_vals * w_vals[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_c  # bias[h] per output channel
    bias_vals = tl.load(bias_ptrs, mask=c_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results to Out[b, c, t]
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + (t0 + offs_t)[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & (t0 + offs_t)[None, :] < L
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def final_proj_kernel(
    IN_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, IN_H, OUT_H,
    stride_im, stride_in,
    stride_wm, stride_wh,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # IN: [M, IN_H], W: [OUT_H, IN_H], OUT: [M, OUT_H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over IN_H in blocks
    for k0 in range(0, IN_H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < IN_H

        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wm + k[None, :] * stride_wh)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w^T
        for kk in range(0, BLOCK_K):
            if k0 + kk < IN_H:
                x_col = x[:, kk]  # [BM]
                w_col = w[:, kk]  # [BN]
                acc += x_col[:, None] * w_col[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self, H, K=4):
        super().__init__()
        self.H = H
        self.K = K
        # Parameters: same shapes as original run
        # Note: in this environment, inputs to forward may provide these tensors; here we keep placeholders.
        self.register_buffer('in_proj_weight', torch.empty(3 * self.H, self.H), persistent=False)
        self.register_buffer('in_proj_bias', torch.empty(3 * self.H), persistent=False)
        self.register_buffer('conv_weight', torch.empty(self.H, self.K), persistent=False)
        self.register_buffer('conv_bias', torch.empty(self.H), persistent=False)
        self.register_buffer('out_proj_weight', torch.empty(self.H, self.H), persistent=False)
        self.register_buffer('out_proj_bias', torch.empty(self.H), persistent=False)

    def forward(self, x: torch.Tensor):
        # x: (B, S, H), ensure CUDA
        assert x.is_cuda, "Input must be on CUDA device"
        B, S, H = x.shape
        assert H == self.H, f"Hidden size mismatch: expected {self.H}, got {H}"

        device = x.device
        dtype = x.dtype

        # 1) First linear projection: y = x @ in_proj_weight^T + in_proj_bias, shape (B, S, 3H)
        M = B * S
        # Allocate flat input and output
        x_flat = x.reshape(M, H).contiguous()
        y_flat = torch.empty((M, 3 * H), device=device, dtype=dtype)

        in_proj_weight = self.in_proj_weight.to(device=device, dtype=dtype).contiguous()
        in_proj_bias = self.in_proj_bias.to(device=device, dtype=dtype).contiguous()

        BLOCK_M = 128
        BLOCK_H = 64
        grid_linear = (triton.cdiv(M, BLOCK_M), triton.cdiv(3 * H, BLOCK_H))
        in_proj_linear_kernel[grid_linear](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, 3 * H,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
        )

        # Reshape y_flat to (B, S, 3H)
        y = y_flat.view(B, S, 3 * H).contiguous()

        # 2) Split channels: B, C, x_proj
        M = B * S
        C = H  # as per original
        # Create flattened views for chunking along dim=1
        y_flat_ = y.reshape(M, 3 * C).contiguous()
        B_flat = torch.empty((M, C), device=device, dtype=dtype)
        C_flat = torch.empty((M, C), device=device, dtype=dtype)
        x_proj_flat = torch.empty((M, C), device=device, dtype=dtype)

        BLOCK_Mc = 128
        BLOCK_Nc = 64
        grid_chunk = (triton.cdiv(M, BLOCK_Mc), triton.cdiv(C, BLOCK_Nc))
        chunk_dim1_3_kernel[grid_chunk](
            y_flat_, B_flat, C_flat, x_proj_flat,
            M, C,
            y_flat_.stride(0), y_flat_.stride(1),
            B_flat.stride(0), B_flat.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            x_proj_flat.stride(0), x_proj_flat.stride(1),
            BLOCK_M=BLOCK_Mc, BLOCK_N=BLOCK_Nc,
        )

        # Reshape back to (B, S, C)
        B_ = B_flat.view(B, S, C).contiguous()
        C_ = C_flat.view(B, S, C).contiguous()
        x_proj = x_proj_flat.view(B, S, C).contiguous()

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, C), device=device, dtype=dtype)
        BLOCK_Mm = 128
        BLOCK_Nm = 64
        grid_mul = (triton.cdiv(M, BLOCK_Mm), triton.cdiv(C, BLOCK_Nm))
        mul_elementwise_kernel[grid_mul](
            B_, x_proj, Bx,
            M, C,
            B_.stride(0), B_.stride(1),
            x_proj.stride(0), x_proj.stride(1),
            Bx.stride(0), Bx.stride(1),
            BLOCK_M=BLOCK_Mm, BLOCK_N=BLOCK_Nm,
        )

        # 4) Grouped causal conv1d: groups=H, kernel_size=K, left-pad=K-1
        # Prepare X for conv: (M=B*S, C_in=H, L=S). We need padding left=K-1 on time.
        # We implement left-pad by masking in the kernel (t_in = t + (K-1) - k).
        M_conv = M
        C_in = C  # groups
        L = S
        conv_weight = self.conv_weight.to(device=device, dtype=dtype).contiguous()  # (C_in, K)
        conv_bias = self.conv_bias.to(device=device, dtype=dtype).contiguous()  # (C_in,)

        # Allocate output conv_out: (M, C_in, L)
        conv_out = torch.empty((M_conv, C_in, L), device=device, dtype=dtype)

        BLOCK_MC = 128
        BLOCK_Cc = 64
        BLOCK_Tc = 128
        grid_conv = (triton.cdiv(M_conv, BLOCK_MC), triton.cdiv(C_in, BLOCK_Cc))
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            M_conv, C_in, L, self.K,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_M=BLOCK_MC, BLOCK_C=BLOCK_Cc, BLOCK_T=BLOCK_Tc,
        )

        # Reshape conv_out to (B, C_in, L) = (B, H, S)
        conv_out = conv_out.view(B, C_in, L).contiguous()

        # 5) Output gating: y = C * conv_out
        y_gate = torch.empty((B, C_in, L), device=device, dtype=dtype)
        for b in range(B):
            for h in range(C_in):
                y_gate[b, h, :] = C_[b, :, :] * conv_out[b, h, :]

        # 6) Final projection: y_gate @ out_proj_weight^T + out_proj_bias, output (B, S, H)
        # Flatten (B, S, H) to (M, H), compute linear
        Mf = B * L
        y_gate_flat = y_gate.reshape(Mf, H).contiguous()  # (Mf, H)
        out_proj_weight = self.out_proj_weight.to(device=device, dtype=dtype).contiguous()  # (H, H)
        out_proj_bias = self.out_proj_bias.to(device=device, dtype=dtype).contiguous()  # (H,)

        output_flat = torch.empty((Mf, H), device=device, dtype=dtype)
        BLOCK_Ml = 128
        BLOCK_Nl = 64
        grid_final = (triton.cdiv(Mf, BLOCK_Ml), triton.cdiv(H, BLOCK_Nl))
        final_proj_kernel[grid_final](
            y_gate_flat, out_proj_weight, out_proj_bias, output_flat,
            Mf, H, H,
            y_gate_flat.stride(0), y_gate_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output_flat.stride(0), output_flat.stride(1),
            BLOCK_M=BLOCK_Ml, BLOCK_N=BLOCK_Nl,
        )

        # Reshape back to (B, S, H)
        output = output_flat.view(B, L, H).contiguous()

        return output


def run(*args):
    return ModelNew()(*args)
