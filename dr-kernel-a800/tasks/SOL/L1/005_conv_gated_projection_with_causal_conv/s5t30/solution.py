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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # X: [M, H], W: [M_OUT, H], OUT: [M, M_OUT]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < M_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K=H in tiles
    for h0 in range(0, H, BLOCK_N):
        k = h0 + offs_n
        k_mask = k < H

        # Load W block [BLOCK_N, ]
        w_ptrs = W_ptr + k * stride_wh
        w = tl.load(w_ptrs, mask=k_mask, other=0.0)

        # Load X block [BLOCK_M, BLOCK_N]
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xh
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += x @ w (outer product across k)
        # Equivalent to acc += sum(x[:, :, None] * w[None, :], axis=1)
        acc += tl.sum(x * w[None, :], axis=1)

    # Add bias: BIAS: [M_OUT], broadcast over M
    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store OUT: [M, M_OUT]
    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_oh
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def chunk_dim1_3_kernel(
    Y_ptr, B_ptr, C_ptr, XPRJ_ptr,
    M, H,  # Y shape: (M, 3H), we output (M, H) for each chunk
    stride_ym, stride_yh,
    stride_bm, stride_bh,
    stride_cm, stride_ch,
    stride_xm, stride_xh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Y: [M, 3H], produce B: [M, H], C: [M, H], XPRJ: [M, H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < H

    base = offs_n * 3  # since 3H

    # B: slice [base, base+H)
    b_ptrs = Y_ptr + offs_m[:, None] * stride_ym + (base + offs_n[None, :]) * stride_yh
    b_vals = tl.load(b_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bh, b_vals, mask=m_mask[:, None] & n_mask[None, :])

    # C: slice [base+H, base+2H)
    c_ptrs = Y_ptr + offs_m[:, None] * stride_ym + (base + H + offs_n[None, :]) * stride_yh
    c_vals = tl.load(c_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_ch, c_vals, mask=m_mask[:, None] & n_mask[None, :])

    # x_proj: slice [base+2H, base+3H)
    x_ptrs = Y_ptr + offs_m[:, None] * stride_ym + (base + 2 * H + offs_n[None, :]) * stride_yh
    x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(XPRJ_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xh, x_vals, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def mul_elementwise_kernel(
    B_ptr, XPRJ_ptr, OUT_ptr,
    M, H,
    stride_bm, stride_bh,
    stride_xm, stride_xh,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # B: [M, H], XPRJ: [M, H], OUT: [M, H]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < H

    b_ptrs = B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bh
    x_ptrs = XPRJ_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xh
    b_vals = tl.load(b_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    out = b_vals * x_vals

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_oh
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    M, C_IN, L, K,
    stride_xM, stride_xC, stride_xL,
    stride_wC, stride_wK,
    stride_oM, stride_oC, stride_oL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # X_ptr: [M, C_IN, L] (M=B*S), W_ptr: [C_IN, K], OUT_ptr: [M, C_IN, L]
    pid_m = tl.program_id(0)  # tile over M (batch*seq)
    pid_c = tl.program_id(1)  # tile over output channels C_IN

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    c_mask = offs_c < C_IN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over output time positions in tiles
    for t0 in range(0, L, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        ti_mask = t < L

        # Accumulate over kernel window k in {0,1,2,3} (no padding)
        for k in range(0, K):
            t_in = t - k  # causal shift
            valid = (t_in >= 0) & (t_in < L) & ti_mask

            # Load X[b, c, t_in] for all b in offs_m, c in offs_c
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xM + offs_c[None, :] * stride_xC + t_in[None, :] * stride_xL)
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & c_mask[None, :] & valid[None, :], other=0.0)  # [BLOCK_M, BLOCK_N]

            # Load W[c, k] for this k, broadcast across M
            w_ptrs = W_ptr + offs_c * stride_wC + k * stride_wK
            w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)  # [BLOCK_N]
            acc += x_vals * w_vals[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_c
    bias_vals = tl.load(bias_ptrs, mask=c_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results to Out[b, c, t]
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_oM + offs_c[None, :] * stride_oC + (t0 + tl.arange(0, BLOCK_T))[None, :] * stride_oL)
    out_mask = m_mask[:, None] & c_mask[None, :] & (t0 + tl.arange(0, BLOCK_T))[None, :] < L
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

    for k0 in range(0, IN_H, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < IN_H
        # Load IN block: [BLOCK_M, BLOCK_K]
        in_ptrs = IN_ptr + (offs_m[:, None] * stride_im + k[None, :] * stride_in)
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W block: [BLOCK_N, BLOCK_K] (we need W^T per tile: [BLOCK_K, BLOCK_N])
        # Here we loop over k and accumulate acc += sum_k (x[:, k] * W[n, k])
        # Implement as acc += x[:, None, :] * (tl.load(W[n, k]) broadcast)
        # More directly: acc += dot(x, W[n, :]) across k
        # But Triton doesn't have a direct tl.dot for 2D. We do outer product loop over k:
        # For each kk in BLOCK_K, load w_vals[n, kk], outer with x[:, kk] and accumulate.
        # However, simpler: load W[n, :] in chunks and outer. Here we do a single k loop:
        # We'll load W per k and use tl.sum(x * w[:, None], axis=1) to accumulate into acc.
        # Note: this is a bit indirect. We'll implement the accumulation with k loop:
        # For each kk, w_vals = W[n, kk], acc += sum(x[:, kk] * w_vals) broadcast over n.
        # This is fine: for each kk, W[n, kk] is scalar per n. So:
        # For each kk in BLOCK_K range, but since k_mask indicates actual k, we can use a Python for k:
        # We'll compute acc += sum over kk in BLOCK_K of (x[:, kk] * w_vals_n), where w_vals_n = W[n, kk] vector for n in tile.
        # To avoid extra loops, we implement direct per-k accumulation using BLOCK_K as single k:
        # Since BLOCK_K is small, we can iterate:
        # Note: Triton requires compile-time loops; we'll iterate kk from 0 to BLOCK_K, masking with k_mask.
        for kk in range(0, BLOCK_K):
            # kk is actually runtime, but we keep iteration up to BLOCK_K and mask with kk < IN_H
            kk_valid = (k0 + kk) < IN_H
            # x[:, kk] is x[:, kk] where kk_valid; else 0
            x_col = tl.load(IN_ptr + (offs_m * stride_im + (k0 + kk) * stride_in), mask=m_mask & kk_valid, other=0.0)  # [BLOCK_M]
            # Load W[n, k0+kk] vector: [BLOCK_N]
            w_ptrs = W_ptr + (offs_n * stride_wm + (k0 + kk) * stride_wh)
            w_vals = tl.load(w_ptrs, mask=offs_n < OUT_H & kk_valid, other=0.0)  # [BLOCK_N]
            # Outer product and accumulate: x_col[:, None] * w_vals[None, :]
            acc += x_col[:, None] * w_vals[None, :]

    # Add bias per output channel
    bias_ptrs = BIAS_ptr + offs_n
    bias_vals = tl.load(bias_ptrs, mask=offs_n < OUT_H, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store OUT: [M, OUT_H]
    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & (offs_n < OUT_H)[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # x: (B, S, H)
        B, S, H = x.shape
        M = B * S

        # 1) in_proj: y = F.linear(x, in_proj_weight, in_proj_bias), y: (B, S, 3H)
        # Flatten x to (M, H) for kernel
        x_flat = x.reshape(M, H).contiguous()
        M_out = 3 * H
        y_flat = torch.empty((M, M_out), device=x.device, dtype=x.dtype)
        # Launch in_proj kernel
        BLOCK_M = 128
        BLOCK_N = 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M_out, BLOCK_N))
        in_proj_linear_kernel[grid](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            M, H, M_out,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # Reshape y_flat to (B, S, 3H)
        y = y_flat.view(B, S, 3 * H).contiguous()

        # 2) Split into B, C, x_proj along dim=1: each (B, S, H)
        # We implement chunking in Triton: produce (B, C, XPRJ) as (M, H) each
        M2 = M  # still M=B*S
        H2 = H
        B_t = torch.empty((M2, H2), device=x.device, dtype=x.dtype)
        C_t = torch.empty((M2, H2), device=x.device, dtype=x.dtype)
        XPRJ_t = torch.empty((M2, H2), device=x.device, dtype=x.dtype)

        BLOCK_M2 = 128
        BLOCK_N2 = 128
        grid2 = (triton.cdiv(M2, BLOCK_M2), triton.cdiv(H2, BLOCK_N2))
        chunk_dim1_3_kernel[grid2](
            y, B_t, C_t, XPRJ_t,
            M2, H2,
            y.stride(0), y.stride(2),
            B_t.stride(0), B_t.stride(1),
            C_t.stride(0), C_t.stride(1),
            XPRJ_t.stride(0), XPRJ_t.stride(1),
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
        )

        # Reshape to (B, S, H) for convenience
        B_ = B_t.view(B, S, H).contiguous()
        C_ = C_t.view(B, S, H).contiguous()
        XPRJ_ = XPRJ_t.view(B, S, H).contiguous()

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((M2, H2), device=x.device, dtype=x.dtype)
        mul_elementwise_kernel[(triton.cdiv(M2, 128), triton.cdiv(H2, 128))](  # grid heuristic
            B_, XPRJ_, Bx,
            M2, H2,
            B_.stride(0), B_.stride(2),
            XPRJ_.stride(0), XPRJ_.stride(2),
            Bx.stride(0), Bx.stride(1),
            BLOCK_M=128, BLOCK_N=128,
        )

        # 4) Grouped causal conv1d: F.conv1d(Bx, conv_weight, conv_bias, groups=H, kernel_size=4)
        # Bx shape (B, S, H) => we treat N=M=B*S, C_in=H, L=S, K=4
        # X_ptr: [M, C_in, L] => we use Bx reshaped to (M, H, S). We'll transpose to (M, S, H) and then view.
        # But to keep it simple, we pass Bx as (B, S, H) and construct X as (M, H, S) by indexing.
        # We'll create a contiguous tensor for X: (M, H, S) by viewing Bx.view(M, H, S) if contiguous, otherwise permute.
        # Safer: permute to (B, H, S) then reshape to (M, H, S).
        Bx_p = Bx.view(B, S, H).contiguous()
        X_3D = Bx_p.permute(0, 2, 1).contiguous()  # (B, H, S)
        X = X_3D.view(M, H, S).contiguous()       # (M, H, S)

        # conv_weight: (C_in, K) where C_in=H, K=4. We need to produce Out (B, H, S) which we flatten to (M, H, S).
        # But our grouped_conv1d kernel expects X as (M, C_in, L) -> here (M, H, S). Let's pass pointers accordingly.
        # We need OUT as (M, C_in, L) -> (M, H, S)
        Out = torch.empty((M, H, S), device=x.device, dtype=x.dtype)

        grid3 = (triton.cdiv(M, 128), triton.cdiv(H, 64), triton.cdiv(S, 128))
        grouped_causal_conv1d_kernel[grid3](
            X, conv_weight, conv_bias, Out,
            M, H, S, 4,  # K=4
            X.stride(0), X.stride(1), X.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_M=128, BLOCK_N=64, BLOCK_T=128,
        )

        # conv_out shape: (B, H, S)
        conv_out = Out.view(B, H, S).contiguous()

        # 5) Output gating: y = C * conv_out
        # C_ shape (B, S, H), conv_out shape (B, H, S)
        # We need to align (S, H) for mul. We can view or permute. Easiest: gate per (b, h) along S.
        # Implement elementwise mul in Triton on reshaped tensors:
        y_mul = torch.empty((M2, H2), device=x.device, dtype=x.dtype)
        # Reshape C_ to (B, S, H), conv_out to (B, H, S), then transpose C_ to (B, H, S) for mul.
        C_bsh = C_.permute(0, 2, 1).contiguous()  # (B, H, S)
        conv_bhs = conv_out  # (B, H, S)
        # Launch elementwise kernel on (B, H, S)
        # We'll flatten to (M2, H2) by viewing as (B*S, H): note H and S are independent dims -> total H*S per (b, h)
        # Instead, we directly compute C * conv_out by broadcasting over S:
        # We can compute per (b, h) along S: for each s, elementwise multiply C[b, h, s] * conv_out[b, h, s].
        # Implement by reshaping:
        # y_mul[b, s, h] = C_[b, s, h] * conv_out[b, h, s] -> we reshape to (B, S, H), then to (B*S, H)
        # But to keep Triton simple, we flatten C_bsh and conv_bhs to (M2, H) and do elementwise mul.
        # However, we must ensure alignment: C_bsh[b, h, s] and conv_out[b, h, s] are the same s. So we can do:
        y_mul_flat = (C_bsh.reshape(M2, H2) * conv_bhs.reshape(M2, H2)).contiguous()

        # 6) Final output projection: y -> out_proj(y)
        # Final y_flat: (B*S, H)
        # out_proj_weight: (H, H), out_proj_bias: (H,)
        output = torch.empty((M2, H2), device=x.device, dtype=x.dtype)
        final_proj_kernel[(triton.cdiv(M2, 128), triton.cdiv(H2, 128), triton.cdiv(H2, 128))](  # heuristic grid
            y_mul_flat, out_proj_weight, out_proj_bias, output,
            M2, H2, H2,
            y_mul_flat.stride(0), y_mul_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
        )

        # Reshape back to (B, S, H)
        output_final = output.view(B, S, H).contiguous()

        return output_final


def run(*args):
    return ModelNew()(*args)
