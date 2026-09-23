import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    M, H, M_out,
    stride_xm, stride_xh,
    stride_wm, stride_wh,
    stride_om, stride_om2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # X: [M, H] row-major, W: [M_out, H] row-major, Out: [M, M_out] row-major
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < M_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < H

        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xh,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wm + k_ids[None, :] * stride_wh,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))

    bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_om2,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def pad_left1d_kernel(
    X_ptr, Out_ptr,
    B, S, H, PAD,
    stride_xb, stride_xs, stride_xh,
    stride_ob, stride_os, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # X: (B, S, H), Out: (B, S+PAD, H)
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    b = pid_b
    s = pid_s

    # tile over H
    offs_h = tl.arange(0, BLOCK_N)
    h_mask = offs_h < H

    # For padded output, s_out in [0, S+PAD)
    # We only need to store to Out at s_out in [PAD, PAD+S)
    s_out = s + PAD

    # Input index along H
    # For s_out < PAD, it's zero; otherwise it maps to s_out - PAD
    in_s = s_out - PAD
    # Valid when in_s in [0, S)
    valid_s_in = in_s >= 0 and in_s < S

    # Load X[b, in_s, h]
    x_val = tl.load(
        X_ptr + b * stride_xb + in_s * stride_xs + offs_h * stride_xh,
        mask=h_mask & (valid_s_in),
        other=0.0
    )
    # Store to Out[b, s_out, h]
    tl.store(
        Out_ptr + b * stride_ob + s_out * stride_os + offs_h * stride_oh,
        x_val,
        mask=h_mask
    )


@triton.jit
def elementwise_mul_kernel(
    A_ptr, B_ptr, Out_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # A: [M, N], B: [M, N], Out: [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
                mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    b = tl.load(B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
                mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    c = a * b
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             c, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, Out_ptr,
    N, C, L, PAD, K,  # K=4
    stride_xn, stride_xc, stride_xt,
    stride_wn, stride_wk,
    stride_on, stride_oc, stride_ot,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # N is batch size of original (B*S), C is channels H, L is original sequence length.
    pid_n = tl.program_id(0)  # over N (B*S)
    pid_c = tl.program_id(1)  # over C (H)

    n = pid_n
    c = pid_c

    offs_t = tl.arange(0, BLOCK_T)
    t_mask = offs_t < L

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # For each kernel element k in [0..K-1], read from X at t_in = t + PAD - k.
    # X is laid out as (N, C, L): n, c, t
    for k in range(0, K):
        t_in = offs_t + PAD - k
        valid = (t_in >= 0) & (t_in < L) & t_mask
        x_val = tl.load(
            X_ptr + n * stride_xn + c * stride_xc + t_in * stride_xt,
            mask=valid,
            other=0.0
        )
        w_val = tl.load(W_ptr + c * stride_wn + k * stride_wk)
        acc += x_val * w_val

    # Add bias for channel c
    bias_val = tl.load(BIAS_ptr + c)
    acc += bias_val

    # Store to Out at (n, c, t). Out corresponds to original non-padded L.
    tl.store(Out_ptr + n * stride_on + c * stride_oc + offs_t * stride_ot, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr, W_out_ptr, BIAS_out_ptr, Out_ptr,
    M, H,
    stride_ym, stride_yh,
    stride_wom, stride_woh,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Y: [M, H] row-major, W_out: [H, H] row-major, Out: [M, H] row-major
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < H

        y = tl.load(
            Y_ptr + offs_m[:, None] * stride_ym + k_ids[None, :] * stride_yh,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            W_out_ptr + offs_n[None, :] * stride_wom + k_ids[:, None] * stride_woh,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(y, tl.trans(w))

    bias = tl.load(BIAS_out_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    tl.store(
        Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation:
        - in_proj: y = F.linear(x, in_proj_weight, in_proj_bias)
        - split into B, C, x_proj and element-wise multiply
        - grouped causal conv1d on Bx with kernel_size=4, groups=H, padding=(3,0)
        - output gating: y = C * conv_out
        - final out-proj: linear on y
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        device = x.device
        B, S, H = x.shape
        M_out = in_proj_weight.shape[0]  # 3*H

        # 1) In-projection: y_flat = x_flat @ W^T + bias, shape (B*S, M_out)
        x_flat = x.reshape(B * S, H).contiguous()
        y_flat = torch.empty((B * S, M_out), dtype=x.dtype, device=device)

        # Tiling parameters
        BLOCK_M_in, BLOCK_N_in, BLOCK_K_in = 64, 64, 32
        grid_in = (triton.cdiv(B * S, BLOCK_M_in), triton.cdiv(M_out, BLOCK_N_in))
        in_proj_linear_kernel[grid_in](
            x_flat, in_proj_weight, in_proj_bias, y_flat,
            B * S, H, M_out,
            x_flat.stride(0), x_flat.stride(1),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_M=BLOCK_M_in, BLOCK_N=BLOCK_N_in, BLOCK_K=BLOCK_K_in,
            num_warps=4, num_stages=2
        )

        # Reshape to (B, S, 3H) and split
        y = y_flat.view(B, S, M_out)
        B_part, C_part, x_proj = torch.chunk(y, 3, dim=2)  # all (B, S, H)

        # 2) Element-wise gating: Bx = B_part * x_proj
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=device)
        BLOCK_M_mul, BLOCK_N_mul = 64, 64
        grid_mul = (B * S, H)
        elementwise_mul_kernel[grid_mul](
            B_part, x_proj, Bx,
            B * S, H,
            B_part.stride(0), B_part.stride(1),
            x_proj.stride(0), x_proj.stride(1),
            Bx.stride(0), Bx.stride(1),
            BLOCK_M=BLOCK_M_mul, BLOCK_N=BLOCK_N_mul,
            num_warps=4, num_stages=2
        )

        # 3) Pad Bx along sequence dimension by PAD=3 on the left
        Bx_padded = torch.empty((B, S + 3, H), dtype=x.dtype, device=device)
        grid_pad = (B, S)
        pad_left1d_kernel[grid_pad](
            Bx, Bx_padded,
            B, S, H, 3,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            BLOCK_M=1, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal conv1d: Bx_padded -> conv_out, groups=H, kernel=4, pad=(3,0)
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=device)

        grid_conv = (B * S, H)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B * S, H, S, 3, 4,  # PAD=3, K=4
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_C=1, BLOCK_T=256,
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C_part * conv_out
        y_gate = C_part * conv_out  # (B, H, S)

        # 6) Final projection: y_gate @ W_out^T + bias_out
        M = B * S
        y_gate_flat = y_gate.reshape(M, H).contiguous()
        out_flat = torch.empty((M, H), dtype=x.dtype, device=device)

        BLOCK_M_out, BLOCK_N_out, BLOCK_K_out = 64, 64, 32
        grid_out = (triton.cdiv(M, BLOCK_M_out), triton.cdiv(H, BLOCK_N_out))
        out_proj_linear_kernel[grid_out](
            y_gate_flat, out_proj_weight, out_proj_bias, out_flat,
            M, H,
            y_gate_flat.stride(0), y_gate_flat.stride(1),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=BLOCK_M_out, BLOCK_N=BLOCK_N_out, BLOCK_K=BLOCK_K_out,
            num_warps=4, num_stages=2
        )

        output = out_flat.view(B, S, H).contiguous()
        return output


def run(*args):
    return ModelNew()(*args)
