import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _matmul_linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias, A: (M, K), B: (K, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    Bsz, S, H,
    stride_ab, stride_as, stride_ah,
    stride_bb, stride_bs, stride_bh,
    stride_cb, stride_cs, stride_ch,
    BLOCK: tl.constexpr,
):
    # Compute C = A * B over (B, S, H), contiguous along last dim
    grid_elems = Bsz * S * H
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < grid_elems

    # Map linear index -> (b, s, h)
    S_times_H = S * H
    b = offs // S_times_H
    rem = offs % S_times_H
    s = rem // H
    h = rem % H

    a_ptrs = A_ptr + b * stride_ab + s * stride_as + h * stride_ah
    b_ptrs = B_ptr + b * stride_bb + s * stride_bs + h * stride_bh
    c_ptrs = C_ptr + b * stride_cb + s * stride_cs + h * stride_ch

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b2 = tl.load(b_ptrs, mask=mask, other=0.0)
    out = a * b2
    tl.store(c_ptrs, out, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, conv_w_ptr, conv_bias_ptr, out_ptr,
    Bsz, S, H,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr,
):
    # One program per (b, h)
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh % H

    # Accumulator for output vector of length S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    # conv_w: (H, 4) -> conv_w[h, k]
    for k in range(4):
        w_k = tl.load(conv_w_ptr + h * 4 + k, mask=tl.full((), True, tl.int1), other=0.0)
        # Loop over s in tiles
        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            mask_s = offs_s < S
            # Padded index: s' = s + 3 - k, within [s0, s0+BLOCK_S-1] + 3 - k must be < S
            # Ensure index within S
            s_prime = offs_s + 3 - k
            mask_s_prime = s_prime < S
            bx_ptrs = Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_prime * stride_bx_s
            bx_vals = tl.load(bx_ptrs, mask=mask_s & mask_s_prime, other=0.0)
            acc += bx_vals * w_k

    # Add bias
    bias_h = tl.load(conv_bias_ptr + h, mask=tl.full((), True, tl.int1), other=0.0)
    acc += bias_h

    # Store output: out[b, h, s] for all s
    for s in range(0, S):
        out_ptr_s = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s
        tl.store(out_ptr_s, acc[s], mask=tl.full((), True, tl.int1))


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are float32 and contiguous
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)  # should be (H, 4)
        conv_bias = conv_bias.contiguous().to(torch.float32)      # (H,)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)      # (H,)

        Bsz, S, H = x.shape
        K_in = H
        N_in = 3 * H
        M_in = Bsz * S

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        A = x.view(M_in, K_in)  # (M_in, K_in)
        B_in = in_proj_weight.t().view(K_in, N_in)  # (K_in, N_in)
        Bias_in = in_proj_bias.view(N_in)           # (N_in,)
        BCx = torch.empty((M_in, N_in), dtype=torch.float32, device=x.device)
        grid_in = (triton.cdiv(M_in, 128), triton.cdiv(N_in, 64))
        _matmul_linear_kernel[grid_in](
            A, B_in, Bias_in, BCx,
            M_in, N_in, K_in,
            A.stride(0), A.stride(1),
            B_in.stride(0), B_in.stride(1),
            BCx.stride(0), BCx.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )
        BCx = BCx.view(Bsz, S, N_in)  # (B, S, 3H)

        # 2) Split BCx into B, C, x_proj along last dim
        B_part = BCx[:, :, :H].contiguous()
        C_part = BCx[:, :, H:2 * H].contiguous()
        x_proj = BCx[:, :, 2 * H:].contiguous()

        # 3) Elementwise gate: Bx = B_part * x_proj
        Bx = torch.empty_like(B_part)
        # Launch elementwise 1D kernel
        total = Bsz * S * H
        _elementwise_mul_2d_kernel[(triton.cdiv(total, 256),)](
            B_part, x_proj, Bx,
            Bsz, S, H,
            B_part.stride(0), B_part.stride(1), B_part.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK=256,
            num_warps=4, num_stages=2,
        )

        # 4) Grouped causal 1D conv: conv_out (B, H, S)
        conv_out = torch.empty((Bsz, H, S), dtype=torch.float32, device=x.device)
        # Launch one program per (b, h)
        _grouped_causal_conv1d_kernel[(Bsz * H,)](
            Bx, conv_weight, conv_bias, conv_out,
            Bsz, S, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=64,
            num_warps=2, num_stages=2,
        )

        # 5) Output gating: y = C_part * conv_out after conv_out.T -> (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = torch.empty_like(conv_out_T)
        total_y = Bsz * S * H
        _elementwise_mul_2d_kernel[(triton.cdiv(total_y, 256),)](
            C_part, conv_out_T, y,
            Bsz, S, H,
            C_part.stride(0), C_part.stride(1), C_part.stride(2),
            conv_out_T.stride(0), conv_out_T.stride(1), conv_out_T.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK=256,
            num_warps=4, num_stages=2,
        )

        # 6) Final out-proj: output = y @ out_proj_weight^T + out_proj_bias
        M_out = Bsz * S
        K_out = H
        N_out = H
        A_out = y.view(M_out, K_out)
        B_out = out_proj_weight.t().contiguous().view(K_out, N_out)
        Bias_out = out_proj_bias.view(N_out)
        output = torch.empty((M_out, N_out), dtype=torch.float32, device=x.device)
        grid_out = (triton.cdiv(M_out, 128), triton.cdiv(N_out, 64))
        _matmul_linear_kernel[grid_out](
            A_out, B_out, Bias_out, output,
            M_out, N_out, K_out,
            A_out.stride(0), A_out.stride(1),
            B_out.stride(0), B_out.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )
        output = output.view(Bsz, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
