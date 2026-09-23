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
    # Compute C = A @ B + Bias, where:
    # A: (M, K) row-major, strides (stride_am, stride_ak)
    # B: (K, N) row-major, strides (stride_bk, stride_bn)
    # C: (M, N) row-major, strides (stride_cm, stride_cn)
    # Bias: (N)
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

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_1d_kernel(
    A_ptr, B_ptr, C_ptr,
    total,
    BLOCK: tl.constexpr,
):
    # 1D elementwise multiplication over flattened arrays
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(C_ptr + offs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, W_ptr, Bias_ptr, C_ptr,
    Bsz, Hsz, Sz,
    stride_bx_b, stride_bx_s, stride_bx_h,
    stride_w_h, stride_w_k,
    stride_c_b, stride_c_h, stride_c_s,
    BLOCK_S: tl.constexpr,
):
    # Grouped causal 1D conv with kernel_size=4 and padding=3, groups=Hsz.
    # Bx_ptr: (Bsz, Hsz, Sz)
    # W_ptr: (Hsz, 4) - conv_weight derived from in_proj_weight's last 4 columns, shape (H, 4).
    # Bias_ptr: (Hsz)
    # C_ptr: (Bsz, Hsz, Sz)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid_b
    h = pid_h

    for s0 in range(0, Sz, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Fixed kernel_size=4, padding=3
        for k in range(4):
            s_eff = offs_s - 3 + k  # causal padding on the left
            in_mask = (s_eff >= 0) & (s_eff < Sz)
            bx_ptrs = Bx_ptr + b * stride_bx_b + s_eff * stride_bx_s + h * stride_bx_h
            bx = tl.load(bx_ptrs, mask=in_mask, other=0.0)
            w = tl.load(W_ptr + h * stride_w_h + k * stride_w_k)
            acc += bx * w

        # Add bias for channel h
        bias = tl.load(Bias_ptr + h)
        acc += bias

        c_ptrs = C_ptr + b * stride_c_b + h * stride_c_h + offs_s * stride_c_s
        store_mask = (offs_s < Sz)
        tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        device = x.device
        B, S, H = x.shape

        # 1) in_proj linear: BCx = in_proj(x), shape (B, S, 3H)
        M = B * S
        N = 3 * H
        A = x.reshape(M, H).contiguous()  # (M, K)
        B_in = in_proj_weight.t().contiguous()  # (K, N) where K=H, N=3H
        BCx = torch.empty((M, N), dtype=torch.float32, device=device)

        grid_in = (triton.cdiv(M, 128), triton.cdiv(N, 64))
        _matmul_linear_kernel[grid_in](
            A, B_in, in_proj_bias, BCx,
            M, N, H,
            A.stride(0), A.stride(1),
            B_in.stride(0), B_in.stride(1),
            BCx.stride(0), BCx.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )
        BCx = BCx.view(B, S, N)

        # Split BCx: B = first H, C = middle H, x_proj = last H
        B_t = BCx[:, :, :H].contiguous()
        C_t = BCx[:, :, H:2 * H].contiguous()
        x_proj = BCx[:, :, 2 * H:].contiguous()

        # 2) Elementwise gating: Bx = B * x_proj, shape (B, S, H)
        total_gate = B * S * H
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        _elementwise_mul_1d_kernel[(triton.cdiv(total_gate, 1024),)](
            B_t.reshape(-1), x_proj.reshape(-1), Bx.reshape(-1),
            total_gate, BLOCK=1024, num_warps=4, num_stages=1,
        )

        # 3) Grouped causal conv: conv_out = conv(Bx) with kernel_size=4, padding=3, groups=H
        # We replicate conv_weight as in the original: last 4 columns of in_proj_weight
        # conv_weight is (H, 4); reshape for kernel view (H, 1, 4) conceptually, but we only need (H, 4).
        # Important: conv_weight must be (H, 4) for groups=H conv.
        Hc, Kc = conv_weight.shape  # should be (H, 4)
        conv_out = torch.empty((B, Hc, S), dtype=torch.float32, device=device)
        grid_conv = (B, Hc)
        _grouped_causal_conv1d_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, Hc, S,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1),  # stride for h and k dims
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128,
            num_warps=2, num_stages=2,
        )

        # 4) Output gating: y = C * conv_out, shape (B, Hc, S)
        # C_t: (B, S, H); conv_out: (B, Hc, S). Multiply elementwise over last dim.
        # Because H == Hc in this setup, we can proceed.
        y = torch.empty((B, Hc, S), dtype=torch.float32, device=device)
        total_gate2 = B * Hc * S
        _elementwise_mul_1d_kernel[(triton.cdiv(total_gate2, 1024),)](
            C_t.reshape(-1), conv_out.reshape(-1), y.reshape(-1),
            total_gate2, BLOCK=1024, num_warps=4, num_stages=1,
        )

        # 5) Final out-proj: output = y @ out_proj_weight^T + out_proj_bias
        # y: (B, Hc, S) => A = y.view(B*S, Hc)
        M_out = B * S
        N_out = Hc  # output channels = Hc
        A_out = y.reshape(M_out, Hc).contiguous()
        B_out = out_proj_weight.t().contiguous()  # (Hc, Hc)
        output = torch.empty((M_out, Hc), dtype=torch.float32, device=device)

        grid_out = (triton.cdiv(M_out, 128), triton.cdiv(Hc, 64))
        _matmul_linear_kernel[grid_out](
            A_out, B_out, out_proj_bias, output,
            M_out, Hc, Hc,
            A_out.stride(0), A_out.stride(1),
            B_out.stride(0), B_out.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )
        output = output.view(B, S, Hc)

        return output


def run(*args):
    return ModelNew()(*args)
