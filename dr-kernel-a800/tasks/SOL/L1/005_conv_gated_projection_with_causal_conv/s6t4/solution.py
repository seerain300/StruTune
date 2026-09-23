import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton matmul: C[M, N] = A[M, K] @ B[K, N] + Bias[N]
@triton.jit
def _matmul_linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
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

    # add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton 1D elementwise multiplication: C = A * B (flattened)
@triton.jit
def _elementwise_mul_1d_kernel(A_ptr, B_ptr, C_ptr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(C_ptr + offs, c, mask=mask)


# Triton grouped causal conv1d: input Bx: (B, H, S) -> output conv_out: (B, H, S)
# conv_weight: (H, 4) derived from in_proj_weight's last 4 columns
@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, conv_w_ptr, conv_bias_ptr, out_ptr,
    B, H, S, K,
    stride_bxm, stride_bxh, stride_bxs,
    stride_owm, stride_owh, stride_ows,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    # Iterate over sequence positions in tiles
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Fixed kernel_size=4, causal left padding handled by masking
        for k in range(0, K):
            pos = offs_s - (k + 1)  # pad left by k+1
            in_bounds = (pos >= 0) & mask_s
            bx_ptrs = Bx_ptr + pid_b * stride_bxm + pid_h * stride_bxh + pos * stride_bxs
            bx_vals = tl.load(bx_ptrs, mask=in_bounds, other=0.0)
            w_val = tl.load(conv_w_ptr + pid_h * 4 + k)  # conv_w is (H, 4) contiguous
            acc += bx_vals * w_val

        # add bias for channel h
        bias_val = tl.load(conv_bias_ptr + pid_h)
        acc += bias_val

        # store to out[b, h, s]
        out_ptrs = out_ptr + pid_b * stride_owm + pid_h * stride_owh + offs_s * stride_ows
        tl.store(out_ptrs, acc, mask=mask_s)


def _triton_in_proj(x, in_proj_weight, in_proj_bias):
    """
    Compute BCx = in_proj(x) where x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H) as float32.
    """
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    x_f = x.to(torch.float32).contiguous()
    in_proj_w_f = in_proj_weight.to(torch.float32).contiguous()
    bias_f = in_proj_bias.to(torch.float32).contiguous() if in_proj_bias is not None else torch.zeros(3 * H, device=x.device)

    A = x_f.reshape(M, K)  # (M, K)
    BkN = in_proj_w_f.T.reshape(K, N)  # (K, N)

    BCx = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, BkN, bias_f, BCx,
        M, N, K,
        A.stride(0), A.stride(1),
        BkN.stride(0), BkN.stride(1),
        BCx.stride(0), BCx.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return BCx.reshape(B, S, 3 * H)


def _triton_elementwise_mul_1d(A, B):
    """
    Elementwise multiply A * B. Returns C with same shape as A (float32).
    """
    A_f = A.to(torch.float32).contiguous()
    B_f = B.to(torch.float32).contiguous()
    total = A_f.numel()
    C = torch.empty_like(A_f, dtype=torch.float32, device=A.device)
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _elementwise_mul_1d_kernel[grid](A_f, B_f, C, total, BLOCK=BLOCK, num_warps=4, num_stages=1)
    return C.reshape(A.shape)


def _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias):
    """
    Bx: (B, H, S) float32, conv_w: (H, 4) float32, conv_bias: (H) float32
    Returns conv_out: (B, H, S) float32
    """
    B, H, S = Bx.shape
    K = 4

    Bx_f = Bx.to(torch.float32).contiguous()
    conv_w_f = conv_w.to(torch.float32).contiguous()
    conv_bias_f = conv_bias.to(torch.float32).contiguous()

    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        Bx_f, conv_w_f, conv_bias_f, conv_out,
        B, H, S, K,
        Bx_f.stride(0), Bx_f.stride(1), Bx_f.stride(2),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        BLOCK_S=128,
        num_warps=4, num_stages=2,
    )
    return conv_out


def _triton_out_proj(y, out_proj_weight, out_proj_bias):
    """
    y: (B, S, H) float32, out_proj_weight: (H, H) float32, out_proj_bias: (H) float32
    Returns output: (B, S, H) float32
    """
    B, S, H = y.shape
    M = B * S
    N = H
    y_f = y.contiguous().to(torch.float32)
    w_f = out_proj_weight.contiguous().to(torch.float32)
    bias_f = out_proj_bias.to(torch.float32).contiguous() if out_proj_bias is not None else torch.zeros(H, device=y.device)

    A = y_f.reshape(M, H)  # (M, K)
    BkN = w_f.T.reshape(H, N)  # (K, N)

    output = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, BkN, bias_f, output,
        M, N, H,
        A.stride(0), A.stride(1),
        BkN.stride(0), BkN.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return output.reshape(B, S, H)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only forward:
        1) in_proj: BCx = in_proj(x) -> (B, S, 3H)
        2) Split: B=BCx[:, :, :H], C=BCx[:, :, H:2H], x_proj=BCx[:, :, 2H:3H]
        3) Gate: Bx = B * x_proj
        4) Grouped causal conv: conv_out = conv1d(Bx, conv_weight, conv_bias, groups=H), kernel_size=4, padding=3
           Note: conv_weight is derived from in_proj_weight[:, -4:].
        5) Gating: y = C * conv_out  (align shapes by transposing conv_out to (B, S, H))
        6) Final projection: out = out_proj(y)
        All operations are implemented via Triton kernels. No PyTorch functional ops are used.
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)
        B = BCx[:, :, :x.shape[-1]]                           # (B, S, H)
        C = BCx[:, :, x.shape[-1]:2 * x.shape[-1]]           # (B, S, H)
        x_proj = BCx[:, :, 2 * x.shape[-1]:]                 # (B, S, H)

        # 2) Elementwise gate
        Bx = _triton_elementwise_mul_1d(B, x_proj)           # (B, S, H)

        # 3) Grouped causal conv: prepare conv_weight from in_proj_weight's last 4 columns
        conv_w = in_proj_weight[:, -4:].to(torch.float32).contiguous()  # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias) # (B, H, S)

        # 4) Output gating: y = C * conv_out; align shapes (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_1d(C, conv_out_T)         # (B, S, H)

        # 5) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
