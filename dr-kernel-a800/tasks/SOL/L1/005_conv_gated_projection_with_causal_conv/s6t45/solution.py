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
    # Compute C = A @ B + Bias, A: (M,K), B: (K,N)
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
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # C = A * B, 2D tiled elementwise
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an), mask=mask, other=1.0)
    b = tl.load(B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn), mask=mask, other=1.0)
    out = a * b
    tl.store(C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), out, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr,  # pointer to Bx_padded, shape (B, H, S+3), row-major: ((b*H + h) * (S+3)) + s
    W_ptr,  # pointer to conv_weight, shape (H, 4)
    Bias_ptr,  # pointer to conv_bias, shape (H,)
    Out_ptr,  # pointer to conv_out, shape (B, H, S)
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_xb, stride_xh, stride_xs,  # strides for X: ((b*H + h) * (S+3)) + s
    stride_outb, stride_outh, stride_outs,  # strides for Out: ((b*H + h) * S) + s
):
    # Each program handles one (b, h) and writes conv_out[b, h, 0:S]
    pid_bh = tl.program_id(0)  # 0..(B*H-1)
    b = pid_bh // H
    h = pid_bh % H

    # Initialize output row
    out_row = tl.zeros((S,), dtype=tl.float32)

    # Accumulate over k in 0..3 with causal left pad of 3
    # X indices: ((b*H + h) * (S+3)) + s_pad, where s_pad = s + 3
    for k in range(4):
        # s index starts from 0..S-1, X[s_pad] valid only if s_pad < S+3
        # Since s_pad = s + 3, and s < S, s_pad in [3, S+2], always valid
        val = 0.0
        for s in range(0, S):
            s_pad = s + 3
            x_idx = (b * H + h) * (S + 3) + s_pad
            val += tl.load(X_ptr + x_idx, mask=True, other=0.0)
        # conv_weight[h, k]
        w_idx = h * 4 + k
        w = tl.load(W_ptr + w_idx, mask=True, other=0.0)
        out_row += val * w

    # Add bias[h]
    bias = tl.load(Bias_ptr + h, mask=True, other=0.0)
    out_row += bias

    # Store out_row to Out[b, h, :]
    # Out strides: ((b*H + h) * S) + s
    for s in range(0, S):
        out_idx = (b * H + h) * S + s
        tl.store(Out_ptr + out_idx, out_row[s])


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias.
    x: (B, S, H) -> (M, K) with M=B*S, K=H
    in_proj_weight: (3H, H) -> (K, N) with N=3H
    Returns BCx: (B, S, 3H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Ensure float32 and contiguous for Triton
    A = x.contiguous().view(M, K).to(torch.float32)
    B_w = in_proj_weight.t().contiguous().view(K, N).to(torch.float32)
    Bias = in_proj_bias.contiguous().view(N).to(torch.float32)

    C = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, B_w, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C.view(B, S, N)


def _triton_elementwise_mul_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Elementwise C = A * B on 2D tensors (B, S, H).
    Returns C with dtype float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B_, S, H = A.shape
    C = torch.empty((B_, S, H), dtype=torch.float32, device=A.device)
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(B_, BLOCK_M), triton.cdiv(H, BLOCK_N))
    _elementwise_mul_2d_kernel[grid](
        A.to(torch.float32), B.to(torch.float32), C,
        B_, H, S,
        A.stride(0), A.stride(2),
        B.stride(0), B.stride(2),
        C.stride(0), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return C


def _grouped_causal_conv1d(Bx_padded: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Grouped causal conv with kernel_size=4, groups=H.
    Bx_padded: (B, H, S+3), float32
    conv_weight: (H, 4), float32
    conv_bias: (H,), float32
    Returns conv_out: (B, H, S), float32
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S_padded = Bx_padded.shape
    S = S_padded - 3
    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx_padded.device)

    # Ensure inputs are float32 and contiguous
    X = Bx_padded.contiguous().view(B * H, S + 3).to(torch.float32)
    W = conv_weight.contiguous().view(H, 4).to(torch.float32)  # (H, 4)
    Bias = conv_bias.contiguous().view(H).to(torch.float32)

    # Launch one program per (b,h)
    grid = (B * H,)
    _grouped_causal_conv1d_kernel[grid](
        X, W, Bias, conv_out,
        B=B, H=H, S=S,
        stride_xb=H, stride_xh=1, stride_xs=1,  # X is row-major: ((b*H + h) * (S+3)) + s
        stride_outb=H, stride_outh=1, stride_outs=1,  # Out is row-major: ((b*H + h) * S) + s
    )
    return conv_out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute output = y @ out_proj_weight^T + out_proj_bias.
    y: (B, S, H) -> (M, K) with M=B*S, K=H
    out_proj_weight: (H, H) -> (K, N) with N=H
    Returns output: (B, S, H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    K = H
    N = H
    M = B * S

    A = y.contiguous().view(M, K).to(torch.float32)
    B_w = out_proj_weight.t().contiguous().view(K, N).to(torch.float32)
    Bias = out_proj_bias.contiguous().view(N).to(torch.float32)

    C = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, B_w, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C.view(B, S, N)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Step 1: in-proj: BCx = x @ in_proj_weight^T + in_proj_bias
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # Split into B, C, x_proj
        B = BCx[:, :, :x.shape[-1]]       # (B, S, H)
        C = BCx[:, :, x.shape[-1]:2 * x.shape[-1]]  # (B, S, H)
        x_proj = BCx[:, :, 2 * x.shape[-1]:]        # (B, S, H)

        # Step 2: Element-wise gating
        Bx = _triton_elementwise_mul_2d(B, x_proj)  # (B, S, H)

        # Step 3: Grouped causal 1D conv, kernel_size=4, groups=H
        # Prepare conv_weight from in_proj_weight's last 4 columns
        conv_w = in_proj_weight[:, -4:].to(torch.float32).contiguous()  # (H, 4)
        # Pre-pad Bx on the sequence dimension with 3 zeros (causal left pad)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # (B, S, H) padded to (B, S, H+3) -> but S+3
        # Note: torch.nn.functional.pad requires int tuple (pad_left, pad_right). We pad left by 3.
        # Ensure dimensions: Bx shape is (B, S, H); pad by 3 on the last dim (H). We want to pad along the sequence S? The original code pads the conv input which is (B, H, S). Here we need to build the padded input explicitly.
        # For grouped conv, input to conv is (B, H, S+3), so we need to pad the sequence length S.
        # However, in our code, Bx is (B, S, H). The original conv is applied to Bx after transposing to (B, H, S) as Bx.T? Wait, original code transposes BCx to (B, S, 3H) -> (B, 3H, S) not directly (B, H, S).
        # To strictly follow the original logic:
        # After computing Bx: shape (B, S, H), original code pads Bx with 3 zeros on the left for causal and then applies F.conv1d with groups=H using conv_weight reshaped to (H, 1, 4).
        # We need to create a tensor of shape (B, H, S+3) for grouped conv: we can use Bx.transpose(-1, -2) to get (B, H, S) and pad along last dim with 3 zeros.
        Bx_for_conv = Bx.transpose(-1, -2).contiguous()  # (B, H, S)
        Bx_padded = torch.nn.functional.pad(Bx_for_conv, (3, 0))  # (B, H, S+3)

        conv_out = _grouped_causal_conv1d(Bx_padded, conv_w, conv_bias)  # (B, H, S)

        # Step 4: Output gating: y = C * conv_out; align shapes (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_2d(C, conv_out_T)         # (B, S, H)

        # Step 5: Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
