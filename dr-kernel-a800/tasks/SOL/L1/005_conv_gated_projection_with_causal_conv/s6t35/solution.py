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
    out_dtype_code: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B + Bias.
    A: (M, K), row-major: stride_am along M, stride_ak along K
    B: (K, N), row-major: stride_bk along K, stride_bn along N
    C: (M, N), row-major: stride_cm along M, stride_cn along N
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load A tiles and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Cast to desired output dtype for final store; keep accumulators in fp32 for stability
        # We'll cast 'a' and 'b' to out_dtype_code before multiply to reduce fp64 overhead if present.
        # However, Triton will handle types; to be safe, cast to fp32 and multiply.
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store result, cast to output dtype
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # acc is float32; cast to desired dtype before store
    # out_dtype_code: 0 -> float16, 1 -> bfloat16, 2 -> float32
    if out_dtype_code == 0:
        acc_out = acc.to(tl.float16)
    elif out_dtype_code == 1:
        acc_out = acc.to(tl.bfloat16)
    else:
        acc_out = acc  # float32
    tl.store(c_ptrs, acc_out, mask=c_mask)


@triton.jit
def _elementwise_mul_1d(
    A_ptr, B_ptr, C_ptr,
    M,  # number of elements to process (we'll iterate), but pass B*S*S-like)
    stride_am, stride_b, stride_cm,
    out_dtype_code: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Elementwise C = A * B for 1D contiguous arrays of length M.
    A_ptr, B_ptr: length M
    C_ptr: length M
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    a = tl.load(A_ptr + offs * stride_am, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs * stride_b, mask=mask, other=0.0)

    # Compute in fp32, then cast to output dtype
    a32 = a.to(tl.float32)
    b32 = b.to(tl.float32)
    prod = a32 * b32

    if out_dtype_code == 0:
        prod_out = prod.to(tl.float16)
    elif out_dtype_code == 1:
        prod_out = prod.to(tl.bfloat16)
    else:
        prod_out = prod  # float32

    tl.store(C_ptr + offs * stride_cm, prod_out, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_pad_ptr, W_ptr, Bias_ptr, Y_ptr,
    B, H, S, K_kernel,
    stride_b, stride_h, stride_s,  # X_pad strides: (B, H, S+3)
    stride_w_h, stride_w_k,        # W strides: (H, 4) but we pass 2D (H, K_kernel)
    stride_y_b, stride_y_h, stride_y_s,  # Y strides: (B, H, S)
    out_dtype_code: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Grouped causal 1D conv for input X_pad: (B, H, S+3), weights W: (H, K_kernel=4), bias: (H).
    Output Y: (B, H, S).
    Each program handles one (b, h), vectorizes over s.
    """
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh % H

    # offs_s over output sequence positions [0..S-1]
    offs_s = tl.arange(0, BLOCK_S)
    s_mask = offs_s < S

    # Initialize accumulator
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over kernel taps k=0..K_kernel-1
    for k in range(0, K_kernel):
        # Input index is s + 3 - k (due to left pad of 3 for kernel_size=4)
        s_in = offs_s + 3 - k
        valid = (s_in >= 0) & (s_in < S) & s_mask
        x_ptrs = X_pad_ptr + b * stride_b + h * stride_h + s_in * stride_s
        x_vals = tl.load(x_ptrs, mask=valid, other=0.0)
        # weight scalar for this h and k
        w_val = tl.load(W_ptr + h * stride_w_h + k * stride_w_k)
        acc += x_vals.to(tl.float32) * w_val.to(tl.float32)

    # Add bias
    bias = tl.load(Bias_ptr + h).to(tl.float32)
    acc += bias

    # Store to Y
    y_ptrs = Y_ptr + b * stride_y_b + h * stride_y_h + offs_s * stride_y_s
    if out_dtype_code == 0:
        acc_out = acc.to(tl.float16)
    elif out_dtype_code == 1:
        acc_out = acc.to(tl.bfloat16)
    else:
        acc_out = acc
    tl.store(y_ptrs, acc_out, mask=s_mask)


def _dtype_code_from_tensor(t: torch.Tensor) -> int:
    # 0: float16, 1: bfloat16, 2: float32
    if t.dtype == torch.float16:
        return 0
    if t.dtype == torch.bfloat16:
        return 1
    return 2


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias.
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H), same dtype as x.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    A = x.contiguous().view(M, K)          # (M, K), dtype=x.dtype
    B_w = in_proj_weight.t().contiguous().view(K, N)  # (K, N), dtype=in_proj_weight.dtype
    Bias = in_proj_bias.contiguous().view(N)          # (N), dtype=in_proj_bias.dtype

    # Output tensor BCx with same dtype as x
    BCx = torch.empty((M, N), dtype=x.dtype, device=x.device)

    out_dtype_code = _dtype_code_from_tensor(x)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, B_w, Bias, BCx,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        BCx.stride(0), BCx.stride(1),
        out_dtype_code,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return BCx.view(B, S, N)


def _triton_elementwise_mul_1d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Elementwise C = A * B for 1D flattened tensors. Returns C with same dtype as A.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert A.shape == B.shape
    M = A.numel()
    C = torch.empty_like(A)
    out_dtype_code = _dtype_code_from_tensor(A)
    BLOCK = 1024
    grid = (triton.cdiv(M, BLOCK),)
    _elementwise_mul_1d[grid](
        A, B, C,
        M,
        1, 1, 1,  # strides not used when 1D contiguous; set to 1
        out_dtype_code,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return C


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_w: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Grouped causal 1D conv on Bx of shape (B, H, S), with conv_w: (H, 4), conv_bias: (H).
    Output: (B, H, S).
    We pre-pad Bx on the sequence dimension with 3 zeros.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    K_kernel = 4  # fixed as in original

    # Pre-pad with 3 zeros on the left along sequence dimension
    Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # (B, H, S+3)

    # Allocate output
    Y = torch.empty((B, H, S), dtype=Bx.dtype, device=Bx.device)

    out_dtype_code = _dtype_code_from_tensor(Bx)

    # Launch one program per (b, h)
    grid = (B * H,)
    _grouped_causal_conv1d_kernel[grid](
        Bx_padded, conv_w, conv_bias, Y,
        B, H, S, K_kernel,
        Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
        conv_w.stride(0), conv_w.stride(1),
        Y.stride(0), Y.stride(1), Y.stride(2),
        out_dtype_code,
        BLOCK_S=128,
        num_warps=2,
        num_stages=2,
    )
    return Y


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute output = y @ out_proj_weight^T + out_proj_bias.
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H), same dtype as y.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    K = H
    M = B * S

    Y_flat = y.contiguous().view(M, K)         # (M, K), dtype=y.dtype
    W_t = out_proj_weight.t().contiguous().view(K, K)  # (K, K), dtype=out_proj_weight.dtype
    Bias = out_proj_bias.contiguous().view(K)    # (K), dtype=out_proj_bias.dtype

    Output = torch.empty((M, K), dtype=y.dtype, device=y.device)

    out_dtype_code = _dtype_code_from_tensor(y)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
    _matmul_linear_kernel[grid](
        Y_flat, W_t, Bias, Output,
        M, K, K,
        Y_flat.stride(0), Y_flat.stride(1),
        W_t.stride(0), W_t.stride(1),
        Output.stride(0), Output.stride(1),
        out_dtype_code,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return Output.view(B, S, K)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that avoids PyTorch functional ops in host code.
        Returns output of shape (B, S, H), same dtype as x.
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias -> (B, S, 3H)
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)
        B, S, N = BCx.shape
        H = N // 3  # hidden_size

        # Split into B, C, x_proj
        B_channel = BCx[:, :, :H]                 # (B, S, H)
        C_channel = BCx[:, :, H:2*H]              # (B, S, H)
        x_proj = BCx[:, :, 2*H:]                  # (B, S, H)

        # 2) Elementwise gating: Bx = B_channel * x_proj
        Bx = _triton_elementwise_mul_1d(B_channel, x_proj)  # (B, S, H)

        # 3) Grouped causal conv: conv_weight is taken from in_proj_weight's last 4 columns
        #    Conv weight derived: (H, 4)
        conv_w = in_proj_weight[:, -4:].to(x.dtype).contiguous()  # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias)  # (B, H, S)

        # 4) Output gating: y = C_channel * conv_out; conv_out shape (B, H, S), C_channel (B, S, H)
        #    Align shapes by transposing conv_out to (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_1d(C_channel, conv_out_T)  # (B, S, H)

        # 5) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
