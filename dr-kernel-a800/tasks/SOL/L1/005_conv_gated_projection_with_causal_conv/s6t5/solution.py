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
    """
    Computes C = A @ B + Bias, where
    - A: (M, K), row-major with strides (stride_am, stride_ak)
    - B: (K, N), row-major with strides (stride_bk, stride_bn) = (1, N) if contiguous
    - C: (M, N), row-major with strides (stride_cm, stride_cn)
    Accumulate in float32, output cast to C dtype via caller.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: shape (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Add bias: bias shape (N,), broadcast over M
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store result to C (cast to output dtype handled by caller)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # We'll let the caller cast to desired dtype before store. Here, we store as float32 and assume
    # caller provides C_ptr with desired dtype; Triton will cast on store.
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_1d_kernel(a_ptr, b_ptr, c_ptr, total_elems, BLOCK: tl.constexpr):
    """
    Elementwise multiply: c[i] = a[i] * b[i], 1D flattened.
    a_ptr, b_ptr: input pointers, float32
    c_ptr: output pointer, float32
    total_elems: int
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(c_ptr + offs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, conv_w_ptr, Bias_ptr, conv_out_ptr,
    B, H, S,
    stride_bx0, stride_bx1, stride_bx2,      # strides for Bx: (B, H, S)
    stride_w0, stride_w1, stride_w2,         # strides for conv_w: (H, 1, 4)
    stride_c0, stride_c1, stride_c2,         # strides for conv_out: (B, H, S)
    BLOCK_S: tl.constexpr,
):
    """
    Grouped causal 1D conv with kernel_size=4 and groups=H.
    Input Bx: (B, H, S), conv_weight: (H, 1, 4), conv_bias: (H)
    Output conv_out: (B, H, S)
    We compute conv_out[b, h, s] = sum_{k=0..3} conv_w[h, 0, k] * Bx[b, h, s - k - 1], with padding on the left.
    Accumulate in float32, store as output dtype handled by caller.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = pid_b
    h = pid_h

    # We process s positions in tiles
    for s_start in range(0, S, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        # Accumulator
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Load conv bias for this h
        bias_h = tl.load(Bias_ptr + h)
        acc += bias_h

        # Loop over kernel size K=4
        for k in range(4):
            # For causal: src index = s - k - 1; guard for s < k+1
            src = offs_s - k - 1
            in_bounds = (src >= 0) & mask_s
            # Pointers for Bx[b, h, src]
            bx_ptrs = Bx_ptr + (b * stride_bx0 + h * stride_bx1 + src * stride_bx2)
            bx = tl.load(bx_ptrs, mask=in_bounds, other=0.0)

            # conv_weight[h, 0, k] (note groups=H => each h has its own conv parameters)
            w_ptr = conv_w_ptr + (h * stride_w0 + 0 * stride_w1 + k * stride_w2)
            w = tl.load(w_ptr)  # scalar
            acc += bx * w

        # Store to conv_out[b, h, offs_s]
        co_ptrs = conv_out_ptr + (b * stride_c0 + h * stride_c1 + offs_s * stride_c2)
        tl.store(co_ptrs, acc, mask=mask_s)


@triton.jit
def _out_proj_linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B + Bias, A: (M, K), B: (K, N).
    Same kernel as _matmul_linear_kernel but used for the final out-proj.
    """
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


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H), dtype same as x
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Prepare inputs: A = x.view(M, H), B = in_proj_weight.T.view(H, 3H)
    A = x.contiguous().view(M, H)
    # in_proj_weight: (3H, H) => B: (H, 3H)
    B_w = in_proj_weight.t().contiguous().view(H, N)
    Bias = in_proj_bias.contiguous().view(N)

    # Output tensor
    C = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Launch Triton matmul kernel
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
        num_warps=4, num_stages=2,
    )
    return C.view(B, S, N)


def _triton_elementwise_mul_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply: c = a * b, 1D flattened.
    a, b: float32 tensors
    returns c: float32 tensor with same shape
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    total = a.numel()
    a_flat = a.contiguous().to(torch.float32).view(-1)
    b_flat = b.contiguous().to(torch.float32).view(-1)
    c_flat = torch.empty_like(a_flat, dtype=torch.float32, device=a.device)

    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _elementwise_mul_1d_kernel[grid](
        a_flat, b_flat, c_flat, total, BLOCK=BLOCK, num_warps=4, num_stages=1,
    )
    return c_flat.view_as(a)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor):
    """
    Triton grouped causal 1D conv: input Bx: (B, H, S) -> output conv_out: (B, H, S)
    conv_weight: (H, 1, 4), conv_bias: (H)
    Kernel size fixed at 4; groups=H. We treat conv_bias as per-out channel bias.
    Returns conv_out: (B, H, S), dtype same as Bx
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    # Ensure contiguous for predictable strides
    Bx = Bx.contiguous()
    conv_w = conv_weight.contiguous()  # (H, 1, 4)
    conv_bias = conv_bias.contiguous().to(torch.float32)  # bias in float32 for stability

    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    # Grid: one program per (b, h)
    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        Bx, conv_w, conv_bias, conv_out,
        B, H, S,
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        conv_w.stride(0), conv_w.stride(1), conv_w.stride(2),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        BLOCK_S=128, num_warps=4, num_stages=2,
    )
    # Cast to original dtype
    if conv_out.dtype != Bx.dtype:
        conv_out = conv_out.to(Bx.dtype)
    return conv_out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton version of out_proj: compute output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns: output: (B, S, H), dtype same as y
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    M = B * S
    K = H
    N = H

    y_flat = y.contiguous().view(M, K)
    out_w_t = out_proj_weight.t().contiguous().view(K, N)  # (K, N)
    Bias = out_proj_bias.contiguous().view(N)

    output = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _out_proj_linear_kernel[grid](
        y_flat, out_w_t, Bias, output,
        M, N, K,
        y_flat.stride(0), y_flat.stride(1),
        out_w_t.stride(0), out_w_t.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return output.view(B, S, N).to(y.dtype)


class ModelNew(nn.Module):
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
        """
        Full Triton implementation of the original run function.
        - in_proj: Triton matmul
        - elementwise gates: Triton elementwise kernels
        - grouped causal conv: Triton kernel
        - out-proj: Triton matmul
        """
        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), dtype=x.dtype

        B = BCx[:, :, :H]                 # (B, S, H)
        C = BCx[:, :, H:2 * H]            # (B, S, H)
        x_proj = BCx[:, :, 2 * H:]        # (B, S, H)

        # 2) Elementwise gate: Bx = B * x_proj
        Bx = _triton_elementwise_mul_1d(B, x_proj)  # (B, S, H), float32 (we'll cast later)

        # 3) Grouped causal conv with kernel_size=4 and groups=H. Conv weight derived from in_proj_weight's last 4 columns.
        # conv_weight in original is (H, 1, 4) derived from in_proj_weight[:, -4:]. We need that exact layout.
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx.to(torch.float32), conv_w, conv_bias.to(torch.float32))  # (B, H, S), float32

        # 4) Output gating: y = C * conv_out, align shapes: conv_out_T = (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H), float32
        y = _triton_elementwise_mul_1d(C.to(torch.float32), conv_out_T)  # (B, S, H), float32

        # 5) Final out-proj: y -> (B, S, H)
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # cast to y.dtype (which is same as x.dtype)

        return output


def run(*args):
    return ModelNew()(*args)
