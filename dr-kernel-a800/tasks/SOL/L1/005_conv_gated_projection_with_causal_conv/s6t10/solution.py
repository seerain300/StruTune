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
    Compute C = A @ B + Bias, where:
      - A: (M, K), row-major
      - B: (K, N), row-major (we pass in_proj_weight.T with shape (K, N))
      - Bias: (N,)
      - C: (M, N)
    We use float32 accumulation. Output C is float32.
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
        # cast to float32 for stable accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_1d_kernel(
    A_ptr, B_ptr, C_ptr,
    M,  # M = B * S
    stride_am, stride_bm, stride_cm,
    BLOCK: tl.constexpr,
):
    """
    Compute C = A * B for flattened 1D vectors of length M.
    A: (M,), B: (M,), C: (M,)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    a = tl.load(A_ptr + offs * stride_am, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs * stride_bm, mask=mask, other=0.0).to(tl.float32)
    c = a * b
    tl.store(C_ptr + offs * stride_cm, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    P_ptr,          # input padded tensor: (B, H, S+3), row-major over (H, S+3)
    Weight_ptr,     # conv weights: (H, 4), row-major
    Bias_ptr,       # conv bias: (H,)
    Out_ptr,        # output tensor: (B, H, S), row-major
    B, H, S,
    stride_bh, stride_bs, stride_bw, stride_bk,
    out_stride_bh, out_stride_bs,
    BLOCK_S: tl.constexpr,
):
    """
    Grouped causal conv with kernel_size=4, groups=H.
    For each (b, h), out[b, h, s] = sum_{k=0..3} P[b, h, s+3-k] * Weight[h, k] + Bias[h]
    P is padded with 3 zeros on the left along sequence (S+3).
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    # if b>=B or h>=H, skip (grid should be (B, H))
    if b >= B or h >= H:
        return

    # loop over sequence positions s=0..S-1
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S

        # initialize accumulator for this (b, h, offs)
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # accumulate over kernel taps k=0..3
        for k in range(4):
            idx = offs + 3 - k  # padding: idx in [3, S+2]
            # load P[b, h, idx]; for invalid idx, masked load uses 0
            p_ptrs = P_ptr + b * stride_bh + h * stride_bs + idx * stride_bw
            p = tl.load(p_ptrs, mask=mask, other=0.0).to(tl.float32)

            # load weight[h, k]
            w_ptrs = Weight_ptr + h * stride_bw + k * stride_bk
            w = tl.load(w_ptrs).to(tl.float32)

            acc += p * w

        # add bias[h]
        bias_val = tl.load(Bias_ptr + h).to(tl.float32)
        acc += bias_val

        # store to Out[b, h, s]
        out_ptrs = Out_ptr + b * out_stride_bh + h * out_stride_bs + offs
        tl.store(out_ptrs, acc, mask=mask)


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias, where:
      - x: (B, S, H), float32, contiguous
      - in_proj_weight: (3H, H), float32, contiguous
      - in_proj_bias: (3H), float32, contiguous
    Returns BCx: (B, S, 3H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # ensure float32 contiguous
    x_c = x.contiguous().to(torch.float32)  # A: (M, K)
    w_t = in_proj_weight.t().contiguous().to(torch.float32)  # (K, N)
    bias = in_proj_bias.contiguous().to(torch.float32)       # (N,)

    C = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        x_c.view(M, K), w_t.view(K, N), bias, C,
        M, N, K,
        x_c.stride(0), x_c.stride(1),
        w_t.stride(0), w_t.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C.view(B, S, N)


def _triton_elementwise_mul_1d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A * B elementwise, A, B: (B, S, H) float32.
    Returns C: (B, S, H) float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    Bsz, S, H = A.shape
    M = Bsz * S
    # flatten
    A_flat = A.contiguous().view(M)
    B_flat = B.contiguous().view(M)
    C_flat = torch.empty_like(A_flat, dtype=torch.float32, device=A.device)

    BLOCK = 1024
    grid = (triton.cdiv(M, BLOCK),)
    _elementwise_mul_1d_kernel[grid](
        A_flat, B_flat, C_flat,
        M,
        A_flat.stride(0), B_flat.stride(0), C_flat.stride(0),
        BLOCK=BLOCK,
    )
    return C_flat.view(Bsz, S, H)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Grouped causal 1D conv with kernel_size=4 and groups=H.
    - Bx: (B, H, S), float32 (sequence last dim), contiguous
    - conv_weight: (H, 4), float32, contiguous (derived from in_proj_weight[:, -4:])
    - conv_bias: (H,), float32, contiguous
    Returns conv_out: (B, H, S), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    # Create padded input: add 3 zeros on the left along sequence
    # Padded along the last dim only, keeping (H) group intact.
    P = torch.nn.functional.pad(Bx, (3, 0))  # (B, H, S+3), float32
    # conv_out: (B, H, S)
    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    BLOCK_S = 128
    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        P.contiguous().to(torch.float32),  # (B, H, S+3)
        conv_weight.contiguous().to(torch.float32),  # (H, 4)
        conv_bias.contiguous().to(torch.float32),     # (H,)
        conv_out,
        B, H, S,
        P.stride(0), P.stride(1), P.stride(2),  # (stride_b, stride_h, stride_s)
        4, 1,                      # conv_weight strides: (stride_bw, stride_bk) but we pass as (H, 4) contiguous => (1, 4)
        conv_out.stride(0), conv_out.stride(1),  # (stride_bh, stride_bs)
        BLOCK_S=BLOCK_S,
    )
    return conv_out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute output = y @ out_proj_weight^T + out_proj_bias, where:
      - y: (B, S, H), float32
      - out_proj_weight: (H, H), float32
      - out_proj_bias: (H,), float32
    Returns output: (B, S, H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    K = H
    N = H
    M = B * S

    y_flat = y.contiguous().to(torch.float32).view(M, K)              # (M, K)
    w_t = out_proj_weight.t().contiguous().to(torch.float32).view(K, N)  # (K, N)
    bias = out_proj_bias.contiguous().to(torch.float32)               # (N,)

    out = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        y_flat, w_t, bias, out,
        M, N, K,
        y_flat.stride(0), y_flat.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out.view(B, S, N)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation of the given fused pipeline:
        1) x -> BCx via in_proj
        2) split BCx into B, C, x_proj; gate: Bx = B * x_proj
        3) grouped causal conv on Bx with kernel_size=4 (groups=H), using conv_weight derived from in_proj_weight's last 4 columns
        4) output gating: y = C * conv_out (align conv_out to (B, S, H))
        5) final out-proj: y -> (B, S, H)
        All computations done via Triton kernels; no PyTorch functional ops in forward.
        """
        # Step 1: in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)        # (B, S, 3H)
        B, S, H = x.shape
        # Split into (B, S, H) along last dim
        B_val = BCx[:, :, :H]
        C_val = BCx[:, :, H:2*H]
        x_proj = BCx[:, :, 2*H:3*H]

        # Step 2: elementwise gate
        Bx = _triton_elementwise_mul_1d(B_val, x_proj)                # (B, S, H)

        # Step 3: grouped causal conv
        # Derive conv_weight from in_proj_weight's last 4 columns
        # Original conv_weight is (H, 1, 4), but we only need (H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias) # (B, H, S)

        # Step 4: output gating: y = C * conv_out (align conv_out to (B, S, H))
        conv_out_T = conv_out.transpose(-1, -2).contiguous()           # (B, S, H)
        y = _triton_elementwise_mul_1d(C_val, conv_out_T)              # (B, S, H)

        # Step 5: final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)   # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
