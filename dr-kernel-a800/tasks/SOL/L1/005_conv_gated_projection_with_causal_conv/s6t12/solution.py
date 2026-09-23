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


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton matmul for in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns: BCx: (B, S, 3H), float32
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    x_flat = x.contiguous().view(M, K).to(torch.float32)
    w_t = in_proj_weight.t().contiguous().view(K, N).to(torch.float32)
    bias = in_proj_bias.contiguous().view(N).to(torch.float32)

    out = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        x_flat, w_t, bias, out,
        M, N, K,
        x_flat.stride(0), x_flat.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out.view(B, S, N)


@triton.jit
def _elementwise_mul_1d(A_ptr, B_ptr, C_ptr, size: tl.int32):
    pid = tl.program_id(0)
    offs = pid * 256 + tl.arange(0, 256)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(C_ptr + offs, c, mask=mask)


@triton.jit
def _pad_left3_kernel(X_ptr, P_ptr, B: tl.int32, N: tl.int32, S: tl.int32, X_stride0: tl.int32, P_stride0: tl.int32, P_stride1: tl.int32, P_stride2: tl.int32):
    # Pad Bx to Bx_padded of shape (B, N, S+3): write zeros to first 3 columns, copy rest
    # X_ptr: (B, N, S), P_ptr: (B, N, S+3)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_s = tl.arange(0, 256)
    for s in range(0, S):
        x_ptrs = X_ptr + pid_b * X_stride0 + pid_n * N + s
        p_base = P_ptr + pid_b * P_stride0 + pid_n * P_stride1
        # left pad s < 3: write 0
        for pad_s in range(3):
            tl.store(p_base + (pad_s + s) * P_stride2, 0.0)
        # copy original
        tl.store(p_base + 3 * P_stride2 + s * P_stride2, tl.load(x_ptrs))
    # fill remaining pad positions if S == 0 (not applicable)


@triton.jit
def _grouped_causal_conv1d_kernel(Bx_pad_ptr, ConvW_ptr, Bias_ptr, ConvOut_ptr,
                                  B: tl.int32, N: tl.int32, S: tl.int32,
                                  Bx_pad_stride0: tl.int32, Bx_pad_stride1: tl.int32, Bx_pad_stride2: tl.int32,
                                  ConvW_stride0: tl.int32, ConvW_stride1: tl.int32,
                                  ConvOut_stride0: tl.int32, ConvOut_stride1: tl.int32, ConvOut_stride2: tl.int32):
    # Each program handles one (b, n) pair, computes conv_out[b, n, 0..S-1]
    b = tl.program_id(0)
    n = tl.program_id(1)
    offs_s = tl.arange(0, 256)  # tile over output sequence positions
    # Accumulator for s positions
    acc = tl.zeros((256,), dtype=tl.float32)
    # Loop over kernel positions k=0..3
    for k in range(4):
        w = tl.load(ConvW_ptr + n * ConvW_stride0 + k * ConvW_stride1).to(tl.float32)
        # For each s, read Bx_pad[b, n, s+3-k] if in range, else 0
        for s_off in range(256):
            s_i = s_off
            pos = s_i + 3 - k
            valid = pos >= 0 and pos < S
            x_val = tl.load(Bx_pad_ptr + b * Bx_pad_stride0 + n * Bx_pad_stride1 + pos * Bx_pad_stride2, mask=valid, other=0.0)
            acc[s_off] += x_val * w
    # Add bias
    bias_n = tl.load(Bias_ptr + n).to(tl.float32)
    acc += bias_n
    # Store conv_out[b, n, 0..S-1]
    conv_out_ptrs = ConvOut_ptr + b * ConvOut_stride0 + n * ConvOut_stride1 + offs_s * ConvOut_stride2
    store_mask = offs_s < S
    tl.store(conv_out_ptrs, acc, mask=store_mask)


@triton.jit
def _out_proj_linear_kernel(Y_ptr, W_ptr, Bias_ptr, Out_ptr,
                            M, K, N,
                            stride_y_m, stride_y_k,
                            stride_w_k, stride_w_n,
                            stride_out_m, stride_out_n,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        y_ptrs = Y_ptr + (offs_m[:, None] * stride_y_m + offs_k[None, :] * stride_y_k)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
        y_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        y = tl.load(y_ptrs, mask=y_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(y, w)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    out_ptrs = Out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def _triton_elementwise_mul_1d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Elementwise A * B, returns C with same shape and float32 dtype.
    A, B: any shape; compute via flat 1D Triton kernel.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    size = A.numel()
    C = torch.empty(size, dtype=torch.float32, device=A.device)
    grid = (triton.cdiv(size, 256),)
    _elementwise_mul_1d[grid](A.contiguous().to(torch.float32), B.contiguous().to(torch.float32), C, size)
    return C.view(A.shape)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation of the given fused pipeline:
        1) x -> BCx via in_proj (F.linear), in_proj_weight: (3H, H), bias: (3H)
        2) split BCx into B, C, x_proj; gate: Bx = B * x_proj
        3) grouped causal conv with kernel_size=4, groups=H, using conv_weight derived from in_proj_weight's last 4 columns
        4) output gating: y = C * conv_out (align conv_out to (B, S, H))
        5) final out-proj: y -> (B, S, H)
        All computations done via Triton kernels; no PyTorch functional ops in forward.
        """
        B, S, H = x.shape

        # Step 1: in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)        # (B, S, 3H), float32

        # Step 2: split into B, C, x_proj
        B_val = BCx[:, :, :H]                                         # (B, S, H)
        C_val = BCx[:, :, H:2*H]                                     # (B, S, H)
        x_proj = BCx[:, :, 2*H:3*H]                                  # (B, S, H)

        # Step 3: elementwise gate
        Bx = _triton_elementwise_mul_1d(B_val, x_proj)               # (B, S, H), float32

        # Step 4: grouped causal conv (kernel_size=4, groups=H)
        # Pad Bx with 3 zeros on the left: Bx_padded (B, H, S+3)
        Bx_padded = torch.empty((B, H, S + 3), dtype=torch.float32, device=x.device)
        # Launch Triton pad kernel
        grid_pad = (B, H)
        _pad_left3_kernel[grid_pad](
            Bx.contiguous().to(torch.float32),
            Bx_padded,
            B, H, S,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
        )

        # conv_weight derived from in_proj_weight's last 4 columns: (H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_b = conv_bias.contiguous().to(torch.float32)               # (H,)

        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)
        # Triton grouped causal conv kernel
        grid_conv = (B, H)
        _grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded,
            conv_w, conv_b, conv_out,
            B, H, S,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_w.stride(0), conv_w.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        )

        # Step 5: output gating: y = C * conv_out
        # To align shapes for elementwise, view conv_out as (B, S, H)
        conv_out_T = conv_out.transpose(1, 2).contiguous()             # (B, S, H)
        y = _triton_elementwise_mul_1d(C_val, conv_out_T)             # (B, S, H), float32

        # Step 6: final out-proj: y (B, S, H) -> output (B, S, H)
        # out_proj_weight: (H, H), out_proj_bias: (H,)
        M = B * S
        K = H
        N = H
        y_flat = y.contiguous().view(M, K).to(torch.float32)
        w_t = out_proj_weight.t().contiguous().view(K, N).to(torch.float32)
        bias = out_proj_bias.contiguous().to(torch.float32)

        out = torch.empty((M, N), dtype=torch.float32, device=x.device)
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid_out = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _out_proj_linear_kernel[grid_out](
            y_flat, w_t, bias, out,
            M, K, N,
            y_flat.stride(0), y_flat.stride(1),
            w_t.stride(0), w_t.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        output = out.view(B, S, N)

        return output


def run(*args):
    return ModelNew()(*args)
