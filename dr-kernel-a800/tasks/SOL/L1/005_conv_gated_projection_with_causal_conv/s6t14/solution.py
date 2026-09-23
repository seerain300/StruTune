import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton matmul for A[M, K] @ B[K, N] + bias, writing C[M, N]
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
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
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


# Triton elementwise multiplication over flattened 1D tensor
@triton.jit
def _elementwise_mul_1d(A_ptr, B_ptr, C_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(C_ptr + offs, c, mask=mask)


def _triton_elementwise_mul_1d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute elementwise A * B using Triton. A and B have same shape and dtype.
    Returns C with the same shape and dtype.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert A.shape == B.shape, "A and B must have the same shape"
    size = A.numel()
    C = torch.empty_like(A, dtype=torch.float32, device=A.device)  # compute in float32
    # For simplicity, ensure contiguous
    A_flat = A.contiguous().view(-1)
    B_flat = B.contiguous().view(-1)
    grid = (triton.cdiv(size, 1024),)
    _elementwise_mul_1d[grid](A_flat, B_flat, C.view(-1), size=size, BLOCK=1024)
    return C.view_as(A)


# Triton grouped causal conv1d with kernel_size=4, groups=H, padding handled via input indexing
# Input Bx: (B, H, S), conv_weight: (H, 4), conv_bias: (H), output conv_out: (B, H, S)
@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, conv_w_ptr, conv_bias_ptr, conv_out_ptr,
    B, H, S,
    stride_bx0, stride_bx1, stride_bx2,
    stride_w0, stride_w1,
    stride_co0, stride_co1, stride_co2,
    BLOCK_S: tl.constexpr,
):
    # one program per (b, h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    offs_s = tl.arange(0, BLOCK_S)
    s_total = B * H
    # Iterate over s from 0..S-1 in chunks of BLOCK_S
    # For each s, compute conv_out[b, h, s] = sum_{k=0..3} Bx[b, h, s+3-k] * conv_w[h, k] + conv_bias[h]
    for s0 in range(0, S, BLOCK_S):
        s = s0 + offs_s
        mask_s = s < S

        # Accumulator for this chunk of S
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Loop over kernel taps k=0..3
        for k in range(4):
            # idx = s + 3 - k
            idx = s + 3 - k
            mask_idx = (idx >= 0) & (idx < S) & mask_s

            bx_ptrs = Bx_ptr + b * stride_bx0 + h * stride_bx1 + idx * stride_bx2
            bx = tl.load(bx_ptrs, mask=mask_idx, other=0.0)  # (BLOCK_S,)

            w_ptrs = conv_w_ptr + h * stride_w0 + k * stride_w1
            w = tl.load(w_ptrs)  # scalar

            acc += bx * w

        bias = tl.load(conv_bias_ptr + h)  # scalar
        acc = acc + bias

        co_ptrs = conv_out_ptr + b * stride_co0 + h * stride_co1 + s * stride_co2
        tl.store(co_ptrs, acc, mask=mask_s)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_w: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of grouped causal conv1d with kernel_size=4 and groups=H.
    Bx: (B, H, S), conv_w: (H, 4), conv_bias: (H)
    Returns conv_out: (B, H, S), float32
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    # Ensure contiguous and float32
    Bx = Bx.contiguous().to(torch.float32)
    conv_w = conv_w.contiguous().to(torch.float32)  # (H, 4)
    conv_bias = conv_bias.contiguous().to(torch.float32)  # (H,)
    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    # Launch one program per (b, h)
    grid = (B * H,)
    _grouped_causal_conv1d_kernel[grid](
        Bx, conv_w, conv_bias, conv_out,
        B, H, S,
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        conv_w.stride(0), conv_w.stride(1),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        BLOCK_S=128,
    )
    return conv_out


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation of the fused pipeline:
        1) x -> BCx via in_proj (Triton matmul)
        2) split BCx into B, C, x_proj; gate: Bx = B * x_proj (Triton elementwise)
        3) grouped causal conv (kernel_size=4, groups=H), using conv_weight derived from in_proj_weight's last 4 columns
           (Triton kernel, no padding required; it indexes s+3-k safely).
        4) output gating: y = C * conv_out (Triton elementwise)
        5) final out-proj (Triton matmul)
        All computations are done via Triton kernels; no PyTorch functional ops in forward.
        """
        # Step 1: in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)        # (B, S, 3H)
        B, S, H = x.shape
        # Split into (B, S, H) along last dim
        B_val = BCx[:, :, :H]
        C_val = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:3 * H]

        # Step 2: elementwise gate
        Bx = _triton_elementwise_mul_1d(B_val, x_proj)                # (B, S, H)

        # Step 3: grouped causal conv (kernel_size=4, groups=H)
        # conv_weight from in_proj_weight's last 4 columns: (H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_bias_h = conv_bias.contiguous().to(torch.float32)          # (H,)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias_h)  # (B, H, S)

        # Step 4: output gating: y = C * conv_out (align conv_out to (B, S, H))
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_1d(C_val, conv_out_T)     # (B, S, H)

        # Step 5: final out-proj
        out = _triton_in_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return out


def run(*args):
    return ModelNew()(*args)
