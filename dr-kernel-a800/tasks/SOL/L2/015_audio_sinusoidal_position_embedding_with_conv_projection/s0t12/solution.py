import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton matmul kernel: A[M, K] x B[K, N] -> C[M, N]
# We will use it to compute the linear projection: x.view(B*T3, 3840) @ conv_out_weight.T
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
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
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise kernel: Y = X * SCALE
@triton.jit
def scale_kernel(
    X_ptr, Y_ptr, NUMEL,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * SCALE + tl.arange(0, SCALE)
    mask = offs < NUMEL
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = x * SCALE
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton elementwise kernel: Y = X + POS where POS has shape [N_COLS] (here N_COLS=1024)
@triton.jit
def add_pos_emb_kernel(
    X_ptr, POS_ptr, Y_ptr, NUMEL,
    N_COLS: tl.constexpr,
):
    # NUMEL = B * T (rows) * N_COLS (cols)
    pid_row = tl.program_id(0)  # over rows B * T
    col = tl.program_id(1)      # over columns 0..N_COLS-1

    base = pid_row * N_COLS + col
    mask = base < NUMEL

    x = tl.load(X_ptr + base, mask=mask, other=0.0)
    pos = tl.load(POS_ptr + col, mask=col < N_COLS, other=0.0)  # pos is [N_COLS], broadcast
    y = x + pos
    tl.store(Y_ptr + base, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        x = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = torch.nn.functional.gelu(x)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x = torch.nn.functional.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = torch.nn.functional.gelu(x)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x = torch.nn.functional.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = torch.nn.functional.gelu(x)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Prepare inputs for Triton matmul: [B*T, 3840] x [3840, 1024]
        x_view = x.to(torch.float32)  # compute in fp32
        B_T = b * t
        K = c * f  # 3840
        N = 1024

        A = x_view.reshape(B_T, K)  # [B*T, K], fp32
        # conv_out_weight shape is [1024, 3840] (output_features x in_features); we need [K, N] = [3840, 1024]
        BT = conv_out_weight.transpose(0, 1).contiguous()  # [3840, 1024], fp32
        C = torch.empty((B_T, N), device=A.device, dtype=torch.float32)

        # Launch Triton matmul
        if TRITON_AVAILABLE:
            grid_mm = (triton.cdiv(B_T, 128), triton.cdiv(N, 64))
            matmul_kernel[grid_mm](
                A, BT, C,
                B_T, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
            )
        else:
            # Fallback: use torch matmul if Triton not available (still in fp32)
            C = A @ BT

        # Scale by embed_scale: Triton elementwise
        Y = torch.empty_like(C)
        if TRITON_AVAILABLE:
            numel = B_T * N
            grid_scale = (triton.cdiv(numel, 1024),)
            scale_kernel[grid_scale](C, Y, numel, SCALE=embed_scale)
        else:
            Y = C * embed_scale

        # Reshape to [B, T3, 1024]
        out = Y.view(b, t, N)

        # Add positional embedding: [max_source_positions, 1024], use only first T rows
        # Create pos slice and add via Triton
        # Note: positional_embedding from get_inputs is [max_source_positions, 1024], dtype matches input_features
        # Here we use float32 for addition, as output is fp32
        pos_emb = positional_embedding[:t, :].to(torch.float32)
        # Make sure pos_emb is contiguous and has shape [T, 1024]
        pos_emb = pos_emb.contiguous()  # [T, 1024]
        # Now Y has shape [B, T, 1024]; flatten rows B*T
        Y_flat = Y.view(B_T, N)
        Y_out_flat = torch.empty_like(Y_flat)
        if TRITON_AVAILABLE:
            numel_add = B_T * N
            grid_add = (B_T, N)
            add_pos_emb_kernel[grid_add](Y_flat, pos_emb.view(-1), Y_out_flat, numel_add, N_COLS=N)
        else:
            Y_out_flat = Y_flat + pos_emb

        out = Y_out_flat.view(b, t, N)
        return out


def run(*args):
    return ModelNew()(*args)
