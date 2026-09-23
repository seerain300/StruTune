import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton matmul kernel: compute C = A^T @ B, where
# A: [M, K], B: [K, N], C: [M, N]
# We will use: A = x.view(B*T, 3840), B = conv_out_weight (shape [3840, 1024])
@triton.jit
def matmul_trans_a_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K]
    stride_bk, stride_bn,   # B is [K, N]
    stride_cm, stride_cn,   # C is [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load A^T tile: A^T has rows indexed by k and cols by m
        a_ptrs = A_ptr + offs_k[:, None] * stride_am + offs_m[None, :] * stride_ak  # [BK, BM]
        # Load B tile: B has rows indexed by k and cols by n
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # [BK, BN]

        a = tl.load(a_ptrs, mask=(offs_k[:, None] < K) & (offs_m[None, :] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # acc += a @ b  => a: [BK, BM], b: [BK, BN] -> result [BM, BN]
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise kernel: scale and add positional embedding
# x_in: [M*N], pos_emb: [M*N], y_out: [M*N]
@triton.jit
def scale_add_kernel(x_ptr, pos_ptr, y_ptr, NUMEL: tl.int32, SCALE: tl.float32):
    pid = tl.program_id(0)
    offs = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offs < NUMEL
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    p = tl.load(pos_ptr + offs, mask=mask, other=0.0)
    y = x * SCALE + p
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        # Extract inputs: input_features, conv2d weights and biases, conv_out_weight, positional_embedding, embed_scale
        input_features = args[0]                      # [B, 1, 80, T_in], dtype bfloat16
        conv2d1_weight = args[1]                     # [OC=384, 1, 3, 3], bfloat16
        conv2d1_bias = args[2]                       # [384], bfloat16
        conv2d2_weight = args[3]                     # [384, 384, 3, 3], bfloat16
        conv2d2_bias = args[4]                       # [384], bfloat16
        conv2d3_weight = args[5]                     # [384, 384, 3, 3], bfloat16
        conv2d3_bias = args[6]                       # [384], bfloat16
        conv_out_weight = args[7]                    # [d_model=1024, conv_out_dim=3840], bfloat16
        positional_embedding = args[8]               # [max_source_positions, 1024], bfloat16
        embed_scale = args[9]                        # float

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape: [B, 384, 10, T3] -> [B, T3, 384*10] with T3 = time_after_conv
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()      # [B, T3, 384, 10]
        x = x.view(b, t, c * f)                     # [B, T3, 3840]

        # Linear projection: Triton matmul with conv_out_weight as [3840, 1024]
        # A = x.view(B*T3, 3840), B = conv_out_weight (3840, 1024)
        B_T = conv_out_weight
        A = x.reshape(b * t, 3840).to(torch.float32)  # cast to float32 for numerical stability
        M = A.shape[0]
        K = A.shape[1]
        N = B_T.shape[1]  # 1024

        # Allocate C
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Launch Triton matmul kernel
        grid = (triton.cdiv(M, 128), triton.cdiv(N, 64))
        matmul_trans_a_kernel[grid](
            A, B_T, C,
            M, N, K,
            A.stride(0), A.stride(1),           # A: [M, K], strides
            B_T.stride(0), B_T.stride(1),       # B_T: [K, N], strides
            C.stride(0), C.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # Scale by embed_scale in Triton
        numel = M * N
        scaled = torch.empty_like(C)
        grid_scale = (triton.cdiv(numel, 1024),)
        scale_add_kernel[grid_scale](
            C, C, scaled,  # use C as pos_ptr (we pass it as identity)
            NUMEL=numel, SCALE=float(embed_scale),
        )

        # Add positional embedding: only first T3 rows are needed
        # Convert to float32 for alignment with C (we computed in float32)
        T3 = t
        pos_emb = positional_embedding[:T3, :].to(torch.float32)  # [T3, 1024]
        pos_flat = pos_emb.reshape(-1).to(C.dtype)                # [T3*1024]
        # We need to match C's numel. If T3 * 1024 > M * N, we must crop; otherwise, pad or add zeros.
        # Since output after scaling has M*N = (B*T3)*1024, and our pos_flat length is T3*1024,
        # we add only up to min(numel, T3*1024). If T3*1024 < numel, we add zeros for the rest.
        pos_len = T3 * 1024
        if pos_len > numel:
            pos_flat = pos_flat[:numel]
        else:
            # pad pos_flat to numel
            pad = torch.zeros(numel - pos_len, dtype=C.dtype, device=C.device)
            pos_flat = torch.cat([pos_flat, pad])

        # Triton elementwise add kernel
        final = torch.empty_like(scaled)
        grid_add = (triton.cdiv(numel, 1024),)
        scale_add_kernel[grid_add](
            scaled, pos_flat, final, NUMEL=numel, SCALE=1.0,  # add pos_emb
        )

        # Reshape to [B, T3, 1024]
        out = final.view(b, T3, N)
        return out


def run(*args):
    return ModelNew()(*args)
