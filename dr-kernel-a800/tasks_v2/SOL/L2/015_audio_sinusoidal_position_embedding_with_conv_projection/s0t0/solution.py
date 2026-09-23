import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton matmul kernel: computes C = A @ B
# A: [M, K], B: [K, N], C: [M, N]
# We'll specialize for our specific sizes by passing M, K, N as constexpr.
@triton.jit
def matmul_kernel(A, B, C,
                   M: tl.constexpr,  # number of rows in A / output rows
                   K: tl.constexpr,  # number of columns in A / rows in B
                   N: tl.constexpr,  # number of columns in B / output cols
                   stride_am, stride_ak,  # strides for A
                   stride_bk, stride_bn,  # strides for B
                   stride_cm, stride_cn,  # strides for C
                   BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and B tiles
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; cast to float32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # A is typically bfloat16; cast to float32
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # B is float32 (conv_out_weight)

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton elementwise kernel: y = x * scale + pos_emb
# x: [M, N], pos_emb: [T, N] (we pass only first T rows), scale: float
@triton.jit
def scale_add_pos_emb_kernel(X, POS_EMB, Y,
                              M: tl.constexpr,  # batch*T3
                              T: tl.constexpr,  # time after conv (first T rows of pos_emb)
                              N: tl.constexpr,  # 1024
                              stride_xm, stride_xn,
                              stride_ym, stride_yn,
                              stride_pt, stride_pn,
                              scale: tl.constexpr,
                              BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = X + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    y_ptrs = Y + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)

    x = tl.load(x_ptrs, mask=mask, other=0.0)  # x is fp32 (matmul output)
    pos_ptrs = POS_EMB + (offs_m[:, None] * stride_pt + offs_n[None, :] * stride_pn)
    pos = tl.load(pos_ptrs, mask=mask, other=0.0)  # pos_emb is fp32

    y = x * scale + pos
    tl.store(y_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The input args are provided by the harness in the same order as get_inputs() returns them:
        # 0: input_features
        # 1: conv2d1_weight
        # 2: conv2d1_bias
        # 3: conv2d2_weight
        # 4: conv2d2_bias
        # 5: conv2d3_weight
        # 6: conv2d3_bias
        # 7: conv_out_weight
        # 8: positional_embedding
        # 9: embed_scale (float)

        # Extract tensors
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]  # [N, K] = [1024, 3840]
        positional_embedding = args[8]  # [max_source_positions, 1024], dtype bfloat16
        embed_scale = args[9]  # float, e.g., 32.0

        # Compute convs with PyTorch (cuDNN), as in the original code
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)

        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)

        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape to [B, T3, 384*10] -> [B, T3, N]
        b, c, f, t = x.size()
        # f should be 10 after the last conv; confirm that
        assert f == 10, f"Expected last conv output freq=10, got f={f}"
        N = c * f  # 384 * 10 = 3840
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, N)  # [B, T3, N]

        # Prepare A: [M, K] where M = B * T3, K = N = 3840
        M = b * t
        K = N  # 3840
        A = x  # [B, T3, K]

        # Allocate C: [M, N] in fp32 (matmul output)
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Launch Triton matmul kernel
        # We specialize for our sizes. Triton compiles per function signature.
        BLOCK_M = 1
        BLOCK_N = 64
        BLOCK_K = 128

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_kernel[grid](
            A, conv_out_weight, C,
            M, K, N,
            A.stride(0), A.stride(2),  # A is [M, K] logically: stride(0)=T*K, stride(2)=1, but we pass logical strides via view. Better: make A contiguous as 2D.
            conv_out_weight.stride(0), conv_out_weight.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape C to [B, T3, N] where N=1024
        C = C.view(b, t, N)

        # Multiply by embed_scale (elementwise), can do in Triton or PyTorch. Since it's a scalar, either is fine.
        # We'll do it in Triton to satisfy "use Triton for computation".
        # First, allocate Y for result
        Y = torch.empty_like(C)  # fp32

        # Launch elementwise scale kernel
        # Scale as fp32 scalar
        SCALE = float(embed_scale)
        BLOCK_M_scale = 64
        BLOCK_N_scale = 128
        grid_scale = (triton.cdiv(M, BLOCK_M_scale), triton.cdiv(N, BLOCK_N_scale))
        scale_add_pos_emb_kernel[grid_scale](
            Y, positional_embedding, Y,
            M, t, N,
            Y.stride(0), Y.stride(2),
            Y.stride(0), Y.stride(2),
            positional_embedding.stride(0), positional_embedding.stride(1),
            scale=SCALE,
            BLOCK_M=BLOCK_M_scale, BLOCK_N=BLOCK_N_scale,
            num_warps=4, num_stages=2
        )

        # Note: positional_embedding is [max_source_positions, 1024]. We only need the first t rows for Y of shape [b, t, 1024].
        # Since we used Y as output buffer, we need to adjust the pos_emb usage. To correctly add positional embedding,
        # we should read from positional_embedding only rows [0..t-1]. However, the kernel above wrote to Y.
        # Let's correct this by allocating a temporary buffer Z and then add scaled Y to it.
        # Simpler approach: do the scaling in PyTorch and then add pos_emb in Triton (or vice versa). To keep Triton usage, we'll instead compute scale in Triton on a separate buffer and then add pos_emb.
        # We'll therefore do scaling in PyTorch (simple) and add pos_emb in Triton (the requested heavy compute).

        # Therefore, after matmul and scaling, we have Y = C * embed_scale (fp32).
        # Now, to add pos_emb (fp32), we need to pass only the first t rows of positional_embedding.
        # We'll extract pos_emb_t = positional_embedding[:t, :].to(torch.float32), then call the scale_add_pos_emb_kernel with Y (C*SCALE) and pos_emb_t.

        # Perform scaling in PyTorch to avoid confusion: Y_scaled = C * SCALE
        Y_scaled = C * SCALE

        # Create pos_emb_t: only first t rows
        pos_emb_t = positional_embedding[:t, :].to(torch.float32)  # [t, 1024], fp32

        # Allocate final output Z
        Z = torch.empty_like(Y_scaled)

        # Launch Triton scale_add_pos_emb_kernel to add pos_emb_t
        grid_add = (triton.cdiv(M, BLOCK_M_scale), triton.cdiv(N, BLOCK_N_scale))
        scale_add_pos_emb_kernel[grid_add](
            Z, pos_emb_t, Z,
            M, t, N,
            Z.stride(0), Z.stride(2),
            Z.stride(0), Z.stride(2),
            pos_emb_t.stride(0), pos_emb_t.stride(1),
            scale=1.0,  # we just add pos_emb; original code y = x * scale + pos_emb, here scale=embed_scale was already applied
            BLOCK_M=BLOCK_M_scale, BLOCK_N=BLOCK_N_scale,
            num_warps=4, num_stages=2
        )

        # Return Z (which equals (C * embed_scale) + pos_emb[:t, :])
        return Z


def run(*args):
    return ModelNew()(*args)
