import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_matmul_kernel(
    x_ptr,        # *bf16, [M] where M=B*S*K flattened row-major
    w_ptr,        # *bf16, [N, K] (conv_out_weight) — we reference as w_T[k, n] = w[n, k]
    y_ptr,        # *bf16, [M*N] flattened
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    stride_x_m, stride_x_k,  # here stride_x_m=K, stride_x_k=1 since x is contiguous in K dimension
    stride_w_n, stride_w_k,  # w strides (for [N,K])
    stride_y_m, stride_y_n,  # y strides (for [M,N])
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # which "row" m in [0, M)
    pid_n = tl.program_id(1)  # which tile of N columns

    offs_m = pid_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator for this row m over BLOCK_N columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # load x segment: x[offs_m, offs_k] -> vector [BLOCK_K]
        x_ptrs = x_ptr + offs_m * stride_x_m + offs_k * stride_x_k
        x_vals = tl.load(x_ptrs, mask=offs_k < K, other=0.0).to(tl.float32)

        # load w segment for each n in tile: w_T[offs_k, offs_n] -> [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + offs_n[None, :] * stride_w_n + offs_k[:, None] * stride_w_k
        mask_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_vals = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.float32)

        # accumulate: acc[offs_n] += sum_k w_vals[:, j] * x_vals[j]
        # we reduce along axis=0 (the BLOCK_K dimension)
        acc += tl.sum(w_vals * x_vals[None, :], axis=0)

    # store results to y row
    y_row_base = y_ptr + offs_m * stride_y_m
    y_ptrs = y_row_base + offs_n * stride_y_n
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=offs_n < N)


@triton.jit
def scale_elementwise_kernel(y_flat_ptr, scale, N_elems, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_elems
    y = tl.load(y_flat_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = y * scale
    tl.store(y_flat_ptr + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def add_pos_emb_kernel(y_flat_ptr, pos_flat_ptr, N_elems, S, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_elems
    # map linear index to (s, n): n = idx % N, s = idx // N
    n = offs % N
    s = offs // N
    y_ptrs = y_flat_ptr + offs
    pos_ptrs = pos_flat_ptr + s * N + n
    y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(pos_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = y + p
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        # Extract inputs: [B, 1, 80, T], followed by conv weights and biases, conv_out_weight, pos emb, embed_scale
        input_features = args[0]  # [B, 1, 80, T]
        conv2d1_weight = args[1]  # [C_out1, C_in, 3, 3] = [384, 1, 3, 3]
        conv2d1_bias = args[2]    # [384]
        conv2d2_weight = args[3]  # [384, 384, 3, 3]
        conv2d2_bias = args[4]    # [384]
        conv2d3_weight = args[5]  # [384, 384, 3, 3]
        conv2d3_bias = args[6]    # [384]
        conv_out_weight = args[7] # [N=1024, K=3840]
        positional_embedding = args[8] # [max_source_positions, 1024], dtype bfloat16
        embed_scale = args[9]     # float, sqrt(1024)=32

        device = input_features.device

        # Stage 1: Conv2d (1 -> 384 channels), stride=2, padding=1, GELU
        x1 = input_features.to(torch.bfloat16)
        y1 = F.conv2d(x1, conv2d1_weight.to(torch.bfloat16), conv2d1_bias.to(torch.bfloat16), stride=2, padding=1)
        y1 = F.gelu(y1, approximate='tanh')

        # Stage 2: Conv2d (384 -> 384 channels), stride=2, padding=1, GELU
        y2 = F.conv2d(y1, conv2d2_weight.to(torch.bfloat16), conv2d2_bias.to(torch.bfloat16), stride=2, padding=1)
        y2 = F.gelu(y2, approximate='tanh')

        # Stage 3: Conv2d (384 -> 384 channels), stride=2, padding=1, GELU
        y3 = F.conv2d(y2, conv2d3_weight.to(torch.bfloat16), conv2d3_bias.to(torch.bfloat16), stride=2, padding=1)
        y3 = F.gelu(y3, approximate='tanh')

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = y3.shape
        time_after_conv = t
        C = c
        freq = f
        K = C * freq  # conv_out_dim = 3840
        x_proj = y3.permute(0, 3, 1, 2).contiguous().view(b, time_after_conv, K)
        x_proj = x_proj.to(torch.bfloat16)

        # Triton Linear projection: y = x @ conv_out_weight, conv_out_weight is [N=1024, K]
        B, S, K = x_proj.shape
        N = conv_out_weight.shape[0]
        x_flat = x_proj.view(-1)                         # [B*S*K]
        w = conv_out_weight.contiguous()                # [N, K], bfloat16
        y_flat = torch.empty((B * S * N,), device=device, dtype=torch.bfloat16)

        grid_mm = (B * S, triton.cdiv(N, 128))
        linear_matmul_kernel[grid_mm](
            x_flat, w, y_flat,
            B * S, K, N,
            K, 1,  # stride_x_m=K, stride_x_k=1 for contiguous row-major x_flat
            w.stride(0), w.stride(1),  # strides for [N,K]
            B * S * N, N,               # strides for [M=N_total, N=N]
            BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2
        )
        y = y_flat.view(B, S, N)  # [B, S, N]

        # Triton scale: y *= embed_scale
        y_flat2 = y.view(-1)
        N_elems = y_flat2.numel()
        grid_scale = (triton.cdiv(N_elems, 1024),)
        scale_elementwise_kernel[grid_scale](y_flat2, float(embed_scale), N_elems, num_warps=4, num_stages=2)

        # Triton add positional embedding [S, N] broadcast over batch
        pos_emb = positional_embedding[:S, :].contiguous().to(torch.bfloat16)  # [S, N]
        N_elems_add = S * N
        grid_add = (triton.cdiv(N_elems_add, 1024),)
        add_pos_emb_kernel[grid_add](y_flat2, pos_emb.view(-1), N_elems_add, S, N, num_warps=4, num_stages=2)

        # Final reshape back to [B, S, N]
        y = y_flat2.view(B, S, N)
        return y


def run(*args):
    return ModelNew()(*args)
