import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_matmul_kernel(
    x_row_ptr, w_ptr, y_ptr,
    B, S, K, N,
    stride_x_row, stride_x_k,
    stride_w_n, stride_w_k,
    stride_y_row, stride_y_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute y_row = x_row @ w, where:
      x_row: [B*S, K], row-major with strides (stride_x_row, stride_x_k)
      w:     [N, K],    row-major with strides (stride_w_n, stride_w_k)
      y_row: [B*S, N], row-major with strides (stride_y_row=1, stride_y_n=1) (we expect y to be [B*S, N] contiguous)
    We accumulate in float32, write out as bfloat16.
    """
    # Each program instance handles one row (one [S, N] output slice)
    row_id = tl.program_id(0)  # 0 .. B*S - 1
    # n tile
    n_block = tl.program_id(1)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    # acc for this row across N-tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load x_row_chunk: shape [BLOCK_K]
        x_chunk = tl.load(
            x_row_ptr + row_id * stride_x_row + k_offsets * stride_x_k,
            mask=k_offsets < K,
            other=0.0
        )
        # Load w_chunk: shape [BLOCK_N, BLOCK_K]
        w_chunk = tl.load(
            w_ptr + n_offsets[:, None] * stride_w_n + k_offsets[None, :] * stride_w_k,
            mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
            other=0.0
        )
        # Accumulate: acc += sum over K-tile of x_chunk * w_chunk
        acc += tl.sum(w_chunk * x_chunk[None, :], axis=1)

    # Store acc to y as bfloat16
    tl.store(y_ptr + row_id * stride_y_row + n_offsets * stride_y_n, acc.to(tl.bfloat16), mask=n_offsets < N)


@triton.jit
def scale_elementwise_kernel(y_ptr, scale: tl.float32, N_elems: tl.int32):
    """
    Elementwise scale: y = y * scale, for y contiguous tensor.
    """
    offsets = tl.arange(0, 1024)
    # Unrolled loop over chunks of 1024 elements
    for base in range(0, N_elems, 1024):
        idx = base + offsets
        y = tl.load(y_ptr + idx, mask=idx < N_elems, other=0.0)
        y = y * scale
        tl.store(y_ptr + idx, y, mask=idx < N_elems)


@triton.jit
def add_pos_emb_kernel(y_ptr, pos_ptr, N_elems: tl.int32):
    """
    Add positional embedding to y: y = y + pos, where pos is [S, N] flattened.
    y is [B, S, N] flattened as row-major. We broadcast pos over batch.
    """
    offsets = tl.arange(0, 1024)
    for base in range(0, N_elems, 1024):
        idx = base + offsets
        y = tl.load(y_ptr + idx, mask=idx < N_elems, other=0.0)
        p = tl.load(pos_ptr + idx, mask=idx < N_elems, other=0.0)
        y = y + p
        tl.store(y_ptr + idx, y, mask=idx < N_elems)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args are: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale (float)
        # We rely on PyTorch for convs and GELU for correctness and speed, and Triton for GEMM and elementwise ops.

        # Extract inputs
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]  # [1024, 3840]
        positional_embedding = args[8]  # [max_source_positions, 1024], bfloat16
        embed_scale = float(args[9])  # float

        # Ensure bfloat16 and contiguous
        input_features = input_features.to(torch.bfloat16).contiguous()
        conv2d1_weight = conv2d1_weight.to(torch.bfloat16).contiguous()
        conv2d1_bias = conv2d1_bias.to(torch.bfloat16).contiguous()
        conv2d2_weight = conv2d2_weight.to(torch.bfloat16).contiguous()
        conv2d2_bias = conv2d2_bias.to(torch.bfloat16).contiguous()
        conv2d3_weight = conv2d3_weight.to(torch.bfloat16).contiguous()
        conv2d3_bias = conv2d3_bias.to(torch.bfloat16).contiguous()
        conv_out_weight = conv_out_weight.to(torch.bfloat16).contiguous()
        positional_embedding = positional_embedding.to(torch.bfloat16).contiguous()

        device = input_features.device

        # Conv 1: (1, 80, T) -> (384, 40, T//2)
        x1 = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x1 = F.gelu(x1)  # PyTorch GELU

        # Conv 2: (384, 40, T//2) -> (384, 20, T//4)
        x2 = F.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x2 = F.gelu(x2)

        # Conv 3: (384, 20, T//4) -> (384, 10, T//8)
        x3 = F.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x3 = F.gelu(x3)

        # Reshape to [B, S, K] where S = time_after_conv, K = 3840
        B, C, H, W = x3.shape  # C=384, H=10, W=T//8
        S = W  # time_after_conv
        K = C * H  # 384 * 10 = 3840
        x3_reshaped = x3.permute(0, 3, 1, 2).contiguous().view(B, S, K)

        # Linear projection via Triton GEMM: y_row = x_row @ conv_out_weight
        # conv_out_weight: [N=1024, K=3840]
        N = conv_out_weight.shape[0]  # 1024
        B2, S2, K2 = x3_reshaped.shape
        assert S2 == S and K2 == K, "Reshaped x must match conv_out_weight's K dimension"

        # Flatten rows for GEMM: [B*S, K]
        x_row = x3_reshaped.view(-1, K).contiguous()  # [B*S, K]
        w = conv_out_weight.contiguous()              # [N, K]
        y_flat = torch.empty((B2 * S2, N), device=device, dtype=torch.bfloat16)  # [B*S, N]

        BLOCK_N = 128
        BLOCK_K = 64
        grid = (B2 * S2, triton.cdiv(N, BLOCK_N))
        linear_matmul_kernel[grid](
            x_row, w, y_flat,
            B2, S2, K, N,
            x_row.stride(0), x_row.stride(1),
            w.stride(0), w.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape back to [B, S, N]
        y = y_flat.view(B2, S2, N)

        # Scale by embed_scale
        y_flat = y.view(-1)
        N_elems = y_flat.numel()
        grid_scale = (triton.cdiv(N_elems, 1024),)
        scale_elementwise_kernel[grid_scale](y_flat, embed_scale, N_elems=N_elems, num_warps=4, num_stages=2)

        # Add positional embedding [S, N], broadcast over batch
        pos_emb = positional_embedding[:S2, :].contiguous()  # [S, N]
        grid_add = (triton.cdiv(N_elems, 1024),)
        add_pos_emb_kernel[grid_add](y_flat, pos_emb.view(-1), N_elems=N_elems, num_warps=4, num_stages=2)

        return y


def run(*args):
    return ModelNew()(*args)
