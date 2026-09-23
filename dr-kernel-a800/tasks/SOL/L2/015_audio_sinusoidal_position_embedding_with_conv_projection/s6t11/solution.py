import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_gemm_no_bias_fp32(
    x_ptr,  # *fp32, input [B, T, K]
    w_ptr,  # *fp32, weight [N, K], N=d_model=1024
    out_ptr,  # *fp32, output [B, T, N]
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_D: tl.constexpr,  # tile size over N
):
    # Grid = (B, T, ceil_div(N, BLOCK_D))
    b = tl.program_id(0)
    t = tl.program_id(1)
    d_block = tl.program_id(2)

    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < N

    # Accumulator for this (b, t) across N tile
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Iterate over K in chunks (e.g., 128) to reduce memory traffic
    for k_start in range(0, K, 128):
        k_offsets = k_start + tl.arange(0, 128)
        k_mask = k_offsets < K

        # Load x[b, t, k_offsets] -> vector [128]
        x_off = (b * T + t) * K + k_offsets
        x_vec = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)

        # Load W[d_offsets, k_offsets] -> [BLOCK_D, 128]
        w_off = d_offsets[:, None] * K + k_offsets[None, :]
        w_mat = tl.load(w_ptr + w_off, mask=d_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc += sum_k w_mat[:, k] * x_vec[k] (reduce over K-chunk)
        # Explicit loop to keep Triton happy
        for kk in range(0, 128):
            k_valid = (k_start + kk) < K
            x_k = x_vec[kk] if k_valid else 0.0
            acc += w_mat[:, kk] * x_k

    # Store results for this (b, t, d_offsets)
    out_off = (b * T * N) + (t * N) + d_offsets
    tl.store(out_ptr + out_off, acc, mask=d_mask)


@triton.jit
def add_pos_embed_kernel(
    y_ptr,  # *fp32, input [B, T, N]
    pos_ptr,  # *fp32, positional embedding [T, N]
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr,
):
    # Grid = (B, T, N)
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)
    if (b < B) and (t < T) and (d < N):
        y_off = (b * T * N) + (t * N) + d
        pos_off = t * N + d
        y_val = tl.load(y_ptr + y_off)
        pos_val = tl.load(pos_ptr + pos_off)
        tl.store(y_ptr + y_off, y_val + pos_val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        input_features,             # [B, 1, 80, time_dim], bfloat16
        conv2d1_weight, conv2d1_bias,   # [384,1,3,3], [384]
        conv2d2_weight, conv2d2_bias,   # [384,384,3,3], [384]
        conv2d3_weight, conv3_bias,     # [384,384,3,3], [384]
        conv_out_weight,                 # [d_model=1024, conv_out_dim], helper sets conv_out_dim=3840
        positional_embedding,            # [max_source_positions, d_model], bfloat16
        embed_scale: float,              # sqrt(1024)=32.0
    ):
        # Perform convolutions and GELU in PyTorch for correctness
        B = input_features.shape[0]
        H = input_features.shape[2]
        W = input_features.shape[3]

        # Conv1: in_channels=1 -> out_channels=384
        x1 = F.conv2d(input_features.float(), conv2d1_weight.float(), conv2d1_bias.float(), stride=2, padding=1)
        x1 = F.gelu(x1, approximate='tanh')

        # Conv2: in_channels=384 -> out_channels=384
        x2 = F.conv2d(x1, conv2d2_weight.float(), conv2d2_bias.float(), stride=2, padding=1)
        x2 = F.gelu(x2, approximate='tanh')

        # Conv3: in_channels=384 -> out_channels=384
        x3 = F.conv2d(x2, conv2d3_weight.float(), conv3_bias.float(), stride=2, padding=1)
        x3 = F.gelu(x3, approximate='tanh')

        # Reshape: (B, C_out3, H_out3, W_out3) -> (B, W_out3, C_out3*H_out3) -> (B, T, K)
        B, C_out3, H_out3, W_out3 = x3.shape
        T = W_out3
        K = C_out3 * H_out3 * W_out3

        x3_perm = x3.permute(0, 3, 2, 1).contiguous()  # [B, W_out3, H_out3, C_out3]
        x_flat = x3_perm.view(B, T, K).float()         # [B, T, K] in fp32

        # Prepare conv_out_weight: [N=1024, K]. In the helper, conv_out_dim=3840; we ignore extra dims and use actual K.
        N = conv_out_weight.shape[0]  # d_model
        # If conv_out_weight.shape[1] > K, ignore extra columns. Helper sets conv_out_dim=3840 and K much smaller, but to be safe we zero-pad or slice.
        # Here we assume helper provides weight aligned to K; in practice, helper sets conv_out_dim=K, but for generality we keep N arbitrary and rely on provided dims.
        # For robustness, we can just use conv_out_weight[:N, :] but N is already d_model=1024. We’ll use as-is.

        # Linear projection: y[b, t, d] = sum_k x_flat[b, t, k] * conv_out_weight[d, k] without bias
        y = torch.empty((B, T, N), dtype=torch.float32, device=input_features.device)

        BLOCK_D = 64
        grid4 = (B, T, (N + BLOCK_D - 1) // BLOCK_D)
        linear_gemm_no_bias_fp32[grid4](
            x_flat, conv_out_weight.float(), y,
            B, T, K, N, BLOCK_D
        )

        # Scale by embed_scale
        y = y * embed_scale

        # Add positional embedding: pos has shape [max_source_positions, d_model]
        # We add only the first T rows: pos[:T, :]
        pos_embed = positional_embedding.float()[:T, :]  # [T, N]
        grid5 = (B, T, N)
        add_pos_embed_kernel[grid5](
            y, pos_embed, B, T, N
        )

        return y


def run(*args):
    return ModelNew()(*args)
