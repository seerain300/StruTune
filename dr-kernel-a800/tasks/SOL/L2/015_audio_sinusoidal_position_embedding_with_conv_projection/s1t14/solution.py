import math
import torch
import triton
import triton.language as tl


# Triton GELU (exact, erf-based) kernel: 1D over flattened tensor
@triton.jit
def gelu_erf_kernel(X_flat_ptr, Y_flat_ptr, numel: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x = tl.load(X_flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(Y_flat_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# Triton kernel for final projection and add scaled positional embedding:
# Input x_flat: (B * t * K), weight_flat: (N * K), output y_flat: (B * t * N)
# We will launch grid over (B, t, tiles over N). Each program handles one (b, t) and a tile of N.
@triton.jit
def gemm_pos_add_kernel(
    X_flat_ptr,            # *bfloat16, shape (B*t*K)
    W_flat_ptr,            # *bfloat16, shape (N*K), we pass conv_out_weight.T flattened
    POS_ptr,               # *bfloat16, shape (1500, N)
    Y_ptr,                 # *bfloat16, shape (B*t*N)
    B, t, K, N,
    embed_scale: tl.float32,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    t_idx = tl.program_id(1)
    pid_n = tl.program_id(2)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Accumulator for this tile of output channels
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Base linear index for x[b, t, k] -> (b * t * K) + k_offsets
        base = b * t * K
        x_vec = tl.load(
            X_flat_ptr + base + k_offsets,
            mask=mask_k,
            other=0.0
        ).to(tl.float32)  # (BLOCK_K,)

        # Load corresponding weight slice W[n, k] -> flattened index n*K + k_offsets
        w_vec = tl.load(
            W_flat_ptr + n_offsets[:, None] * K + k_offsets[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float32)  # (BLOCK_N, BLOCK_K)

        # Accumulate outer product: (BLOCK_N, BLOCK_K) @ (BLOCK_K,) -> (BLOCK_N,)
        acc += tl.sum(w_vec * x_vec[None, :], axis=1)

    # Add scaled positional embedding: POS[t_idx, n_offsets]
    # POS shape is (1500, N), so we index row t_idx and columns n_offsets
    pos = tl.load(
        POS_ptr + t_idx * N + n_offsets,
        mask=mask_n,
        other=0.0
    ).to(tl.float32)
    acc = acc + embed_scale * pos  # broadcasting

    # Store to Y[b, t, n_offsets]
    base_y = b * t * N
    tl.store(Y_ptr + base_y + n_offsets, acc.to(tl.bfloat16), mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features: torch.Tensor,           # (B, 1, 80, T), bfloat16, CUDA
        conv2d1_weight: torch.Tensor,           # (384, 1, 3, 3), bfloat16
        conv2d1_bias: torch.Tensor,             # (384,), bfloat16
        conv2d2_weight: torch.Tensor,           # (384, 384, 3, 3), bfloat16
        conv2d2_bias: torch.Tensor,             # (384,), bfloat16
        conv2d3_weight: torch.Tensor,           # (384, 384, 3, 3), bfloat16
        conv2d3_bias: torch.Tensor,             # (384,), bfloat16
        conv_out_weight: torch.Tensor,          # (d_model, conv_out_dim) where d_model=1024, conv_out_dim=1024 in this setup
        positional_embedding: torch.Tensor,     # (1500, d_model), bfloat16
        embed_scale: float,
    ):
        # Ensure tensors are on CUDA and dtype is bfloat16
        assert input_features.is_cuda, "All tensors must be on CUDA device."
        assert input_features.dtype == torch.bfloat16, "Expected bfloat16 tensors."
        conv2d1_weight = conv2d1_weight.to(input_features.device)
        conv2d1_bias = conv2d1_bias.to(input_features.device)
        conv2d2_weight = conv2d2_weight.to(input_features.device)
        conv2d2_bias = conv2d2_bias.to(input_features.device)
        conv2d3_weight = conv2d3_weight.to(input_features.device)
        conv2d3_bias = conv2d3_bias.to(input_features.device)
        conv_out_weight = conv_out_weight.to(input_features.device)
        positional_embedding = positional_embedding.to(input_features.device)

        # Stage 1: conv1 + GELU
        x = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        # GELU conv1
        x_flat = x.reshape(-1)
        x_gelu = torch.empty_like(x_flat, dtype=torch.bfloat16, device=input_features.device)
        grid_gelu1 = (triton.cdiv(x_flat.numel(), 1024),)
        gelu_erf_kernel[grid_gelu1](x_flat, x_gelu, x_flat.numel(), BLOCK=1024)
        x = x_gelu.view_as(x)
        # Stage 2: conv2 + GELU
        x = torch.nn.functional.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x_flat = x.reshape(-1)
        x_gelu = torch.empty_like(x_flat, dtype=torch.bfloat16, device=input_features.device)
        grid_gelu2 = (triton.cdiv(x_flat.numel(), 1024),)
        gelu_erf_kernel[grid_gelu2](x_flat, x_gelu, x_flat.numel(), BLOCK=1024)
        x = x_gelu.view_as(x)
        # Stage 3: conv3 + GELU
        x = torch.nn.functional.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x_flat = x.reshape(-1)
        x_gelu = torch.empty_like(x_flat, dtype=torch.bfloat16, device=input_features.device)
        grid_gelu3 = (triton.cdiv(x_flat.numel(), 1024),)
        gelu_erf_kernel[grid_gelu3](x_flat, x_gelu, x_flat.numel(), BLOCK=1024)
        x = x_gelu.view_as(x)

        # Reshape: (B, 384, 40, t) -> (B, t, 384*40) where t = floor((T - 3)/2 + 1) after 3 convs
        B, Cout, H, T = x.shape  # after conv3, Cout=384, H=40
        t = (T - 3) // 2 + 1
        x = x.permute(0, 3, 1, 2).contiguous().view(B, t, Cout * H)

        # Final linear projection and add scaled positional embedding (Triton)
        K = x.shape[2]  # 15360 in the original setup, but we expect conv_out_dim=1024 here
        B_out, t_out, K_actual = x.shape
        N = conv_out_weight.shape[0]  # d_model = 1024
        # Flatten x for kernel
        x_flat = x.reshape(B_out * t_out * K_actual)

        # Prepare weight as (N*K_actual) flattened
        W_flat = conv_out_weight.transpose(0, 1).contiguous().view(N * K_actual)

        # Allocate output
        y_flat = torch.empty(B_out * t_out * N, device=input_features.device, dtype=torch.bfloat16)

        # Launch Triton GEMM + positional add
        grid = (B_out, t_out, triton.cdiv(N, 128))
        gemm_pos_add_kernel[grid](
            x_flat, W_flat, positional_embedding, y_flat,
            B_out, t_out, K_actual, N,
            embed_scale=embed_scale,
            BLOCK_N=128, BLOCK_K=1024,
            num_warps=4, num_stages=2
        )

        y = y_flat.view(B_out, t_out, N)

        return y


def run(*args):
    return ModelNew()(*args)
