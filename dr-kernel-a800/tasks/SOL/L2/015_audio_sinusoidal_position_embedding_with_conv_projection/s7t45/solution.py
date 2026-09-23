import math
import torch
import triton
import triton.language as tl


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def linear_matmul_kernel(
    x_ptr,  # *ptr to X [M, K], row-major contiguous
    w_ptr,  # *ptr to W [N, K], row-major contiguous (conv_out_weight)
    y_ptr,  # *ptr to Y [M, N], row-major contiguous (bf16)
    M, K, N,
    stride_x_row, stride_x_k,
    stride_w_n, stride_w_k,
    stride_y_row, stride_y_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # grid = (M, tiles of N)
    pid_m = tl.program_id(0)  # row index in X (M = B*S)
    pid_n = tl.program_id(1)  # tile along N

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        # Load X row slice: x[pid_m, kk] -> [BLOCK_K]
        x_ptrs = x_ptr + pid_m * stride_x_row + kk * stride_x_k
        x_vals = tl.load(x_ptrs, mask=(kk < K), other=0.0).to(tl.float32)

        # Load W block: w[n, kk] -> [BLOCK_N, BLOCK_K]
        w_ptrs = w_ptr + n[:, None] * stride_w_n + kk[None, :] * stride_w_k
        w_mask = (n[:, None] < N) & (kk[None, :] < K)
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # Accumulate: acc += sum_k w[n, k] * x[pid_m, k]
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Store Y[pid_m, n] in bfloat16
    y_ptrs = y_ptr + pid_m * stride_y_row + n * stride_y_n
    y_mask = n < N
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=y_mask)


@triton.jit
def scale_elementwise_kernel(y_ptr, scale, N_elems, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elems
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y = y * scale
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def add_pos_emb_kernel(y_ptr, pos_ptr, N_elems, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elems
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    pos = tl.load(pos_ptr + offsets, mask=mask, other=0.0)
    y = y + pos
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def forward(self, *args):
        # Arguments: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        input_features = args[0].to(self.device).to(torch.bfloat16)
        conv2d1_weight = args[1].to(self.device).to(torch.bfloat16)
        conv2d1_bias = args[2].to(self.device).to(torch.bfloat16)
        conv2d2_weight = args[3].to(self.device).to(torch.bfloat16)
        conv2d2_bias = args[4].to(self.device).to(torch.bfloat16)
        conv2d3_weight = args[5].to(self.device).to(torch.bfloat16)
        conv2d3_bias = args[6].to(self.device).to(torch.bfloat16)
        conv_out_weight = args[7].to(self.device)  # [N=1024, K=3840], use float32 for matmul
        positional_embedding = args[8].to(self.device).to(torch.bfloat16)
        embed_scale = float(args[9])

        # Ensure inputs are on device and bfloat16 for convs
        B, _, H, W = input_features.shape
        T = W  # original time dimension

        # conv1: [B, 1, 80, T] -> [B, 384, 40, T//2]
        x1 = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        # GELU (PyTorch) to match original behavior
        x1 = F.gelu(x1)

        # conv2: [B, 384, 40, T//2] -> [B, 384, 20, T//4]
        x2 = F.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x2 = F.gelu(x2)

        # conv3: [B, 384, 20, T//4] -> [B, 384, 10, T//8]
        x3 = F.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x3 = F.gelu(x3)

        # Reshape to (B, S, K) where S = time_after_conv = T//8, K = 384*10 = 3840
        S = x3.shape[3]  # time_after_conv
        C = x3.shape[1]  # 384
        F_ = 10
        x = x3.permute(0, 3, 1, 2).contiguous().view(B, S, C * F_)  # [B, S, K]

        # Linear projection: (B*S, K) @ (N, K)^T -> (B*S, N) in Triton
        B2 = B
        S_ = S
        K = C * F_  # 3840
        N = conv_out_weight.shape[0]  # 1024

        x_row = x.reshape(B2 * S_, K).contiguous()  # [M, K]
        # Output in bfloat16
        y = torch.empty((B2 * S_, N), device=self.device, dtype=torch.bfloat16)

        grid_linear = (B2 * S_, _ceil_div(N, 128))
        linear_matmul_kernel[grid_linear](
            x_row, conv_out_weight, y,
            B2 * S_, K, N,
            x_row.stride(0), x_row.stride(1),
            conv_out_weight.stride(0), conv_out_weight.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale
        y_flat = y.view(-1)
        N_elems = y_flat.numel()
        grid_scale = (_ceil_div(N_elems, 1024),)
        scale_elementwise_kernel[grid_scale](y_flat, embed_scale, N_elems, BLOCK_SIZE=1024, num_warps=4, num_stages=2)

        # Add positional embedding [S, N], broadcast over batch
        pos_emb = positional_embedding[:S, :].contiguous()  # [S, N], bfloat16
        grid_add = (_ceil_div(N_elems, 1024),)
        add_pos_emb_kernel[grid_add](y_flat, pos_emb.view(-1), N_elems, BLOCK_SIZE=1024, num_warps=4, num_stages=2)

        # Reshape back to (B, S, N)
        y_final = y.view(B, S, N)
        return y_final


def run(*args):
    return ModelNew()(*args)
