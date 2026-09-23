import math
import torch
import triton
import triton.language as tl


# Triton LayerNorm: per-row normalization over H columns.
# Inputs: x (N, H) bfloat16; ln_weight (H) bfloat16; ln_bias (H) bfloat16
# Output: y (N, H) bfloat16
@triton.jit
def layer_norm_kernel(
    x_ptr,           # *ptr to input [N, H], bfloat16
    y_ptr,           # *ptr to output [N, H], bfloat16
    ln_weight_ptr,   # *ptr to ln_weight [H], bfloat16
    ln_bias_ptr,     # *ptr to ln_bias [H], bfloat16
    N,               # number of rows
    H: tl.constexpr, # hidden_size (1536)
    eps,             # epsilon as float32
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= N:
        return
    row_offset = row_id * H

    # Pass 1: compute sum
    sum_ = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)
    mean = sum_ / H

    # Pass 2: compute sum of squares
    sumsq = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    var = sumsq / H - mean * mean
    var = tl.maximum(var, 0.0)  # numerical safety
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: normalize and apply affine, store bfloat16
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        gamma = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * gamma + beta
        tl.store(y_ptr + row_offset + cols, y.to(tl.bfloat16), mask=mask)


# Triton Linear kernel: C[M, N] = A[M, K] @ W_T[K, N] + Bias[N]
# A: (M, K) bfloat16; W_T: (K, N) float32; Bias: (N) float32; C: (M, N) float32
@triton.jit
def linear_kernel(
    A_ptr,          # *ptr to A [M, K], bfloat16
    W_ptr,          # *ptr to W_T [K, N], float32
    Bias_ptr,       # *ptr to bias [N], float32
    C_ptr,          # *ptr to output [M, N], float32
    M,              # number of rows in A
    K,              # K dimension
    N,              # number of output columns
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K] as float32
        a_ptrs = A_ptr + m[:, None] * K + k[None, :]
        a_mask = (m[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load W_T tile [BLOCK_K, BLOCK_N] as float32
        w_ptrs = W_ptr + k[:, None] * N + n[None, :]
        w_mask = (k[:, None] < K) & (n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # already float32

        # Accumulate
        acc += tl.dot(a, w)

    # Add bias
    bias = tl.load(Bias_ptr + n, mask=(n < N), other=0.0)  # [BLOCK_N]
    acc += bias[None, :]

    # Store result
    c_ptrs = C_ptr + m[:, None] * N + n[None, :]
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton GELU kernel: y = 0.5 * x * (1 + erf(x / sqrt(2)))
@triton.jit
def gelu_kernel(
    x_ptr,  # *ptr input [M, N], float32
    y_ptr,  # *ptr output [M, N], float32
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (m[:, None] < M) & (n[None, :] < N)
    x = tl.load(x_ptr + m[:, None] * N + n[None, :], mask=mask, other=0.0)
    inv_sqrt2 = 0.7071067811865475  # 1 / sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(y_ptr + m[:, None] * N + n[None, :], y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 1536
        self.hidden_size_expanded = 6144
        self.out_hidden_size = 3584

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,   # (H, H) = (6144, 6144)
                fc1_bias: torch.Tensor,     # (H,) = (6144,)
                fc2_weight: torch.Tensor,   # (Out, H) = (3584, 6144)
                fc2_bias: torch.Tensor,     # (Out,) = (3584,)
                eps: float):
        # Ensure CUDA tensors
        assert hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, "All tensors must be CUDA."

        # Spatial shuffle: reconstruct per-grid patches (PyTorch data movement)
        patches = []
        offset = 0
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_patches_this = t * h * w
            patches.append(hidden[offset: offset + num_patches_this])
            offset += num_patches_this
        hidden = torch.cat(patches, dim=0)  # (num_merged_patches, 1536)

        # Step 1: LayerNorm (per row, H=1536) using Triton
        hidden_norm = torch.empty_like(hidden)  # bfloat16 output
        ln_weight_f32 = ln_weight.to(torch.float32)
        ln_bias_f32 = ln_bias.to(torch.float32)
        N = hidden.shape[0]
        H = self.hidden_size
        BLOCK_SIZE = 256  # reasonable tile
        grid_ln = (N,)
        layer_norm_kernel[grid_ln](
            hidden.to(torch.bfloat16), hidden_norm, ln_weight_f32, ln_bias_f32,
            N, H, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Step 2: First Linear (A: N x H, W_T: H x H) -> B: N x H, float32
        W_T1 = fc1_weight.transpose(0, 1).contiguous()  # (H, H), float32 by default
        M = hidden_norm.shape[0]
        K = H
        N2 = H
        B = torch.empty((M, N2), dtype=torch.float32, device=hidden_norm.device)

        # Tiling parameters
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N2, BLOCK_N))
        linear_kernel[grid1](
            hidden_norm.to(torch.bfloat16), W_T1.to(torch.float32), fc1_bias.to(torch.float32), B,
            M, K, N2,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Step 3: GELU activation using Triton (float32 input, float32 output)
        B_gelu = torch.empty_like(B, dtype=torch.float32)
        BLOCK_M2 = 128
        BLOCK_N2 = 64
        grid_gelu = (triton.cdiv(M, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        gelu_kernel[grid_gelu](
            B, B_gelu, M, N2,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
        )

        # Step 4: Second Linear (B_gelu: N x H, V_T: H x Out) -> C: N x Out, float32
        V_T2 = fc2_weight.transpose(0, 1).contiguous()  # (H, Out), float32
        Out = self.out_hidden_size
        C = torch.empty((M, Out), dtype=torch.float32, device=hidden_norm.device)

        BLOCK_M3 = 128
        BLOCK_N3 = 64
        BLOCK_K3 = 64
        grid2 = (triton.cdiv(M, BLOCK_M3), triton.cdiv(Out, BLOCK_N3))
        linear_kernel[grid2](
            B_gelu, V_T2.to(torch.float32), fc2_bias.to(torch.float32), C,
            M, K, Out,
            BLOCK_M=BLOCK_M3, BLOCK_N=BLOCK_N3, BLOCK_K=BLOCK_K3,
        )

        # Return in bfloat16 to match original behavior
        return C.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
