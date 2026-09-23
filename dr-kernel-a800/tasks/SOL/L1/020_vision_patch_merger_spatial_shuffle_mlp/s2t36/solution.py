import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_affine_kernel(
    x_ptr,            # *bf16, input [NUM_PATCHES, HIDDEN_SIZE]
    out_ptr,          # *bf16, output [NUM_PATCHES, HIDDEN_SIZE]
    ln_weight_ptr,    # *bf16, [HIDDEN_SIZE]
    ln_bias_ptr,      # *bf16, [HIDDEN_SIZE]
    HIDDEN_SIZE: tl.constexpr,
    NUM_PATCHES: tl.constexpr,
    eps,              # float32
):
    row = tl.program_id(0)
    if row >= NUM_PATCHES:
        return
    offs = tl.arange(0, HIDDEN_SIZE)
    # Load row
    x = tl.load(x_ptr + row * HIDDEN_SIZE + offs)
    x_fp32 = x.to(tl.float32)
    mean = tl.sum(x_fp32, axis=0) / HIDDEN_SIZE
    diff = x_fp32 - mean
    var = tl.sum(diff * diff, axis=0) / HIDDEN_SIZE
    inv_std = tl.math.rsqrt(var + eps)
    norm = diff * inv_std
    w = tl.load(ln_weight_ptr + offs).to(tl.float32)
    b = tl.load(ln_bias_ptr + offs).to(tl.float32)
    out_fp32 = norm * w + b
    tl.store(out_ptr + row * HIDDEN_SIZE + offs, out_fp32)


@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr, Bias_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D grid over tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * N + (m0 + tl.arange(0, BLOCK_M))[:, None] * N + k_range[None, :]
        a_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_range[:, None] * N + (n0 + tl.arange(0, BLOCK_N))[None, :]
        b_mask = (k0 + tl.arange(0, BLOCK_K))[:, None] < K
        b_mask = b_mask & (tl.arange(0, BLOCK_N)[None, :] < BLOCK_N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + (n0 + tl.arange(0, BLOCK_N)), mask=(n0 + tl.arange(0, BLOCK_N)) < N, other=0.0)
    acc = acc + bias[None, :]

    # Store C
    c_ptrs = C_ptr + (m0 + tl.arange(0, BLOCK_M))[:, None] * N + (n0 + tl.arange(0, BLOCK_N))[None, :]
    c_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M
    c_mask = c_mask & ((n0 + tl.arange(0, BLOCK_N))[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def gelu_kernel_fp32(
    inp_ptr, out_ptr,
    M, N
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m
    n0 = pid_n
    # 1D launch over M*N; each program processes one element
    idx = tl.program_id(0)
    if idx >= M * N:
        return
    row = idx // N
    col = idx % N
    x = tl.load(inp_ptr + row * N + col)
    # tanh-based GELU approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(t))
    tl.store(out_ptr + row * N + col, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, fc1_weight: torch.Tensor, fc1_bias: torch.Tensor, fc2_weight: torch.Tensor, fc2_bias: torch.Tensor, eps: float):
        # Ensure CUDA tensors (evaluation harness provides CUDA inputs)
        assert hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, "All tensors must be on CUDA for Triton."
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()

        num_patches, hidden_size = hidden.shape  # hidden_size = 1536

        # LayerNorm + affine (fp32 compute)
        hidden_norm_fp32 = torch.empty((num_patches, hidden_size), dtype=torch.float32, device=hidden.device)
        layernorm_affine_kernel[(num_patches,)](
            hidden, hidden_norm_fp32, ln_weight.to(torch.float32), ln_bias.to(torch.float32),
            HIDDEN_SIZE=hidden_size, NUM_PATCHES=num_patches, eps=float(eps),
            num_warps=4,
        )

        # fc1: (num_patches, 1536) @ (1536, 1536) -> (num_patches, 1536)
        # Note: Original code uses hidden_size_expanded=6144; however provided fc1_weight is [1536,1536]. We use given fc1_weight.
        # The output shape of fc1 is M x fc1_weight.shape[1] = num_patches x 1536.
        out1 = torch.empty((num_patches, fc1_weight.shape[1]), dtype=torch.float32, device=hidden.device)
        M = num_patches
        N1 = fc1_weight.shape[1]  # 1536
        K1 = fc1_weight.shape[0]  # 1536
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid_fc1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        gemm_bias_kernel[grid_fc1](
            hidden_norm_fp32, fc1_weight, out1, fc1_bias,
            M, N1, K1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # GELU activation in Triton (elementwise on fp32)
        out1_gelu = torch.empty_like(out1, dtype=torch.float32, device=out1.device)
        grid_gelu = (M, N1)
        gelu_kernel_fp32[grid_gelu](out1, out1_gelu, M, N1, num_warps=4)

        # fc2: (num_patches, 1536) @ (3584, 1536) -> (num_patches, 3584)
        out2 = torch.empty((num_patches, fc2_weight.shape[0]), dtype=torch.float32, device=hidden.device)
        M = num_patches
        N2 = fc2_weight.shape[0]  # 3584
        K2 = fc1_weight.shape[1]  # 1536
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 64, 32
        grid_fc2 = (triton.cdiv(M, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        gemm_bias_kernel[grid_fc2](
            out1_gelu, fc2_weight, out2, fc2_bias,
            M, N2, K2,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4,
        )

        return out2


def run(*args):
    return ModelNew()(*args)
