import math
import torch
import triton
import triton.language as tl


# Triton kernel: LayerNorm over last dimension + affine
# Inputs:
#   x_ptr: *bf16, shape (M, K)
#   w_ptr: *bf16, shape (K,)
#   b_ptr: *bf16, shape (K,)
#   y_ptr: *bf16, shape (M, K)
@triton.jit
def _layer_norm_affine_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                               M: tl.constexpr, K: tl.constexpr,
                               eps: tl.float32, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # First pass: sum and sum of squares in FP32
    sum_val = 0.0
    sum_sq = 0.0
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / K
    var = sum_sq / K - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize + affine and store
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        z = y * w + b
        # Store as BF16
        tl.store(y_ptr + row * K + offs, z.to(tl.bfloat16), mask=mask)


# Triton kernel: pack normalized hidden into expanded features
# Inputs:
#   x_norm_ptr: *bf16, shape (M, K), normalized hidden
#   out_ptr:    *bf16, shape (M_out, 4*K)
# M_out is a runtime arg passed as constexpr to keep grid positive
@triton.jit
def _pack_2x2_to_expanded_kernel(x_norm_ptr, out_ptr,
                                  M_out: tl.constexpr, K: tl.constexpr):
    row_out = tl.program_id(0)  # program over output rows
    M = M_out * 4  # original num_patches
    # Determine which 2x2 block this output row comes from
    patch_idx = row_out * 4
    # For T=1, each output row corresponds to a single 2x2 block, four positions
    for seg in range(4):
        in_row = patch_idx + seg
        offs = tl.arange(0, K)
        # load from x_norm[in_row, :]
        x = tl.load(x_norm_ptr + in_row * K + offs).to(tl.float32)
        # store into out[row_out, seg*K : (seg+1)*K]
        out_start = seg * K
        tl.store(out_ptr + row_out * (4 * K) + (out_start + offs), x.to(tl.bfloat16))


# Triton kernel: GEMM + bias (A[M, K] x W[K, N]^T + bias) -> C[M, N] (BF16 output)
@triton.jit
def _gemm_bias_kernel(A_ptr, W_ptr, BIAS_ptr, C_ptr,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)  # tile id over M
    pid_n = tl.program_id(1)  # tile id over N

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + rm[:, None] * K + rk[None, :]
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load W tile: we want W[k, n] for each k in BLOCK_K and n in BLOCK_N
        # W is stored as (K, N) row-major, so index k*stride_k + n*stride_n
        w_ptrs = W_ptr + rk[:, None] * N + rn[None, :]
        w_mask = (rk[:, None] < K) & (rn[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # acc += a @ w.T
        acc += tl.dot(a, tl.trans(w))

    # Add bias: BIAS[n]
    bias_ptrs = BIAS_ptr + rn
    bias = tl.load(bias_ptrs, mask=(rn < N), other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Write back to C in BF16
    c_ptrs = C_ptr + rm[:, None] * N + rn[None, :]
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# Triton kernel: GELU activation (tanh approximation)
@triton.jit
def _gelu_kernel(x_ptr, y_ptr,
                 M: tl.constexpr, N: tl.constexpr,
                 BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    # We iterate over columns in chunks of BLOCK_N
    for n_start in range(0, N, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        # tanh-based GELU approximation:
        # gelu(x) ~ 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c = 0.7978845608028654  # sqrt(2/pi)
        x3 = x * x * x
        inner = c * (x + 0.044715 * x3)
        y = 0.5 * x * (1.0 + tl.tanh(inner))
        tl.store(y_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants for Triton tiles
        self.BLOCK_ln = 256
        self.BLOCK_pack = 1024  # unused in kernel, but we pass K
        self.BLOCK_M = 128
        self.BLOCK_N = 128
        self.BLOCK_K = 64
        self.BLOCK_N_gelu = 256

    def forward(self,
                hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float):
        """
        hidden: (num_patches, 1536), bfloat16, device=GPU
        ln_weight, ln_bias: (1536,), bfloat16
        fc1_weight: (6144, 6144), bfloat16
        fc1_bias: (6144,), bfloat16
        fc2_weight: (3584, 6144), bfloat16
        fc2_bias: (3584,), bfloat16
        """
        assert hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and \
               fc1_weight.is_cuda and fc1_bias.is_cuda and \
               fc2_weight.is_cuda and fc2_bias.is_cuda, "All tensors must be CUDA tensors"

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536
        device = hidden.device

        # 1) LayerNorm + affine (Triton kernel)
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        grid_ln = (num_patches,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=num_patches, K=hidden_size, eps=1e-6, BLOCK=self.BLOCK_ln,
            num_warps=4, num_stages=2
        )

        # 2) Pack normalized hidden into expanded features: (num_patches//4, 4*1536)
        M_out = num_patches // 4
        assert M_out * 4 == num_patches, "num_patches must be divisible by 4 for packing"
        packed = torch.empty((M_out, 4 * hidden_size), dtype=torch.bfloat16, device=device)
        grid_pack = (M_out,)
        _pack_2x2_to_expanded_kernel[grid_pack](
            ln_out, packed,
            M_out=M_out, K=hidden_size,
            num_warps=4, num_stages=2
        )

        # 3) First linear layer (Triton GEMM + bias): (M_out, 6144)
        M_merged = M_out  # num_merged_patches (from axes)
        K1 = packed.shape[1]  # 4*hidden_size = 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((M_merged, N1), dtype=torch.bfloat16, device=device)
        grid_fc1 = (triton.cdiv(M_merged, self.BLOCK_M), triton.cdiv(N1, self.BLOCK_N))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M=M_merged, N=N1, K=K1,
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=4, num_stages=3
        )

        # 4) GELU activation (Triton kernel)
        K_after_gelu = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
        grid_gelu = (M_merged, triton.cdiv(K_after_gelu, self.BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M=M_merged, N=K_after_gelu,
            BLOCK_N=self.BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # 5) Second linear layer (Triton GEMM + bias): (M_merged, 3584)
        M_out2 = M_merged
        N2 = fc2_weight.shape[0]  # 3584
        output = torch.empty((M_out2, N2), dtype=torch.bfloat16, device=device)
        grid_fc2 = (triton.cdiv(M_out2, self.BLOCK_M), triton.cdiv(N2, self.BLOCK_N))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, output,
            M=M_out2, N=N2, K=K_after_gelu,
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=4, num_stages=3
        )

        return output


def run(*args):
    return ModelNew()(*args)
