import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    hidden_ptr, ln_w_ptr, ln_b_ptr, out_ptr,
    M: tl.constexpr, K: tl.constexpr, eps: tl.float32,
    BLOCK: tl.constexpr,
):
    # One program per row (patch)
    row = tl.program_id(0)
    # Compute sum and sum of squares across hidden dimension in blocks
    sum_val = 0.0
    sum_sq = 0.0
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(hidden_ptr + row * K + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / K
    var = sum_sq / K - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(hidden_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        # Store as bfloat16
        tl.store(out_ptr + row * K + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _pack_2x2_rows_kernel(
    ln_ptr, packed_ptr,
    M: tl.constexpr, K: tl.constexpr, Kexp: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per output row r
    r = tl.program_id(0)
    # Write 4 segments of size K into packed[r, :]
    base = r * K
    # Segment 0: row=base, col=0
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        val = tl.load(ln_ptr + base + offs, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(packed_ptr + r * Kexp + offs, val, mask=mask)
    # Segment 1: row=base, col=1
    base_row1 = base + K
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        val = tl.load(ln_ptr + base_row1 + offs, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(packed_ptr + r * Kexp + K + offs, val, mask=mask)
    # Segment 2: row=base + 2*K, col=0
    base_row2 = base + 2 * K
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        val = tl.load(ln_ptr + base_row2 + offs, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(packed_ptr + r * Kexp + 2 * K + offs, val, mask=mask)
    # Segment 3: row=base + 2*K, col=1
    base_row3 = base_row2 + K
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        val = tl.load(ln_ptr + base_row3 + offs, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(packed_ptr + r * Kexp + 3 * K + offs, val, mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr, W_ptr, Bias_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: programs over tiles of output [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A_tile: (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * K) + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # W_tile as B[k, n] = W[n, k]: (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + (offs_n[None, :] * K) + offs_k[:, None]
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, w)

    # Add bias: broadcast over rows
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store as BF16
    c_ptrs = C_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _gelu_kernel(
    inp_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D launch over rows and column tiles
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_n < N) & (row < M)

    x = tl.load(inp_ptr + row * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    # tanh-based GELU approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    gelu = 0.5 * x * (1.0 + tl.tanh(c0 * (x + c1 * x3)))
    tl.store(out_ptr + row * N + offs_n, gelu.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps):
        device = hidden.device
        # Step 1: LayerNorm + affine in Triton
        M = hidden.shape[0]  # num_patches
        K = hidden.shape[1]  # hidden_size
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256
        _layer_norm_affine_kernel[(M,)](
            hidden, ln_weight, ln_bias, ln_out,
            M=M, K=K, eps=eps,
            BLOCK=BLOCK_ln, num_warps=4, num_stages=2
        )

        # Step 2: Pack rows into (M_out, 4*K), Triton
        M_out = M // 4  # invariant from inputs
        Kexp = 4 * K
        packed = torch.empty((M_out, Kexp), dtype=torch.bfloat16, device=device)
        BLOCK_pack = 256
        _pack_2x2_rows_kernel[(M_out,)](
            ln_out, packed,
            M=M_out, K=K, Kexp=Kexp,
            BLOCK=BLOCK_pack, num_warps=4, num_stages=2
        )

        # Step 3: fc1 GEMM + bias (6144->6144), Triton
        M_in = M_out  # rows
        K1 = packed.shape[1]  # 4*K = 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((M_in, N1), dtype=torch.bfloat16, device=device)

        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 256, 64
        grid_fc1 = (triton.cdiv(M_in, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M=M_in, N=N1, K=K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # Step 4: GELU activation (approx), Triton
        K_after = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (M_in, triton.cdiv(K_after, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M=M_in, N=K_after, BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # Step 5: fc2 GEMM + bias (6144->3584), Triton
        M_merged = M_in  # num_merged_patches (from axes)
        N2 = fc2_weight.shape[0]  # 3584
        K2 = fc1_after_gelu.shape[1]  # 6144
        fc2_out = torch.empty((M_merged, N2), dtype=torch.bfloat16, device=device)

        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 256, 64
        grid_fc2 = (triton.cdiv(M_merged, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, fc2_out,
            M=M_merged, N=N2, K=K2,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return fc2_out


def run(*args):
    return ModelNew()(*args)
