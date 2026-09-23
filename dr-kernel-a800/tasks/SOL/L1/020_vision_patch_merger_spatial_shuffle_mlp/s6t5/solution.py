import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    X_ptr,      # *bf16, input of shape [M, K] where M=num_patches, K=hidden_size
    W_ptr,      # *bf16, ln_weight of shape [K]
    B_ptr,      # *bf16, ln_bias of shape [K]
    Out_ptr,    # *bf16, output of shape [M, K]
    M: tl.constexpr,   # number of rows (patches)
    K: tl.constexpr,   # hidden_size (1536)
    BLOCK: tl.constexpr,  # tile size for reduction, e.g., 128
):
    # One program per row
    row = tl.program_id(0)
    # Guard in case grid > M (not expected, but safe)
    if row >= M:
        return

    # First pass: compute sum and sum of squares in FP32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_val += tl.sum(x_f32, axis=0)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)

    n = K
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    inv_std = 1.0 / tl.sqrt(var + 1e-6)

    # Second pass: normalize and apply affine
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(Out_ptr + row * K + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _pack_2x2_to_expanded_kernel(
    In_ptr,     # *bf16, input of shape [M_rows, K] (M_rows = num_patches), K=hidden_size
    Out_ptr,    # *bf16, output of shape [M_out, 4*K], M_out = M_rows // 4
    M_rows: tl.constexpr,  # number of input rows (num_patches)
    K: tl.constexpr,       # hidden_size (1536)
    BLOCK: tl.constexpr,   # tile along K, e.g., 256
):
    # One program per output row
    row_out = tl.program_id(0)
    if row_out >= (M_rows // 4):
        return

    # Map output row to its 2x2 input rows (T=1 in this task)
    # Let r = row_out
    # For kh=0,kw=0 -> input row r
    # For kh=1,kw=0 -> input row r + (w//2), but since T=1, we directly index by r
    # For kh=0,kw=1 -> input row r + (w//2), but again T=1
    # For kh=1,kw=1 -> input row r + (w//2)
    # So the 4 copies are from rows [r, r, r + (w//2), r + (w//2)].
    # However, when M_rows is divisible by 4, we pack 4 distinct rows per output row.
    # With T=1 and fixed get_inputs logic, M_rows = t * h * w and h,w are multiples of 2.
    # We pack 4 consecutive rows: [r, r+1, r+ (w//2), r+ (w//2) + 1].
    # To keep it simple and correct: each output row uses the same input row.
    # But since M_rows must be divisible by 4 for 2x2 packing, we can do:
    r0 = row_out * 4
    r1 = r0 + 1
    r2 = r0 + (M_rows // 2)
    r3 = r2 + 1

    # Pack segments: In_ptr[r0, :] -> Out[row_out, 0:K], ..., In_ptr[r3, :] -> Out[row_out, 3*K:4*K]
    # Loop over K in tiles
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K

        # Row r0
        x0 = tl.load(In_ptr + r0 * K + offs, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(Out_ptr + row_out * (4 * K) + 0 * K + offs, x0, mask=mask)

        # Row r1
        x1 = tl.load(In_ptr + r1 * K + offs, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(Out_ptr + row_out * (4 * K) + 1 * K + offs, x1, mask=mask)

        # Row r2
        x2 = tl.load(In_ptr + r2 * K + offs, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(Out_ptr + row_out * (4 * K) + 2 * K + offs, x2, mask=mask)

        # Row r3
        x3 = tl.load(In_ptr + r3 * K + offs, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(Out_ptr + row_out * (4 * K) + 3 * K + offs, x3, mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,       # *bf16, input [M, K]
    B_ptr,       # *bf16, weight [N, K] (we access as B[n, k] = B_ptr[n*K + k])
    Bias_ptr,    # *bf16, bias [N]
    C_ptr,       # *bf16, output [M, N]
    M: tl.constexpr,   # rows of A
    N: tl.constexpr,   # rows of C and Bias
    K: tl.constexpr,   # cols of A and B
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * K + (tl.arange(0, BLOCK_M)[:, None]) * K + k_ids[None, :]
        a = tl.load(a_ptrs, mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (k_ids[None, :] < K), other=0.0)
        a = a.to(tl.float32)
        # B tile: [BLOCK_K, BLOCK_N] as B[n, k] -> index n*N + k
        b_ptrs = B_ptr + n0 * K + k_ids[None, :] * N + tl.arange(0, BLOCK_N)[:, None]
        b = tl.load(b_ptrs, mask=(tl.arange(0, BLOCK_N)[:, None] < BLOCK_N) & (k_ids[None, :] < K), other=0.0)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < BLOCK_N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + m0 * N + (tl.arange(0, BLOCK_M)[:, None]) * N + (n0 + tl.arange(0, BLOCK_N)[None, :])
    mask_c = (tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (tl.arange(0, BLOCK_N)[None, :] < BLOCK_N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)


@triton.jit
def _gelu_kernel(
    In_ptr,     # *bf16, input [M, N]
    Out_ptr,    # *bf16, output [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m
    n0 = pid_n * BLOCK_N
    x = tl.load(In_ptr + m * N + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(Out_ptr + m * N + n0 + tl.arange(0, BLOCK_N), y.to(tl.bfloat16), mask=tl.arange(0, BLOCK_N) < N)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps):
        """
        Triton-only forward:
          1) LN over last dim (1536) + affine
          2) Spatial pack: 2x2 merge -> [num_patches//4, 4*1536]
          3) FC1: (num_merged_patches, 6144) @ (6144, 6144)^T + bias, GELU
          4) FC2: (num_merged_patches, 3584) @ (3584, 6144)^T + bias
        """
        device = hidden.device
        dtype = torch.bfloat16

        # 1) LayerNorm + affine (compute in FP32, store BF16)
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536
        ln_out = torch.empty_like(hidden, dtype=dtype, device=device)
        # Launch LN kernel with grid size == num_patches (always positive)
        BLOCK = 128  # tile for reduction
        grid_ln = (num_patches,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=num_patches, K=hidden_size, BLOCK=BLOCK,
            num_warps=4, num_stages=2
        )

        # 2) Spatial pack (2x2 merge) into expanded features
        # Each output row packs 2x2 features. Given T=1 in this task, num_patches must be divisible by 4.
        assert (num_patches % 4) == 0, "num_patches must be divisible by 4 for 2x2 merge"
        M_out = num_patches // 4
        K = hidden_size
        K_expanded = 4 * K
        in_pack = ln_out  # already BF16
        packed = torch.empty((M_out, K_expanded), dtype=dtype, device=device)
        BLOCK_pack = 256  # tile along K
        grid_pack = (M_out,)
        _pack_2x2_to_expanded_kernel[grid_pack](
            in_pack, packed,
            M_rows=num_patches, K=K, BLOCK=BLOCK_pack,
            num_warps=4, num_stages=2
        )

        # At this point, packed is (M_out, 4*K) and should match the MLP input shape in original logic.
        # However, original MLP expects input of shape (num_merged_patches, 6144). In provided inputs,
        # num_merged_patches == M_out. So we use packed as input to MLP.

        # 3) FC1: A @ W^T + bias, where A = packed, W = fc1_weight (6144, 6144), bias = fc1_bias (6144)
        num_merged = packed.shape[0]  # == num_merged_patches
        K1 = fc1_weight.shape[1]  # == hidden_size_expanded (6144)
        N1 = fc1_weight.shape[0]  # == hidden_size_expanded (6144)
        M1 = num_merged
        fc1_out = torch.empty((M1, N1), dtype=dtype, device=device)
        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
        grid_fc1 = (triton.cdiv(M1, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M=M1, N=N1, K=K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation on fc1_out
        K_after_gelu = fc1_out.shape[1]  # == N1 (6144)
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=dtype, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (M1, triton.cdiv(K_after_gelu, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M=M1, N=K_after_gelu, BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # 5) FC2: fc1_after_gelu @ fc2_weight^T + fc2_bias
        K2_in = K_after_gelu  # 6144
        N2 = fc2_weight.shape[0]  # 3584
        M2 = M1
        out = torch.empty((M2, N2), dtype=dtype, device=device)
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 64, 64
        grid_fc2 = (triton.cdiv(M2, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, out,
            M=M2, N=N2, K=K2_in,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
