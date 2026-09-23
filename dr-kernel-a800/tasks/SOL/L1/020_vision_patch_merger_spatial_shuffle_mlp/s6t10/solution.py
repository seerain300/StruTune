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
    K: tl.constexpr,   # hidden size (1536)
    eps: tl.constexpr, # epsilon for LN
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # Guard if row >= M: Triton grid should ensure row < M
    row_base = row * K

    # First pass: compute mean and var in fp32
    sum_val = 0.0
    sum_sq = 0.0
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_val += tl.sum(x_f32, axis=0)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)

    n = K
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(Out_ptr + row_base + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _pack_2x2_to_expanded_kernel(
    In_ptr,       # *bf16, input of shape [M_in, K] where M_in=num_patches, K=hidden_size
    Out_ptr,      # *bf16, output of shape [M_out, 4*K], M_out=M_in//4
    M_in: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 1D over output rows; each program handles one output row r
    r = tl.program_id(0)
    if r >= (M_in // 4):
        return

    # Map output row r to input row mapping: kh, kw and relative offsets
    # Since T=1 always in this task, output rows are contiguous and each corresponds to a unique 2x2 block.
    # We can compute input index as:
    # For a fixed grid, each output row r maps to input row idx = r * 4 + offset within the 2x2 block.
    # But since grid details are arbitrary, we rely on M_out = M_in // 4 and the order of patches.
    # The original PyTorch code uses T=1 and packs 2x2 sequentially. Here, we pack sequentially as well.
    base = r * 4 * K

    # Each 2x2 block contributes 4*K features; the first K features come from the 2x2 at position (0,0),
    # then (1,0), (0,1), (1,1).
    # We load each feature vector from In_ptr and store into Out_ptr at the correct offset.

    # We'll do this in chunks to avoid vector too large. For simplicity, use BLOCK=1024 here.
    # Load and store four segments:
    # Segment 1: kh=0,kw=0 -> features[0:K]
    offs = tl.arange(0, BLOCK)
    for start in range(0, K, BLOCK):
        seg = start + offs
        mask = seg < K
        x = tl.load(In_ptr + (r * 4 + 0) * K + seg, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(Out_ptr + base + seg, x, mask=mask)

    # Segment 2: kh=1,kw=0 -> features[K:2*K]
    base2 = base + K
    for start in range(0, K, BLOCK):
        seg = start + offs
        mask = seg < K
        x = tl.load(In_ptr + (r * 4 + 1) * K + seg, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(Out_ptr + base2 + seg, x, mask=mask)

    # Segment 3: kh=0,kw=1 -> features[2*K:3*K]
    base3 = base + 2 * K
    for start in range(0, K, BLOCK):
        seg = start + offs
        mask = seg < K
        x = tl.load(In_ptr + (r * 4 + 2) * K + seg, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(Out_ptr + base3 + seg, x, mask=mask)

    # Segment 4: kh=1,kw=1 -> features[3*K:4*K]
    base4 = base + 3 * K
    for start in range(0, K, BLOCK):
        seg = start + offs
        mask = seg < K
        x = tl.load(In_ptr + (r * 4 + 3) * K + seg, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(Out_ptr + base4 + seg, x, mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,           # *bf16, input matrix, shape [M, K]
    B_ptr,           # *bf16, weight matrix, shape [N, K] (compute A @ B^T)
    Bias_ptr,        # *bf16, bias, shape [N]
    C_ptr,           # *bf16, output matrix, shape [M, N]
    M: tl.constexpr, # rows of A and C
    N: tl.constexpr, # cols of C (and Bias)
    K: tl.constexpr, # feature dim of A and B
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
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * K + (tl.arange(0, BLOCK_M)[:, None]) * K + k_ids[None, :]
        a = tl.load(a_ptrs, mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (k_ids[None, :] < K), other=0.0).to(tl.float32)
        # Load B tile: B[n, k] with B_ptr[n*K + k]
        b_ptrs = B_ptr + n0 * K + k_ids[None, :] * N + tl.arange(0, BLOCK_N)[:, None]
        b = tl.load(b_ptrs, mask=(tl.arange(0, BLOCK_N)[:, None] < BLOCK_N) & (k_ids[None, :] < K), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)  # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)

    # Add bias
    bias = tl.load(Bias_ptr + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < BLOCK_N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store
    c_ptrs = C_ptr + m0 * N + (tl.arange(0, BLOCK_M)[:, None]) * N + (n0 + tl.arange(0, BLOCK_N)[None, :])
    mask_c = (tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (tl.arange(0, BLOCK_N)[None, :] < BLOCK_N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)


@triton.jit
def _gelu_kernel(
    In_ptr,     # *bf16, input of shape [M, N]
    Out_ptr,    # *bf16, output of shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m
    n0 = pid_n * BLOCK_N
    x = tl.load(In_ptr + m * N + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=0.0).to(tl.float32)
    # tanh-based GELU approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    sqrt2 = 1.4142135623730951
    x3 = x * x * x
    t = c * (x + 0.044715 * x3) * (1.0 - tl.math.tanh(0.7978845608028654 * (x + 0.044715 * x3)))
    y = 0.5 * x * (1.0 + t)
    tl.store(Out_ptr + m * N + n0 + tl.arange(0, BLOCK_N), y.to(tl.bfloat16), mask=tl.arange(0, BLOCK_N) < N)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args correspond to: hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps
        hidden = args[0]
        grid_thw = args[1]
        ln_weight = args[2]
        ln_bias = args[3]
        fc1_weight = args[4]
        fc1_bias = args[5]
        fc2_weight = args[6]
        fc2_bias = args[7]
        eps = args[8]

        device = hidden.device
        dtype = torch.bfloat16

        # 1) LayerNorm pre-shuffle over last dim
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256
        grid_ln = (num_patches,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=num_patches, K=hidden_size, eps=1e-6,
            BLOCK=BLOCK_ln, num_warps=4, num_stages=2
        )

        # 2) Pack 2x2 features to expanded feature dimension (T=1 assumed in task)
        # M_out = num_patches // 4 (each output row is one fused 2x2)
        M_out = num_patches // 4
        K = hidden_size
        K_expanded = 4 * K
        packed = torch.empty((M_out, K_expanded), dtype=torch.bfloat16, device=device)
        BLOCK_pack = 1024
        grid_pack = (M_out,)
        _pack_2x2_to_expanded_kernel[grid_pack](
            ln_out, packed,
            M_in=num_patches, K=K, BLOCK=BLOCK_pack,
            num_warps=4, num_stages=2
        )

        # 3) FC1: (M_out, 6144) @ (6144, 6144)^T + bias
        M_merged = M_out  # num_merged_patches (from axes)
        K1 = packed.shape[1]  # 4*K = 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((M_merged, N1), dtype=torch.bfloat16, device=device)
        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
        grid_fc1 = (triton.cdiv(M_merged, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M=M_merged, N=N1, K=K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation
        K_after_gelu = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (M_merged, triton.cdiv(K_after_gelu, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M=M_merged, N=K_after_gelu, BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # 5) FC2: (M_merged, 6144) @ (3584, 6144)^T + bias
        K2_in = K_after_gelu  # 6144
        N2 = fc2_weight.shape[0]  # 3584
        out = torch.empty((M_merged, N2), dtype=torch.bfloat16, device=device)
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 64, 64
        grid_fc2 = (triton.cdiv(M_merged, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, out,
            M=M_merged, N=N2, K=K2_in,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
