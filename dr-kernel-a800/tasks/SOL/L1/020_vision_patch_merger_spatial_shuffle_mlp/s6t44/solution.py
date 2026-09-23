import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    x_ptr,                # *bf16 (M, K)
    ln_w_ptr,             # *bf16 (K,)
    ln_b_ptr,             # *bf16 (K,)
    y_ptr,                # *bf16 (M, K)
    M: tl.int32,          # number of rows
    K: tl.int32,          # hidden size
    eps: tl.float32,      # epsilon
    BLOCK: tl.constexpr,  # loop block over K
):
    # one program per row
    pid = tl.program_id(axis=0)
    # accumulate sum and sum of squares in FP32
    sum_val = 0.0
    sum_sq = 0.0
    for k in range(0, K, BLOCK):
        offs_k = k + tl.arange(0, BLOCK)
        mask = offs_k < K
        x = tl.load(x_ptr + pid * K + offs_k, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / K
    var = sum_sq / K - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize, affine, store
    for k in range(0, K, BLOCK):
        offs_k = k + tl.arange(0, BLOCK)
        mask = offs_k < K
        x = tl.load(x_ptr + pid * K + offs_k, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(ln_w_ptr + offs_k, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_b_ptr + offs_k, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        # store as BF16
        tl.store(y_ptr + pid * K + offs_k, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _pack_2x2_to_expanded_kernel(
    x_ptr,                # *bf16 (M_out, K)
    out_ptr,              # *bf16 (M_out, 4*K)
    M_out: tl.int32,      # number of output rows
    K: tl.int32,          # hidden size
    BLOCK: tl.constexpr,  # tile over K for loads
):
    # one program per output row
    pid = tl.program_id(axis=0)
    fourK = 4 * K

    # segment offsets for the 4 contiguous segments
    offsets = tl.arange(0, BLOCK)
    base_row = pid * K

    # segment 0: top-left
    for k in range(0, K, BLOCK):
        src_idx = base_row + k + offsets
        dst_idx = (pid * fourK) + (0 * K + k + offsets)
        mask = (k + offsets) < K
        x = tl.load(x_ptr + src_idx, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + dst_idx, x, mask=mask)

    # segment 1: top-right
    for k in range(0, K, BLOCK):
        src_idx = base_row + K + k + offsets
        dst_idx = (pid * fourK) + (1 * K + k + offsets)
        mask = (k + offsets) < K
        x = tl.load(x_ptr + src_idx, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + dst_idx, x, mask=mask)

    # segment 2: bottom-left
    for k in range(0, K, BLOCK):
        src_idx = base_row + 2 * K + k + offsets
        dst_idx = (pid * fourK) + (2 * K + k + offsets)
        mask = (k + offsets) < K
        x = tl.load(x_ptr + src_idx, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + dst_idx, x, mask=mask)

    # segment 3: bottom-right
    for k in range(0, K, BLOCK):
        src_idx = base_row + 3 * K + k + offsets
        dst_idx = (pid * fourK) + (3 * K + k + offsets)
        mask = (k + offsets) < K
        x = tl.load(x_ptr + src_idx, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + dst_idx, x, mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,                # *bf16 (M, K), row-major
    W_ptr,                # *bf16 (K, N), row-major
    bias_ptr,             # *bf16 (N,)
    C_ptr,                # *bf16 (M, N)
    M: tl.int32,          # rows of A
    N: tl.int32,          # cols of W / C
    K: tl.int32,          # cols of A / rows of W
    stride_am: tl.int32,  # stride of A along M
    stride_ak: tl.int32,  # stride of A along K
    stride_wk: tl.int32,  # stride of W along K
    stride_wn: tl.int32,  # stride of W along N
    stride_cm: tl.int32,  # stride of C along M
    stride_cn: tl.int32,  # stride of C along N
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (tiles over M, tiles over N)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # output tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (mask_m[:, None]) & (mask_k[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # load W tile: [BLOCK_K, BLOCK_N] from W[K, N]
        w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (mask_k[:, None]) & (mask_n[None, :])
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # FMA
        acc += tl.dot(a, w)

    # add bias
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += b[None, :]

    # store result in BF16
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (mask_m[:, None]) & (mask_n[None, :])
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _gelu_kernel(
    x_ptr,                # *bf16 (M, N)
    y_ptr,                # *bf16 (M, N)
    M: tl.int32,          # rows
    N: tl.int32,          # cols
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # row id
    pid_n = tl.program_id(axis=1)  # col tile id

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_n < N) & (pid_m < M)

    x = tl.load(x_ptr + pid_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    # tanh-based GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    gelu = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
    tl.store(y_ptr + pid_m * N + offs_n, gelu.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float):
        """
        Triton-only implementation:
        - LayerNorm + affine (in Triton, FP32 compute, BF16 store)
        - Spatial packing (in Triton, data movement)
        - fc1: GEMM + bias (in Triton)
        - GELU (in Triton)
        - fc2: GEMM + bias (in Triton)
        """
        assert hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, "All tensors must be on CUDA for Triton."

        device = hidden.device
        dtype = hidden.dtype  # bfloat16
        M = hidden.shape[0]  # num_patches
        K = hidden.shape[1]  # 1536
        eps_val = eps

        # 1) LayerNorm + affine
        ln_out = torch.empty((M, K), dtype=torch.bfloat16, device=device)
        BLOCK = 256
        grid_ln = (M,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=M, K=K, eps=eps_val,
            BLOCK=BLOCK, num_warps=4, num_stages=2
        )

        # 2) Pack to expanded features
        M_out = M // 4  # guaranteed by get_inputs
        expanded_K = 4 * K  # 6144
        packed = torch.empty((M_out, expanded_K), dtype=torch.bfloat16, device=device)
        BLOCK_pack = 1024
        grid_pack = (M_out,)
        _pack_2x2_to_expanded_kernel[grid_pack](
            ln_out, packed,
            M_out=M_out, K=K,
            BLOCK=BLOCK_pack, num_warps=4, num_stages=2
        )

        # 3) fc1: (M_out, 6144) @ (6144, 6144)^T + bias -> (M_out, 6144)
        # Shapes:
        # A: packed (M_out, 6144)
        # W: fc1_weight (6144, 6144)
        # bias: fc1_bias (6144,)
        # output: fc1_out (M_out, 6144)
        A = packed
        W = fc1_weight
        bias = fc1_bias
        N1 = W.shape[1]  # 6144
        M1 = A.shape[0]  # M_out
        K1 = A.shape[1]  # 6144

        fc1_out = torch.empty((M1, N1), dtype=torch.bfloat16, device=device)

        # Tiling
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_fc1 = (triton.cdiv(M1, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _gemm_bias_kernel[grid_fc1](
            A, W, bias, fc1_out,
            M=M1, N=N1, K=K1,
            stride_am=A.stride(0), stride_ak=A.stride(1),
            stride_wk=W.stride(0), stride_wn=W.stride(1),
            stride_cm=fc1_out.stride(0), stride_cn=fc1_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 4) GELU
        # GELU of fc1_out -> (M_out, 6144)
        M_out_2 = fc1_out.shape[0]
        N2 = fc1_out.shape[1]  # 6144
        fc1_gelu = torch.empty((M_out_2, N2), dtype=torch.bfloat16, device=device)

        BLOCK_N_gelu = 256
        grid_gelu = (M_out_2, triton.cdiv(N2, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_gelu,
            M=M_out_2, N=N2,
            BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # 5) fc2: (M_out, 6144) @ (3584, 6144)^T + bias -> (M_out, 3584)
        # A: fc1_gelu (M_out, 6144)
        # W: fc2_weight (3584, 6144)
        # bias: fc2_bias (3584,)
        M2 = M_out_2
        K2 = fc1_gelu.shape[1]  # 6144
        N2_out = fc2_weight.shape[0]  # 3584

        fc2_out = torch.empty((M2, N2_out), dtype=torch.bfloat16, device=device)
        BLOCK_M2 = 64
        BLOCK_N2 = 64
        BLOCK_K2 = 64

        grid_fc2 = (triton.cdiv(M2, BLOCK_M2), triton.cdiv(N2_out, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_gelu, fc2_weight, fc2_bias, fc2_out,
            M=M2, N=N2_out, K=K2,
            stride_am=fc1_gelu.stride(0), stride_ak=fc1_gelu.stride(1),
            stride_wk=fc2_weight.stride(0), stride_wn=fc2_weight.stride(1),
            stride_cm=fc2_out.stride(0), stride_cn=fc2_out.stride(1),
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return fc2_out


def run(*args):
    return ModelNew()(*args)
