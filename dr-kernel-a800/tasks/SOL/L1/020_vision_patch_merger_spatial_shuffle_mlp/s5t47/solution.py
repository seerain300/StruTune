import math
import torch
import triton
import triton.language as tl


# Triton LayerNorm over last dimension of length H, one program per row
@triton.jit
def layernorm_kernel(
    x_ptr,            # *ptr to input patches (N, H), bfloat16
    y_ptr,            # *ptr to output patches (N, H), bfloat16
    ln_weight_ptr,    # *ptr to ln_weight (H), bfloat16
    ln_bias_ptr,      # *ptr to ln_bias (H), bfloat16
    N,                # number of rows (num_patches)
    H: tl.constexpr,  # hidden_size (1536)
    eps,              # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= N:
        return
    row_offset = row_id * H

    # First pass: sum
    sum_ = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)
    mean = sum_ / H

    # Second pass: sum of squared deviation
    var_sum = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        var_sum += tl.sum((x - mean) * (x - mean), axis=0)
    var = var_sum / H
    rstd = 1.0 / tl.sqrt(var + eps)

    # Third pass: normalize + affine and store
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


# Triton kernel: expand hidden (H=1536) -> expanded (E=4*H=6144) via spatial 2x2 mapping
@triton.jit
def expand_to_hidden_expanded_kernel(
    hidden_ptr,          # *ptr to input hidden (N, H), bfloat16
    expanded_ptr,        # *ptr to output expanded (N, E), bfloat16
    N,                   # number of rows (num_patches)
    H: tl.constexpr,     # hidden_size (1536)
    E: tl.constexpr,     # expanded size (6144)
    BLOCK_OUT: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= N:
        return
    # For each expanded position, map back to original channel index
    for off in range(0, E, BLOCK_OUT):
        out_cols = off + tl.arange(0, BLOCK_OUT)
        mask = out_cols < E
        # Derive dh, dw, c from out_cols: out_cols = (dh * H + c) * 4 + dw
        # dh = out_cols // (H * 4)
        dh = out_cols // (H * 4)
        rem = out_cols - dh * (H * 4)
        c = rem // 4
        dw = rem - c * 4  # not used in read, but ensures exact mapping
        # Compute original hidden index: idx = row_id * H + c
        idx = row_id * H + c
        val = tl.load(hidden_ptr + idx, mask=mask, other=0.0)
        tl.store(expanded_ptr + row_id * E + out_cols, val.to(tl.bfloat16), mask=mask)


# Triton GEMM + bias: C[M, N] = A[M, K] @ W_T[N, K] + bias[N]
@triton.jit
def triton_linear_kernel(
    A_ptr,   # *ptr to A (M, K), bfloat16, but we load as float32 for math
    W_ptr,   # *ptr to W (N, K), bfloat16
    B_ptr,   # *ptr to output (M, N), float32
    M, N, K,
    stride_am, stride_ak,
    stride_wm, stride_wk,
    stride_b,
    eps,  # not used, placeholder
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        w_ptrs = W_ptr + offs_n[None, :] * stride_wm + (k + offs_k[:, None]) * stride_wk

        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k + offs_k[:, None] < K)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # a and w are bfloat16; cast to float32 for math
        a = a.to(tl.float32)
        w = w.to(tl.float32)
        # Accumulate
        acc += tl.dot(a, w)  # (BLOCK_M, BLOCK_N)

    # Add bias
    bias = tl.load(W_ptr + offs_n * stride_wm + stride_b, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Write back
    out_ptrs = B_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# Triton GELU activation: y = 0.5 * x * (1 + erf(x / sqrt(2)))
@triton.jit
def gelu_kernel(
    x_ptr,    # *ptr to input (M, N), float32
    y_ptr,    # *ptr to output (M, N), float32
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
    # GELU using erf
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)


# Triton kernel: second linear layer D[M, OUT_N] = B[M, K] @ V_T[OUT_N, K] + bias
@triton.jit
def triton_linear2_kernel(
    B_ptr,   # *ptr to B (M, K), float32
    V_ptr,   # *ptr to V (OUT_N, K), bfloat16
    D_ptr,   # *ptr to output (M, OUT_N), float32
    M, OUT_N, K,
    stride_bm, stride_bk,
    stride_vm, stride_vk,
    stride_db,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        b_ptrs = B_ptr + offs_m[:, None] * stride_bm + (k + offs_k[None, :]) * stride_bk
        v_ptrs = V_ptr + offs_n[None, :] * stride_vm + (k + offs_k[:, None]) * stride_vk

        b_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        v_mask = (offs_n[None, :] < OUT_N) & (k + offs_k[:, None] < K)

        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # already float32 input
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        acc += tl.dot(b, v)

    # Add bias (bias is the last column in V_ptr for each OUT_N; we pass stride for bias via stride_db)
    bias = tl.load(V_ptr + offs_n * stride_vm + stride_db, mask=offs_n < OUT_N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    out_ptrs = D_ptr + offs_m[:, None] * OUT_N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_N)
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters here; we rely on inputs to provide weights/biases

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        # Ensure tensors are on CUDA for Triton
        device = hidden.device
        N = hidden.shape[0]
        H = hidden.shape[1]  # 1536
        E = 4 * H  # 6144

        # 1) Triton LayerNorm
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        layernorm_kernel[(N,)](
            hidden, hidden_norm, ln_weight, ln_bias, N, H, eps, BLOCK_SIZE=256
        )

        # 2) Triton spatial expand to hidden_expanded (6144) per grid
        expanded = torch.empty((N, E), dtype=torch.bfloat16, device=device)
        expand_to_hidden_expanded_kernel[(N,)](
            hidden_norm, expanded, N, H, E, BLOCK_OUT=1024
        )

        # 3) Triton first linear: A = expanded (N, 6144), W = fc1_weight.T (6144, 6144)
        M1 = expanded.shape[0]
        K1 = expanded.shape[1]  # 6144
        N1 = fc1_weight.shape[0]  # 6144
        # Create W_T in (K, N) = (6144, 6144) from fc1_weight (N, N)
        W_T = fc1_weight.t()  # bfloat16
        # Output B1 in float32
        B1 = torch.empty((M1, N1), dtype=torch.float32, device=device)

        grid_m = (M1 + 64 - 1) // 64
        grid_n = (N1 + 64 - 1) // 64
        triton_linear_kernel[(grid_m, grid_n)](
            expanded, W_T, B1, M1, N1, K1,
            1, K1,  # stride_am, stride_ak for A
            N1, K1,  # stride_wm, stride_wk for W_T
            N1 - 1,  # bias offset at last column
            eps,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 4) Triton GELU activation
        B1_gelu = torch.empty_like(B1, dtype=torch.float32, device=device)
        gelu_kernel[(grid_m, grid_n)](
            B1, B1_gelu, M1, N1, BLOCK_M=64, BLOCK_N=64
        )

        # 5) Triton second linear: B2 = B1_gelu (M1, 6144), V = fc2_weight.T (6144, 3584)
        M2 = M1
        K2 = B1_gelu.shape[1]  # 6144
        OUT_N = fc2_weight.shape[0]  # 3584
        V_T = fc2_weight.t()  # bfloat16, (6144, 3584)
        D = torch.empty((M2, OUT_N), dtype=torch.float32, device=device)

        grid_m2 = (M2 + 64 - 1) // 64
        grid_n2 = (OUT_N + 64 - 1) // 64
        triton_linear2_kernel[(grid_m2, grid_n2)](
            B1_gelu, V_T, D, M2, OUT_N, K2,
            1, K2,  # stride_bm, stride_bk
            OUT_N, K2,  # stride_vm, stride_vk
            K2 - 1,  # bias offset at last column
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 6) Return in bfloat16 to match original behavior
        return D.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
