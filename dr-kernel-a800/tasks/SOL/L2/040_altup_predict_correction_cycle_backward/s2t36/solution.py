import torch
import triton
import triton.language as tl


# Kernel: compute per-row variance (mean of squares) over N columns for X[M, N]
# M is number of rows (e.g., B*T), N is hidden_size. Outputs var[M] = mean(x^2)
@triton.jit
def var_mean_f32(X_ptr, Out_ptr, M, N, stride_xm, stride_xn):
    pid = tl.program_id(0)
    total = 0.0
    # Loop over columns in tiles of 128
    for start in range(0, N, 128):
        offs = start + tl.arange(0, 128)
        mask = offs < N
        row_ptr = X_ptr + pid * stride_xm + offs * stride_xn
        x = tl.load(row_ptr, mask=mask, other=0.0)
        total += tl.sum(x * x, axis=0)
    mean = total / N
    tl.store(Out_ptr + pid, mean)


# Kernel: compute rstd per row: rstd = 1 / sqrt(var + eps)
@triton.jit
def rsqrt_f32(Var_ptr, Rstd_ptr, size, eps):
    pid = tl.program_id(0)
    if pid < size:
        v = tl.load(Var_ptr + pid)
        rstd = 1.0 / tl.sqrt(v + eps)
        tl.store(Rstd_ptr + pid, rstd)


# Kernel: elementwise tanh over 1D tensor In_ptr of length 'size'
@triton.jit
def tanh_f32(In_ptr, Out_ptr, size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < size
    x = tl.load(In_ptr + offs, mask=mask, other=0.0)
    y = tl.tanh(x)
    tl.store(Out_ptr + offs, y)


# Kernel: GEMV Y[M] = X[M, N] @ W[N, K]^T
# X is [M, N] with strides (stride_xm, stride_xn), W is [N, K] with strides (stride_wm, stride_wn), Y is [M]
@triton.jit
def gemv_f32(X_ptr, W_ptr, Y_ptr, M, N, K, stride_xm, stride_xn, stride_wm, stride_wn):
    pid = tl.program_id(0)
    acc = 0.0
    # Loop over columns in tiles
    for start in range(0, N, 128):
        offs_n = start + tl.arange(0, 128)
        mask_n = offs_n < N
        x = tl.load(X_ptr + pid * stride_xm + offs_n * stride_xn, mask=mask_n, other=0.0)  # [128]
        # Accumulate dot with each row of W
        for j in range(0, K):
            w_j = tl.load(W_ptr + j * stride_wm + offs_n * stride_wn, mask=mask_n, other=0.0)
            acc += tl.sum(x * w_j, axis=0)
    tl.store(Y_ptr + pid, acc)


# Kernel: Batched matrix multiply C[M, N] = A[M, K] @ B[N, K]^T
# A is [M, K], B is [N, K], C is [M, N]. We use 2D tiling with BLOCK_M, BLOCK_N, BLOCK_K.
@triton.jit
def bmm_f32(A_ptr, B_ptr, C_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        b = tl.load(
            B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=mask_m[:, None] & mask_n[None, :]
    )


class ModelNew(torch.nn.Module):
    def __init__(self, rms_norm_eps: float):
        super().__init__()
        self.rms_norm_eps = float(rms_norm_eps)

    def forward(
        self,
        grad_corrected: torch.Tensor,
        hidden_states: torch.Tensor,
        activated: torch.Tensor,
        prediction_coef_weight: torch.Tensor,  # not used (no torch ops in forward)
        correction_coef_weight: torch.Tensor,  # not used
        router_weight: torch.Tensor,           # shape [H, 3]
        norm_weight: torch.Tensor,             # shape [H]
        altup_active_idx: int,
        rms_norm_eps: float,
    ):
        # hidden_states: [B, H, T], activated: [B, H, T], H=2304 fixed
        B


def run(*args):
    return ModelNew()(*args)
