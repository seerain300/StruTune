import torch
import triton
import triton.language as tl


# Triton GEMM kernel: C[M, N] = A[M, K] @ B[N, K]^T (no bias)
@triton.jit
def matmul_no_bias_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton RMSNorm per row (last dim): X[M, N] -> Y[M, N], normalize by row RMS and scale by per-column W[N]
@triton.jit
def rmsnorm_row_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_w,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    # We process one row per program. M is number of rows (e.g., B*S).
    offs_n = tl.arange(0, BLOCK_N)
    x = tl.load(X_ptr + pid_m * stride_xm + offs_n * stride_xn, mask=offs_n < N, other=0.0)
    sq = x * x
    sum_sq = tl.sum(sq, axis=0)
    mean = sum_sq / N
    eps = 1e-6
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    w = tl.load(W_ptr + offs_n * stride_w, mask=offs_n < N, other=1.0)
    x_norm = x * inv_rms * w
    tl.store(Y_ptr + pid_m * stride_ym + offs_n * stride_yn, x_norm, mask=offs_n < N)


# Triton kernel for Rotated Positional Embedding (RoPE) rotation on 128-dim vectors.
# Input X[B, S, 128], output Y[B, S, 128], using cos/sin of length 128.
@triton.jit
def rotate_128_kernel(
    X_ptr, Y_ptr, cos_ptr, sin_ptr,
    B, S,
    stride_xb, stride_xs, stride_xd,
    stride_yb, stride_ys, stride_yd,
    stride_cos, stride_sin,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    if pid_b >= B or pid_s >= S:
        return

    offs_d = tl.arange(0, 128)
    x = tl.load(X_ptr + pid_b * stride_xb + pid_s * stride_xs + offs_d * stride_xd, mask=offs_d < 128, other=0.0)
    cos = tl.load(cos_ptr + offs_d * stride_cos, mask=offs_d < 128, other=1.0)
    sin = tl.load(sin_ptr + offs_d * stride_sin, mask=offs_d < 128, other=0.0)

    q1 = x[:64]
    q2 = x[64:]
    rotated_half = -q2 * sin + q1 * cos
    y = x * cos + rotated_half * sin

    tl.store(Y_ptr + pid_b * stride_yb + pid_s * stride_ys + offs_d * stride_yd, y, mask=offs_d < 128)


# Triton GEMM kernel for output projection: Y[M, N] = A[M, K] @ W[N, K]^T (no bias)
@triton.jit
def output_proj_kernel(
    A_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(a, w)

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self, num_attention_heads=96, head_dim=128, num_key_value_heads=8, num_key_value_groups=12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

    def forward(self, hidden_states, q_proj_weight, q_norm_weight, k_proj_weight, k_norm_weight, v_proj_weight, v_norm_weight, o_proj_weight, q_norm_weight_q, k_norm_weight_k, cos, sin):
        # hidden_states: [B, S, H], H = num_attention_heads * head_dim = 96 * 128 = 12288
        # Note: In the original run, v_norm_weight is unused; we ignore it here.
        B, S, H = hidden_states.shape
        D = self.head_dim  # 128

        # 1) Dense linear projections via Triton GEMM (no bias)
        query = torch.empty((B, S, H), dtype=hidden_states.dtype, device=hidden_states.device)
        matmul_no_bias_kernel[(B, H), (H, H)](
            hidden_states, q_proj_weight, query,
            B, H, H,
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            query.stride(0), query.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        key = torch.empty((B, S, H), dtype=hidden_states.dtype, device=hidden_states.device)
        matmul_no_bias_kernel[(B, H), (H, H)](
            hidden_states, k_proj_weight, key,
            B, H, H,
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            key.stride(0), key.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        value


def run(*args):
    return ModelNew()(*args)
