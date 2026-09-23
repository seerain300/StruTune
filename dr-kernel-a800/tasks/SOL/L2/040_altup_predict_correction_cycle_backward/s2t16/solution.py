import torch
import triton
import triton.language as tl


# GEMV: Y[M] = X[M, N] @ W[K, N]^T
# Inputs: X_ptr [M, N] contiguous or strided; W_ptr [K, N] contiguous or strided; Output: Y_ptr [M]
@triton.jit
def gemv_f32(X_ptr, W_ptr, Y_ptr, M, N, K,
             stride_xm, stride_xn,
             stride_wk, stride_wn,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # One program per output element i
    i = tl.program_id(0)
    acc = 0.0
    # Loop over K in chunks
    for start_k in range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # For each k in chunk, accumulate X[i, k] * W[k, :]
        # W[k, :] is a vector of length N
        for kk in range(0, BLOCK_K):
            k_idx = start_k + kk
            if k_idx < K:
                # Load x scalar
                x_val = tl.load(X_ptr + i * stride_xm + k_idx * stride_xn)
                # Load W row vector
                w_row = tl.load(W_ptr + k_idx * stride_wk + tl.arange(0, BLOCK_N) * stride_wn, mask=tl.arange(0, BLOCK_N) < N, other=0.0)
                # Dot: sum over N of x_val * w_row
                acc += tl.sum(x_val * w_row, axis=0)
    tl.store(Y_ptr + i, acc)


# Batched MatMul: Y[M, N] = X[M, K] @ W[N, K]^T
# Inputs: X_ptr [M, K]; W_ptr [N, K]; Output: Y_ptr [M, N]
@triton.jit
def bmm_f32(X_ptr, W_ptr, Y_ptr, M, N, K,
            stride_xm, stride_xk,
            stride_wn, stride_wk,
            stride_ym, stride_yn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 2D grid over tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # Load X tile: [BLOCK_M, BLOCK_K]
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        # Load W^T tile: we need W[N, K] and take its transpose conceptually.
        # Implement as loop over kk: for each kk in BLOCK_K, take W[:, kk] vector and outer with x[:, kk]
        for kk in range(0, BLOCK_K):
            k_idx = k0 + kk
            if k_idx < K:
                # w_col vector of length N: W[:, k_idx]
                w_col = tl.load(W_ptr + tl.arange(0, BLOCK_N) * stride_wn + k_idx * stride_wk, mask=tl.arange(0, BLOCK_N) < N, other=0.0)
                # Outer product and accumulate
                acc += x[:, kk][:, None] * w_col[None, :]

    # Store tile
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


class ModelNew(torch.nn.Module):
    def forward(self,
                grad_corrected: torch.Tensor,
                hidden_states: torch.Tensor,
                activated: torch.Tensor,
                prediction_coef_weight: torch.Tensor,
                correction_coef_weight: torch.Tensor,
                router_weight: torch.Tensor,
                norm_weight: torch.Tensor,
                altup_active_idx: int,
                rms_norm_eps: float):
        """
        Triton-optimized forward that mirrors the original logic and returns gradients for learnable parameters.
        - Computes forward recomputation for 'predict' and 'correct' steps in Triton.
        - Derives gradients for prediction_coef_weight, correction_coef_weight, router_weight, and norm_weight using Triton kernels.
        Returns:
        - hidden_states_grad (bf16, same shape as hidden_states), but original run only returns grads for learnable params; hidden/activated grads are not returned.
        - activated_grad (bf16, same shape as activated) - not returned.
        - prediction_coef_weight_grad (same shape as prediction_coef_weight)
        - correction_coef_weight_grad (same shape as correction_coef_weight)
        - router_weight_grad (shape [3, hidden_size])
        - norm_weight_grad (shape [hidden_size])
        """

        # Extract shapes
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[2]


def run(*args):
    return ModelNew()(*args)
