import torch
import triton
import triton.language as tl


# Triton kernels

@triton.jit
def sum_squares_reduce_kernel(x_ptr, out_ptr, H: tl.constexpr, BLOCK_H: tl.constexpr):
    """
    For each (b, s), reduce sum(x[b, s, :])^2 across H and write to out[b*S].
    We launch one program per (b, s) and accumulate into a scalar via atomic_add.
    """
    pid = tl.program_id(axis=0)  # index over (b, s)
    total = 0.0
    for h0 in range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < H
        x = tl.load(x_ptr + pid * H + offs, mask=mask, other=0.0)
        sq = x * x
        total += tl.sum(sq, axis=0)
    tl.atomic_add(out_ptr + pid, total)


@triton.jit
def rsqrt_kernel(inp_ptr, out_ptr, N, eps, BLOCK_SIZE: tl.constexpr):
    """
    Compute inv_std = 1/sqrt(inp + eps) for a vector of length N.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    inv_std = 1.0 / tl.sqrt(x + eps)
    tl.store(out_ptr + offsets, inv_std, mask=mask)


@triton.jit
def tanh_kernel(inp_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute tanh for a vector of length N using exp:
    tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    z = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    e2z = tl.exp(2.0 * z)
    y = (e2z - 1.0) / (e2z + 1.0)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def matvec_kernel(A_ptr, W_ptr, Out_ptr, M, N, K,
                  stride_a0, stride_a1, stride_w0, stride_w1,
                  BLOCK_N: tl.constexpr):
    """
    Implement GEMV: Out[M,K] = A[M,N] @ W[N,K]
    Launch with M=1 for each (b,s) row; K is the output dim (e.g., 9).
    Grid: (M, K). We loop over N in tiles of BLOCK_N and accumulate scalar acc into Out[m,k].
    """
    pid_m = tl.program_id(axis=0)  # row index m
    pid_k = tl.program_id(axis=1)  # feature index k
    acc = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)
        mask_n = n_idx < N
        # Load A[m, n_idx]
        a_row_ptr = A_ptr + pid_m * stride_a0 + n_idx * stride_a1
        a = tl.load(a_row_ptr, mask=mask_n, other=0.0)
        # Load W[n_idx, k]
        w_col_ptr = W_ptr + n_idx * stride_w0 + pid_k * stride_w1
        w = tl.load(w_col_ptr, mask=mask_n, other=0.0)
        acc += tl.sum(a * w, axis=0)
    out_index = pid_m * K + pid_k
    tl.store(Out_ptr + out_index, acc)


@triton.jit
def bmm_assemble_per_bs_kernel(X_ptr, W_ptr, Out_ptr, H, N, K,
                                stride_x0, stride_x1,  # X[H, N]
                                stride_w0, stride_w1,  # W[N, K]
                                stride_out0, stride_out1,  # Out[H, K]
                                BLOCK_N: tl.constexpr):
    """
    For a single (b, s) group, compute Out[h, k] = sum_n X[h, n] * W[n, k], h in [0, H), k in [0, K).
    Launch grid (H, K). Loop over N in tiles BLOCK_N and accumulate into Out[H, K].
    This avoids torch.bmm by performing per-(b,s) dot products over H against a 9x9 matrix.
    """
    pid_h = tl.program_id(axis=0)  # row index h
    pid_k = tl.program_id(axis=1)  # feature index k
    acc = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)
        mask_n = n_idx < N
        x_row_ptr = X_ptr + pid_h * stride_x0 + n_idx * stride_x1
        x = tl.load(x_row_ptr, mask=mask_n, other=0.0)
        w_col_ptr = W_ptr + n_idx * stride_w0 + pid_k * stride_w1
        w = tl.load(w_col_ptr, mask=mask_n, other=0.0)
        acc += tl.sum(x * w, axis=0)
    out_index = pid_h * K + pid_k
    tl.store(Out_ptr + out_index, acc)


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
                rms_norm_eps: float,
                ):
        """
        Triton-optimized forward recomputation for the 'predict' and 'correct' steps.
        We reconstruct necessary vectors from inputs, perform heavy math in Triton kernels,
        and return gradients matching the original signature. We avoid torch.bmm entirely.
        """

        # Dimensions
        H = hidden_states.shape[-1]  # 2304
        B = hidden_states.shape[1]
        S = hidden_states.shape[2]
        N_modalities = 9  # output dim

        device = hidden_states.device

        # 1) Compute variance and rstd per (b, s) using Triton
        # hidden_states_flat: [


def run(*args):
    return ModelNew()(*args)
