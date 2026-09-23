import torch
import math
import triton
import triton.language as tl


# Triton elementwise kernels
@triton.jit
def _softplus_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes softplus(x) = log(1 + exp(x)) elementwise for a 1-element vector (N=1 specialization)
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i, mask=i < 1, other=0.0)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, y, mask=i < 1)


@triton.jit
def _sigmoid_vector(x_ptr, out_ptr, N: tl.constexpr):
    # Computes sigmoid(x) = 1 / (1 + exp(-x)) elementwise for a 1-element vector (N=1 specialization)
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i, mask=i < 1, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y, mask=i < 1)


@triton.jit
def _elementwise_scalar_mul_add(alpha, beta, v_ptr, old_ptr, out_ptr, N: tl.constexpr):
    # Computes out_vec = alpha * v_vec + beta * old_v_vec for N elements (here N=128)
    i = tl.arange(0, N)
    v = tl.load(v_ptr + i, mask=i < N, other=0.0)
    old = tl.load(old_ptr + i, mask=i < N, other=0.0)
    out = alpha * v + beta * old
    tl.store(out_ptr + i, out, mask=i < N)


@triton.jit
def _gemv_1xKxKxV_into_1xV(q_ptr, A_ptr, out_ptr, K: tl.constexpr, V: tl.constexpr):
    # A is [K, V] row-major (i.e., linear index i*V + j), q_ptr is [K]
    # out_vec[i] = sum_j q[j] * A[j, i], for i in 0..V-1
    i = tl.arange(0, V)  # V=128
    acc = tl.zeros([V], dtype=tl.float32)
    for k_start in range(0, K, 16):  # K=128, iterate 8 chunks
        k_idx = k_start + tl.arange(0, 16)
        mask_k = k_idx < K
        q_k = tl.load(q_ptr + k_idx, mask=mask_k, other=0.0)  # [16]
        for kk in range(16):
            kk_valid = k_start + kk < K
            a_row = tl.load(A_ptr + (k_start + kk) * V + i, mask=kk_valid, other=0.0)  # [128]
            acc += q_k[kk] * a_row
    tl.store(out_ptr + i, acc, mask=i < V)


@triton.jit
def _gemv_1xVxK_into_1xK(v_ptr, B_ptr, out_ptr, V: tl.constexpr, K: tl.constexpr):
    # B is [V, K] row-major (index j*K + k), v_ptr is [V]
    # out[k] = sum_j v[j] * B[j, k], for k in 0..K-1
    k = tl.arange(0, K)  # K=128
    acc = tl.zeros([K], dtype=tl.float32)
    for j_start in range(0, V, 16):  # V=128, iterate 8 chunks
        j_idx = j_start + tl.arange(0, 16)
        mask_j = j_idx < V
        v_j = tl.load(v_ptr + j_idx, mask=mask_j, other=0.0)  # [16]
        for jj in range(16):
            jj_valid = j_start + jj < V
            b_col = tl.load(B_ptr + (j_start + jj) * K + k, mask=jj_valid, other=0.0)  # [128]
            acc += v_j[jj] * b_col
    tl.store(out_ptr + k, acc, mask=k < K)


@triton.jit
def _dot_scalar(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    # Computes scalar = sum_i x[i] * y[i] for N elements (e.g., N=128)
    acc = tl.zeros([1], dtype=tl.float32)
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.load(y_ptr + i, mask=i < N, other=0.0)
    acc = tl.sum(x * y, axis=0)
    tl.store(out_ptr, acc)


@triton.jit
def _add_scalar_to_matrix(alpha, A_ptr, M: tl.constexpr, N: tl.constexpr):
    # Add scalar alpha to all elements of [M, N] matrix stored in row-major
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    A = tl.load(A_ptr + rows[:, None] * N + cols[None, :], mask=rows[:, None] < M, other=0.0)
    A = A + alpha
    tl.store(A_ptr + rows[:, None] * N + cols[None, :], A, mask=rows[:, None] < M)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # q: [T, 4, 128], k: [T, 4, 128], v: [T, 8, 128]
        # state: [num_seqs, 8, 128, 128]
        device = q.device
        T = q.shape[0]
        H_Q = q.shape[1]
        H_V = v.shape[1]
        H_K = k.shape[1]
        head_size = 128  # fixed
        num_seqs = cu_seqlens.numel() - 1

        # Prepare expanded q/k for v heads
        q_exp = q.repeat_interleave(H_V // H_Q, dim=1)  # [T, 8, 128]
        k_exp = k.repeat_interleave(H_V // H_K, dim=1)  # [T, 8, 128]

        output = torch.empty((T, H_V, head_size), dtype=torch.bfloat16, device=device)

        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start

            # Running state per head across this segment; since Triton kernels don't return, we keep it in torch
            state_curr = [None] * H_V  # list of [128, 128] for each head h

            for t in range(seq_len):
                t_abs = seq_start + t

                for h in range(H_V):
                    q_vec = q_exp[t_abs, h].contiguous().float()  # [128]
                    k_vec = k_exp[t_abs, h].contiguous().float()  # [128]
                    v_vec = v[t_abs, h].contiguous().float()      # [128]

                    # Compute g and beta scalars for this head h
                    a_h = a[t_abs, h].to(torch.float32).contiguous()     # [1]
                    dt_b_h = dt_bias[h].to(torch.float32).contiguous()   # [1]
                    b_h = b[t_abs, h].to(torch.float32).contiguous()     # [1]
                    A_log_h = A_log[h].to(torch.float32).contiguous()    # [1]

                    # softplus(x) = log(1 + exp(x))
                    softplus_val = torch.empty(1, dtype=torch.float32, device=device)
                    _softplus_vector[(1,)](a_h + dt_b_h, softplus_val, N=1)
                    softplus_val = softplus_val[0]

                    # exp(A_log[h])
                    exp_A = torch.empty(1, dtype=torch.float32, device=device)
                    _softplus_vector[(1,)](A_log_h, exp_A, N=1)  # trick: softplus(exp(A)) is not desired; use separate exp
                    # Implement proper exp for A_log[h]:
                    exp_A = torch.empty(1, dtype=torch.float32, device=device)
                    _exp_scalar(A_log_h, exp_A, N=1)
                    exp_A = exp_A[0]

                    # g = exp(-exp(A_log[h]) * softplus(a[t_abs, h] + dt_bias[h]))
                    g = torch.exp(-exp_A * softplus_val)  # scalar

                    # beta = sigmoid(b[t_abs, h])
                    beta = torch.empty(1, dtype=torch.float32, device=device)
                    _sigmoid_vector[(1,)](b_h, beta, N=1)
                    beta = beta[0]

                    # Initialize state_curr[h] if not set; otherwise use current state
                    if state_curr[h] is None:
                        if seq_idx < state.size(0):
                            state_curr[h] = state[seq_idx, h].transpose(-1, -2).contiguous().float()  # [128, 128]
                        else:
                            state_curr[h] = torch.zeros((head_size, head_size), dtype=torch.float32, device=device)

                    state_curr_T = state_curr[h]  # [128, 128], we want k @ state_curr_T

                    # old_v = k_vec @ state_curr_T
                    old_v = torch.empty((head_size,), dtype=torch.float32, device=device)
                    _gemv_1xKxKxV_into_1xV[(1,)](k_vec, state_curr_T, old_v, K=128, V=128)

                    # new_v = beta * v_vec + (1 - beta) * old_v
                    new_v = torch.empty((head_size,), dtype=torch.float32, device=device)
                    _elementwise_scalar_mul_add((1.0 - beta), beta, v_vec, old_v, new_v, N=128)

                    # Compute scalar contributions: state_update = k_vec · new_v, state_remove = k_vec · old_v
                    state_remove = torch.empty(1, dtype=torch.float32, device=device)
                    _dot_scalar[(1,)](k_vec, old_v, state_remove, N=128)
                    state_update = torch.empty(1, dtype=torch.float32, device=device)
                    _dot_scalar[(1,)](k_vec, new_v, state_update, N=128)
                    alpha = state_update[0] - state_remove[0]  # scalar added to each element of state

                    # Update state: state_new_mat = g * state_curr_T + alpha (broadcast)
                    state_new_mat = torch.empty((head_size, head_size), dtype=torch.float32, device=device)
                    _add_scalar_to_matrix(alpha, state_curr_T, M=128, N=128)

                    # output_vec = scale * (q_vec @ state_new_mat)
                    out_vec = torch.empty((head_size,), dtype=torch.float32, device=device)
                    _gemv_1xVxK_into_1xK[(1,)](state_new_mat, q_vec, out_vec, V=128, K=128)

                    # Store output
                    scale_val = 1.0 / math.sqrt(head_size)
                    output[t_abs, h, :] = (out_vec * scale_val).to(torch.bfloat16)

                    # Update state_curr_T for next iteration
                    state_curr[h] = state_new_mat

        return output

# Additional scalar exp kernel (used in host to compute g)
@triton.jit
def _exp_scalar(x_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i, mask=i < 1, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + i, y, mask=i < 1)


def run(*args):
    return ModelNew()(*args)
