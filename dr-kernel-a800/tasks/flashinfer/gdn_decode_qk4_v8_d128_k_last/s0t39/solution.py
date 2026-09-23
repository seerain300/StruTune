import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    B, H,
    stride_A, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
    stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load scalars
    A = tl.load(A_log_ptr + h_idx * stride_A).to(tl.float32)
    a = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h).to(tl.float32)
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt).to(tl.float32)
    bb = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32)
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a + dt))
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))
    # Store to [B, H]
    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_t_b, stride_t_h,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # k[b, h, :] as [K]
    k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
    # state[b, h, :, :] as [V, K]
    state_mat = tl.load(
        state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + tl.arange(0, V)[:, None] * stride_s_v + tl.arange(0, K)[None, :] * stride_s_k,
        mask=(tl.arange(0, V)[:, None] < V) & (tl.arange(0, K)[None, :] < K),
        other=0.0,
    )
    # tmp_old_v = sum_j k[j] * state[j, :]
    tmp = tl.sum(k_vec[None, :] * state_mat, axis=1)  # shape [V], then sum
    tmp = tl.sum(tmp, axis=0)  # scalar
    tl.store(tmp_ptr + b_idx * stride_t_b + h_idx * stride_t_h, tmp)


@triton.jit
def kernel_update_and_output(
    state_ptr, v_ptr, k_ptr, q_ptr, g_ptr, beta_ptr, tmp_ptr,
    new_state_ptr, output_ptr,
    B, H, V, K,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_v_b, stride_v_h, stride_v_v,
    stride_k_b, stride_k_h, stride_k_k,
    stride_q_b, stride_q_h, stride_q_k,
    stride_g_b, stride_g_h,
    stride_b_b, stride_b_h,
    stride_t_b, stride_t_h,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load scalars and vectors
    g_val = tl.load(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h).to(tl.float32)
    beta_val = tl.load(beta_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32)
    tmp_old = tl.load(tmp_ptr + b_idx * stride_t_b + h_idx * stride_t_h).to(tl.float32)
    k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0).to(tl.float32)
    q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0).to(tl.float32)

    # Load v[b, h, :] as [V]
    v_vec = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + tl.arange(0, V), mask=tl.arange(0, V) < V, other=0.0).to(tl.float32)

    # Load state[b, h, :, :] as [V, K]
    state_mat = tl.load(
        state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + tl.arange(0, V)[:, None] * stride_s_v + tl.arange(0, K)[None, :] * stride_s_k,
        mask=(tl.arange(0, V)[:, None] < V) & (tl.arange(0, K)[None, :] < K),
        other=0.0,
    ).to(tl.float32)

    # Prepare new_state_mat [V, K] and compute output scalar
    new_state_mat = tl.zeros((V, K), dtype=tl.float32)
    # Compute per-column j update
    for j in range(0, V):  # Triton supports Python range; V is constexpr for kernel specialization
        v_j = v_vec[j]
        const_term = beta_val * v_j + (1.0 - beta_val) * tmp_old  # scalar
        # For each i in K: new_state_mat[j, i] = g * state_mat[j, i] - sum_k k[k] * (sum_i state_mat[i, j]) + sum_k k[k] * const_term
        sum_k_state_row_i = tl.zeros((K,), dtype=tl.float32)
        for i in range(0, K):
            state_j_i = state_mat[j, i]
            sum_k_state_row_i[i] = tl.sum(k_vec * state_mat[:, i], axis=0)  # compute sum_k k[k] * state_mat[k, i]
            new_state_mat[j, i] = g_val * state_j_i - (k_vec.dot(sum_k_state_row_i)) + k_vec.dot(const_term)

    # Store new_state[b, h, :, :]
    tl.store(
        new_state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + tl.arange(0, V)[:, None] * stride_s_v + tl.arange(0, K)[None, :] * stride_s_k,
        new_state_mat,
        mask=(tl.arange(0, V)[:, None] < V) & (tl.arange(0, K)[None, :] < K),
    )

    # Compute output[b, h] = scale * q[b,h] @ new_state[b,h]
    # output is scalar; allocate output_ptr as [B, H] and store
    output_val = tl.sum(q_vec * tl.sum(new_state_mat, axis=1), axis=0)  # sum over K to get q @ new_state
    tl.store(output_ptr + b_idx * stride_g_b + h_idx * stride_g_h, output_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure inputs are on CUDA device and contiguous, use float32
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All inputs must be CUDA tensors"
        assert q.shape[0] == 1 and q.shape[2] == 4, "q must have shape [B, 1, QH, K] with QH=4"
        assert k.shape[0] == 1 and k.shape[2] == 4, "k must have shape [B, 1, KH, K] with KH=4"
        assert v.shape[0] == 1 and v.shape[2] == 8, "v must have shape [B, 1, VH, V] with VH=8"
        assert state.shape[0] == 1 and state.shape[1] == 8 and state.shape[3] == 128, "state must have shape [B, H, V, K] with H=8, V=128, K=128"

        B = q.shape[0]
        H = state.shape[1]
        V = state.shape[2]
        K = state.shape[3]

        # Cast to float32 for kernels
        q_f = q.float().contiguous()     # [B, 1, QH, K] -> specialize on QH=4
        k_f = k.float().contiguous()     # [B, 1, KH, K] -> specialize on KH=4
        v_f = v.float().contiguous()     # [B, 1, VH, V] -> specialize on VH=8
        state_f = state.float().contiguous()  # [B, H, V, K]

        # Allocate outputs
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)
        tmp = torch.empty((B, H), dtype=torch.float32, device=device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)
        output = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch kernels
        grid = (B, H)
        # 1) Compute g and beta
        kernel_g_beta[grid](
            A_log.float().contiguous(), a.float().contiguous(), dt_bias.float().contiguous(), b.float().contiguous(),
            g, beta,
            B, H,
            A_log.stride(0), a.stride(0), a.stride(1), dt_bias.stride(0), b.stride(0), b.stride(1),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
            num_warps=1,
        )
        # 2) Compute tmp_old_v[b,h] = dot(k[b,h], state[b,h])
        kernel_tmp_old_v[grid](
            k_f, state_f, tmp,
            B, H, V, K,
            k_f.stride(0), k_f.stride(1), k_f.stride(2),
            state_f.stride(0), state_f.stride(1), state_f.stride(2), state_f.stride(3),
            tmp.stride(0), tmp.stride(1),
            num_warps=1,
        )
        # 3) Update new_state and output[b,h] per (b,h)
        kernel_update_and_output[grid](
            state_f, v_f, k_f, q_f, g, beta, tmp,
            new_state, output,
            B, H, V, K,
            state_f.stride(0), state_f.stride(1), state_f.stride(2), state_f.stride(3),
            v_f.stride(0), v_f.stride(1), v_f.stride(2),
            k_f.stride(0), k_f.stride(1), k_f.stride(2),
            q_f.stride(0), q_f.stride(1), q_f.stride(2),
            g.stride(0), g.stride(1),
            beta.stride(0), beta.stride(1),
            tmp.stride(0), tmp.stride(1),
            num_warps=1,
        )

        # Return: output cast to bfloat16 as [B, 1, H], new_state as [B, H, V, K]
        output_b1H = output.unsqueeze(1).to(torch.bfloat16)
        return output_b1H, new_state


def run(*args):
    return ModelNew()(*args)
