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
    # Each program handles one (b, h)
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
    # Store to g[b, h] and beta[b, h]
    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr,
    tmp_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,  # k is [B, H, K]
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,  # state is [B, H, V, K]
    stride_t_b, stride_t_h,
    num_warps: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Compute dot(k[b,h], state[b,h]) over V, K
    # Load k_vec [K]
    k_offsets = tl.arange(0, K)
    k_ptr_base = k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
    k_vec = tl.load(k_ptr_base + k_offsets * stride_k_k).to(tl.float32)  # [K]
    # Load state_mat [V, K]
    state_ptr_base = state_ptr + b_idx * stride_s_b + h_idx * stride_s_h
    state_mat = tl.load(state_ptr_base + tl.arange(0, V)[:, None] * stride_s_v + k_offsets[None, :] * stride_s_k,
                        mask=(tl.arange(0, V)[:, None] < V) & (k_offsets[None, :] < K),
                        other=0.0).to(tl.float32)  # [V, K]
    # tmp_old_v = sum_j k[j] * state[j, :]
    # Since state_mat loaded with k fixed, this is equivalent to sum over rows of state_mat multiplied by k_vec:
    # We'll compute it as a dot using reduction over K: for each j, tmp_j += k_vec @ state_mat[j, :]
    tmp_val = tl.zeros((), dtype=tl.float32)
    # Reduction over K
    for i in range(K):
        tmp_val += k_vec[i] * tl.sum(state_mat[:, i])
    tl.store(tmp_ptr + b_idx * stride_t_b + h_idx * stride_t_h, tmp_val)


@triton.jit
def kernel_update_and_output(
    g_ptr, beta_ptr, tmp_ptr, k_ptr, q_ptr, v_ptr, state_ptr, new_state_ptr, out_ptr,
    B, H, V, K,
    stride_g_b, stride_g_h,
    stride_beta_b, stride_beta_h,
    stride_tmp_b, stride_tmp_h,
    stride_k_b, stride_k_h, stride_k_k,  # k is [B, H, K]
    stride_q_b, stride_q_h, stride_q_k,  # q is [B, H, K]
    stride_v_b, stride_v_h, stride_v_v, stride_v_k,  # v is [B, H, V]
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,  # state is [B, H, V, K]
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,  # new_state is [B, H, V, K]
    stride_out_b, stride_out_h,
    num_warps: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load scalars and vectors
    g_val = tl.load(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h).to(tl.float32)
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)
    tmp_old = tl.load(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h).to(tl.float32)

    k_offsets = tl.arange(0, K)
    k_ptr_base = k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
    k_vec = tl.load(k_ptr_base + k_offsets * stride_k_k).to(tl.float32)  # [K]

    q_ptr_base = q_ptr + b_idx * stride_q_b + h_idx * stride_q_h
    q_vec = tl.load(q_ptr_base + k_offsets * stride_q_k).to(tl.float32)  # [K]

    v_ptr_base = v_ptr + b_idx * stride_v_b + h_idx * stride_v_h
    v_vec = tl.load(v_ptr_base + tl.arange(0, V) * stride_v_v + tl.zeros((1,), dtype=tl.int32) * stride_v_k,  # mask not needed since V is static here
                    mask=tl.arange(0, V) < V,
                    other=0.0).to(tl.float32)  # [V]

    # Load state_mat [V, K] for this (b, h)
    state_ptr_base = state_ptr + b_idx * stride_s_b + h_idx * stride_s_h
    state_mat = tl.load(state_ptr_base + tl.arange(0, V)[:, None] * stride_s_v + k_offsets[None, :] * stride_s_k,
                        mask=(tl.arange(0, V)[:, None] < V) & (k_offsets[None, :] < K),
                        other=0.0).to(tl.float32)  # [V, K]

    # Compute new_state_mat [V, K] elementwise:
    # new_state[j, :] = g * state[j, :] - k @ (k @ state[:, j]) + k @ (beta * v[j] + (1 - beta) * tmp_old)
    # Initialize
    new_state_mat = state_mat * g_val  # elementwise g * state

    # Compute k @ state_j for each j, then update
    # Note: k @ state_j means sum_i k[i] * state[i, j]
    for j in range(V):
        state_j_vec = state_mat[j, :]  # [K]
        # First term k @ state_j
        k_dot_j = 0.0
        for i in range(K):
            k_dot_j += k_vec[i] * state_j_vec[i]
        const_term = (1.0 - beta_val) * tmp_old + beta_val * v_vec[j]
        new_state_mat[j, :] -= k_dot_j
        new_state_mat[j, :] += const_term

    # Store new_state[b, h] as [V, K] to new_state_ptr[b, h, :, :]
    ns_ptr_base = new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h
    # Store with mask (in case K,V are not multiples of BLOCK size, but here we use exact V,K)
    tl.store(ns_ptr_base + tl.arange(0, V)[:, None] * stride_ns_v + k_offsets[None, :] * stride_ns_k,
             new_state_mat,
             mask=(tl.arange(0, V)[:, None] < V) & (k_offsets[None, :] < K))

    # Compute output[b, h] = scale * q @ new_state_mat
    # q @ new_state_mat means sum_i q[i] * new_state_mat[i, :]
    out_val = 0.0
    for i in range(K):
        out_val += q_vec[i] * tl.sum(new_state_mat[i, :])
    tl.store(out_ptr + b_idx * stride_out_b + h_idx * stride_out_h, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device and dtype
        device = q.device
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
        assert state.dim() == 4
        B, _, QH, Kq = q.shape
        _, _, KH, _ = k.shape
        _, _, VH, V = v.shape
        Bstate, H, Vstate, Ks = state.shape
        assert Bstate == B and Vstate == V and Ks == Kq, "q, k, v, state K/V dimensions must match"
        assert H == (QH if QH == 4 else None), "QH must be 4 per original assert; adjust if different"

        # Convert all to float32 for Triton kernels
        q_f = q.to(torch.float32).contiguous()
        k_f = k.to(torch.float32).contiguous()
        v_f = v.to(torch.float32).contiguous()
        state_f = state.to(torch.float32).contiguous()
        A_log_f = A_log.to(torch.float32).contiguous()
        a_f = a.to(torch.float32).contiguous()  # [B, 1, H]
        dt_bias_f = dt_bias.to(torch.float32).contiguous()
        b_f = b.to(torch.float32).contiguous()

        # Allocate outputs
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)
        tmp_old = torch.empty((B, H), dtype=torch.float32, device=device)
        new_state = torch.empty((B, H, V, Kq), dtype=torch.float32, device=device)
        out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch kernels
        grid = (B, H)
        # kernel_g_beta
        kernel_g_beta[grid](
            A_log_f, a_f, dt_bias_f, b_f,
            g, beta,
            B, H,
            A_log_f.stride(0), a_f.stride(0), a_f.stride(1), dt_bias_f.stride(0), b_f.stride(0), b_f.stride(1),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
            num_warps=1,
        )
        # kernel_tmp_old_v
        kernel_tmp_old_v[grid](
            k_f, state_f,
            tmp_old,
            B, H, V, Kq,
            k_f.stride(0), k_f.stride(1), k_f.stride(2),  # k is [B, H, K]
            state_f.stride(0), state_f.stride(1), state_f.stride(2), state_f.stride(3),  # state is [B, H, V, K]
            tmp_old.stride(0), tmp_old.stride(1),
            num_warps=1,
        )
        # kernel_update_and_output
        kernel_update_and_output[grid](
            g, beta, tmp_old, k_f, q_f, v_f, state_f, new_state, out,
            B, H, V, Kq,
            g.stride(0), g.stride(1),
            beta.stride(0), beta.stride(1),
            tmp_old.stride(0), tmp_old.stride(1),
            k_f.stride(0), k_f.stride(1), k_f.stride(2),  # k [B, H, K]
            q_f.stride(0), q_f.stride(1), q_f.stride(2),  # q [B, H, K]
            v_f.stride(0), v_f.stride(1), v_f.stride(2), v_f.stride(3),  # v [B, H, V]
            state_f.stride(0), state_f.stride(1), state_f.stride(2), state_f.stride(3),  # state [B, H, V, K]
            new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),  # new_state [B, H, V, K]
            out.stride(0), out.stride(1),
            num_warps=4,
        )

        # Prepare outputs as in original: output [B, 1, H] (cast to bfloat16), new_state [B, H, V, K]
        output_out = out.unsqueeze(1).to(torch.bfloat16)
        return output_out, new_state


def run(*args):
    return ModelNew()(*args)
