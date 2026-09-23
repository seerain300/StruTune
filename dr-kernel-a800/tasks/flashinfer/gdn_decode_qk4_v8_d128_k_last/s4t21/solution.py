import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_g_kernel(dt_bias_ptr, a_ptr, A_log_ptr, g_ptr, H: tl.constexpr):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    h = pid % H
    a_val = tl.load(a_ptr + pid)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val
    s = tl.log(1.0 + tl.exp(x))
    e = tl.exp(tl.load(A_log_ptr + h))
    g = tl.exp(-e * s)
    tl.store(g_ptr + pid, g)


@triton.jit
def compute_beta_kernel(b_ptr, beta_ptr, H: tl.constexpr):
    pid = tl.program_id(axis=0)
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + pid, beta)


@triton.jit
def dot_k_state_kernel(k_ptr, state_ptr, old_v_ptr, V: tl.constexpr, K: tl.constexpr):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j
    tl.store(old_v_ptr, acc)


@triton.jit
def dot_k_gstate_kernel(k_ptr, state_ptr, g_ptr, state_remove_ptr, V: tl.constexpr, K: tl.constexpr):
    g_val = tl.load(g_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += k_j * (state_ij * g_val)
    tl.store(state_remove_ptr, acc)


@triton.jit
def dot_k_newv_kernel(k_ptr, v_ptr, old_v_ptr, beta_ptr, state_update_ptr, V: tl.constexpr, K: tl.constexpr):
    beta_val = tl.load(beta_ptr)
    old_v_val = tl.load(old_v_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (beta_val * v_i + (1.0 - beta_val) * old_v_val)
    tl.store(state_update_ptr, acc)


@triton.jit
def h_state_vec_kernel(state_ptr, g_ptr, state_remove_ptr, state_update_ptr, h_state_ptr, V: tl.constexpr, K: tl.constexpr):
    g_val = tl.load(g_ptr)
    state_remove_val = tl.load(state_remove_ptr)
    state_update_val = tl.load(state_update_ptr)
    for i in range(V):
        total = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            total += state_ij * g_val
        h_state_vec_i = total - state_remove_val + state_update_val
        tl.store(h_state_ptr + i, h_state_vec_i)


@triton.jit
def dot_q_hstate_kernel(q_ptr, h_state_ptr, output_ptr, V: tl.constexpr, K: tl.constexpr):
    acc = 0.0
    for j in range(K):
        q_j = tl.load(q_ptr + j)
        for i in range(V):
            h_i = tl.load(h_state_ptr + i)
            acc += q_j * h_i
    tl.store(output_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Shapes: q: [B,1,4,128], k: [B,1,4,128], v: [B,1,8,128], state: [B,8,128,128]
        device = q.device
        # Convert inputs to float32 for computation
        q32 = q.squeeze(1).to(torch.float32).contiguous()  # [B,4,128]
        k32 = k.squeeze(1).to(torch.float32).contiguous()  # [B,4,128]
        v32 = v.squeeze(1).to(torch.float32).contiguous()  # [B,8,128]
        a32 = a.squeeze(1).to(torch.float32).contiguous()  # [B,H]
        b32 = b.squeeze(1).to(torch.float32).contiguous()  # [B,H]
        state32 = state.to(torch.float32).contiguous()     # [B,8,128,128]
        dt_bias32 = dt_bias.to(torch.float32).contiguous() # [H]
        A_log32 = A_log.to(torch.float32).contiguous()     # [H]

        B, H = a32.shape
        K = q32.shape[1]  # 128
        V = state32.shape[2]  # 128

        # Allocate intermediates
        g = torch.empty(B * H, dtype=torch.float32, device=device)
        beta = torch.empty(B * H, dtype=torch.float32, device=device)
        out_scalar = torch.empty(B * H, dtype=torch.float32, device=device)

        # 1) Compute g and beta
        grid0 = (B * H,)
        compute_g_kernel[grid0](dt_bias32, a32.reshape(-1), A_log32, g, H=H)
        compute_beta_kernel[grid0](b32.reshape(-1), beta, H=H)

        # Process each (b,h)
        for pid in range(B * H):
            b_idx = pid // H
            h_idx = pid % H

            q_vec = q32[b_idx, h_idx]          # [K]
            k_vec = k32[b_idx, h_idx]          # [K]
            v_vec = v32[b_idx, h_idx]          # [V]
            state_mat = state32[b_idx, h_idx]  # [V,K]

            # 2) old_v = dot(k_vec, state_mat)
            old_v = torch.empty((), dtype=torch.float32, device=device)
            dot_k_state_kernel[(1,)](k_vec, state_mat.reshape(-1), old_v, V=V, K=K)

            # 3) state_remove = dot(k_vec, g * state_mat)
            state_remove = torch.empty((), dtype=torch.float32, device=device)
            dot_k_gstate_kernel[(1,)](k_vec, state_mat.reshape(-1), g[pid], state_remove, V=V, K=K)

            # 4) state_update = dot(k_vec, beta * v + (1 - beta) * old_v)
            state_update = torch.empty((), dtype=torch.float32, device=device)
            dot_k_newv_kernel[(1,)](k_vec, v_vec, old_v, beta[pid], state_update, V=V, K=K)

            # 5) h_state_vec[i] = sum_j state_mat[i,j] * g - state_remove + state_update
            h_state = torch.empty(V, dtype=torch.float32, device=device)
            h_state_vec_kernel[(1,)](state_mat.reshape(-1), g[pid], state_remove, state_update, h_state, V=V, K=K)

            # 6) output_scalar[b,h] = scale * (q_vec @ h_state_vec)
            if scale is None or scale == 0.0:
                scale_val = 1.0 / math.sqrt(K)
            else:
                scale_val = float(scale)
            out_scalar[pid] = scale_val * dot_q_hstate_kernel[(1,)](q_vec, h_state, out_scalar.new_empty((1,)), V=V, K=K)[0]

        # Return output as [B,1,H] in bfloat16
        return out_scalar.view(B, H).unsqueeze(1).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
