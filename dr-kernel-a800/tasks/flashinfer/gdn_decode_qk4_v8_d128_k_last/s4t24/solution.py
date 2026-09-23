import math
import torch
import triton
import triton.language as tl


# Kernel: compute g = exp(-exp(A_log[h]) * softplus(x)), where x = a[b,h] + dt_bias[h]
# softplus(x) = log(1 + exp(x))
@triton.jit
def softplus_and_exp_kernel(
    dt_bias_ptr,    # float32 [H]
    a_ptr,          # float32 [B*H]
    A_log_ptr,      # float32 [H]
    g_out_ptr,      # float32 [B*H]
    B: tl.constexpr, H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    h = pid % H
    a_val = tl.load(a_ptr + pid)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val
    s = tl.log(1.0 + tl.exp(x))
    e = tl.exp(tl.load(A_log_ptr + h))
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Kernel: compute beta[b,h] = 1 / (1 + exp(-b[b,h]))
@triton.jit
def sigmoid_kernel(
    b_ptr,          # float32 [B*H]
    beta_out_ptr,   # float32 [B*H]
    B: tl.constexpr, H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Kernel: scalar old_v = dot(k_vec[K], state_mat[V*K])
@triton.jit
def dot_k_state_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    old_v_ptr,      # float32 [1]
    V: tl.constexpr, K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j
    tl.store(old_v_ptr, acc)


# Kernel: scalar state_remove = dot(k_vec, g * state_mat)
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    g_ptr,          # float32 [1] scalar g for this b,h
    state_remove_ptr,  # float32 [1]
    V: tl.constexpr, K: tl.constexpr,
):
    g_val = tl.load(g_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j * g_val
    tl.store(state_remove_ptr, acc)


# Kernel: scalar state_update = dot(k_vec, beta * v_vec + (1 - beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    old_v_ptr,      # float32 [1]
    beta_ptr,       # float32 [1] scalar beta for this b,h
    state_update_ptr,   # float32 [1]
    V: tl.constexpr, K: tl.constexpr,
):
    beta_val = tl.load(beta_ptr)
    old_v_val = tl.load(old_v_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (beta_val * v_i + (1.0 - beta_val) * old_v_val)
    tl.store(state_update_ptr, acc)


# Kernel: vector h_state_vec[i] = sum_j state[b,h,i,j] * g - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_ptr,          # float32 [1] scalar g for this b,h
    state_remove_ptr,   # float32 [1]
    state_update_ptr,   # float32 [1]
    h_state_ptr,         # float32 [V]
    V: tl.constexpr, K: tl.constexpr,
):
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


# Kernel: scalar output[b,h] = scale * q_vec @ h_state_vec
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    output_ptr,     # float32 [1]
    V: tl.constexpr, K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        q_j = tl.load(q_ptr + j)
        for i in range(V):
            h_i = tl.load(h_state_ptr + i)
            acc += q_j * h_i
    tl.store(output_ptr, acc)


# Kernel: write new_state[b,h] as [V,K], broadcasting h_state_vec across K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [V*K]
    V: tl.constexpr, K: tl.constexpr,
):
    # Write h_state_vec across K columns
    for i in range(V):
        val = tl.load(h_state_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, val)


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device and dtype
        device = q.device
        dtype_compute = torch.float32

        # Shapes: q: [B,1,Hq,128], k: [B,1,Hk,128], v: [B,1,Hv,128], state: [B,Hv,128,128]
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q,k,v must be 4D"
        assert state.dim() == 4, "state must be 4D [B, H, V, K]"
        B_q, T_q, Hq, K = q.shape
        B_k, T_k, Hk, Kk = k.shape
        B_v, T_v, Hv, V = v.shape
        B_s, H, V_s, K_s = state.shape
        assert B_q == B_k == B_v == B_s, "Batch sizes must match"
        assert Hq == 4 and Hk == 4 and Hv == 8, "Expected num_q_heads=4, num_k_heads=4, num_v_heads=8"
        assert T_q == 1 and T_k == 1 and T_v == 1, "T must be 1"
        assert K == 128 and V == 128 and H == 8 and V_s == 128 and K_s == 128, "Expected dimensions 128"
        B = B_q

        # Prepare inputs as float32
        a_f32 = a.float().reshape(B * H)
        dt_bias_f32 = dt_bias.float()
        A_log_f32 = A_log.float()
        b_f32 = b.float().reshape(B * H)

        # Allocate outputs
        g_out = torch.empty(B * H, device=device, dtype=torch.float32)
        beta_out = torch.empty(B * H, device=device, dtype=torch.float32)

        # Launch kernels
        softplus_and_exp_kernel[(B * H,)](dt_bias_f32, a_f32, A_log_f32, g_out, B=B, H=H)
        sigmoid_kernel[(B * H,)](b_f32, beta_out, B=B, H=H)

        # Flatten k and v to [K] for dot-products (per (b,h))
        # Use .contiguous() to ensure 1D vector
        k_vec = k.reshape(B, H, K).contiguous()          # [B, H, K]
        state_mat = state.reshape(B, H, V, K).contiguous()  # [B, H, V, K]
        v_vec = v.reshape(B, H, V).contiguous()          # [B, H, V]

        # We'll compute output tensor [B,1,H] and new_state [B,H,V,128]
        output_f32 = torch.empty(B * H, device=device, dtype=torch.float32)
        new_state = torch.empty((B, H, V, K), device=device, dtype=torch.float32)

        # For each (b,h), compute scalar old_v, state_remove, state_update, h_state_vec, output, and write new_state
        for b_idx in range(B):
            for h_idx in range(H):
                pid = b_idx * H + h_idx

                # Prepare pointers for current (b,h)
                k_bh = k_vec[b_idx, h_idx]  # [K]
                state_bh = state_mat[b_idx, h_idx]  # [V, K] flattened
                v_bh = v_vec[b_idx, h_idx]  # [V]

                # 1) old_v = k @ state
                old_v = torch.empty(1, device=device, dtype=torch.float32)
                dot_k_state_kernel[(1,)](k_bh, state_bh, old_v, V=V, K=K)

                # 2) state_remove = k @ (g * state)
                g_val = torch.empty(1, device=device, dtype=torch.float32)
                # g_out[pid] is scalar g for this (b,h)
                g_scalar = g_out[pid]
                tl.store(g_val, g_scalar)
                state_remove = torch.empty(1, device=device, dtype=torch.float32)
                dot_k_gstate_kernel[(1,)](k_bh, state_bh, g_val, state_remove, V=V, K=K)

                # 3) state_update = k @ (beta * v + (1 - beta) * old_v)
                beta_val = beta_out[pid]
                old_v_val = old_v[0]
                state_update = torch.empty(1, device=device, dtype=torch.float32)
                dot_k_newv_kernel[(1,)](k_bh, v_bh, old_v_val, beta_val, state_update, V=V, K=K)

                # 4) h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update
                h_state = torch.empty(V, device=device, dtype=torch.float32)
                h_state_vec_kernel[(1,)](state_bh, g_scalar, state_remove, state_update, h_state, V=V, K=K)

                # 5) output[b,h] = scale * q[b,h] @ h_state
                q_bh = q[b_idx, 0, h_idx]  # [K]
                output_f32[pid] = scale * dot_q_hstate_kernel[(1,)](q_bh, h_state, output_f32.new_empty(1), V=V, K=K)[0]

                # 6) write new_state[b,h] = h_state_vec broadcast across K
                new_state_bh = new_state[b_idx, h_idx]  # [V, K]
                write_new_state_kernel[(1,)](h_state, new_state_bh, V=V, K=K)

        # Return output as bfloat16 with shape [B,1,H]
        output = output_f32.view(B, H).unsqueeze(1)  # [B,1,H]
        return output.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
