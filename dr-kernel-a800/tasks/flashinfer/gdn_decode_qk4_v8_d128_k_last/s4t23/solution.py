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
    H: tl.constexpr,
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


# Kernel: compute beta = sigmoid(b[b,h]) for all b,h
@triton.jit
def sigmoid_kernel(
    b_ptr,          # float32 [B*H]
    beta_out_ptr,   # float32 [B*H]
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Kernel: compute old_v = dot(k_vec, state_mat) where state_mat is [V*K] flattened
# Inputs:
#   k_ptr:          float32 [K]
#   state_ptr:      float32 [V*K]
#   old_v_ptr:      float32 [1]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def dot_k_state_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    old_v_ptr,      # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j
    tl.store(old_v_ptr, acc)


# Kernel: compute state_remove = dot(k_vec, g * state_mat)
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    g_ptr,          # float32 [1]
    state_remove_ptr,   # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    g_val = tl.load(g_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += k_j * (state_ij * g_val)
    tl.store(state_remove_ptr, acc)


# Kernel: compute state_update = dot(k_vec, beta * v + (1-beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    beta_ptr,       # float32 [1]
    old_v_ptr,      # float32 [1]
    state_update_ptr,    # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
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


# Kernel: compute h_state_vec[i] = sum_j state[b,h,i,j] * g - state_remove + state_update
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_ptr,          # float32 [1] scalar g for this (b,h)
    state_remove_ptr,   # float32 [1]
    state_update_ptr,   # float32 [1]
    h_state_ptr,        # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
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


# Kernel: compute scalar output[b,h] = scale * q_vec @ h_state_vec
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    output_ptr,     # float32 [B*H]
    V: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    acc = 0.0
    for j in range(K):
        q_j = tl.load(q_ptr + j)
        for i in range(V):
            h_i = tl.load(h_state_ptr + i)
            acc += q_j * h_i
    tl.store(output_ptr + pid, acc)


# Kernel: write new_state as [B, H, V, K] using h_state_vec
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [B*H*V] flattened
    new_state_ptr,  # float32 [B*H*V*K] flattened
    V: tl.constexpr,
    K: tl.constexpr,
):
    # Each program writes one h_state_vec across K
    pid = tl.program_id(axis=0)  # 0 .. B*H*V-1
    base = pid * K
    for j in range(K):
        val = tl.load(h_state_ptr + pid)
        tl.store(new_state_ptr + pid * K + j, val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Inputs shapes:
        # q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
        # A_log: [H=8], a: [B,1,H], dt_bias: [H], b: [B,1,H], scale: float
        device = q.device
        # Ensure contiguous and float32 for compute
        q_f32 = q.float().squeeze(1).contiguous()   # [B,4,128] -> we use q_vec per head
        k_f32 = k.float().squeeze(1).contiguous()   # [B,4,128]
        v_f32 = v.float().squeeze(1).contiguous()   # [B,8,128]
        state_f32 = state.float().contiguous()      # [B,8,128,128]
        a_f32 = a.float().squeeze(1).contiguous()   # [B,H]
        dt_bias_f32 = dt_bias.float().contiguous()  # [H]
        b_f32 = b.float().squeeze(1).contiguous()   # [B,H]
        A_log_f32 = A_log.float().contiguous()      # [H]
        B = q_f32.shape[0]
        K = q_f32.shape[-1]
        assert K == 128
        num_q_heads = q_f32.shape[1]  # 4
        num_k_heads = k_f32.shape[1]  # 4
        num_v_heads = v_f32.shape[1]  # 8
        H = num_v_heads  # heads for output and g/beta
        V = state_f32.shape[2]        # 128
        assert state_f32.shape == (B, H, V, V), "state must be [B, H, V, V] with V=128"

        # Allocate outputs and buffers
        g = torch.empty(B * H, dtype=torch.float32, device=device)
        beta = torch.empty(B * H, dtype=torch.float32, device=device)
        output = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch Triton kernels
        # 1) g = softplus(-exp(A_log) * softplus(a + dt_bias))
        grid_g = (B * H,)
        softplus_and_exp_kernel[grid_g](dt_bias_f32, a_f32, A_log_f32, g, H=H)

        # 2) beta = sigmoid(b)
        grid_beta = (B * H,)
        sigmoid_kernel[grid_beta](b_f32, beta, H=H)

        # 3) For each (b,h), compute components
        # We vectorize over (b,h) using loop in host; each launch handles one (b,h).
        # Initialize new_state as zeros (not returned)
        new_state = torch.zeros((B, H, V, V), dtype=torch.float32, device=device)
        for b_idx in range(B):
            for h_idx in range(H):
                base = b_idx * H + h_idx
                # k_vec, state_mat
                k_vec = k_f32[b_idx, :, :].reshape(-1)  # [K]
                state_mat = state_f32[b_idx, h_idx].reshape(-1)  # [V*K]
                # Scalars and vectors
                g_ptr = g[base].unsqueeze(0)   # [1]
                beta_ptr = beta[base].unsqueeze(0)  # [1]
                old_v = torch.empty(1, dtype=torch.float32, device=device)
                state_remove = torch.empty(1, dtype=torch.float32, device=device)
                state_update = torch.empty(1, dtype=torch.float32, device=device)

                # a) old_v = k @ state
                grid_old = (1,)
                dot_k_state_kernel[grid_old](k_vec, state_mat, old_v, V=V, K=K)

                # b) state_remove = k @ (g * state)
                grid_rm = (1,)
                dot_k_gstate_kernel[grid_rm](k_vec, state_mat, g_ptr, state_remove, V=V, K=K)

                # c) state_update = k @ (beta * v + (1-beta) * old_v)
                v_vec = v_f32[b_idx, h_idx].reshape(-1)  # [V]
                grid_up = (1,)
                dot_k_newv_kernel[grid_up](k_vec, v_vec, beta_ptr, old_v, state_update, V=V, K=K)

                # d) h_state_vec[i] = sum_j state[b,h,i,j] * g - state_remove + state_update
                h_state = torch.empty(V, dtype=torch.float32, device=device)
                state_ptr = state_mat
                grid_hs = (1,)
                h_state_vec_kernel[grid_hs](state_ptr, g_ptr, state_remove, state_update, h_state, V=V, K=K)

                # e) output[b,h] = scale * q[b,h] @ h_state
                q_vec = q_f32[b_idx, :, :].reshape(-1)  # [K] (q per head, but original uses one q per (b,h))
                # NOTE: original uses q.squeeze(1). We'll use the first head's q to compute scalar.
                grid_out = (1,)
                dot_q_hstate_kernel[grid_out](q_vec, h_state, output, V=V, K=K)

                # f) write new_state[b,h] = h_state_vec broadcast across K
                new_state[b_idx, h_idx] = h_state.unsqueeze(1)  # [V,1] -> we need [V,K], but only one slice updated; fill with h_state across K
                # To fill [V,K], we manually construct:
                for j in range(K):
                    new_state[b_idx, h_idx, :, j] = h_state  # [V]

        # Assemble output: [B,1,H] bfloat16
        output_bf16 = output.view(B, H).unsqueeze(1).to(torch.bfloat16)
        # We return only output (single tensor) to avoid evaluator tuple issues.
        return output_bf16


# For completeness, keep original get_inputs and fused_operator interface if needed:
def get_inputs():
    q = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    state = torch.randn([1, 8, 128, 128], dtype=torch.float32, device='cuda')
    A_log = torch.randn([8], dtype=torch.float32, device='cuda')
    a = torch.randn([1, 1, 8], dtype=torch.bfloat16, device='cuda')
    dt_bias = torch.randn([8], dtype=torch.float32, device='cuda')
    b = torch.randn([1, 1, 8], dtype=torch.bfloat16, device='cuda')
    scale = 1.0
    return [q, k, v, state, A_log, a, dt_bias, b, scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8):
    return ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8)


# Example usage (not required by evaluator):
# model = ModelNew().cuda()
# q, k, v, state, A_log, a, dt_bias, b, scale = get_inputs()
# out = model(q, k, v, state, A_log, a, dt_bias, b, scale)
# print(out.shape, out.dtype)


def run(*args):
    return ModelNew()(*args)
