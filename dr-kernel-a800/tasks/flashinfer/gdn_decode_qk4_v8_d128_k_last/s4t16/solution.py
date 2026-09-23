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


# Kernel: compute beta = sigmoid(b[b,h])
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
    g_scalar,       # float32
    state_ptr,      # float32 [V*K]
    state_remove_ptr,# float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j * g_scalar
    tl.store(state_remove_ptr, acc)


# Kernel: compute state_update = dot(k_vec, beta * v_vec + (1-beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    beta_scalar,    # float32
    v_ptr,          # float32 [V]
    old_v_scalar,   # float32
    state_update_ptr,# float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (beta_scalar * v_i + (1.0 - beta_scalar) * old_v_scalar)
    tl.store(state_update_ptr, acc)


# Kernel: compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
# Inputs:
#   state_ptr:      float32 [V*K]
#   g_scalar:       float32
#   state_remove:   float32 scalar
#   state_update:   float32 scalar
#   h_state_ptr:    float32 [V]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_scalar,       # float32
    state_remove,   # float32
    state_update,   # float32
    h_state_ptr,    # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        acc = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * g_scalar
        h_state_ptr[i] = acc - state_remove + state_update


# Kernel: output[b,h] = scale * dot(q_vec, h_state_vec)
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [V]
    h_state_ptr,    # float32 [V]
    out_ptr,        # float32 [1]
    scale,          # float32
    V: tl.constexpr,
):
    acc = 0.0
    for i in range(V):
        q_i = tl.load(q_ptr + i)
        h_i = tl.load(h_state_ptr + i)
        acc += q_i * h_i
    tl.store(out_ptr, acc * scale)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [V*K]
    K: tl.constexpr,
    V: tl.constexpr,
):
    for i in range(V):
        val = tl.load(h_state_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, val)


class ModelNew(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        # Capture inputs; forward will use them
        # Note: harness will not call __init__, but this keeps signature intact.
        self.q = args[0]
        self.k = args[1]
        self.v = args[2]
        self.state = args[3]
        self.A_log = args[4]
        self.a = args[5]
        self.dt_bias = args[6]
        self.b = args[7]
        self.scale = args[8]

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Allocate outputs and run Triton kernels; no torch ops here.
        # Convert to float32 for compute
        q_f32 = q.float().contiguous()
        k_f32 = k.float().contiguous()
        v_f32 = v.float().contiguous()
        a_f32 = a.float().contiguous()
        b_f32 = b.float().contiguous()
        state_f32 = state.float().contiguous()
        A_log_f32 = A_log.float().contiguous()
        dt_bias_f32 = dt_bias.float().contiguous()

        # B, H, V, K derived from inputs
        B, T, num_q_heads, K = q_f32.shape
        _, _, num_k_heads, _ = k_f32.shape
        _, _, num_v_heads, V = v_f32.shape
        H = num_v_heads  # per original: H=8
        assert T == 1
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8 and K == 128 and V == 128

        # Compute g and beta (sizes: [B*H])
        g = torch.empty(B * H, dtype=torch.float32, device=q_f32.device)
        beta = torch.empty(B * H, dtype=torch.float32, device=q_f32.device)

        grid_g = (B * H,)
        softplus_and_exp_kernel[grid_g](dt_bias_f32, a_f32, A_log_f32, g, H)

        grid_beta = (B * H,)
        sigmoid_kernel[grid_beta](b_f32, beta, H)

        # Prepare output and new_state
        output = torch.empty(B, 1, H, dtype=torch.float32, device=q_f32.device)  # will be cast to bfloat16
        new_state = torch.empty(B, H, V, K, dtype=torch.float32, device=q_f32.device)

        # Launch per-(b,h) computations
        for b_idx in range(B):
            for h_idx in range(H):
                base = b_idx * H + h_idx
                # Allocate scalars
                old_v_buf = torch.empty(1, dtype=torch.float32, device=q_f32.device)
                state_remove_buf = torch.empty(1, dtype=torch.float32, device=q_f32.device)
                state_update_buf = torch.empty(1, dtype=torch.float32, device=q_f32.device)
                # h_state_vec [V]
                h_state = torch.empty(V, dtype=torch.float32, device=q_f32.device)

                # k_vec [K], state_mat [V*K]
                k_vec = k_f32[b_idx, 0, h_idx].contiguous()  # shape [K]
                state_mat = state_f32[b_idx, h_idx].contiguous().view(V * K)  # [V*K]

                # 1) old_v = k @ state
                grid_dot1 = (1,)
                dot_k_state_kernel[grid_dot1](k_vec, state_mat, old_v_buf, V, K)

                # 2) state_remove = k @ (g * state)
                g_scalar = g[base]
                grid_dot2 = (1,)
                dot_k_gstate_kernel[grid_dot2](k_vec, g_scalar, state_mat, state_remove_buf, V, K)

                # 3) state_update = k @ (beta * v + (1-beta) * old_v)
                beta_scalar = beta[base]
                v_vec = v_f32[b_idx, 0, h_idx].contiguous()  # [V]
                old_v_scalar = old_v_buf[0]
                grid_dot3 = (1,)
                dot_k_newv_kernel[grid_dot3](k_vec, beta_scalar, v_vec, old_v_scalar, state_update_buf, V, K)

                # 4) h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
                grid_vec = (V,)
                h_state_vec_kernel[grid_vec](state_mat, g_scalar, state_remove_buf[0], state_update_buf[0], h_state, V, K)

                # 5) output[b,h] = scale * (q @ h_state)
                q_vec = q_f32[b_idx, 0, h_idx].contiguous()  # [V]
                out_buf = torch.empty(1, dtype=torch.float32, device=q_f32.device)
                grid_q = (V,)
                dot_q_hstate_kernel[grid_q](q_vec, h_state, out_buf, float(scale), V)
                output[b_idx, 0, h_idx] = out_buf[0]

                # 6) write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
                new_state_mat = new_state[b_idx, h_idx].contiguous().view(V * K)
                grid_write = (V, K)
                write_new_state_kernel[grid_write](h_state, new_state_mat, K, V)

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
