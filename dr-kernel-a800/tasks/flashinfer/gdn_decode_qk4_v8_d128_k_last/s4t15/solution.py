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
    s = tl.log(1.0 + tl.exp(x))   # softplus
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
# Inputs:
#   k_ptr:          float32 [K]
#   g_scalar_ptr:   float32 [1] (scalar for this (b,h))
#   state_ptr:      float32 [V*K]
#   state_remove_ptr: float32 [1]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    g_scalar_ptr,   # float32 [1] scalar
    state_ptr,      # float32 [V*K]
    state_remove_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    g_scalar = tl.load(g_scalar_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j * g_scalar
    tl.store(state_remove_ptr, acc)


# Kernel: compute state_update = dot(k_vec, beta * v_vec + (1 - beta) * old_v_scalar)
# Inputs:
#   k_ptr:          float32 [K]
#   beta_ptr:       float32 [1] (scalar for this (b,h))
#   v_ptr:          float32 [V]
#   old_v_scalar_ptr: float32 [1]
#   state_update_ptr: float32 [1]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    beta_ptr,       # float32 [1] (scalar)
    v_ptr,          # float32 [V]
    old_v_scalar_ptr,  # float32 [1]
    state_update_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    beta = tl.load(beta_ptr)
    old_v = tl.load(old_v_scalar_ptr)
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (beta * v_i + (1.0 - beta) * old_v)
    tl.store(state_update_ptr, acc)


# Kernel: compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
# Inputs:
#   state_ptr:      float32 [V*K]
#   g_scalar_ptr:   float32 [1]
#   state_remove_ptr: float32 [1]
#   state_update_ptr: float32 [1]
#   h_state_ptr:    float32 [V]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_scalar_ptr,   # float32 [1]
    state_remove_ptr,    # float32 [1]
    state_update_ptr,    # float32 [1]
    h_state_ptr,    # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    g_scalar = tl.load(g_scalar_ptr)
    state_remove = tl.load(state_remove_ptr)
    state_update = tl.load(state_update_ptr)
    for i in range(V):
        acc = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * g_scalar
        h_state_i = acc - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_i)


# Kernel: compute output_scalar[b,h] = scale * dot(q_vec, h_state_vec)
# Inputs:
#   q_ptr:          float32 [K]
#   h_state_ptr:    float32 [V]
#   output_ptr:     float32 [1]
#   scale_scalar_ptr: float32 [1]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    output_ptr,     # float32 [1]
    scale_ptr,      # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    scale = tl.load(scale_ptr)
    acc = 0.0
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        for j in range(K):
            q_j = tl.load(q_ptr + j)
            acc += h_i * q_j
    tl.store(output_ptr, acc * scale)


# Kernel: write new_state[b,h] as [V,K], broadcasting h_state_vec across K
# Inputs:
#   h_state_ptr:    float32 [V]
#   new_state_ptr:  float32 [V*K] (row-major)
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [V*K]
    V: tl.constexpr,
    K: tl.constexpr,
):
    # one program per row i, across K columns
    i = tl.program_id(axis=0)  # 0 .. V-1
    if i >= V:
        return
    for j in range(K):
        h_i = tl.load(h_state_ptr + i)
        tl.store(new_state_ptr + i * K + j, h_i)


class ModelNew(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        # Expect 9 positional arguments: q,k,v,state,A_log,a,dt_bias,b,scale
        assert len(args) == 9, "Expected 9 positional arguments: q,k,v,state,A_log,a,dt_bias,b,scale"
        self.q, self.k, self.v, self.state, self.A_log, self.a, self.dt_bias, self.b, self.scale = args

        # Shapes (as in original assertions)
        B, T_q, num_q_heads, K = self.q.shape
        _, T_k, num_k_heads, _ = self.k.shape
        _, _, num_v_heads, V = self.v.shape
        assert T_q == 1 and T_k == 1, "T must be 1"
        assert num_q_heads == 4 and num_k_heads == 4 and num_v_heads == 8
        assert K == 128 and V == 128

        self.B = B
        self.H = num_v_heads  # heads dimension
        self.V = V
        self.K = K

    def forward(self):
        device = self.q.device
        dtype_compute = torch.float32

        # Convert inputs to float32 for stable computation
        q_f32 = self.q.float()                      # [B,1,4,128]
        k_f32 = self.k.float()                     # [B,1,4,128]
        v_f32 = self.v.float()                     # [B,1,8,128]
        if self.state is None:
            state_f32 = torch.zeros(self.B, self.H, self.V, self.K, dtype=dtype_compute, device=device)
        else:
            state_f32 = self.state.float()

        # Prepare parameter tensors
        A_log = self.A_log.float()                # [H]
        a = self.a.float()                        # [B,1,H]
        dt_bias = self.dt_bias.float()            # [H]
        b = self.b.float()                        # [B,1,H]

        # Compute g[b,h] and beta[b,h]
        g = torch.empty(self.B * self.H, dtype=dtype_compute, device=device)
        beta = torch.empty(self.B * self.H, dtype=dtype_compute, device=device)

        grid_g = (self.B * self.H,)
        softplus_and_exp_kernel[grid_g](dt_bias, a.reshape(-1).contiguous(), A_log, g, H=self.H)

        beta_kernel = sigmoid_kernel[grid_g](b.reshape(-1).contiguous(), beta, H=self.H)

        # Allocate outputs
        output_scalar = torch.empty((self.B, self.H), dtype=dtype_compute, device=device)
        new_state = torch.empty((self.B, self.H, self.V, self.K), dtype=dtype_compute, device=device)

        # Compute scale if None or 0.0
        scale = self.scale
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(self.K)
        scale_tensor = torch.tensor([scale], dtype=dtype_compute, device=device)

        # For each (b,h): perform Triton dot computations and write outputs
        for b_idx in range(self.B):
            for h_idx in range(self.H):
                # 1) Compute old_v = k[b,h] @ state[b,h]
                k_vec = k_f32[b_idx, 0, h_idx].contiguous()           # [K]
                state_bh = state_f32[b_idx, h_idx].contiguous()       # [V,K]
                old_v = torch.empty((), dtype=dtype_compute, device=device)  # 1-element buffer
                dot_k_state_kernel[(1,)](k_vec, state_bh.reshape(-1), old_v, V=self.V, K=self.K)

                # 2) Compute state_remove = k @ (g * state)
                g_scalar = torch.empty((), dtype=dtype_compute, device=device)
                g_scalar[:] = g[b_idx * self.H + h_idx]
                state_remove = torch.empty((), dtype=dtype_compute, device=device)
                dot_k_gstate_kernel[(1,)](k_vec, g_scalar, state_bh.reshape(-1), state_remove, V=self.V, K=self.K)

                # 3) Compute state_update = k @ (beta * v + (1 - beta) * old_v)
                beta_scalar = torch.empty((), dtype=dtype_compute, device=device)
                beta_scalar[:] = beta[b_idx * self.H + h_idx]
                v_vec = v_f32[b_idx, 0, h_idx].contiguous()           # [V]
                state_update = torch.empty((), dtype=dtype_compute, device=device)
                dot_k_newv_kernel[(1,)](k_vec, beta_scalar, v_vec, old_v, state_update, V=self.V, K=self.K)

                # 4) Compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
                h_state = torch.empty(self.V, dtype=dtype_compute, device=device)
                h_state_vec_kernel[(1,)](state_bh.reshape(-1), g_scalar, state_remove, state_update, h_state, V=self.V, K=self.K)

                # 5) Compute output_scalar[b,h] = scale * q[b,h] @ h_state
                q_vec = q_f32[b_idx, 0, h_idx].contiguous()           # [K]
                out = torch.empty((), dtype=dtype_compute, device=device)
                dot_q_hstate_kernel[(1,)](q_vec, h_state, out, scale_tensor, V=self.V, K=self.K)
                output_scalar[b_idx, h_idx] = out[0]

                # 6) Write new_state[b,h] as [V,K], broadcasting h_state_vec across K
                new_state_row_ptr = new_state[b_idx, h_idx].reshape(-1)  # [V*K]
                write_new_state_kernel[(self.V,)](h_state, new_state_row_ptr, V=self.V, K=self.K)

        # Return output [B,1,H] in bfloat16 and new_state [B,H,V,K]
        output = output_scalar.unsqueeze(1).to(torch.bfloat16)        # [B,1,H] bfloat16
        return output, new_state


def run(*args):
    return ModelNew()(*args)
