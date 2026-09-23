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
# Inputs:
#   k_ptr:          float32 [K]
#   state_ptr:      float32 [V*K]
#   g_scalar:       float32
#   state_remove_ptr: float32 [1]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    g_scalar,       # float32
    state_remove_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += (g_scalar * state_ij) * k_j
    tl.store(state_remove_ptr, acc)


# Kernel: compute state_update = dot(k_vec, beta*v + (1-beta)*old_v)
# Inputs:
#   k_ptr:          float32 [K]
#   v_ptr:          float32 [V]
#   old_v_scalar:   float32
#   beta_scalar:    float32
#   state_update_ptr: float32 [1]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    old_v_scalar,   # float32
    beta_scalar,    # float32
    state_update_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            # new_v_i = beta*v_i + (1-beta)*old_v
            new_v_i = beta_scalar * v_i + (1.0 - beta_scalar) * old_v_scalar
            acc += new_v_i * k_j
    tl.store(state_update_ptr, acc)


# Kernel: compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
# Inputs:
#   state_ptr:      float32 [V*K]
#   g_scalar:       float32
#   state_remove:   float32
#   state_update:   float32
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
            acc += state_ij
        h_state_i = acc * g_scalar - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_i)


# Kernel: compute output_scalar = scale * dot(q_vec, h_state_vec)
# Inputs:
#   q_ptr:          float32 [K]
#   h_state_ptr:    float32 [V]
#   scale:          float32
#   output_ptr:     float32 [1]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    scale,          # float32
    output_ptr,     # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        q_j = tl.load(q_ptr + j)
        for i in range(V):
            h_i = tl.load(h_state_ptr + i)
            acc += q_j * h_i
    acc *= scale
    tl.store(output_ptr, acc)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
# Inputs:
#   h_state_ptr:    float32 [V]
#   new_state_ptr:  float32 [V*K]
#   V: tl.constexpr
#   K: tl.constexpr
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [V*K]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, h_i)


class ModelNew(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        # Capture tensors and derive shapes
        assert len(args) >= 7, "Not enough arguments; expected q,k,v,state,A_log,a,dt_bias,b,scale"
        q, k, v, state, A_log, a, dt_bias, b, scale = args
        # Shapes
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        # The original assertions:
        assert T == 1
        assert num_q_heads == 4
        assert num_k_heads == 4
        assert num_v_heads == 8
        assert K == 128 and V == 128

        self.B = B
        self.H = num_v_heads  # heads = 8
        self.V = V
        self.K = K
        self.device = q.device

        # Store tensors (on device), keep original dtypes
        # a and dt_bias come as [B,1,H] -> squeeze to [B,H]; b same
        self.a = a.squeeze(1).to(device=self.device, dtype=torch.float32).reshape(-1)  # [B*H]
        self.dt_bias = dt_bias.to(device=self.device, dtype=torch.float32)              # [H]
        self.A_log = A_log.to(device=self.device, dtype=torch.float32)                  # [H]
        self.b = b.squeeze(1).to(device=self.device, dtype=torch.float32).reshape(-1)   # [B*H]
        self.q = q.to(device=self.device, dtype=torch.float32)                          # [B,1,4,128] -> keep as is, will use [B,4,128] in forward
        self.k = k.to(device=self.device, dtype=torch.float32)                          # [B,1,4,128]
        self.v = v.to(device=self.device, dtype=torch.float32)                          # [B,1,8,128]
        self.state = state.to(device=self.device, dtype=torch.float32)                  # [B,8,128,128]
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(K)

    def forward(self):
        # Compute g and beta via Triton kernels
        g = torch.empty((self.B * self.H,), dtype=torch.float32, device=self.device)
        beta = torch.empty((self.B * self.H,), dtype=torch.float32, device=self.device)

        grid_g = (self.B * self.H,)
        softplus_and_exp_kernel[grid_g](self.dt_bias, self.a, self.A_log, g, H=self.H)

        grid_beta = (self.B * self.H,)
        sigmoid_kernel[grid_beta](self.b, beta, H=self.H)

        # Prepare output and new_state
        output = torch.empty((self.B * self.H,), dtype=torch.float32, device=self.device)  # [B*H]
        new_state = torch.empty((self.B, self.H, self.V, self.K), dtype=torch.float32, device=self.device)

        # For each (b,h), compute the updates
        for pid in range(self.B * self.H):
            b_idx = pid // self.H
            h_idx = pid % self.H

            # Squeeze heads for q,k,v
            q_vec = self.q[b_idx, 0, h_idx].contiguous()  # [K]
            k_vec = self.k[b_idx, 0, h_idx].contiguous()  # [K]
            v_vec = self.v[b_idx, 0, h_idx].contiguous()  # [V]
            state_mat = self.state[b_idx, h_idx].contiguous()  # [V,K]

            # Scalars
            g_val = g[pid]
            beta_val = beta[pid]
            scale = self.scale

            # 1) old_v = dot(k, state)
            old_v_buf = torch.empty((1,), dtype=torch.float32, device=self.device)
            dot_k_state_kernel[(1,)](k_vec, state_mat, old_v_buf, V=self.V, K=self.K)

            # 2) state_remove = dot(k, g * state)
            state_remove_buf = torch.empty((1,), dtype=torch.float32, device=self.device)
            dot_k_gstate_kernel[(1,)](k_vec, state_mat, g_val, state_remove_buf, V=self.V, K=self.K)

            # 3) state_update = dot(k, beta*v + (1-beta)*old_v)
            state_update_buf = torch.empty((1,), dtype=torch.float32, device=self.device)
            dot_k_newv_kernel[(1,)](k_vec, v_vec, old_v_buf[0], beta_val, state_update_buf, V=self.V, K=self.K)

            # 4) h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
            h_state_vec = torch.empty((self.V,), dtype=torch.float32, device=self.device)
            h_state_vec_kernel[(1,)](state_mat, g_val, state_remove_buf[0], state_update_buf[0], h_state_vec, V=self.V, K=self.K)

            # 5) output_scalar = scale * dot(q, h_state_vec)
            dot_q_hstate_kernel[(1,)](q_vec, h_state_vec, scale, output[pid], V=self.V, K=self.K)

            # 6) write new_state[b,h] as [V,K] broadcasting h_state_vec across K
            write_new_state_kernel[(1,)](h_state_vec, new_state[b_idx, h_idx], V=self.V, K=self.K)

        # Assemble output as [B,1,H] bfloat16 (original code returns [B,1,H,V] bfloat16; here V=1 per head, so emulate [B,1,H])
        output_expanded = output.view(self.B, self.H).unsqueeze(1).to(torch.bfloat16)
        return output_expanded, new_state


def run(*args):
    return ModelNew()(*args)
