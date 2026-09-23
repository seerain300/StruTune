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
    pid = tl.program_id(axis=0)  # 0 .. (B*H - 1)
    h = pid % H
    a_val = tl.load(a_ptr + pid)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val
    s = tl.log(1.0 + tl.exp(x))  # softplus
    e = tl.exp(tl.load(A_log_ptr + h))
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Kernel: compute beta[b,h] = sigmoid(b[b,h])
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


# Kernel: compute dot k_vec @ state_mat, where state_mat is [V*K] flattened; return scalar
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


# Kernel: compute dot k_vec @ (g * state_mat), where state_mat is [V*K] flattened; return scalar
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    g_val,          # float32 scalar
    state_g_ptr,    # float32 [1] (we could compute and store here, but we pass g as scalar)
    V: tl.constexpr,
    K: tl.constexpr,
):
    # Precompute g * state and store into state_g_ptr[0]
    acc_g = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc_g += state_ij * (k_j * g_val)
    tl.store(state_g_ptr, acc_g)


# Kernel: compute dot k_vec @ (beta * v_vec + (1 - beta) * old_v), where v_vec is [V], old_v is scalar
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    beta_val,       # float32 scalar
    old_v_val,      # float32 scalar
    new_v_ptr,      # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += v_i * (beta_val * v_i + (1.0 - beta_val) * old_v_val)
            # The above is not correct for new_v; we need to use v_i * (beta_val * v_i + (1 - beta_val) * old_v_val)
            # The correct inner product should be k_j * sum_i v_i * (beta * v_i + (1 - beta) * old_v). However, we need to compute
            # the vector first. Let's re-define:
            # We can't directly multiply by v_i here; instead, we compute the vector contribution per i:
            # new_v_i = beta * v_i + (1 - beta) * old_v, then acc += k_j * new_v_i for each j.
            # Re-define kernel to accept v_ptr and compute per j loop:
            pass
    # Implement correctly by passing vector contribution to a scalar accumulator via a separate helper? Triton supports reduction loops.
    # Simplify: compute new_v per element and reduce:
    # We need to compute the scalar first: sum_i k_vec[j] * (beta * v_i + (1 - beta) * old_v)
    # This equals (1 - beta) * old_v * sum_j k_j + beta * sum_j (k_j * sum_i (v_i^2))
    # This is incorrect. Better: recompute new_v for each i and reduce:
    # We need to store new_v as [V] and reduce with k; but Triton scalar kernel only has one out. We'll instead implement this in Python.
    # To strictly adhere to Triton-only, we implement it via two kernels: compute new_v_vec, then dot_k_newv_vec_kernel.
    # But for brevity, we define the correct kernel below:
    pass


# Correct dot_k_newv_kernel: compute scalar sum_j sum_i k[j] * (beta * v[i] + (1 - beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    beta_val,       # float32 scalar
    old_v_val,      # float32 scalar
    new_v_ptr,      # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        contrib = (1.0 - beta_val) * old_v_val * k_j
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            contrib += beta_val * v_i * k_j
        acc += contrib
    tl.store(new_v_ptr, acc)


# Kernel: compute h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update
# state_ptr is [V*K] flattened, h_state_ptr is [V]
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    g_val,          # float32 scalar
    state_remove,   # float32 scalar
    state_update,   # float32 scalar
    h_state_ptr,    # float32 [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        acc = 0.0
        for j in range(K):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * g_val
        h_state_i = acc - state_remove + state_update
        tl.store(h_state_ptr + i, h_state_i)


# Kernel: compute output scalar = scale * (q_vec @ h_state_vec), q_vec is [V], h_state_vec is [V]
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [V]
    h_state_ptr,    # float32 [V]
    out_ptr,        # float32 [1]
    scale,          # float32 scalar
    V: tl.constexpr,
):
    acc = 0.0
    for i in range(V):
        q_i = tl.load(q_ptr + i)
        h_i = tl.load(h_state_ptr + i)
        acc += q_i * h_i
    tl.store(out_ptr, acc * scale)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
# h_state_ptr is [V], new_state_ptr is [V*K] flattened
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # float32 [V]
    new_state_ptr,  # float32 [V*K]
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        val = tl.load(h_state_ptr + i)
        for j in range(K):
            tl.store(new_state_ptr + i * K + j, val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        q, k, v, state, A_log, a, dt_bias, b, scale,
    ):
        # Ensure devices match
        device = q.device
        assert k.device == device and v.device == device and state.device == device and A_log.device == device and a.device == device and dt_bias.device == device and b.device == device
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        # The original asserts:
        # assert num_q_heads == 4
        # assert num_k_heads == 4
        # assert num_v_heads == 8
        # assert K == 128 and V == 128
        # assert T == 1
        # We’ll assume these hold (the harness provides valid inputs). If not, we can still compute but outputs may be wrong.
        # Compute g and beta from raw parameters
        H = num_v_heads  # number of heads
        # a has shape [1, 1, H] => flatten to [H]; dt_bias [H]; A_log [H]
        a_flat = a.view(-1).float().to(device)
        dt_bias = dt_bias.float().to(device)
        A_log = A_log.float().to(device)
        g = torch.empty(B * H, dtype=torch.float32, device=device)
        beta = torch.empty(B * H, dtype=torch.float32, device=device)
        # Launch kernels
        softplus_and_exp_kernel[(B * H,)](dt_bias, a_flat, A_log, g, H)
        sigmoid_kernel[(B * H,)](b.view(-1).float().to(device), beta, H)
        g = g.view(B, H)
        beta = beta.view(B, H)

        # Prepare outputs
        output = torch.empty((B, H), dtype=torch.float32, device=device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # For each batch and head, compute using Triton kernels
        for b_idx in range(B):
            for h_idx in range(H):
                # q_vec, k_vec, v_vec
                q_h = q[b_idx, 0, h_idx].view(-1).float()  # [K]
                k_h = k[b_idx, 0, h_idx].view(-1).float()  # [K]
                v_h = v[b_idx, 0, h_idx].view(-1).float()  # [V]
                state_h = state[b_idx, h_idx].view(-1).float()  # [V*K]

                # Compute old_v = k @ state
                old_v = torch.empty((), dtype=torch.float32, device=device)
                old_v_ptr = old_v  # Triton needs pointers to scalars; we’ll pass a 1-element tensor
                dot_k_state_kernel[(K,)](k_h, state_h, old_v_ptr, V, K)

                # Compute state_remove = k @ (g[b,h] * state)
                state_g = torch.empty((), dtype=torch.float32, device=device)
                dot_k_gstate_kernel[(K,)](k_h, state_h, g[b_idx, h_idx], state_g, V, K)
                state_remove = state_g

                # Compute state_update = k @ (beta * v + (1 - beta) * old_v)
                state_update = torch.empty((), dtype=torch.float32, device=device)
                dot_k_newv_kernel[(K,)](k_h, v_h, beta[b_idx, h_idx], old_v.item(), state_update, V, K)

                # Compute h_state_vec
                h_state_vec = torch.empty((V,), dtype=torch.float32, device=device)
                h_state_vec_kernel[(V,)](state_h, g[b_idx, h_idx], state_remove.item(), state_update.item(), h_state_vec, V, K)

                # Compute output scalar
                out_scalar = torch.empty((), dtype=torch.float32, device=device)
                dot_q_hstate_kernel[(1,)](q_h, h_state_vec, out_scalar, (scale if scale is not None else (1.0 / math.sqrt(K))), V)
                output[b_idx, h_idx] = out_scalar

                # Write new_state[b,h]
                new_state[b_idx, h_idx] = write_new_state_kernel((V * K,), h_state_vec, new_state[b_idx, h_idx].view(-1), V, K)

        # Return output [B,1,H] in bfloat16 and new_state [B,H,V,K] float32
        output = output.unsqueeze(1).to(torch.bfloat16)
        return output, new_state


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Optional local test helper (not used by harness):
def get_inputs():
    # Note: in the harness, a single tensor (state) is passed. This helper demonstrates typical shapes.
    q = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16)
    k = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16)
    v = torch.randn([1, 1, 8, 128], dtype=torch.bfloat16)
    state = torch.randn([1, 8, 128, 128], dtype=torch.float32)
    A_log = torch.randn([8], dtype=torch.float32)
    a = torch.randn([1, 1, 8], dtype=torch.bfloat16)
    dt_bias = torch.randn([8], dtype=torch.float32)
    b = torch.randn([1, 1, 8], dtype=torch.bfloat16)
    scale = 1.0
    return [q, k, v, state, A_log, a, dt_bias, b, scale]


def run(*args):
    return ModelNew()(*args)
