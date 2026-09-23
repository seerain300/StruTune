import math
import torch
import triton
import triton.language as tl


# Kernel: compute g = exp(-exp(A_log[h]) * softplus(x)), where x = a[b,h] + dt_bias[h]
# softplus(x) = log(1 + exp(x))
@triton.jit
def softplus_and_exp_kernel(
    dt_bias_ptr,    # float32 [H]
    a_ptr,          # float32 [B*H] (a.squeeze(1) flattened)
    A_log_ptr,      # float32 [H]
    g_out_ptr,      # float32 [B*H]
    B: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H - 1
    h = pid % H
    b = pid // H
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
    b_ptr,          # float32 [B*H] (b.squeeze(1) flattened)
    beta_out_ptr,   # float32 [B*H]
    B: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    h = pid % H
    b = pid // H
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Kernel: compute old_v = dot(k[b,h], state[b,h]) where state is flattened [V*K]
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


# Kernel: compute state_remove = dot(k[b,h], g[b,h] * state[b,h])
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # float32 [K]
    state_ptr,      # float32 [V*K]
    g_val,          # float32 scalar g[b,h]
    state_remove_ptr,  # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j * g_val
    tl.store(state_remove_ptr, acc)


# Kernel: compute state_update = dot(k[b,h], beta[b,h] * v[b,h] + (1 - beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    old_v_val,      # float32 scalar
    beta_val,       # float32 scalar
    state_update_ptr,   # float32 [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for j in range(K):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += k_j * (beta_val * v_i + (1.0 - beta_val) * old_v_val)
    tl.store(state_update_ptr, acc)


# Kernel: compute h_state_vec[i] = sum_j state[b,h,i,j] * g - state_remove + state_update, i in [0..V-1]
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # float32 [V*K]
    h_state_ptr,    # float32 [V]
    g_val,          # float32 scalar
    state_remove_val,    # float32 scalar
    state_update_val,    # float32 scalar
    V: tl.constexpr,
    K: tl.constexpr,
):
    for i in range(V):
        s = 0.0
        for j in range(K):
            s += tl.load(state_ptr + i * K + j)
        h_state_ptr[i] = s * g_val - state_remove_val + state_update_val


# Kernel: compute output_scalar[b,h] = scale * dot(q[b,h], h_state_vec)
@triton.jit
def dot_q_hstate_kernel(
    q_ptr,          # float32 [K]
    h_state_ptr,    # float32 [V]
    output_ptr,     # float32 [1]
    scale,          # float32 scalar
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    for i in range(V):
        h_i = tl.load(h_state_ptr + i)
        for j in range(K):
            q_j = tl.load(q_ptr + j)
            acc += h_i * q_j
    acc = acc * scale
    tl.store(output_ptr, acc)


# Kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
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


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Gated Delta Net decode reference implementation (k-last layout).
        State layout: [B, H, V, K] (k-last, K dimension at the end)
        Gate computation:
          g = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
          beta = sigmoid(b[b,h])
        Delta rule update:
          state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
          output = scale * q @ state_new
        """
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        num_heads = num_v_heads
        device = q.device

        assert num_q_heads == 4
        assert num_k_heads == 4
        assert num_v_heads == 8
        assert K == 128 and V == 128
        assert T == 1

        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(K)

        # Compute g and beta from raw parameters
        x = a.squeeze(1).float() + dt_bias.float()  # [B, 1, H] -> [B*H]
        g = torch.exp(-torch.exp(A_log.float()) * torch.nn.functional.softplus(x))  # [B, 1, H]
        beta = torch.sigmoid(b.float())  # [B, 1, H]

        q_f32 = q.squeeze(1).float()
        k_f32 = k.squeeze(1).float()
        v_f32 = v.squeeze(1).float()
        g_f32 = g.squeeze(1).float()
        beta_f32 = beta.squeeze(1).float()

        if state is not None:
            state_f32 = state.float()
        else:
            state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

        # Prepare outputs
        new_state = torch.empty_like(state_f32, dtype=torch.float32, device=device)
        output = torch.empty((B, num_heads, V), dtype=torch.float32, device=device)

        # For each (b, h)
        for b_idx in range(B):
            for h_idx in range(num_heads):
                q_h = q_f32[b_idx, h_idx]                     # [K]
                k_h = k_f32[b_idx, h_idx]                     # [K]
                v_h = v_f32[b_idx, h_idx]                     # [V]
                state_old = state_f32[b_idx, h_idx].contiguous()  # [V, K]
                g_val = g_f32[b_idx, h_idx].item()            # scalar
                beta_val = beta_f32[b_idx, h_idx].item()      # scalar

                # Compute old_v
                old_v = torch.zeros(1, dtype=torch.float32, device=device)
                dot_k_state_kernel[(1,)](k_h, state_old.view(-1), old_v, V, K)

                # Compute state_remove
                state_remove = torch.zeros(1, dtype=torch.float32, device=device)
                dot_k_gstate_kernel[(1,)](k_h, state_old.view(-1), g_val, state_remove, V, K)

                # Compute state_update
                state_update = torch.zeros(1, dtype=torch.float32, device=device)
                dot_k_newv_kernel[(1,)](k_h, v_h, old_v.item(), beta_val, state_update, V, K)

                # Compute h_state_vec
                h_state = torch.empty((V,), dtype=torch.float32, device=device)
                h_state_vec_kernel[(V,)](state_old.view(-1), h_state, g_val, state_remove.item(), state_update.item(), V, K)

                # Compute output scalar
                out_scalar = torch.zeros(1, dtype=torch.float32, device=device)
                dot_q_hstate_kernel[(1,)](q_h, h_state, out_scalar, scale, V, K)
                output[b_idx, h_idx] = out_scalar[0]

                # Write new_state[b,h] = h_state broadcasted across K
                new_state[b_idx, h_idx] = write_new_state_kernel((V*K,), h_state, new_state[b_idx, h_idx].view(-1), V, K)

        # Output shape: [B,1,H] as bfloat16 to match original
        output = output.unsqueeze(1).to(torch.bfloat16)
        return output, new_state


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward: all computation in Triton kernels.
        Assumes inputs are tensors matching the original run helper:
          - q: [B,1,4,128]
          - k: [B,1,4,128]
          - v: [B,1,8,128]
          - state: [B,8,128,128]
          - A_log: [8]
          - a: [B,1,8]
          - dt_bias: [8]
          - b: [B,1,8]
          - scale: float or None
        Output:
          - output: [B,1,8], bfloat16
          - new_state: [B,8,128,128], float32
        """
        # Ensure inputs are on same device and contiguous, convert to float32 for compute
        device = q.device
        B, T_q, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        num_heads = num_v_heads

        # Flatten a and b for kernels
        a_flat = a.squeeze(1).contiguous().float().view(-1)  # [B*H]
        b_flat = b.squeeze(1).contiguous().float().view(-1)  # [B*H]
        A_log = A_log.contiguous().float()                   # [H]
        dt_bias = dt_bias.contiguous().float()               # [H]
        q_f32 = q.squeeze(1).contiguous().float()            # [B,H,K]
        k_f32 = k.squeeze(1).contiguous().float()            # [B,H,K]
        v_f32 = v.squeeze(1).contiguous().float()            # [B,H,V]
        state_f32 = state.contiguous().float()               # [B,H,V,K]

        # Compute g and beta via Triton
        g = torch.empty((B * num_heads,), dtype=torch.float32, device=device)
        beta = torch.empty((B * num_heads,), dtype=torch.float32, device=device)
        softplus_and_exp_kernel[(B * num_heads,)](dt_bias, a_flat, A_log, g, B, num_heads)
        sigmoid_kernel[(B * num_heads,)](b_flat, beta, B, num_heads)

        # Allocate outputs
        output = torch.empty((B, num_heads, V), dtype=torch.float32, device=device)
        new_state = torch.empty((B, num_heads, V, K), dtype=torch.float32, device=device)

        # Process each (b,h)
        for b_idx in range(B):
            for h_idx in range(num_heads):
                q_h = q_f32[b_idx, h_idx]                     # [K]
                k_h = k_f32[b_idx, h_idx]                     # [K]
                v_h = v_f32[b_idx, h_idx]                     # [V]
                state_old = state_f32[b_idx, h_idx].contiguous()  # [V,K]
                g_val = g[b_idx * num_heads + h_idx].item()   # scalar
                beta_val = beta[b_idx * num_heads + h_idx].item()  # scalar

                # Compute old_v = k @ state_old
                old_v = torch.zeros(1, dtype=torch.float32, device=device)
                dot_k_state_kernel[(1,)](k_h, state_old.view(-1), old_v, V, K)

                # Compute state_remove = k @ (g * state_old)
                state_remove = torch.zeros(1, dtype=torch.float32, device=device)
                dot_k_gstate_kernel[(1,)](k_h, state_old.view(-1), g_val, state_remove, V, K)

                # Compute state_update = k @ (beta * v + (1 - beta) * old_v)
                state_update = torch.zeros(1, dtype=torch.float32, device=device)
                dot_k_newv_kernel[(1,)](k_h, v_h, old_v.item(), beta_val, state_update, V, K)

                # Compute h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update
                h_state = torch.empty((V,), dtype=torch.float32, device=device)
                h_state_vec_kernel[(V,)](state_old.view(-1), h_state, g_val, state_remove.item(), state_update.item(), V, K)

                # Compute output[b,h] = scale * (q @ h_state)
                out_scalar = torch.zeros(1, dtype=torch.float32, device=device)
                dot_q_hstate_kernel[(1,)](q_h, h_state, out_scalar, (scale if scale is not None else (1.0 / math.sqrt(K))), V, K)
                output[b_idx, h_idx] = out_scalar[0]

                # Write new_state[b,h] as broadcast of h_state across K
                new_state[b_idx, h_idx] = write_new_state_kernel((V*K,), h_state, new_state[b_idx, h_idx].view(-1), V, K)

        # Output shape [B,1,H] as bfloat16
        output = output.unsqueeze(1).to(torch.bfloat16)
        return output, new_state


def get_inputs():
    q = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    state = torch.randn([1, 8, 128, 128], dtype=torch.float32, device='cuda')
    A_log = torch.randn([8], dtype=torch.float32, device='cuda')
    a = torch.randn([1, 1, 8], dtype=torch.bfloat16, device='cuda')
    dt_bias = torch.randn([8], dtype=torch.float32, device='cuda')
    b = torch.randn([1, 1, 8], dtype=torch.bfloat16, device='cuda')
    scale = 1.0  # float32 scalar
    return [q, k, v, state, A_log, a, dt_bias, b, scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
