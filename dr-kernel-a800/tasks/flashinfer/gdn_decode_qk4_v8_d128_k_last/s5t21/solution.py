import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a[0,0,h] + dt_bias[h])) for h in [0..H-1]
    a_ptr: [1, 1, H] flattened to [H] (we pass a[0,1,:])
    dt_bias_ptr: [H]
    A_log_ptr: [H]
    g_ptr: [H]
    """
    h = tl.program_id(0)
    a_val = tl.load(a_ptr + h)
    dt_val = tl.load(dt_bias_ptr + h)
    A_val = tl.load(A_log_ptr + h)
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * sp)
    tl.store(g_ptr + h, g_val)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, H):
    """
    Compute beta[h] = sigmoid(b[0,0,h]) for h in [0..H-1]
    b_ptr: [1, 1, H] flattened to [H] (we pass b[0,1,:])
    beta_ptr: [H]
    """
    h = tl.program_id(0)
    b_val = tl.load(b_ptr + h)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + h, beta_val)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, out_ptr, new_state_ptr,
    B, H, V, K, scale
):
    """
    For each (b, h), update new_state[b,h] and compute output[b,h].
    q_ptr: [B, H, K]
    k_ptr: [B, H, K]
    v_ptr: [B, H, V]
    state_ptr: [B, H, V, K] (input state)
    g_ptr: [H]
    beta_ptr: [H]
    out_ptr: [B, H] (float32)
    new_state_ptr: [B, H, V, K] (float32, linearized)
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Base offsets for q/k/v after squeezing T=1
    q_base = b * (H * K) + h * K
    k_base = b * (H * K) + h * K
    v_base = b * (H * V) + h * V

    # Load vectors q_h, k_h
    q_vec = [0.0] * K
    k_vec = [0.0] * K
    for i in range(0, K):
        q_vec[i] = tl.load(q_ptr + q_base + i)
        k_vec[i] = tl.load(k_ptr + k_base + i)

    # Load per-head params
    g_val = tl.load(g_ptr + h)
    beta_val = tl.load(beta_ptr + h)

    # Load state[b,h] as [V, K]
    state_base = b * (H * V * K) + h * (V * K)
    old_state = [[0.0] * K for _ in range(0, V)]
    for i in range(0, V):
        for k in range(0, K):
            old_state[i][k] = tl.load(state_ptr + state_base + i * K + k)

    # old_v = k @ state_old (reduce over K)
    old_v = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_vec[k] * old_state[i][k]
        old_v[i] = sum_k

    # new_v = beta * v + (1 - beta) * old_v
    new_v = [0.0] * V
    for i in range(0, V):
        v_elem = tl.load(v_ptr + v_base + i)  # v[b,h,i]
        new_v[i] = beta_val * v_elem + (1.0 - beta_val) * old_v[i]

    # state_remove = k @ old_state (reduce over K)
    state_remove = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_vec[k] * old_state[i][k]
        state_remove[i] = sum_k

    # state_update = k @ new_v (reduce over K)
    state_update = [0.0] * V
    for i in range(0, V):
        sum_k = 0.0
        for k in range(0, K):
            sum_k += k_vec[k] * new_v[i]
        state_update[i] = sum_k

    # Write new_state[b,h] into new_state_ptr at offset b*H*V*K + h*V*K
    new_state_base = b * (H * V * K) + h * (V * K)
    for i in range(0, V):
        for k in range(0, K):
            val = old_state[i][k] - state_remove[i] + state_update[i]
            tl.store(new_state_ptr + new_state_base + i * K + k, val)

    # Compute output[b,h] = scale * (sum_i q[i] * (sum_k new_state[i,k]))
    row_sum = 0.0
    for k in range(0, K):
        for i in range(0, V):
            row_sum += new_state_ptr[new_state_base + i * K + k]
    out_val = sum(q_vec) * row_sum
    out_val *= scale
    tl.store(out_ptr + b * H + h, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only forward:
        - Compute g and beta via Triton kernels
        - Update new_state and compute output via Triton kernel
        Returns: (output [B,H,V] in bfloat16), new_state [B,H,V,K] in float32
        """
        # Triton requires CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "All inputs must be CUDA tensors for Triton"

        # Shapes after squeezing T=1
        B = q.shape[0]
        H = v.shape[1]  # num_v_heads = 8
        V = v.shape[2]  # 128
        K = q.shape[2]  # 128

        # Flatten pointers for Triton kernels
        a_flat = a.reshape(-1)             # [B,1,H] -> [H] (we pass a[0,1,:])
        dt_bias_flat = dt_bias             # [H]
        A_log_flat = A_log                 # [H]
        b_flat = b.reshape(-1)             # [B,1,H] -> [H] (we pass b[0,1,:])

        # Allocate outputs for g and beta
        g = torch.empty(H, dtype=torch.float32, device=q.device)
        beta = torch.empty(H, dtype=torch.float32, device=q.device)

        # Launch kernels to compute g and beta
        _compute_g_kernel[(H,)](a_flat, dt_bias_flat, A_log_flat, g, H, num_warps=1)
        _compute_beta_kernel[(H,)](b_flat, beta, H, num_warps=1)

        # Allocate output and new_state buffers
        out = torch.empty(B * H, dtype=torch.float32, device=q.device)
        new_state = torch.empty(B * H * V * K, dtype=torch.float32, device=q.device)

        # Prepare q, k, v after squeeze T=1 -> shapes [B,4,K] and [B,8,V]
        # The Triton kernel expects [B,H,K] and [B,H,V]. We assume H==8 as in the reference setup.
        assert H == 8, "This Triton implementation currently assumes H=8 (num_v_heads=8)"
        q_exp = q      # [B,4,K]
        k_exp = k      # [B,4,K]
        v_exp = v      # [B,8,V]

        # Launch kernel to update all (b,h)
        _update_all_kernel[(B, H)](
            q_exp, k_exp, v_exp, state, g, beta, out, new_state,
            B, H, V, K, float(scale), num_warps=1
        )

        # Reshape output to [B,H] and broadcast to [B,H,V] (output in bfloat16)
        out = out.view(B, H)
        out_broadcast = out.unsqueeze(-1).expand(B, H, V).contiguous().to(torch.bfloat16)

        # Reshape new_state from linear to [B,H,V,K] float32
        new_state = new_state.view(B, H, V, K).contiguous()

        return out_broadcast, new_state


def run(*args):
    return ModelNew()(*args)
