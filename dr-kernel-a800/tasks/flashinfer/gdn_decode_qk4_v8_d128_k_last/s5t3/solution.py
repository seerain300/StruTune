import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, H):
    """
    Compute g[h] = exp(-exp(A_log[h]) * softplus(a[0,0,h] + dt_bias[h])) for h in [0..H-1]
    a_ptr: flattened [H] float32
    dt_bias_ptr: [H] float32
    A_log_ptr: [H] float32
    g_ptr: [H] float32
    """
    h = tl.program_id(0)
    a_val = tl.load(a_ptr + h)         # a[0, 0, h]
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias[h]
    A_val = tl.load(A_log_ptr + h)     # A_log[h]
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * sp)
    tl.store(g_ptr + h, g_val)


@triton.jit
def _compute_beta_kernel(b_ptr, beta_ptr, H):
    """
    Compute beta[h] = sigmoid(b[0,0,h]) for h in [0..H-1]
    b_ptr: flattened [H] float32
    beta_ptr: [H] float32
    """
    h = tl.program_id(0)
    b_val = tl.load(b_ptr + h)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_ptr + h, beta_val)


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, out_ptr,
    B, H, V, K, scale
):
    """
    For each (b, h), update new_state[b,h] and compute output[b,h].
    q_ptr: [B, H, K] float32
    k_ptr: [B, H, K] float32
    v_ptr: [B, H, V] float32
    state_ptr: [B, H, V, K] float32 (input state)
    g_ptr: [H] float32
    beta_ptr: [H] float32
    out_ptr: [B, H] float32
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Base offsets
    q_base = b * (H * K) + h * K
    k_base = b * (H * K) + h * K
    v_base = b * (H * V) + h * V

    # Load vectors q[h] and k[h]
    q_vec = [0.0] * K
    k_vec = [0.0] * K
    for i in range(0, K):
        q_vec[i] = tl.load(q_ptr + q_base + i)
        k_vec[i] = tl.load(k_ptr + k_base + i)

    # Load gate and beta
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

    # Write new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None]
    new_state_base = b * (H * V * K) + h * (V * K)
    for i in range(0, V):
        for k in range(0, K):
            val = old_state[i][k] - state_remove[i] + state_update[i]
            tl.store(state_ptr + new_state_base + i * K + k, val)

    # Compute output[b,h] = scale * sum_i q[i] * (sum_k new_state[b,h,i,k])
    out_val = 0.0
    for i in range(0, V):
        row_sum = 0.0
        for k in range(0, K):
            row_sum += tl.load(state_ptr + new_state_base + i * K + k)
        out_val += q_vec[i] * row_sum
    out_val *= scale
    tl.store(out_ptr + b * H + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Triton-only implementation, robust to argument order and names.
        Returns: (output [B, H] cast to bfloat16, new_state [B, H, V, K] float32)
        """
        # We do not assume any specific positional index for q/k/v; instead, search all inputs
        device = None
        q = None
        k = None
        v = None
        state = None
        A_log = None
        a = None
        dt_bias = None
        b = None
        scale = None

        for t in args:
            if isinstance(t, torch.Tensor):
                if device is None:
                    device = t.device
                if t.shape == ():
                    # scalar tensor
                    scale = t.item()
                # Identify q: [B, 1, 4, K], after squeeze(1) -> [B, 4, K]
                elif t.shape[1:] == (4, 128) and t.ndim == 4:
                    q = t
                # Identify k: [B, 1, 4, K] -> [B, 4, K]
                elif t.shape[1:] == (4, 128) and t.ndim == 4:
                    k = t
                # Identify v: [B, 1, 8, 128] -> [B, 8, 128]
                elif t.shape[1:] == (8, 128) and t.ndim == 4:
                    v = t
                # Identify state: [B, 8, 128, 128]
                elif t.shape[1:] == (8, 128, 128) and t.ndim == 4:
                    state = t
                # Identify A_log: [8]
                elif t.shape == (8,) and t.ndim == 1:
                    A_log = t
                # Identify a: [B, 1, 8]
                elif t.shape == (B, 1, 8) and t.ndim == 3:
                    a = t
                # Identify dt_bias: [8]
                elif t.shape == (8,) and t.ndim == 1:
                    dt_bias = t
                # Identify b: [B, 1, 8]
                elif t.shape == (B, 1, 8) and t.ndim == 3:
                    b = t
            elif isinstance(t, (int, float)):
                if scale is None:
                    scale = t

        assert q is not None and k is not None and v is not None and state is not None and \
               A_log is not None and a is not None and dt_bias is not None and b is not None and \
               scale is not None, "Failed to identify required tensors or scalar in inputs"

        # Ensure T=1 squeeze
        q = q.squeeze(1)  # [B, 4, K]
        k = k.squeeze(1)  # [B, 4, K]
        v = v.squeeze(1)  # [B, 8, V]

        B = q.shape[0]
        H = v.shape[1]  # num_v_heads = 8
        V = v.shape[2]  # 128
        K = q.shape[2]  # 128

        #


def run(*args):
    return ModelNew()(*args)
