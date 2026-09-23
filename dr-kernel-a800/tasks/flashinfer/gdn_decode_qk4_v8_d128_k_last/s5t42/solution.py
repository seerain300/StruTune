import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, g_ptr, beta_ptr, H, num_b):
    """
    Compute per-head g[h] and beta[h] for all h in [0..H-1] and b in [0..num_b-1].
    a_ptr: [num_b, H] bf16 -> we load and cast to fp32 for math
    b_ptr: [num_b, H] bf16 -> we load and cast to fp32
    A_log_ptr: [H] fp32
    dt_bias_ptr: [H] fp32
    g_ptr: [num_b, H] fp32
    beta_ptr: [num_b, H] fp32
    """
    # We do one program per (b,h)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H
    if b >= num_b:
        return

    # Load a[b,h], b[b,h] as fp32
    a_val = tl.load(a_ptr + b * H + h)
    b_val = tl.load(b_ptr + b * H + h)
    a_val = a_val.to(tl.float32)
    b_val = b_val.to(tl.float32)

    # Load A_log[h], dt_bias[h] as fp32
    A_log_val = tl.load(A_log_ptr + h)
    dt_bias_val = tl.load(dt_bias_ptr + h)

    # Compute softplus and g
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_bias_val))
    g = tl.exp(-(tl.exp(A_log_val)) * sp)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b * H + h, g)
    tl.store(beta_ptr + b * H + h, beta)


@triton.jit
def _update_single_bh_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    g_ptr, beta_ptr,
    out_ptr, new_state_ptr,
    B, H, V, K, scale
):
    """
    One program computes for a single (b,h):
    - Loads g[h] and beta[h]
    - Computes old_v, new_v, old_state, state_remove, state_update
    - Writes output[b,h] and new_state[b,h,:] flattened
    """
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H
    if b >= B:
        return

    # Load scalars g[h], beta[h]
    g_val = tl.load(g_ptr + b * H + h)  # fp32
    beta_val = tl.load(beta_ptr + b * H + h)  # fp32

    # Load vectors
    # k_h: [K]
    k_h = tl.load(k_ptr + b * H + h + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)  # fp32
    # v_h: [V]
    v_h = tl.load(v_ptr + b * H + h + tl.arange(0, V), mask=tl.arange(0, V) < V, other=0.0)  # fp32

    # state[b,h,:,:] -> [V, K] vectorized (row-major with stride K)
    # We access row i (i in [0..V-1]) as base + i*K, then load K elements
    # old_v = sum_k k_h[k] * state[b,h,i,k]
    old_v = 0.0
    for k in range(0, K):
        s_row_k = tl.load(state_ptr + b * (H * V * K) + h * (V * K) + k * V + tl.arange(0, V),
                          mask=tl.arange(0, V) < V, other=0.0)
        old_v += k_h[k] * tl.sum(s_row_k, axis=0)

    new_v = beta_val * v_h + (1.0 - beta_val) * old_v  # [V]

    # old_state = g * state[b,h] (elementwise across V rows)
    for i in range(0, V):
        row_base = b * (H * V * K) + h * (V * K) + i * K
        s_row = tl.load(state_ptr + row_base + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
        old_state_row = g_val * s_row  # [K]

        # state_remove[i, :] = sum_k k_h[k] * old_state_row[k]
        state_remove = 0.0
        for k in range(0, K):
            state_remove += k_h[k] * old_state_row[k]

        # state_update[i, :] = sum_k k_h[k] * new_v[k]
        state_update = 0.0
        for k in range(0, K):
            state_update += k_h[k] * new_v[k]

        new_row = old_state_row - state_remove + state_update  # [K]
        # Write new_state[b,h,i,:] (row-major)
        tl.store(state_ptr + row_base, new_row)

    # Compute output[b,h] = scale * sum_v q[b,h,v] * new_state[b,h,v]
    q_h = tl.load(q_ptr + b * H + h + tl.arange(0, V), mask=tl.arange(0, V) < V, other=0.0)
    out_val = 0.0
    for i in range(0, V):
        row_base = b * (H * V * K) + h * (V * K) + i * K
        new_row = tl.load(state_ptr + row_base + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)  # this is new_state, but we also need q_h[i]
        out_val += q_h[i] * tl.sum(new_row, axis=0)

    out_val = out_val * scale
    # Store output[b,h] linearized as out_ptr[b*H + h]
    tl.store(out_ptr + b * H + h, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, K], k: [B, 1, 4, K], v: [B, 1, 8, V], state: [B, 1, 8, V, K]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        Returns: output [B, H, V] bfloat16 and new_state [B, H, V, K] float32
        """
        device = q.device
        dtype = q.dtype

        # Squeeze batch dimension (B=1 per provided get_inputs)
        B_q = q.shape[0]
        H = v.shape[1]  # num_v_heads
        K = q.shape[3]
        V = v.shape[2]

        # Ensure inputs are contiguous and flatten batch to 1 for simplicity (B=1 in provided inputs)
        # We keep general shapes but in provided workload B=1, so no complex grid needed.
        q_ = q.contiguous().view(B_q, 4, K)
        k_ = k.contiguous().view(B_q, 4, K)
        v_ = v.contiguous().view(B_q, H, V)
        state_ = state.contiguous().view(B_q, H, V, K)  # [B, H, V, K]

        # Cast parameters to fp32 for Triton math
        a_fp32 = a.to(torch.float32).contiguous()  # [B, 1, H] -> flatten later
        b_fp32 = b.to(torch.float32).contiguous()
        A_log_fp32 = A_log.to(torch.float32).contiguous()
        dt_bias_fp32 = dt_bias.to(torch.float32).contiguous()

        # Flatten a,b to [B*H]
        H_val = H
        B_num = B_q
        a_flat = a_fp32.view(B_num, H_val)  # [B,H]
        b_flat = b_fp32.view(B_num, H_val)

        # Allocate outputs for g and beta: [B,H]
        g_vec = torch.empty(B_num * H_val, dtype=torch.float32, device=device)
        beta_vec = torch.empty(B_num * H_val, dtype=torch.float32, device=device)

        # Launch compute g/beta kernel: one program per (b,h)
        grid = (B_num * H_val,)
        _compute_g_beta_kernel[grid](a_flat, b_flat, A_log_fp32, dt_bias_fp32, g_vec, beta_vec, H_val, B_num)

        # Now update all (b,h) with Triton kernel
        # Allocate output [B,H,V] and new_state [B,H,V,K]
        out = torch.empty(B_num * H_val * V, dtype=torch.float32, device=device)
        new_state = state_.clone()  # we will write into this tensor in-place

        grid2 = (B_num * H_val,)
        _update_single_bh_kernel[grid2](
            q_, k_, v_, new_state,
            g_vec, beta_vec,
            out, new_state,  # new_state is used as write-back
            B_num, H_val, V, K, float(scale)
        )

        # Reshape and return
        out_bf16 = out.view(B_num, H_val, V).to(torch.bfloat16)  # [B,H,V] bfloat16
        new_state = new_state.view(B_num, H_val, V, K)  # [B,H,V,K] float32

        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
