import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr,
                                g_ptr, beta_ptr,
                                B: tl.constexpr, H: tl.constexpr):
    # One program per (b, h)
    pid = tl.program_id(0)
    b_idx = pid // H
    h = pid % H
    if b_idx >= B:
        return
    a_val = tl.load(a_ptr + b_idx * H + h)      # a: [B,H]
    dt_val = tl.load(dt_bias_ptr + h)           # dt_bias: [H]
    A_val = tl.load(A_log_ptr + h)              # A_log: [H]
    b_val = tl.load(b_ptr + b_idx * H + h)      # b: [B,H]

    # softplus(x) = log(1 + exp(x))
    splus = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * splus)
    sig = 1.0 / (1.0 + tl.exp(-b_val))          # sigmoid
    tl.store(g_ptr + b_idx * H + h, g_val)
    tl.store(beta_ptr + b_idx * H + h, sig)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, state_ptr, out_ptr,
                         K: tl.constexpr, V: tl.constexpr):
    # Compute out_vec[k] = sum_v state[v,k] * k[v] for k in [0..K-1]
    # state_ptr points to a contiguous [V*K] view of [V,K]
    for k in tl.static_range(0, K):
        acc = 0.0
        for v in tl.static_range(0, V):
            sv = tl.load(state_ptr + v * K + k)  # state[v, k]
            kv = tl.load(k_ptr + v)              # k[v]
            acc += sv * kv
        tl.store(out_ptr + k, acc)


@triton.jit
def _vec_matmul_scalar(k_ptr, vec_ptr, out_ptr, SIZE: tl.constexpr):
    # Compute scalar = sum_{i=0}^{SIZE-1} k[i] * vec[i]
    acc = 0.0
    for i in tl.static_range(0, SIZE):
        ki = tl.load(k_ptr + i)
        vi = tl.load(vec_ptr + i)
        acc += ki * vi
    tl.store(out_ptr, acc)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr,
                           V: tl.constexpr, K: tl.constexpr, H: tl.constexpr):
    # One program per (b, h) — we can use out_ptr indexing to store scalar per (b,h)
    pid = tl.program_id(0)
    b_idx = pid // H
    h = pid % H
    if b_idx >= 0 and h < H:
        acc = 0.0
        # Iterate over V and K tiles
        for v in tl.static_range(0, V):
            for k in tl.static_range(0, K):
                qk = tl.load(q_ptr + b_idx * K + k)               # q[b, h, k] flattened (q is [B,K])
                ns = tl.load(new_state_ptr + b_idx * H * V * K + h * V * K + v * K + k)  # new_state[b,h,v,k]
                acc += qk * ns
        # Compute scale inside kernel: scale = 1 / sqrt(K)
        scale = 1.0 / tl.sqrt(K)
        acc = acc * scale
        tl.store(out_ptr + b_idx * H + h, acc)


def _run(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Triton-only implementation of the reference logic.
    q: [B, 1, num_q_heads, K]
    k: [B, 1, num_k_heads, K]
    v: [B, 1, num_v_heads, V]
    state: [B, num_v_heads, V, K]
    A_log: [num_v_heads]
    a: [B, 1, num_v_heads]
    dt_bias: [num_v_heads]
    b: [B, 1, num_v_heads]
    scale: float or None (unused here; computed inside Triton)
    Returns:
      output: [B, 1, H, V] in bfloat16
      new_state: [B, H, V, K] in float32
    """
    device = q.device
    # Input shapes
    B_q, _, num_q_heads, K = q.shape
    B_k, _, num_k_heads, Kk = k.shape
    B_v, _, num_v_heads, V = v.shape
    B_s, num_v_heads_s, V_s, K_s = state.shape
    assert B_q == B_k == B_v == B_s, "Batch sizes must match"
    assert K == Kk == K_s == 128, "K must be 128"
    assert V == V_s == 128, "V must be 128"
    B = B_q
    H = num_v_heads  # number of heads in the algorithm

    # Cast to float32 and contiguous
    q = q.contiguous().float()                 # [B, 1, num_q_heads, K]
    k = k.contiguous().float()                 # [B, 1, num_k_heads, K]
    v = v.contiguous().float()                 # [B, 1, num_v_heads, V]
    state = state.contiguous().float()         # [B, num_v_heads, V, K]
    a = a.contiguous().float().view(B, H)      # [B, H]
    dt_bias = dt_bias.contiguous().float()     # [H]
    b = b.contiguous().float().view(B, H)      # [B, H]
    A_log = A_log.contiguous().float()         # [H]

    # Allocate outputs
    g_out = torch.empty((B, H), dtype=torch.float32, device=device)
    beta_out = torch.empty((B, H), dtype=torch.float32, device=device)
    output_f = torch.empty((B, H), dtype=torch.float32, device=device)
    new_state_f = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

    # Launch gate computation kernel
    _compute_g_and_beta_kernel[(B * H,)](
        a, dt_bias, b, A_log,
        g_out, beta_out,
        B=B, H=H
    )

    # For each (b, h), compute new state and output
    for b_idx in range(B):
        for h in range(H):
            # Extract vectors and matrices
            # q and k are [B, 1, Hq/K, K] -> take K-length vector
            q_h = q[b_idx, 0, :].contiguous()        # [K]
            k_h = k[b_idx, 0, :].contiguous()        # [K]
            state_h = state[b_idx, h, :, :].contiguous()    # [V, K]
            v_h = v[b_idx, 0, h, :].contiguous()     # [V]

            g_val = g_out[b_idx, h]
            beta_val = beta_out[b_idx, h]

            # Compute old_v = k_h @ state_h
            old_v = torch.empty((K,), dtype=torch.float32, device=device)
            _vec_matmul_tile_vec[(1,)](
                k_h, state_h.view(-1), old_v,
                K=K, V=V
            )

            # Compute new_v = beta * v_h + (1 - beta) * old_v
            new_v = beta_val * v_h + (1.0 - beta_val) * old_v  # [K]

            # Compute state_remove = k_h @ old_v (scalar)
            state_remove = torch.empty((), dtype=torch.float32, device=device)
            _vec_matmul_scalar[(1,)](
                k_h, old_v, state_remove,
                SIZE=K
            )

            # Compute state_update = k_h @ new_v (scalar)
            state_update = torch.empty((), dtype=torch.float32, device=device)
            _vec_matmul_scalar[(1,)](
                k_h, new_v, state_update,
                SIZE=K
            )

            # Update new state elementwise: new_state = g * state - state_remove + state_update
            new_state_f[b_idx, h] = (g_val * state_h) - state_remove + state_update  # broadcast scalars

            # Compute output = (1/sqrt(K)) * q_h @ new_state_h
            _output_scalar_kernel[(1,)](
                q_h, new_state_f[b_idx, h],  # [V, K]
                output_f[b_idx, h].contiguous(),
                V=V, K=K, H=H
            )

    # Return output in bfloat16 as per original get_inputs behavior: [B, 1, H, V]
    output = output_f.unsqueeze(1).to(torch.bfloat16)
    return output, new_state_f


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        output, new_state = _run(q, k, v, state, A_log, a, dt_bias, b, scale)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
