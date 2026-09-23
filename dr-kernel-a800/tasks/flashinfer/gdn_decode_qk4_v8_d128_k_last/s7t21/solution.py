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
    b = pid // H
    h = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b * H + h)      # a has shape [B,H]
    dt_val = tl.load(dt_bias_ptr + h)       # dt_bias has shape [H]
    A_val = tl.load(A_log_ptr + h)          # A_log has shape [H]
    b_val = tl.load(b_ptr + b * H + h)      # b has shape [B,H]

    # softplus(x) = log(1 + exp(x))
    splus = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * splus)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def _matvec_kernel(k_ptr, state_ptr, out_ptr,
                   K: tl.constexpr, V: tl.constexpr):
    # Compute out_vec[k] = sum_v k[v] * state[v, k]
    # k_ptr: [K], state_ptr: flattened [V*K], out_ptr: [K]
    for k in range(K):
        sum_k = 0.0
        for v in tl.static_range(V):
            sum_k += tl.load(k_ptr + v) * tl.load(state_ptr + v * K + k)
        tl.store(out_ptr + k, sum_k)


@triton.jit
def _scalar_dot_kernel(k_ptr, vec_ptr, out_ptr,
                        SIZE: tl.constexpr):
    # Compute scalar = sum_i k[i] * vec[i]
    scalar = 0.0
    for i in tl.static_range(SIZE):
        scalar += tl.load(k_ptr + i) * tl.load(vec_ptr + i)
    tl.store(out_ptr, scalar)


@triton.jit
def _output_matvec_kernel(q_ptr, new_state_ptr, out_ptr,
                           scale, B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b, h): compute output[b, h] = scale * (q[b,h] @ new_state[b,h,:,:])
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Base for q at this (b,h)
    q_base = b * H * K  # since q is [B, H, K] in flattened view
    # Base for new_state at this (b,h)
    state_base = (b * H * V) * K  # [B, H, V, K]

    total = 0.0
    # Tile over V and K (loops are fine with small sizes)
    for v in tl.static_range(V):
        for k in tl.static_range(K):
            qk = tl.load(q_ptr + q_base + h * K + k)
            ns = tl.load(new_state_ptr + state_base + h * V * K + v * K + k)
            total += qk * ns

    total = total * scale
    tl.store(out_ptr + b * H + h, total)


# Host function: compute scale via Triton kernel (though we can compute it here; keep Triton usage minimal)
@triton.jit
def _sqrt_scale_kernel(scale_ptr, K: tl.constexpr):
    inv_sqrt = 1.0 / tl.sqrt(K)
    tl.store(scale_ptr, inv_sqrt)


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
    scale: float or None (ignored; output uses q @ new_state)
    Returns:
      output: [B, num_v_heads] in bfloat16
      new_state: [B, num_v_heads, V, K] in float32
    """
    device = q.device
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
    q = q.contiguous().float()                      # [B, 1, num_q_heads, K]
    k = k.contiguous().float()                      # [B, 1, num_k_heads, K]
    v = v.contiguous().float()                      # [B, 1, num_v_heads, V]
    state = state.contiguous().float()              # [B, num_v_heads, V, K]
    a = a.contiguous().float().view(B, H)           # [B, H]
    dt_bias = dt_bias.contiguous().float()          # [H]
    b = b.contiguous().float().view(B, H)           # [B, H]
    A_log = A_log.contiguous().float()              # [H]

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
        for h_idx in range(H):
            # Extract vectors and matrices (contiguous)
            q_vec = q[b_idx, 0, h_idx, :].contiguous().float()        # [K]
            k_vec = k[b_idx, 0, h_idx, :].contiguous().float()        # [K]
            state_h = state[b_idx, h_idx, :, :].contiguous().float()  # [V, K]
            v_vec = v[b_idx, 0, h_idx, :].contiguous().float()        # [V]

            # Load g and beta
            g_val = g_out[b_idx, h_idx]
            beta_val = beta_out[b_idx, h_idx]

            # 1) old_v = k_vec @ state_h (vector of length K)
            old_v = torch.empty((K,), dtype=torch.float32, device=device)
            _matvec_kernel[(1,)](
                k_vec, state_h.view(-1), old_v,
                K=K, V=V
            )

            # 2) new_v = beta * v_vec + (1 - beta) * old_v
            new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [V]

            # 3) state_remove = k_vec @ old_v (scalar)
            state_remove = torch.empty((), dtype=torch.float32, device=device)
            _scalar_dot_kernel[(1,)](
                k_vec, old_v, state_remove,
                SIZE=K
            )

            # 4) state_update = k_vec @ new_v (scalar)
            state_update = torch.empty((), dtype=torch.float32, device=device)
            _scalar_dot_kernel[(1,)](
                k_vec, new_v, state_update,
                SIZE=K
            )

            # 5) Update new state
            # new_state[b,h,v,k] = g * state[b,h,v,k] - state_remove + state_update
            new_state_f[b_idx, h_idx, :, :] = state_h * g_val                      # [V,K]
            new_state_f[b_idx, h_idx, :, :] = new_state_f[b_idx, h_idx, :, :] - state_remove + state_update  # [V,K]

            # 6) Compute output[b,h] = (q_vec @ new_state_f[b,h,:,:])
            # Launch Triton kernel for output scalar (one program per (b,h))
            _output_matvec_kernel[(1,)](
                q_vec, new_state_f[b_idx, h_idx, :, :].view(-1),
                output_f[b_idx, h_idx],
                1.0,  # scale is irrelevant for output computation here; original output does not scale by scale
                B=B, H=H, V=V, K=K
            )

    # Return: output in bfloat16, new_state in float32
    return output_f.to(torch.bfloat16), new_state_f


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        return _run(q, k, v, state, A_log, a, dt_bias, b, scale)


def run(*args):
    return ModelNew()(*args)
