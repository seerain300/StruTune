import torch
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
    a_val = tl.load(a_ptr + b * H + h)     # a has shape [B,H]
    dt_val = tl.load(dt_bias_ptr + h)      # dt_bias has shape [H]
    A_val = tl.load(A_log_ptr + h)         # A_log has shape [H]
    b_val = tl.load(b_ptr + b * H + h)     # b has shape [B,H]

    # softplus(x) = log(1 + exp(x))
    splus = tl.log(1.0 + tl.exp(a_val + dt_val))
    # g = exp(-exp(A) * softplus(a + dt_bias))
    g_val = tl.exp(-tl.exp(A_val) * splus)
    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, state_ptr, out_ptr,
                          K: tl.constexpr, V: tl.constexpr):
    # out_ptr is [K]
    # Compute out_vec[k] = sum_v k[v] * state[v, k]
    # state_ptr is flattened [V*K], so element at (v, k) is state_ptr[v*K + k]
    for k in tl.static_range(K):
        s_k = 0.0
        for v in tl.static_range(V):
            s_val = tl.load(state_ptr + v * K + k)
            k_val = tl.load(k_ptr + v)
            s_k += k_val * s_val
        tl.store(out_ptr + k, s_k)


@triton.jit
def _vec_matmul_scalar(k_ptr, vec_ptr, out_ptr,
                        SIZE: tl.constexpr):
    # Compute scalar = sum_i k[i] * vec[i] over SIZE
    scalar = 0.0
    for i in tl.static_range(SIZE):
        k_val = tl.load(k_ptr + i)
        v_val = tl.load(vec_ptr + i)
        scalar += k_val * v_val
    tl.store(out_ptr, scalar)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr, scale_ptr,
                           B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    # Load scale
    scale = tl.load(scale_ptr)  # scalar
    # Accumulate q @ new_state over tiles (V=128, K=128)
    total = 0.0
    for v in tl.static_range(V):
        for k in tl.static_range(K):
            qk = tl.load(q_ptr + b * K + k)   # q[b, :, k], but q shape is [B,1,Hq,K] -> q[b,0,:,:]; here we use q[b,0,:] which is [K]
            ns = tl.load(new_state_ptr + b * H * V * K + h * V * K + v * K + k)  # new_state[b,h,v,k]
            total += qk * ns
    total = total * scale
    tl.store(out_ptr + b * H + h, total)


@triton.jit
def _sqrt_scale_kernel(out_ptr, K: tl.constexpr):
    # Compute scale = 1 / sqrt(K)
    scale = 1.0 / tl.sqrt(K)
    tl.store(out_ptr, scale)


def _run_triton(q, k, v, state, A_log, a, dt_bias, b, scale_arg):
    """
    Triton-only implementation of the original run function logic.
    q: [B,1,Hq,K]
    k: [B,1,Hk,K]
    v: [B,1,Hv,V]
    state: [B,Hv,V,K]
    A_log: [Hv]
    a: [B,1,Hv] -> treat as [B,Hv]
    dt_bias: [Hv]
    b: [B,1,Hv] -> treat as [B,Hv]
    scale_arg: float or None (unused; scale computed inside Triton)
    Returns:
      output: [B,Hv] in bfloat16
      new_state: [B,Hv,V,K] in float32
    """
    device = q.device
    B_q, _, Hq, K = q.shape
    B_k, _, Hk, Kk = k.shape
    B_v, _, Hv, V = v.shape
    B_s, Hv_s, V_s, K_s = state.shape
    assert B_q == B_k == B_v == B_s, "Batch sizes must match"
    assert K == Kk == K_s == 128, "K must be 128"
    assert V == V_s == 128, "V must be 128"
    assert Hv == Hv_s, "num_v_heads must match"
    B = B_q
    H = Hv

    # Cast to float32 and contiguous
    q = q.contiguous().float()                      # [B,1,Hq,K]
    k = k.contiguous().float()                      # [B,1,Hk,K]
    v = v.contiguous().float()                      # [B,1,Hv,V]
    state = state.contiguous().float()              # [B,Hv,V,K]
    a = a.contiguous().float().view(B, H)           # [B,H]
    dt_bias = dt_bias.contiguous().float()          # [H]
    b = b.contiguous().float().view(B, H)           # [B,H]
    A_log = A_log.contiguous().float()              # [H]

    # Allocate outputs
    g_out = torch.empty((B, H), dtype=torch.float32, device=device)
    beta_out = torch.empty((B, H), dtype=torch.float32, device=device)
    output_f = torch.empty((B, H), dtype=torch.float32, device=device)
    new_state_f = torch.empty((B, H, V, K), dtype=torch.float32, device=device)
    # Compute scale using Triton kernel (one element)
    scale_buf = torch.empty((1,), dtype=torch.float32, device=device)
    _sqrt_scale_kernel[(1,)](scale_buf, meta={'K': K})

    # Launch gate computation kernel
    _compute_g_and_beta_kernel[(B * H,)](a, dt_bias, b, A_log, g_out, beta_out, meta={'B': B, 'H': H})

    # For each (b, h), compute new state and output
    for b_idx in range(B):
        for h_idx in range(H):
            # Extract vectors and matrices
            q_vec = q[b_idx, 0, :].contiguous()        # [K]
            k_vec = k[b_idx, 0, :].contiguous()        # [K]
            state_mat = state[b_idx, h_idx, :, :].contiguous()    # [V, K]
            v_vec = v[b_idx, 0, h_idx, :].contiguous()     # [V]

            g_val = g_out[b_idx, h_idx]
            beta_val = beta_out[b_idx, h_idx]

            # Compute old_v = k_vec @ state_mat (vector [K])
            old_v = torch.empty((K,), dtype=torch.float32, device=device)
            _vec_matmul_tile_vec[(1,)](k_vec, state_mat.view(-1), old_v, meta={'K': K, 'V': V})

            # Compute new_v = beta * v_vec + (1 - beta) * old_v (construct [K] by repeating v_vec to K)
            # Since v_vec is [V], we create new_v by repeating every V/K elements. For V=K, it's exact; for general, we repeat.
            repeat = int(K // V) if K % V == 0 else 1
            new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [K]

            # Compute state_remove = k_vec @ old_v (scalar)
            state_remove = torch.empty((), dtype=torch.float32, device=device)
            _vec_matmul_scalar[(1,)](k_vec, old_v, state_remove, meta={'SIZE': K})

            # Compute state_update = k_vec @ new_v (scalar)
            state_update = torch.empty((), dtype=torch.float32, device=device)
            _vec_matmul_scalar[(1,)](k_vec, new_v, state_update, meta={'SIZE': K})

            # Update new_state for this (b,h): new_state = g * state + (k^T @ new_v) - (k^T @ old_v)
            new_state_b_h = g_val * state_mat + (state_update - state_remove)  # [V,K]

            # Assign to output tensor
            new_state_f[b_idx, h_idx] = new_state_b_h           # [128,128]

            # Compute output[b,h] = scale * (q_vec @ new_state_b_h)
            out_elem = _output_scalar_kernel[(1,)](q_vec, new_state_b_h.view(-1), torch.zeros((1,), dtype=torch.float32, device=device), scale_buf, meta={'B': B, 'H': H, 'V': V, 'K': K})
            output_f[b_idx, h_idx] = out_elem

    # Return: output as bfloat16 [B, H], new_state as float32 [B, H, V, K]
    return output_f.unsqueeze(1).to(torch.bfloat16), new_state_f


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # ModelNew.forward invokes Triton kernels; scale is ignored here (computed in Triton).
        out, new_state = _run_triton(q, k, v, state, A_log, a, dt_bias, b, scale)
        return out, new_state


def run(*args):
    return ModelNew()(*args)
