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
    a_val = tl.load(a_ptr + b * H + h)       # a[b, h]
    dt_val = tl.load(dt_bias_ptr + h)        # dt_bias[h]
    A_val = tl.load(A_log_ptr + h)           # A_log[h]

    # x = a + dt_bias
    x = a_val + dt_val
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    # g = exp(-exp(A_log) * softplus(a + dt_bias))
    g = tl.exp(-tl.exp(A_val) * sp)
    # beta = sigmoid(b[b, h]) = 1 / (1 + exp(-b))
    b_val = tl.load(b_ptr + b * H + h)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # Store
    tl.store(g_ptr + b * H + h, g)
    tl.store(beta_ptr + b * H + h, beta)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, state_flat_ptr, out_ptr,
                         K: tl.constexpr):
    # out[k] = sum_{v=0..V-1} k[v] * state[v, k]
    # We don't have V here because V is implied by state_flat_ptr length V*K.
    # This kernel expects state_flat_ptr to be of length V*K and out_ptr of length K.
    for k in tl.static_range(0, K):
        acc = 0.0
        # We need to iterate over all v to accumulate contributions to out[k]
        # We can do that by iterating over m in [0, V*K), deriving v = m // K, kk = m % K.
        # When kk == k, add k[v] * state[v, k].
        for m in tl.static_range(0, K * V):
            v = m // K
            kk = m % K
            if kk == k:
                k_val = tl.load(k_ptr + v)       # k[v]
                state_val = tl.load(state_flat_ptr + m)  # state[v, k]
                acc += k_val * state_val
        tl.store(out_ptr + k, acc)


@triton.jit
def _vec_matmul_scalar(vec_ptr, k_ptr, out_ptr, N: tl.constexpr):
    # scalar = sum_{i=0..N-1} vec[i] * k[i]
    acc = 0.0
    for i in tl.static_range(0, N):
        vec_i = tl.load(vec_ptr + i)
        k_i = tl.load(k_ptr + i)
        acc += vec_i * k_i
    tl.store(out_ptr, acc)


@triton.jit
def _sqrt_scale_kernel(scale_ptr, K: tl.constexpr):
    # Compute scale = 1/sqrt(K)
    scale = 1.0 / tl.sqrt(K)
    tl.store(scale_ptr, scale)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Provided shapes (B=1, Hq=4, Hk=4, Hv=8, K=V=128):
        assert q.shape == (1, 1, 4, 128) and k.shape == (1, 1, 4, 128) and v.shape == (1, 1, 8, 128) and state.shape == (1, 1, 8, 128, 128)

        B, _, Hq, K = q.shape
        _, _, Hv, V = v.shape
        assert Hq == Hv, "Hq must equal Hv"
        Hv = Hq

        device = q.device

        # Prepare tensors as float32 and contiguous
        q_f = q.squeeze(1).to(torch.float32).contiguous()  # [4, 128]
        k_f = k.squeeze(1).to(torch.float32).contiguous()  # [4, 128]
        v_f = v.squeeze(1).to(torch.float32).contiguous()  # [8, 128]
        state_f = state.to(torch.float32).contiguous()     # [8, 128, 128]
        A_log_f = A_log.to(torch.float32).contiguous()     # [8]
        a_f = a.squeeze(1).to(torch.float32).contiguous()  # [1, 8]
        dt_bias_f = dt_bias.to(torch.float32).contiguous() # [8]
        b_f = b.squeeze(1).to(torch.float32).contiguous()  # [1, 8]

        # Allocate outputs
        g_out = torch.empty(B * Hv, device=device, dtype=torch.float32)   # [Hv]
        beta_out = torch.empty(B * Hv, device=device, dtype=torch.float32) # [Hv]
        output_f = torch.empty(B * Hv, device=device, dtype=torch.float32) # [Hv]
        new_state_f = torch.empty((B, Hv, V, K), device=device, dtype=torch.float32)  # [1, Hv, V, K]

        # Launch g and beta computation kernel
        _compute_g_and_beta_kernel[(B * Hv,)](a_f, dt_bias_f, b_f, A_log_f,
                                              g_out, beta_out,
                                              B=B, H=Hv)

        # Compute scale = 1/sqrt(K) in Triton
        scale_buf = torch.empty(1, device=device, dtype=torch.float32)
        _sqrt_scale_kernel[(1,)](scale_buf, K=K)

        # Process each (b, h) pair: update state and compute output (only one batch item, B=1)
        for b_idx in range(B):
            for h_idx in range(Hv):
                h = h_idx  # head index

                # Prepare vectors
                k_vec = k_f[h]  # [128]
                v_vec = v_f[h]  # [128]
                state_mat = state_f[h]  # [128, 128]
                q_vec = q_f[h]  # [128]

                # 1) old_v = k @ state_old (vector [K])
                old_v = torch.empty(K, device=device, dtype=torch.float32)
                # Flatten state to [V*K] for the kernel
                state_flat = state_mat.reshape(-1)  # [V*K] = [128*128]
                _vec_matmul_tile_vec[(K,)](k_vec, state_flat, old_v, K=K, V=V)

                # 2) beta and g
                g_val = g_out[b_idx * Hv + h]
                beta_val = beta_out[b_idx * Hv + h]

                # 3) new_v = beta * v + (1 - beta) * old_v
                new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [128]

                # 4) state_remove = k @ old_v (scalar)
                state_remove = torch.empty((), device=device, dtype=torch.float32)
                _vec_matmul_scalar[(K,)](old_v, k_vec, state_remove, N=K)

                # 5) state_update = k @ new_v (scalar)
                state_update = torch.empty((), device=device, dtype=torch.float32)
                _vec_matmul_scalar[(K,)](new_v, k_vec, state_update, N=K)

                # 6) new_state = g * state_old - state_remove + state_update
                state_old_flat = state_mat.reshape(-1)  # [V*K]
                new_state_row = (g_val * state_old_flat) - state_remove + state_update  # [V*K]
                new_state_f[b_idx, h] = new_state_row.reshape(V, K)

                # 7) output = scale * (q @ new_state)
                # Compute q @ new_state via scalar product over K (since new_state_row is flattened [V*K], and we want sum q[k] * new_state[k]):
                # Note: new_state_row corresponds to flattened


def run(*args):
    return ModelNew()(*args)
