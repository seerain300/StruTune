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
    a_val = tl.load(a_ptr + b * H + h)     # a has shape [B,H]
    dt_val = tl.load(dt_bias_ptr + h)      # dt_bias has shape [H]
    A_val = tl.load(A_log_ptr + h)         # A_log has shape [H]
    b_val = tl.load(b_ptr + b * H + h)     # b has shape [B,H]

    # softplus(x) = log(1 + exp(x))
    splus = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * splus)
    sig = 1.0 / (1.0 + tl.exp(-b_val))     # sigmoid

    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, sig)


@triton.jit
def _vec_matmul_tile(a_ptr, state_ptr, out_ptr,
                     K: tl.constexpr, V: tl.constexpr):
    # Compute out[k] = sum_v a[k] * state[v, k], for k in [0..K-1], v in [0..V-1]
    # a is 1D of length K; state is [V, K] linearized; out is 1D of length K
    for k in tl.static_range(0, K):
        s = 0.0
        for v in tl.static_range(0, V):
            # state linear index = v * K + k
            s += tl.load(state_ptr + v * K + k) * tl.load(a_ptr + k)
        tl.store(out_ptr + k, s)


@triton.jit
def _vec_matmul_scalar(a_ptr, vec_ptr, out_ptr,
                       SIZE: tl.constexpr):
    # Compute scalar = sum_i a[i] * vec[i]
    scalar = 0.0
    for i in tl.static_range(0, SIZE):
        scalar += tl.load(a_ptr + i) * tl.load(vec_ptr + i)
    tl.store(out_ptr, scalar)


@triton.jit
def _output_scalar(q_ptr, new_state_ptr, out_ptr,
                   scale, V: tl.constexpr, K: tl.constexpr):
    # Compute scalar = scale * sum_v sum_k q[k] * new_state[v, k]
    scalar = 0.0
    for v in tl.static_range(0, V):
        row_sum = 0.0
        for k in tl.static_range(0, K):
            qk = tl.load(q_ptr + k)
            ns = tl.load(new_state_ptr + v * K + k)
            row_sum += qk * ns
        scalar += scale * row_sum
    tl.store(out_ptr, scalar)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the reference logic.
        q: [B, 1, num_q_heads, K], bfloat16
        k: [B, 1, num_k_heads, K], bfloat16
        v: [B, 1, num_v_heads, V], bfloat16
        state: [B, num_v_heads, V, K], float32
        A_log: [num_v_heads], float32
        a: [B, 1, num_v_heads], bfloat16
        dt_bias: [num_v_heads], float32
        b: [B, 1, num_v_heads], bfloat16
        scale: float or None (if 0 or None, use 1/sqrt(K))
        Returns:
          output: [B, num_v_heads] bfloat16
          new_state: [B, num_v_heads, V, K] float32
        """
        device = q.device
        # Shapes from get_inputs:
        # q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
        B_q, _, num_q_heads, K = q.shape
        B_k, _, num_k_heads, Kk = k.shape
        B_v, _, num_v_heads, V = v.shape
        B_s, num_v_heads_s, V_s, K_s = state.shape
        assert B_q == B_k == B_v == B_s, "Batch sizes must match"
        assert K == Kk == K_s == 128, "K must be 128"
        assert V == V_s == 128, "V must be 128"
        B = B_q
        H = num_v_heads  # number of heads used in the algorithm

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
            for h in range(H):
                # Extract vectors and matrices
                q_h = q[b_idx, 0, :].contiguous()        # [K]
                k_h = k[b_idx, 0, :].contiguous()        # [K]
                state_h = state[b_idx, h, :, :].contiguous()    # [V, K]
                v_h = v[b_idx, 0, h, :].contiguous()     # [V]

                g_val = g_out[b_idx, h]
                beta_val = beta_out[b_idx, h]

                # Compute old_v = k_h @ state_h
                old_v = torch.empty((K,), dtype=torch.float32, device=device)
                _vec_matmul_tile[(1,)](
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

                # Update new state: new_state = g * state - state_remove + state_update
                # Initialize with g * state
                new_state_f[b_idx, h] = g_val * state[b_idx, h].clone()
                # Adjust by scalars: subtract state_remove and add state_update
                new_state_f[b_idx, h] -= state_remove
                new_state_f[b_idx, h] += state_update

                # Compute output[b,h] = scale * (q_h @ new_state_h) via Triton
                # Flatten new_state[b,h] to [V*K]
                state_ptr = new_state_f[b_idx, h].view(-1)
                # Choose scale: if None or 0.0, use 1/sqrt(K)
                scale_val = 1.0 / math.sqrt(K) if (scale is None or scale == 0.0 or scale == 0.0) else float(scale)
                out_scalar = torch.empty((), dtype=torch.float32, device=device)
                _output_scalar[(1,)](
                    q_h, state_ptr, out_scalar,
                    scale_val, V=V, K=K
                )
                output_f[b_idx, h] = out_scalar

        # Return output in bfloat16 and new_state in float32
        output = output_f.unsqueeze(1).to(torch.bfloat16)
        return output, new_state_f


def run(*args):
    return ModelNew()(*args)
