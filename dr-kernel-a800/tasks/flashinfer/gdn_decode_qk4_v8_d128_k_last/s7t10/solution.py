import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr,
                                g_ptr, beta_ptr,
                                B: tl.constexpr, H: tl.constexpr):
    """
    Compute g and beta per (b, h).
    Inputs:
      a_ptr: [B, H] float32
      dt_bias_ptr: [H] float32
      b_ptr: [B, H] float32
      A_log_ptr: [H] float32
      g_ptr: [B, H] float32 output
      beta_ptr: [B, H] float32 output
    """
    pid = tl.program_id(0)
    b_idx = pid // H
    h = pid % H
    # safety
    if b_idx >= B:
        return
    a_val = tl.load(a_ptr + b_idx * H + h)
    dt_val = tl.load(dt_bias_ptr + h)
    A_val = tl.load(A_log_ptr + h)
    b_val = tl.load(b_ptr + b_idx * H + h)

    # softplus(x) = log(1 + exp(x))
    splus = tl.log(1.0 + tl.exp(a_val + dt_val))
    # g = exp(-exp(A) * softplus(a + dt))
    g_val = tl.exp(-tl.exp(A_val) * splus)
    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_ptr + b_idx * H + h, g_val)
    tl.store(beta_ptr + b_idx * H + h, beta_val)


@triton.jit
def _vec_matmul_tile_kernel(k_ptr, state_ptr, out_ptr,
                            K: tl.constexpr, V: tl.constexpr):
    """
    out[i] = sum_v k[i] * state[v, i], for i in [0..K-1]
    k_ptr: [K]
    state_ptr: [V, K] row-major
    out_ptr: [K]
    """
    for i in tl.static_range(0, K):
        acc = tl.zeros((), dtype=tl.float32)
        for v in tl.static_range(0, V):
            kv = tl.load(k_ptr + i)
            state_val = tl.load(state_ptr + v * K + i)
            acc += kv * state_val
        tl.store(out_ptr + i, acc)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr, scale,
                          V: tl.constexpr, K: tl.constexpr):
    """
    Compute output scalar = scale * sum_v sum_k q[k] * new_state[v, k]
    q_ptr: [K]
    new_state_ptr: [V, K] row-major
    out_ptr: [1] scalar
    """
    acc = tl.zeros((), dtype=tl.float32)
    for v in tl.static_range(0, V):
        for k in tl.static_range(0, K):
            qk = tl.load(q_ptr + k)
            ns = tl.load(new_state_ptr + v * K + k)
            acc += qk * ns
    total = acc * scale
    tl.store(out_ptr, total)


class ModelNew(torch.nn.Module):
    def __init__(self, K=128, V=128, NUM_HEADS=8, NUM_Q_HEADS=4, NUM_K_HEADS=4, NUM_V_HEADS=8):
        super().__init__()
        self.K = K
        self.V = V
        self.NUM_HEADS = NUM_HEADS
        self.NUM_Q_HEADS = NUM_Q_HEADS
        self.NUM_K_HEADS = NUM_K_HEADS
        self.NUM_V_HEADS = NUM_V_HEADS

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward:
        - Computes g and beta per (batch, head) using Triton.
        - For each (batch b, head h):
          - q_h, k_h, v_h, state_h
          - old_v = k_h @ state_h
          - new_v = beta[h] * v_h + (1 - beta[h]) * old_v
          - state_remove = k_h @ old_v
          - state_update = k_h @ new_v
          - new_state_h = g[h] * state_h - state_remove + state_update
          - output[b, h] = scale * (q_h @ new_state_h)
        Returns:
          - output: [B, 1, NUM_HEADS, V] in bfloat16
          - new_state: [B, NUM_HEADS, V, K] in float32
        """
        # Shapes: q: [B, 1, 4, K], k: [B, 1, 4, K], v: [B, 1, 8, V], state: [B, 8, V, K]
        B = q.shape[0]
        device = q.device

        # Ensure contiguous and float32 for Triton
        a_bh = a[:, 0, :].contiguous().float()        # [B, H]
        dt_bias_h = dt_bias.contiguous().float()      # [H]
        A_log_h = A_log.contiguous().float()          # [H]
        b_bh = b[:, 0, :].contiguous().float()        # [B, H]

        # Allocate outputs for g and beta
        g_out = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute g and beta
        _compute_g_and_beta_kernel[(B * self.NUM_HEADS,)](
            a_bh, dt_bias_h, b_bh, A_log_h,
            g_out, beta_out,
            B=B, H=self.NUM_HEADS
        )

        # Prepare inputs: squeeze T=1
        q_b = q.squeeze(1).contiguous().float()       # [B, 4, K]
        k_b = k.squeeze(1).contiguous().float()       # [B, 4, K]
        v_b = v.squeeze(1).contiguous().float()       # [B, 8, V]
        state_b = state.contiguous().float()          # [B, 8, V, K]

        output_f = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)
        new_state_f = torch.empty((B, self.NUM_HEADS, self.V, self.K), dtype=torch.float32, device=device)

        for b_idx in range(B):
            for h in range(self.NUM_HEADS):
                # Extract vectors and matrices
                q_h = q_b[b_idx, h, :]               # [K]
                k_h = k_b[b_idx, h, :]               # [K]
                state_h = state_b[b_idx, h, :, :]    # [V, K]
                v_h = v_b[b_idx, h, :]               # [V]
                g_val = g_out[b_idx, h]
                beta_val = beta_out[b_idx, h]

                # Compute old_v = k_h @ state_h (V=128)
                old_v = torch.empty((self.K,), dtype=torch.float32, device=device)
                _vec_matmul_tile_kernel[(self.K,)](
                    k_h, state_h, old_v,
                    K=self.K, V=self.V
                )

                # Compute new_v = beta * v_h + (1 - beta) * old_v
                new_v = beta_val * v_h + (1.0 - beta_val) * old_v  # [K]

                # Compute state_remove = k_h @ old_v (scalar)
                state_remove = torch.empty((), dtype=torch.float32, device=device)
                _vec_matmul_tile_kernel[(1,)](
                    k_h, old_v, state_remove,
                    K=self.K, V=1
                )

                # Compute state_update = k_h @ new_v (scalar)
                state_update = torch.empty((), dtype=torch.float32, device=device)
                _vec_matmul_tile_kernel[(1,)](
                    k_h, new_v, state_update,
                    K=self.K, V=1
                )

                # Update new state elementwise: new_state = g * state - state_remove + state_update
                new


def run(*args):
    return ModelNew()(*args)
