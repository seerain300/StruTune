import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr,
                                g_ptr, beta_ptr,
                                B: tl.constexpr, H: tl.constexpr):
    """
    Compute g[b, h] and beta[b, h] per (b, h).
    a_ptr: [B, H], dt_bias_ptr: [H], b_ptr: [B, H], A_log_ptr: [H]
    g_ptr: [B, H], beta_ptr: [B, H]
    """
    pid = tl.program_id(0)
    b_idx = pid // H
    h = pid % H
    if b_idx >= B:
        return

    a_val = tl.load(a_ptr + b_idx * H + h)     # float32
    dt_val = tl.load(dt_bias_ptr + h)          # float32
    A_val = tl.load(A_log_ptr + h)             # float32
    b_val = tl.load(b_ptr + b_idx * H + h)     # float32

    # softplus(x) = log(1 + exp(x))
    splus = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * splus)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_ptr + b_idx * H + h, g_val)
    tl.store(beta_ptr + b_idx * H + h, beta_val)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, state_ptr, out_ptr,
                         K: tl.constexpr, V: tl.constexpr):
    """
    Compute out[k] = sum_{v=0..V-1} state[v,k] * k[k] for k in [0..K-1].
    out_ptr: [K]
    """
    for k in tl.static_range(0, K):
        total = 0.0
        kk = tl.load(k_ptr + k)
        for v in tl.static_range(0, V):
            sv = tl.load(state_ptr + v * K + k)
            total += sv * kk
        tl.store(out_ptr + k, total)


@triton.jit
def _vec_matmul_tile_scalar(k_ptr, state_ptr, out_ptr,
                            K: tl.constexpr, V: tl.constexpr):
    """
    Compute scalar = sum_{k=0..K-1} k[k] * state[k,k] for V=1 using tiling over K.
    out_ptr: [1] single-element tensor
    """
    total = 0.0
    for k in tl.static_range(0, K):
        kk = tl.load(k_ptr + k)
        sk = tl.load(state_ptr + k * V + k)  # since V=1, this is state[k,0]
        total += kk * sk
    tl.store(out_ptr, total)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr,
                          V: tl.constexpr, K: tl.constexpr, scale):
    """
    Compute out = scale * (q @ new_state), where q is [V] and new_state is [V,K].
    """
    total = 0.0
    for v in tl.static_range(0, V):
        qv = tl.load(q_ptr + v)
        for k in tl.static_range(0, K):
            svk = tl.load(new_state_ptr + v * K + k)
            total += qv * svk
    total *= scale
    tl.store(out_ptr, total)


class ModelNew(torch.nn.Module):
    def __init__(self, num_q_heads=4, num_k_heads=4, num_v_heads=8, K=128, V=128):
        super().__init__()
        self.NUM_Q_HEADS = num_q_heads
        self.NUM_K_HEADS = num_k_heads
        self.NUM_V_HEADS = num_v_heads
        self.K = K
        self.V = V

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, num_q_heads, K], k: [B, 1, num_k_heads, K], v: [B, 1, num_v_heads, V], 
        state: [B, num_v_heads, V, K], A_log: [num_v_heads], a: [B, 1, num_v_heads], 
        dt_bias: [num_v_heads], b: [B, 1, num_v_heads], scale: float or None
        Returns (output [B, num_v_heads], new_state [B, num_v_heads, V, K])
        """
        device = q.device
        B = q.shape[0]
        # Reshape without using squeeze to avoid dimension-1 size issues
        q = q.contiguous().float().view(B, self.NUM_Q_HEADS, self.K)        # [B, num_q_heads, K]
        k = k.contiguous().float().view(B, self.NUM_K_HEADS, self.K)        # [B, num_k_heads, K]
        v = v.contiguous().float().view(B, self.NUM_V_HEADS, self.V)        # [B, num_v_heads, V]
        state = state.contiguous().float().view(B, self.NUM_V_HEADS, self.V, self.K)  # [B, num_v_heads, V, K]
        a = a.contiguous().float().view(B, self.NUM_V_HEADS)                # [B, num_v_heads]
        dt_bias = dt_bias.contiguous().float()                              # [num_v_heads]
        b = b.contiguous().float().view(B, self.NUM_V_HEADS)                # [B, num_v_heads]
        A_log = A_log.contiguous().float()                                  # [num_v_heads]

        # Allocate outputs
        g_out = torch.empty((B, self.NUM_V_HEADS), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, self.NUM_V_HEADS), dtype=torch.float32, device=device)
        output_f = torch.empty((B, self.NUM_V_HEADS), dtype=torch.float32, device=device)
        new_state_f = torch.empty((B, self.NUM_V_HEADS, self.V, self.K), dtype=torch.float32, device=device)

        # Launch kernel to compute g and beta
        _compute_g_and_beta_kernel[(B * self.NUM_V_HEADS,)](
            a, dt_bias, b, A_log,
            g_out, beta_out,
            B=B, H=self.NUM_V_HEADS
        )

        # For each (b, h), compute new state and output
        for b_idx in range(B):
            for h in range(self.NUM_V_HEADS):
                # Extract vectors and matrices
                q_h = q[b_idx, h, :].contiguous()                           # [K]
                k_h = k[b_idx, h, :].contiguous()                           # [K]
                state_h = state[b_idx, h, :, :].contiguous()                # [V, K]
                v_h = v[b_idx, h, :].contiguous()                           # [V]

                g_val = g_out[b_idx, h]
                beta_val = beta_out[b_idx, h]

                # Compute old_v = k_h @ state_h
                old_v = torch.empty((self.K,), dtype=torch.float32, device=device)
                _vec_matmul_tile_vec[(self.K,)](
                    k_h, state_h, old_v,
                    K=self.K, V=self.V
                )

                # Compute new_v = beta * v_h + (1 - beta) * old_v
                new_v = beta_val * v_h + (1.0 - beta_val) * old_v           # [K]

                # Compute state_remove = k_h @ old_v (scalar)
                state_remove = torch.empty((), dtype=torch.float32, device=device)
                _vec_matmul_tile_scalar[(1,)](
                    k_h, old_v, state_remove,
                    K=self.K, V=1
                )

                # Compute state_update = k_h @ new_v (scalar)
                state_update = torch.empty((), dtype=torch.float32, device=device)
                _vec_matmul_tile_scalar[(1,)](
                    k_h, new_v, state_update,
                    K=self.K, V=1
                )

                # Update new state elementwise: new_state = g * state - state_remove + state_update
                for v_idx in range(self.V):
                    row = state_h[v_idx, :]                                  # [K]
                    new_row = g_val * row - state_remove + state_update
                    new_state_f[b_idx, h, v_idx, :] = new_row                # assign vector

                # Compute output[b, h] = scale * (q_h @ new_state_h)
                scale_val = float(scale) if scale is not None else (1.0 / math.sqrt(self.K))
                out_scalar = torch.empty((), dtype=torch.float32, device=device)
                _output_scalar_kernel[(1,)](
                    q_h, new_state_f[b_idx, h, :, :], out_scalar,
                    V=self.V, K=self.K, scale=scale_val
                )
                output_f[b_idx, h] = out_scalar

        # Match original output dtype behavior: output is bfloat16, new_state kept float32
        output_out = output_f.unsqueeze(1).to(torch.bfloat16)
        return output_out, new_state_f


def run(*args):
    return ModelNew()(*args)
