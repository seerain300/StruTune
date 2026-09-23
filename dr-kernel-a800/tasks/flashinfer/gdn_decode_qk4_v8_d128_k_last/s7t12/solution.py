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
    # Load scalars
    a_val = tl.load(a_ptr + b_idx * H + h)      # a has shape [B,H]
    dt_val = tl.load(dt_bias_ptr + h)           # dt_bias has shape [H]
    A_val = tl.load(A_log_ptr + h)              # A_log has shape [H]
    b_val = tl.load(b_ptr + b_idx * H + h)      # b has shape [B,H]

    # softplus(x) = log(1 + exp(x))
    splus = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * splus)
    sig = 1.0 / (1.0 + tl.exp(-b_val))          # sigmoid

    tl.store(g_ptr + b_idx * H + h, g_val)
    tl.store(beta_ptr + b_idx * H + h, sig)


@triton.jit
def _vec_matmul_tile_kernel(k_ptr, state_ptr, out_ptr,
                            K: tl.constexpr, V: tl.constexpr):
    # Compute out[k] = sum_v k[k] * state[v,k] for k in [0..K-1], v in [0..V-1]
    # k_ptr: [K], state_ptr: [V, K] row-major, out_ptr: [K]
    for k in tl.static_range(0, K):
        acc = 0.0
        for v in tl.static_range(0, V):
            state_elem = tl.load(state_ptr + v * K + k)  # element at row v, col k
            acc += state_elem * tl.load(k_ptr + k)       # k[k]
        tl.store(out_ptr + k, acc)


@triton.jit
def _vec_matmul_scalar_kernel(k_ptr, vec_ptr, out_ptr,
                              K: tl.constexpr):
    # Compute scalar = sum_k k[k] * vec[k]
    acc = 0.0
    for k in tl.static_range(0, K):
        acc += tl.load(k_ptr + k) * tl.load(vec_ptr + k)
    tl.store(out_ptr, acc)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr, scale,
                          V: tl.constexpr, K: tl.constexpr):
    # output = scale * sum_v sum_k q[v] * new_state[v, k]
    acc = 0.0
    for v in tl.static_range(0, V):
        for k in tl.static_range(0, K):
            acc += tl.load(q_ptr + v * K + k) * tl.load(new_state_ptr + v * K + k)
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
        # Shapes (as per evaluator inputs):
        # q: [B, 1, NUM_Q_HEADS, K]
        # k: [B, 1, NUM_K_HEADS, K]
        # v: [B, 1, NUM_V_HEADS, V]
        # state: [B, NUM_V_HEADS, V, K] (k-last)
        B = q.shape[0]
        device = q.device

        # Convert to float32 and make contiguous for Triton
        a_bh = a[:, 0, :].contiguous().float()        # [B, H]
        dt_bias_h = dt_bias.contiguous().float()      # [H]
        A_log_h = A_log.contiguous().float()          # [H]
        b_bh = b[:, 0, :].contiguous().float()        # [B, H]

        q_b = q.squeeze(1).contiguous().float()       # [B, 4, K]
        k_b = k.squeeze(1).contiguous().float()       # [B, 4, K]
        v_b = v.squeeze(1).contiguous().float()       # [B, 8, V]
        state_b = state.contiguous().float()          # [B, 8, V, K]

        # Output buffers
        output_f = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)  # [B, H]
        new_state_f = torch.empty((B, self.NUM_HEADS, self.V, self.K), dtype=torch.float32, device=device)  # [B, H, V, K]

        # Compute g and beta per (b, h)
        g_out = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)
        _compute_g_and_beta_kernel[(B * self.NUM_HEADS,)](
            a_bh, dt_bias_h, b_bh, A_log_h, g_out, beta_out,
            B=B, H=self.NUM_HEADS
        )

        # Per (b, h) updates
        for b_idx in range(B):
            for h in range(self.NUM_HEADS):
                q_h = q_b[b_idx, h, :]               # [K]
                k_h = k_b[b_idx, h, :]               # [K]
                state_h = state_b[b_idx, h, :, :]    # [V, K]
                v_h = v_b[b_idx, h, :]               # [V]

                g_val = g_out[b_idx, h]
                beta_val = beta_out[b_idx, h]

                # old_v = k_h @ state_h (V=128)
                old_v = torch.empty((self.K,), dtype=torch.float32, device=device)
                _vec_matmul_tile_kernel[(self.K,)](
                    k_h, state_h, old_v,
                    K=self.K, V=self.V
                )

                # new_v = beta * v_h + (1 - beta) * old_v
                new_v = beta_val * v_h + (1.0 - beta_val) * old_v  # [K]

                # state_remove = k_h @ old_v (scalar)
                state_remove = torch.empty((), dtype=torch.float32, device=device)
                _vec_matmul_scalar_kernel[(1,)](
                    k_h, old_v, state_remove,
                    K=self.K
                )

                # state_update = k_h @ new_v (scalar)
                state_update = torch.empty((), dtype=torch.float32, device=device)
                _vec_matmul_scalar_kernel[(1,)](
                    k_h, new_v, state_update,
                    K=self.K
                )

                # new_state = g * state - state_remove + state_update
                # We update new_state_f[b, h, :, :]
                new_state_row = g_val * state_h - (state_remove - state_update)  # broadcasting scalar
                new_state_f[b_idx, h, :, :] = new_state_row

                # output[b, h] = scale * (q_h @ new_state_row)
                out_scalar = torch.empty((), dtype=torch.float32, device=device)
                _output_scalar_kernel[(1,)](
                    q_h, new_state_row, out_scalar, scale,
                    V=self.V, K=self.K
                )
                output_f[b_idx, h] = out_scalar

        # Return output in bfloat16 to match original casting behavior, and new state in float32
        output_bf16 = output_f.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H]
        return output_bf16, new_state_f


def run(*args):
    return ModelNew()(*args)
