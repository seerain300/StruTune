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
    a_val = tl.load(a_ptr + b_idx * H + h)   # a: [B, H]
    dt_val = tl.load(dt_bias_ptr + h)        # dt_bias: [H]
    A_val = tl.load(A_log_ptr + h)           # A_log: [H]
    b_val = tl.load(b_ptr + b_idx * H + h)   # b: [B, H]

    # softplus(x) = log(1 + exp(x))
    splus = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * splus)
    sig = 1.0 / (1.0 + tl.exp(-b_val))       # sigmoid

    tl.store(g_ptr + b_idx * H + h, g_val)
    tl.store(beta_ptr + b_idx * H + h, sig)


@triton.jit
def _vec_matmul_tile_kernel(k_ptr, state_ptr, out_ptr,
                            K: tl.constexpr, V: tl.constexpr):
    # Compute out[k] = sum_v k[k] * state[v,k] for v in [0..V-1]
    # k: [K], state: [V, K], out: [K]
    out = tl.zeros((K,), dtype=tl.float32)
    for v in tl.static_range(0, V):
        k_vec = tl.load(k_ptr + tl.arange(0, K))
        state_row = tl.load(state_ptr + v * K + tl.arange(0, K))
        out += k_vec * state_row
    tl.store(out_ptr, out)


@triton.jit
def _vec_matmul_scalar_kernel(k_ptr, vec_ptr, out_ptr, K: tl.constexpr):
    # Compute scalar = sum_k k[k] * vec[k]
    acc = tl.zeros((), dtype=tl.float32)
    k_vec = tl.load(k_ptr + tl.arange(0, K))
    v_vec = tl.load(vec_ptr + tl.arange(0, K))
    acc = tl.sum(k_vec * v_vec, axis=0)
    tl.store(out_ptr, acc)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr, scale,
                          V: tl.constexpr, K: tl.constexpr):
    # output = scale * sum_v sum_k q[k] * new_state[v, k]
    acc = tl.zeros((), dtype=tl.float32)
    for v in tl.static_range(0, V):
        q_vec = tl.load(q_ptr + tl.arange(0, K))
        ns_vec = tl.load(new_state_ptr + v * K + tl.arange(0, K))
        acc += tl.sum(q_vec * ns_vec, axis=0)
    total = acc * scale
    tl.store(out_ptr, total)


class ModelNew(torch.nn.Module):
    def __init__(self, K=128, V=128):
        super().__init__()
        self.K = K
        self.V = V

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Shapes: q: [B, 1, H_q, K], k: [B, 1, H_k, K], v: [B, 1, H_v, V], state: [B, H_v, V, K]
        B = q.shape[0]
        device = q.device

        # Prepare tensors as float32 for Triton compute
        a_bh = a[:, 0, :].contiguous().float()          # [B, H_v]
        dt_bias_h = dt_bias.contiguous().float()        # [H_v]
        A_log_h = A_log.contiguous().float()            # [H_v]
        b_bh = b[:, 0, :].contiguous().float()          # [B, H_v]

        q_s = q.squeeze(1).contiguous().float()         # [B, H_q, K]
        k_s = k.squeeze(1).contiguous().float()         # [B, H_k, K]
        v_s = v.squeeze(1).contiguous().float()         # [B, H_v, V]
        state_s = state.contiguous().float()            # [B, H_v, V, K]

        # Dynamic head counts
        q_heads = q_s.shape[1]
        k_heads = k_s.shape[1]
        v_heads = v_s.shape[1]

        # Outputs
        g_out = torch.empty((B, v_heads), dtype=torch.float32, device=device)   # [B, H_v]
        beta_out = torch.empty((B, v_heads), dtype=torch.float32, device=device)  # [B, H_v]

        # Launch Triton kernel to compute g and beta
        _compute_g_and_beta_kernel[(B * v_heads,)](
            a_bh, dt_bias_h, b_bh, A_log_h, g_out, beta_out,
            B=B, H=v_heads
        )

        # Prepare output and new state buffers
        output_f = torch.empty((B, v_heads), dtype=torch.float32, device=device)  # [B, H_v]
        new_state_f = torch.empty((B, v_heads, self.V, self.K), dtype=torch.float32, device=device)  # [B, H_v, V, K]

        # For each (b, h), compute and update
        for b_idx in range(B):
            for h in range(v_heads):
                # Extract vectors and matrices for this head
                q_h = q_s[b_idx, h, :]                  # [K], float32
                k_h = k_s[b_idx, h, :]                  # [K], float32
                state_h = state_s[b_idx, h, :, :]       # [V, K], float32
                v_h = v_s[b_idx, h, :]                  # [V], float32
                g_val = g_out[b_idx, h]                 # scalar
                beta_val = beta_out[b_idx, h]           # scalar

                # Compute old_v = k_h @ state_h
                old_v = torch.empty((self.K,), dtype=torch.float32, device=device)
                _vec_matmul_tile_kernel[(self.K,)](
                    q_h, state_h, old_v,
                    K=self.K, V=self.V
                )

                # Compute new_v = beta * v_h + (1 - beta) * old_v
                new_v = beta_val * v_h + (1.0 - beta_val) * old_v  # [V]

                # Compute state_remove = k_h @ old_v (scalar)
                state_remove = torch.empty((), dtype=torch.float32, device=device)
                _vec_matmul_scalar_kernel[(1,)](
                    k_h, old_v, state_remove,
                    K=self.K
                )

                # Compute state_update = k_h @ new_v (scalar)
                state_update = torch.empty((), dtype=torch.float32, device=device)
                _vec_matmul_scalar_kernel[(1,)](
                    k_h, new_v, state_update,
                    K=self.K
                )

                # Compute new_state elementwise: new_state = g * state - state_remove + state_update
                state_scaled = g_val * state_h
                new_state_h = state_scaled - state_remove + state_update  # broadcast scalar to [V, K]
                new_state_f[b_idx, h, :, :] = new_state_h

                # Compute output[b,h] = scale * (q_h @ new_state_h)
                # Implement q @ new_state_h scalar via Triton reduction kernel
                output_scalar = torch.empty((), dtype=torch.float32, device=device)
                _output_scalar_kernel[(1,)](
                    q_h, new_state_h, output_scalar, scale, V=self.V, K=self.K
                )
                output_f[b_idx, h] = output_scalar

        # Return output in bfloat16 and new_state in float32
        output_bf16 = output_f.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H_v]
        return output_bf16, new_state_f


def run(*args):
    return ModelNew()(*args)
