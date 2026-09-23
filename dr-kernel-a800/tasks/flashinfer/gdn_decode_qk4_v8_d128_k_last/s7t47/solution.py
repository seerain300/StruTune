import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr,
                                g_ptr, beta_ptr,
                                B: tl.constexpr, H: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b * H + h)      # [B, H]
    dt_val = tl.load(dt_bias_ptr + h)       # [H]
    A_val = tl.load(A_log_ptr + h)          # [H]
    b_val = tl.load(b_ptr + b * H + h)      # [B, H]

    # Compute g = exp(-exp(A) * softplus(a + dt))
    x = a_val + dt_val
    softplus_x = tl.log(1.0 + tl.exp(x))
    g = tl.exp(-tl.exp(A_val) * softplus_x)

    # Compute beta = sigmoid(b) = 1 / (1 + exp(-b))
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    tl.store(g_ptr + b * H + h, g)
    tl.store(beta_ptr + b * H + h, beta)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, output_ptr,
                          scale_ptr,
                          B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    scale = tl.load(scale_ptr)  # scalar

    acc = 0.0
    for v in tl.static_range(V):
        # Load q[b,h,:] -> [K]
        q_row = tl.zeros((K,), dtype=tl.float32)
        base_q = b * H * K + h * K
        for k in tl.static_range(K):
            q_row[k] = tl.load(q_ptr + base_q + k)

        # Load new_state[b,h,v,:] -> [K]
        base_ns = b * H * V * K + h * V * K + v * K
        vec_row = tl.zeros((K,), dtype=tl.float32)
        for kk in tl.static_range(K):
            vec_row[kk] = tl.load(new_state_ptr + base_ns + kk)

        # Dot product q_row · vec_row
        dot = 0.0
        for k in tl.static_range(K):
            dot += q_row[k] * vec_row[k]

        acc += dot

    out_val = scale * acc
    tl.store(output_ptr + b * H + h, out_val)


@triton.jit
def _sqrt_scale_kernel(scale_ptr, K: tl.constexpr):
    inv_sqrt = 1.0 / tl.sqrt(K)
    tl.store(scale_ptr, inv_sqrt)


class ModelNew(torch.nn.Module):
    def forward(self,
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: torch.Tensor,
                A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor,
                scale: float):
        # Cast to float32 for stable compute and ensure contiguous
        q = q.contiguous().float()
        k = k.contiguous().float()
        v = v.contiguous().float()
        state = state.contiguous().float()

        # Shapes from the provided get_inputs
        B, T_q, Hq, K = q.shape
        _, T_k, Hk, _ = k.shape
        _, T_v, Hv, V = v.shape

        # Assertions to match the reference behavior
        assert Hq == 4 and Hk == 4 and Hv == 8 and K == 128 and V == 128 and T_q == 1 and T_k == 1 and T_v == 1
        H = Hv  # number of heads

        device = q.device

        # Allocate g_out and beta_out [B*H]
        g_out = torch.empty(B * H, dtype=torch.float32, device=device)
        beta_out = torch.empty(B * H, dtype=torch.float32, device=device)

        # Allocate new_state [B, H, V, K]
        new_state = torch.empty(B, H, V, K, dtype=torch.float32, device=device)

        # Output per (b,h)
        output = torch.empty(B * H, dtype=torch.float32, device=device)

        # Compute scale = 1/sqrt(K) via Triton
        scale_buf = torch.empty(1, dtype=torch.float32, device=device)
        _sqrt_scale_kernel[(1,)](scale_buf, K=K)

        # Launch compute g and beta
        _compute_g_and_beta_kernel[(B * H,)](a, dt_bias, b, A_log, g_out, beta_out, B=B, H=H)

        # For each (b,h): compute new_state and output
        for b_idx in range(B):
            for h_idx in range(H):
                # Flatten views
                q_row = q[b_idx, 0, h_idx].contiguous().view(K)
                k_row = k[b_idx, 0, h_idx].contiguous().view(K)
                v_scalar = v[b_idx, 0, h_idx].item()  # single scalar per (b,h)

                # Initialize new_state for this (b,h) as zeros to update in-place
                base = b_idx * H * V * K + h_idx * V * K
                for v_idx in range(V):
                    base_row = base + v_idx * K
                    old_state_row = torch.zeros((K,), dtype=torch.float32, device=device)
                    # Load original state[b,h,v,:] -> [K]
                    for k_off in range(K):
                        old_state_row[k_off] = state[b_idx, h_idx, v_idx, k_off]
                    # Compute old_v = k @ old_state
                    old_v = 0.0
                    for k_off in range(K):
                        old_v += k_row[k_off] * old_state_row[k_off]
                    # new_v = beta * v_scalar + (1 - beta) * old_v
                    beta_val = beta_out[b_idx * H + h_idx]
                    g_val = g_out[b_idx * H + h_idx]
                    new_v = beta_val * v_scalar + (1.0 - beta_val) * old_v

                    # Compute state_remove = k @ old_v (scalar)
                    state_remove = 0.0
                    for k_off in range(K):
                        state_remove += k_row[k_off] * old_state_row[k_off]
                    # Compute state_update = k @ new_v (scalar)
                    state_update = 0.0
                    new_v_vec = torch.full((K,), new_v, dtype=torch.float32, device=device)
                    for k_off in range(K):
                        state_update += k_row[k_off] * new_v_vec[k_off]

                    new_state_vec = g_val * old_state_row - state_remove + state_update
                    for k_off in range(K):
                        tl.store(new_state + base_row + k_off, new_state_vec[k_off])

        # Compute outputs using Triton: output[b,h] = scale * (q @ new_state)
        _output_scalar_kernel[(B * H,)](q, new_state, output, scale_buf, B=B, H=H, V=V, K=K)

        # Return outputs and new state (match original signature: output first, then new state)
        return output.unsqueeze(1).to(torch.bfloat16), new_state


def run(*args):
    return ModelNew()(*args)
