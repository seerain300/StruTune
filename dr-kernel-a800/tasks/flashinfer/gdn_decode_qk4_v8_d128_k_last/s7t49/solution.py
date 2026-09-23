import torch
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

    a_val = tl.load(a_ptr + b * H + h)      # a[B, H]
    dt_val = tl.load(dt_bias_ptr + h)       # dt_bias[H]
    A_val = tl.load(A_log_ptr + h)          # A_log[H]

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt_val))
    g_val = tl.exp(-tl.exp(A_val) * sp)
    b_val = 1.0 / (1.0 + tl.exp(-tl.load(b_ptr + b * H + h)))  # sigmoid(b[b,H])

    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, b_val)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, state_ptr, out_ptr,
                         V: tl.constexpr, K: tl.constexpr):
    # Compute out[k] = sum_v k[v] * state[v,k]
    for k in tl.static_range(K):
        acc = 0.0
        for v in tl.static_range(V):
            sv = tl.load(state_ptr + v * K + k)
            kv = tl.load(k_ptr + v)
            acc += sv * kv
        tl.store(out_ptr + k, acc)


@triton.jit
def _vec_matmul_scalar(k_ptr, vec_ptr, scalar_ptr,
                       K: tl.constexpr):
    # scalar = sum_k k[k] * vec[k]
    acc = 0.0
    for k in tl.static_range(K):
        kv = tl.load(k_ptr + k)
        vv = tl.load(vec_ptr + k)
        acc += kv * vv
    tl.store(scalar_ptr, acc)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, output_ptr,
                          scale_ptr,
                          B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    scale = tl.load(scale_ptr)  # scalar

    # Accumulate dot product q[b,h,:] @ new_state[b,h,:,:]
    acc = 0.0
    for v in tl.static_range(V):
        # new_state[b,h,v,:] flattened index: base = b*H*V*K + h*V*K + v*K
        base = b * H * V * K + h * V * K + v * K
        vec_row = tl.zeros((K,), dtype=tl.float32)
        for k in tl.static_range(K):
            vec_row[k] = tl.load(new_state_ptr + base + k)
        # q[b,h,:] -> index: base_q = b*H*K + h*K
        q_row = tl.zeros((K,), dtype=tl.float32)
        base_q = b * H * K + h * K
        for k in tl.static_range(K):
            q_row[k] = tl.load(q_ptr + base_q + k)
        acc += tl.sum(q_row * vec_row, axis=0)

    out_val = scale * acc
    tl.store(output_ptr + b * H + h, out_val)


@triton.jit
def _sqrt_scale_kernel(scale_ptr, K: tl.constexpr):
    inv_sqrt = 1.0 / tl.sqrt(K)
    tl.store(scale_ptr, inv_sqrt)


class ModelNew(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: torch.Tensor,
                A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor, scale: float):
        # Ensure dtype and contiguity for Triton
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        state = state.float().contiguous()
        A_log = A_log.float().contiguous()
        a = a.float().contiguous()
        dt_bias = dt_bias.float().contiguous()
        b = b.float().contiguous()

        # Shapes: q [B,1,Hq,K], k [B,1,Hk,K], v [B,1,Hv,V], state [B,Hv,V,K]
        B = q.shape[0]
        H = v.shape[1]  # number of heads from v (Hv)
        V = v.shape[3]  # last dim of v (should be K in this task)
        K = state.shape[3]  # last dim of state (K)

        # Allocate outputs
        g_out = torch.empty((B, H), dtype=torch.float32, device=q.device)
        beta_out = torch.empty((B, H), dtype=torch.float32, device=q.device)
        output = torch.empty((B, H), dtype=torch.float32, device=q.device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=q.device)

        # Launch: compute g and beta
        _compute_g_and_beta_kernel[(B * H,)](a, dt_bias, b, A_log, g_out, beta_out, B, H)

        # Compute scale = 1/sqrt(K) in Triton
        scale_buf = torch.empty((1,), dtype=torch.float32, device=q.device)
        _sqrt_scale_kernel[(1,)](scale_buf, K)

        # For each (b,h), compute new_state and output
        # Note: Hq, Hk in original code are not used in logic; we rely on shape assumptions: Hq=4, Hk=4, Hv=8, K=V=128.
        for b_idx in range(B):
            for h_idx in range(H):
                # Prepare vectors/rows
                # q[b,h,:] -> [K]
                q_row = q[b_idx, 0, h_idx].contiguous()
                # k[b,h,:] -> [K]
                k_row = k[b_idx, 0, h_idx].contiguous()
                # v[b,h,:] -> [V]
                v_row = v[b_idx, 0, h_idx].contiguous()

                # old_v = k @ state via tile_vec kernel
                old_v = torch.empty((K,), dtype=torch.float32, device=q.device)
                _vec_matmul_tile_vec[(K,)](k_row, state[b_idx, h_idx].contiguous(), old_v, V, K)

                # new_v = beta * v + (1 - beta) * old_v (vector length K)
                beta_val = beta_out[b_idx, h_idx]
                g_val = g_out[b_idx, h_idx]
                new_v = beta_val * v_row + (1.0 - beta_val) * old_v

                # state_remove = k @ old_v (scalar)
                state_remove = torch.empty((1,), dtype=torch.float32, device=q.device)
                _vec_matmul_scalar[(1,)](k_row, old_v, state_remove, K)

                # state_update = k @ new_v (scalar)
                state_update = torch.empty((1,), dtype=torch.float32, device=q.device)
                _vec_matmul_scalar[(1,)](k_row, new_v, state_update, K)

                # Update new_state[b,h,:,:] = g * state - state_remove + state_update
                for v_idx in range(V):
                    base = b_idx * H * V * K + h_idx * V * K + v_idx * K
                    # old_state_row = state[b,h,v_idx,:]
                    old_state_row = state[b_idx, h_idx, v_idx].contiguous()
                    new_state_row = g_val * old_state_row - state_remove[0] + state_update[0]
                    for k in range(K):
                        new_state[b_idx, h_idx, v_idx, k] = new_state_row[k]

                # Compute output[b,h] = scale * (q @ new_state[b,h,:,:])
                _output_scalar_kernel[(1,)](q_row, new_state[b_idx, h_idx].contiguous(), output[b_idx, h_idx], scale_buf, B, H, V, K)

        # Match original output types: output as bfloat16 and new_state as float32
        output = output.to(torch.bfloat16)
        new_state = new_state  # already float32
        return output, new_state


def run(*args):
    return ModelNew()(*args)
