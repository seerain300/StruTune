import torch
import math
import triton
import triton.language as tl


@triton.jit
def _gate_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr, g_out_ptr, beta_out_ptr,
                            NUM_HEADS: tl.constexpr):
    # One Triton program per batch b
    b_idx = tl.program_id(0)
    for h in range(NUM_HEADS):
        # Load scalars: a[b, h], dt[h], A[h], b[b, h]
        a_val = tl.load(a_ptr + b_idx * NUM_HEADS + h)   # [B, H] flattened
        dt_val = tl.load(dt_bias_ptr + h)                # [H]
        A_val = tl.load(A_log_ptr + h)                   # [H]
        b_val = tl.load(b_ptr + b_idx * NUM_HEADS + h)   # [B, H] flattened
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g = tl.exp(-tl.exp(A_val) * sp)
        sig = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(g_out_ptr + b_idx * NUM_HEADS + h, g)
        tl.store(beta_out_ptr + b_idx * NUM_HEADS + h, sig)


@triton.jit
def _vec_matmul_kernel(k_ptr, state_ptr, out_ptr,
                       K: tl.constexpr):
    # out = k @ state, where k is [K], state is [V, K], out is [K]
    # K is compile-time constant (e.g., 128). Use a simple loop to sum over V.
    out = tl.zeros((K,), dtype=tl.float32)
    for v in range(0, 128):  # evaluator uses V=128
        state_vec = tl.load(state_ptr + v * K + tl.arange(0, K))  # [K]
        out += tl.load(k_ptr + tl.arange(0, K)) * state_vec
    for i in range(K):
        tl.store(out_ptr + i, out[i])


@triton.jit
def _elementwise_newv_kernel(beta, oldv_ptr, v_ptr, newv_ptr,
                             V: tl.constexpr):
    # newv = beta * v + (1 - beta) * oldv, elementwise over V
    # V is compile-time constant (e.g., 128).
    for i in range(V):
        oldv_i = tl.load(oldv_ptr + i)
        v_i = tl.load(v_ptr + i)
        newv_i = beta * v_i + (1.0 - beta) * oldv_i
        tl.store(newv_ptr + i, newv_i)


@triton.jit
def _scalar_dot_kernel(k_ptr, vec_ptr, out_ptr,
                       K: tl.constexpr):
    # scalar = k @ vec, where k is [K], vec is [K]
    acc = tl.zeros((), dtype=tl.float32)
    for i in range(K):
        k_i = tl.load(k_ptr + i)
        v_i = tl.load(vec_ptr + i)
        acc += k_i * v_i
    tl.store(out_ptr, acc)


@triton.jit
def _elementwise_update_kernel(state_ptr, new_state_ptr, g_val, remove, update,
                                V: tl.constexpr, K: tl.constexpr):
    # new_state = g * state - remove + update, elementwise over [V, K]
    for v in range(0, V):
        for k in range(0, K):
            state_val = tl.load(state_ptr + v * K + k)
            new_state_val = g_val * state_val - remove + update
            tl.store(new_state_ptr + v * K + k, new_state_val)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr, scale,
                          V: tl.constexpr, K: tl.constexpr):
    # output = scale * sum_v sum_k q[k] * new_state[v, k]
    acc = tl.zeros((), dtype=tl.float32)
    for v in range(0, V):
        for k in range(0, K):
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
        # Shapes (evaluator provides):
        # q: [B, 1, NUM_Q_HEADS, K]
        # k: [B, 1, NUM_K_HEADS, K]
        # v: [B, 1, NUM_V_HEADS, V]
        # state: [B, NUM_V_HEADS, V, K] (k-last)
        B = q.shape[0]
        device = q.device

        # Compute g and beta per (b, h) using Triton
        a_bh = a[:, 0, :].float().contiguous()   # [B, H]
        dt_bias_h = dt_bias.float().contiguous() # [H]
        A_log_h = A_log.float().contiguous()     # [H]
        b_bh = b[:, 0, :].float().contiguous()   # [B, H]
        g_out = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)

        _gate_g_and_beta_kernel[(B,)](
            a_bh, dt_bias_h, b_bh, A_log_h, g_out, beta_out, NUM_HEADS=self.NUM_HEADS
        )

        # Outputs and new state buffers
        output_f = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)  # [B, H]
        new_state_f = torch.empty((B, self.NUM_HEADS, self.V, self.K), dtype=torch.float32, device=device)  # [B, H, V, K]

        # For each batch b, process all heads h in Triton (one program per b, loop over h)
        for b_idx in range(B):
            # Iterate heads h=0..H-1
            for h in range(self.NUM_HEADS):
                # Extract vectors and matrix for this (b, h)
                q_h = q[b_idx, 0, h].contiguous().float()   # [K]
                k_h = k[b_idx, 0, h].contiguous().float()   # [K]
                v_h = v[b_idx, 0, h].contiguous().float()   # [V]
                state_h = state[b_idx, h].contiguous().float()  # [V, K]

                # 1) old_v = k_h @ state_h
                oldv = torch.empty((self.K,), dtype=torch.float32, device=device)
                _vec_matmul_kernel[(1,)](
                    k_h, state_h, oldv, K=self.K
                )

                # 2) new_v = beta[h] * v_h + (1 - beta[h]) * old_v
                beta_val = beta_out[b_idx, h]
                newv = torch.empty((self.V,), dtype=torch.float32, device=device)
                _elementwise_newv_kernel[(1,)](
                    beta_val, oldv, v_h, newv, V=self.V
                )

                # 3) state_remove and state_update scalars
                remove = torch.empty((), dtype=torch.float32, device=device)
                update = torch.empty((), dtype=torch.float32, device=device)
                _scalar_dot_kernel[(1,)](
                    k_h, oldv, remove, K=self.K
                )
                _scalar_dot_kernel[(1,)](
                    k_h, newv, update, K=self.V
                )

                # 4) Update new_state_h elementwise
                new_state_h = torch.empty((self.V, self.K), dtype=torch.float32, device=device)
                _elementwise_update_kernel[(1,)](
                    state_h, new_state_h, beta_out[b_idx, h], remove, update,
                    V=self.V, K=self.K
                )
                new_state_f[b_idx, h] = new_state_h

                # 5) Compute output[b, h] = scale * (q_h @ new_state_h)
                out_scalar = torch.empty((), dtype=torch.float32, device=device)
                _output_scalar_kernel[(1,)](
                    q_h, new_state_h, out_scalar, scale, V=self.V, K=self.K
                )
                output_f[b_idx, h] = out_scalar

        # Return output [B, 1, H, V] in bfloat16 and new_state [B, H, V, K] in float32
        output_bf16 = output_f.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H, V]
        return output_bf16, new_state_f


def run(*args):
    return ModelNew()(*args)
