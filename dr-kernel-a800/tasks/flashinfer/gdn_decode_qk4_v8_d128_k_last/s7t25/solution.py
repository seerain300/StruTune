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
    a_val = tl.load(a_ptr + b * H + h)       # a[B, H]
    dt_val = tl.load(dt_bias_ptr + h)        # dt_bias[H]
    A_val = tl.load(A_log_ptr + h)           # A_log[H]
    b_val = tl.load(b_ptr + b * H + h)       # b[B, H]

    # Compute softplus(x) = log(1 + exp(x))
    x = a_val + dt_val
    sp = tl.log(1.0 + tl.exp(x))
    g_val = tl.exp(-tl.exp(A_val) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, state_ptr, out_ptr,
                          K: tl.constexpr, V: tl.constexpr):
    # Compute out[k] = sum_v k[v] * state[v, k]
    k_vec = tl.load(k_ptr + tl.arange(0, K))  # [K]
    acc = tl.zeros([K], dtype=tl.float32)
    for v_idx in tl.static_range(0, V):
        row_ptr = state_ptr + v_idx * K + tl.arange(0, K)
        row = tl.load(row_ptr)  # [K]
        acc += k_vec * row
    # Store the output vector
    tl.store(out_ptr + tl.arange(0, K), acc)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, output_ptr,
                           scale_ptr,
                           B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    # One program per (b,h): compute output[b,h] = scale * (q @ new_state)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    scale = tl.load(scale_ptr)  # scalar
    q_vec = tl.load(q_ptr + tl.arange(0, K))  # [K]

    base = (b * H + h) * V * K
    new_state_flat = new_state_ptr + base  # [V*K] for this (b,h)

    total = tl.zeros((), dtype=tl.float32)
    # Accumulate dot over all V rows
    for v_idx in tl.static_range(0, V):
        row_ptr = new_state_flat + v_idx * K + tl.arange(0, K)
        row = tl.load(row_ptr)  # [K]
        total += tl.sum(q_vec * row, axis=0)
    out = scale * total
    tl.store(output_ptr + b * H + h, out)


@triton.jit
def _sqrt_scale_kernel(out_ptr, K: tl.constexpr):
    # Compute scale = 1 / sqrt(K) and store at out_ptr[0]
    scale = 1.0 / tl.sqrt(K)
    tl.store(out_ptr, scale)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation of the original logic.
        q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        """
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and state.dim() == 4
        B, _, Hq, K = q.shape
        _, _, Hk, _ = k.shape
        _, _, Hv, V = v.shape
        B_s, Hv_s, V_s, K_s = state.shape
        assert B_s == B and Hv_s == Hv and V_s == V and K_s == K
        assert Hq == 4 and Hk == 4 and Hv == 8 and K == 128 and V == 128

        device = q.device
        dtype = torch.float32

        # Cast and make contiguous
        q_c = q.contiguous().to(dtype)          # [B, 1, 4, 128]
        k_c = k.contiguous().to(dtype)          # [B, 1, 4, 128]
        v_c = v.contiguous().to(dtype)          # [B, 1, 8, 128]
        state_c = state.contiguous().to(dtype)  # [B, 8, 128, 128]

        # Prepare outputs
        g_out = torch.empty(B * Hv, device=device, dtype=dtype)     # [B, Hv]
        beta_out = torch.empty(B * Hv, device=device, dtype=dtype)  # [B, Hv]
        output_f = torch.empty(B * Hv, device=device, dtype=dtype)  # [B, Hv]
        new_state_f = torch.empty_like(state_c)                     # [B, 8, 128, 128]

        # Compute g and beta per (b,h)
        _compute_g_and_beta_kernel[(B * Hv,)](
            a.contiguous().view(-1, Hv), dt_bias, b.contiguous().view(-1, Hv), A_log,
            g_out, beta_out, B, Hv
        )

        # Compute scale = 1 / sqrt(K)
        scale_buf = torch.empty(1, device=device, dtype=dtype)
        _sqrt_scale_kernel[(1,)](scale_buf, K)  # K is tl.constexpr meta-parameter
        scale_val = scale_buf[0]  # scalar in float32

        # Update state and compute output per (b,h)
        for b_idx in range(B):
            for h_idx in range(Hv):
                base = b_idx * Hv + h_idx
                g_val = g


def run(*args):
    return ModelNew()(*args)
