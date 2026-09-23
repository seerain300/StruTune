import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(
    A_log_ptr,   # [H_v] float32
    a_ptr,       # [total_seq_len, H_v] float32
    dt_bias_ptr, # [H_v] float32
    b_ptr,       # [total_seq_len, H_v] float32
    g_ptr,       # [total_seq_len, H_v] float32
    beta_ptr,    # [total_seq_len, H_v] float32
    H_v: tl.constexpr,
    total_seq_len: tl.constexpr,
):
    # Each program handles one hv; loop over tokens to fill g[b, hv] and beta[b, hv]
    pid_hv = tl.program_id(0)
    # softplus(x) = log(1 + exp(x))
    for b in range(0, total_seq_len):
        a_val = tl.load(a_ptr + b * H_v + pid_hv)
        dt_val = tl.load(dt_bias_ptr + pid_hv)
        b_val = tl.load(b_ptr + b * H_v + pid_hv)
        # g = exp(-exp(A_log[hv]) * softplus(a[b, hv] + dt_bias[hv]))
        exp_A = tl.exp(tl.load(A_log_ptr + pid_hv))
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-exp_A * sp)
        # beta = sigmoid(b[b, hv]) = 1 / (1 + exp(-b))
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(g_ptr + b * H_v + pid_hv, g_val)
        tl.store(beta_ptr + b * H_v + pid_hv, beta_val)


@triton.jit
def _state_update_kernel(
    k_ptr,           # [total_seq_len, H_v, D] float32
    v_ptr,           # [total_seq_len, H_v, D] float32
    state_ptr,       # [H_v, D, D] float32
    g_ptr,           # [total_seq_len, H_v] float32
    beta_ptr,        # [total_seq_len, H_v] float32
    new_state_ptr,   # [H_v, D, D] float32
    D: tl.constexpr,
    total_seq_len: tl.constexpr,
    H_v: tl.constexpr,
):
    # 2D grid over (t, hv)
    pid_t = tl.program_id(0)
    pid_hv = tl.program_id(1)

    # Load params for this (t, hv)
    g_val = tl.load(g_ptr + pid_t * H_v + pid_hv)
    beta_val = tl.load(beta_ptr + pid_t * H_v + pid_hv)

    # Compute old_v = k[t, hv, :] @ state[hv, :, :]
    old_v = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        k_i = tl.load(k_ptr + pid_t * (H_v * D) + pid_hv * D + i)
        row = tl.load(state_ptr + pid_hv * D * D + i * D + tl.arange(0, D))  # [D]
        old_v += k_i * row
    # new_v = beta * v + (1 - beta) * old_v
    v_vec = tl.load(v_ptr + pid_t * (H_v * D) + pid_hv * D + tl.arange(0, D))  # [D]
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

    # Compute kT_old = sum_i k[t, hv, i] * old_v[i], kT_newv = sum_i k[t, hv, i] * new_v[i]
    kT_old = 0.0
    kT_newv = 0.0
    for i in range(0, D):
        k_i = tl.load(k_ptr + pid_t * (H_v * D) + pid_hv * D + i)
        kT_old += k_i * old_v[i]
        kT_newv += k_i * new_v[i]

    # Update state: new_state = g * state - kT_old + kT_newv
    for i in range(0, D):
        old_row = tl.load(state_ptr + pid_hv * D * D + i * D + tl.arange(0, D))  # [D]
        new_row = g_val * old_row - kT_old + kT_newv
        tl.store(new_state_ptr + pid_hv * D * D + i * D + tl.arange(0, D), new_row)


@triton.jit
def _output_kernel(
    q_ptr,           # [total_seq_len, H_q, D] float32
    new_state_ptr,   # [H_v, D, D] float32
    scale,           # float32 scalar
    out_ptr,         # [total_seq_len, H_v, D] float32
    D: tl.constexpr,
    H_q: tl.constexpr,
    H_v: tl.constexpr,
    total_seq_len: tl.constexpr,
):
    # 2D grid over (t, hv)
    pid_t = tl.program_id(0)
    pid_hv = tl.program_id(1)

    # Form q_exp: in provided setup H_v // H_q == 2, so q_exp = q[t, hv//2, :]
    ratio = H_v // H_q
    base = pid_hv // ratio
    q_vec = tl.load(q_ptr + pid_t * (H_q * D) + base * D + tl.arange(0, D))  # [D]

    # out_row = scale * q_vec @ new_state[hv, :, :]
    out_row = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        row = tl.load(new_state_ptr + pid_hv * D * D + i * D + tl.arange(0, D))  # [D]
        out_row += q_vec * row

    out_row = out_row * scale
    tl.store(out_ptr + pid_t * (H_v * D) + pid_hv * D + tl.arange(0, D), out_row)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-ONLY implementation of the original forward.
        Returns: output [total_seq_len, H_v, D] bfloat16, new_state [H_v, D, D] float32.
        """
        total_seq_len, H_q, D = q.shape
        H_v = v.shape[1]
        assert H_q == 4 and H_v == 8 and D == 128, "This Triton implementation assumes H_q=4, H_v=8, D=128."

        # Ensure contiguous and dtype
        q = q.contiguous().float()   # compute in float32
        k = k.contiguous().float()
        v = v.contiguous().float()
        A_log = A_log.contiguous().float()
        a = a.contiguous().float()
        dt_bias = dt_bias.contiguous().float()
        b = b.contiguous().float()
        state = state.contiguous().float()  # [H_v, D, D]
        device = q.device

        # Compute scale if None or zero
        if scale is None or scale == 0.0:
            scale = 1.0 / math.sqrt(D)

        # Allocate outputs
        output = torch.empty((total_seq_len, H_v, D), dtype=torch.float32, device=device)  # will cast to bfloat16 after
        new_state = torch.empty((H_v, D, D), dtype=torch.float32, device=device)

        # Kernel 1: compute g and beta
        g = torch.empty((total_seq_len, H_v), dtype=torch.float32, device=device)
        beta = torch.empty((total_seq_len, H_v), dtype=torch.float32, device=device)
        grid_g = (H_v,)
        _compute_g_beta_kernel[grid_g](
            A_log, a, dt_bias, b, g, beta,
            H_v=H_v, total_seq_len=total_seq_len,
        )

        # Kernel 2: update state for each token; 2D grid (t, hv)
        grid_update = (total_seq_len, H_v)
        _state_update_kernel[grid_update](
            k, v, state, g, beta, new_state,
            D=D, total_seq_len=total_seq_len, H_v=H_v,
        )

        # Kernel 3: output for each token; 2D grid (t, hv)
        grid_out = (total_seq_len, H_v)
        _output_kernel[grid_out](
            q, new_state, scale, output,
            D=D, H_q=H_q, H_v=H_v, total_seq_len=total_seq_len,
        )

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
