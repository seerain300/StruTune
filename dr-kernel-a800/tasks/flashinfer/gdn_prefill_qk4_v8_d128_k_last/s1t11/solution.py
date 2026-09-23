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
    # 1D grid over hv
    pid_hv = tl.program_id(0)
    # Loop over tokens b to compute g and beta for each
    for b in range(0, total_seq_len):
        a_val = tl.load(a_ptr + b * H_v + pid_hv)
        dt = tl.load(dt_bias_ptr + pid_hv)
        b_val = tl.load(b_ptr + b * H_v + pid_hv)
        x = a_val + dt
        sp = tl.log(1.0 + tl.exp(x))  # softplus(x) = log(1 + exp(x))
        g_val = tl.exp(-tl.exp(A_log_ptr + pid_hv) * sp)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(g_ptr + b * H_v + pid_hv, g_val)
        tl.store(beta_ptr + b * H_v + pid_hv, beta_val)


@triton.jit
def _state_update_kernel(
    k_ptr,       # [total_seq_len, H_v, D] float32
    v_ptr,       # [total_seq_len, H_v, D] float32
    state_ptr,   # [H_v, D, D] float32
    g_ptr,       # [total_seq_len, H_v] float32
    beta_ptr,    # [total_seq_len, H_v] float32
    new_state_ptr,  # [H_v, D, D] float32
    D: tl.constexpr,
    H_v: tl.constexpr,
    total_seq_len: tl.constexpr,
):
    # 2D grid over (t, hv)
    pid_t = tl.program_id(0)
    pid_hv = tl.program_id(1)

    # Load k vector for this token and head
    k_vec = tl.load(k_ptr + pid_t * (H_v * D) + pid_hv * D + tl.arange(0, D))  # [D]
    # Compute old_v = k @ state (reduce over D)
    old_v = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        row = tl.load(state_ptr + pid_hv * D * D + i * D + tl.arange(0, D))  # [D]
        old_v += k_vec * row

    # Load v vector
    v_vec = tl.load(v_ptr + pid_t * (H_v * D) + pid_hv * D + tl.arange(0, D))  # [D]
    # Compute new_v = beta * v + (1 - beta) * old_v
    beta_val = tl.load(beta_ptr + pid_t * H_v + pid_hv)
    g_val = tl.load(g_ptr + pid_t * H_v + pid_hv)
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

    # Compute kT_old = sum_i k[i] * old_v[i], kT_newv = sum_i k[i] * new_v[i]
    kT_old = tl.sum(k_vec * old_v, axis=0)
    kT_newv = tl.sum(k_vec * new_v, axis=0)

    # Update state: new_state = g * state - kT_old + kT_newv
    for i in range(0, D):
        row = tl.load(state_ptr + pid_hv * D * D + i * D + tl.arange(0, D))
        new_row = g_val * row - kT_old + kT_newv
        tl.store(new_state_ptr + pid_hv * D * D + i * D + tl.arange(0, D), new_row)


@triton.jit
def _output_kernel(
    q_ptr,       # [total_seq_len, H_q, D] float32
    new_state_ptr, # [H_v, D, D] float32
    scale,       # float32 scalar
    out_ptr,     # [total_seq_len, H_v, D] float32
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

    # out = scale * q_vec @ new_state[hv, :, :]
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
        Assumes: H_q=4, H_v=8, D=128.
        """
        total_seq_len, H_q, D = q.shape
        H_v = v.shape[1]
        assert H_q == 4 and H_v == 8 and D == 128, "This Triton implementation assumes H_q=4, H_v=8, D=128."

        # Ensure dtypes are float32 for computation; original uses float() on tensors
        q32 = q.float()
        k32 = k.float()
        v32 = v.float()
        a32 = a.float()
        dt_bias32 = dt_bias.float()
        b32 = b.float()
        A_log32 = A_log.float()
        # Compute scale on host without torch to satisfy TRITON-ONLY requirement
        if scale is None:
            scale = 1.0 / math.sqrt(D)
        else:
            scale = float(scale)

        # Allocate outputs and buffers
        g = torch.empty((total_seq_len, H_v), dtype=torch.float32, device=q.device)
        beta = torch.empty((total_seq_len, H_v), dtype=torch.float32, device=q.device)
        out = torch.empty((total_seq_len, H_v, D), dtype=torch.float32, device=q.device)
        new_state = torch.empty((H_v, D, D), dtype=torch.float32, device=q.device)

        # Launch kernels
        # Kernel 1: compute g and beta
        grid1 = (H_v,)
        _compute_g_beta_kernel[grid1](A_log32, a32, dt_bias32, b32, g, beta, H_v=H_v, total_seq_len=total_seq_len)

        # Kernel 2: state update for each token and head
        grid2 = (total_seq_len, H_v)
        _state_update_kernel[grid2](
            k32, v32, state, g, beta, new_state,
            D=D, H_v=H_v, total_seq_len=total_seq_len
        )

        # Kernel 3: output
        grid3 = (total_seq_len, H_v)
        _output_kernel[grid3](
            q32, new_state, scale, out,
            D=D, H_q=H_q, H_v=H_v, total_seq_len=total_seq_len
        )

        # Return output as bfloat16 and new_state as float32 to match original behavior
        return out.to(torch.bfloat16), new_state


def run(*args):
    return ModelNew()(*args)
