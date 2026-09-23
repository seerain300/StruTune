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
    # 1D grid over head index hv
    pid_hv = tl.program_id(0)
    # Load A_log for this hv
    A_log = tl.load(A_log_ptr + pid_hv)
    # Loop over tokens b
    for b in range(total_seq_len):
        a_val = tl.load(a_ptr + b * H_v + pid_hv)
        db = tl.load(dt_bias_ptr + pid_hv)
        x = a_val + db
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(x))
        g_val = tl.exp(-tl.exp(A_log) * sp)
        b_val = tl.load(b_ptr + b * H_v + pid_hv)
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
    total_seq_len: tl.constexpr,
    H_v: tl.constexpr,
):
    # 2D grid over (t, hv)
    pid_t = tl.program_id(0)
    pid_hv = tl.program_id(1)

    # Load parameters for this token/hv
    g_val = tl.load(g_ptr + pid_t * H_v + pid_hv)
    beta_val = tl.load(beta_ptr + pid_t * H_v + pid_hv)

    # Compute old_v = k @ state (reduce over D)
    old_v = tl.zeros((D,), dtype=tl.float32)
    # Iterate over rows i in state, each row length D
    for i in range(0, D):
        state_vec = tl.load(state_ptr + pid_hv * D * D + i * D + tl.arange(0, D))
        k_i = tl.load(k_ptr + pid_t * (H_v * D) + pid_hv * D + i)
        old_v += state_vec * k_i

    # Compute new_v = beta * v + (1 - beta) * old_v
    v_vec = tl.load(v_ptr + pid_t * (H_v * D) + pid_hv * D + tl.arange(0, D))
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

    # Compute kT_old and kT_newv: sum_i k[t, hv, i] * old_v[i]
    kT_old = 0.0
    kT_newv = 0.0
    for i in range(0, D):
        k_i = tl.load(k_ptr + pid_t * (H_v * D) + pid_hv * D + i)
        kT_old += k_i * old_v[i]
        kT_newv += k_i * new_v[i]

    # Update state: new_state = g * state - kT_old + kT_newv
    for i in range(0, D):
        old_row = tl.load(state_ptr + pid_hv * D * D + i * D + tl.arange(0, D))
        new_row = g_val * old_row - kT_old + kT_newv
        tl.store(new_state_ptr + pid_hv * D * D + i * D + tl.arange(0, D), new_row)

    # No need to store scalars; new_state_ptr holds updated state.


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

    # Form q_exp: in provided setup H_v // H_q == 2, so q_exp = concat(q[t, 0, :], q[t, 1, :])
    ratio = H_v // H_q
    base = pid_hv // ratio
    q_vec = tl.load(q_ptr + pid_t * (H_q * D) + base * D + tl.arange(0, D))

    # matmul: out[t, hv, :] = scale * q_vec @ new_state[hv, :, :]
    out_row = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        row = tl.load(new_state_ptr + pid_hv * D * D + i * D + tl.arange(0, D))
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

        device = q.device

        # Compute g and beta via Triton kernel
        g = torch.empty((total_seq_len, H_v), dtype=torch.float32, device=device)
        beta = torch.empty((total_seq_len, H_v), dtype=torch.float32, device=device)
        grid_g = (H_v,)
        _compute_g_beta_kernel[grid_g](
            A_log.float(), a.float(), dt_bias.float(), b.float(), g, beta,
            H_v=H_v, total_seq_len=total_seq_len
        )

        # Prepare inputs for Triton kernels
        k_exp = k.float().contiguous()  # [T, H_v, D]
        v_exp = v.float().contiguous()  # [T, H_v, D]
        # state layout: [H_v, D, D] (k-last)
        state_klast = state.float().contiguous()  # [H_v, D, D]
        new_state = torch.empty((H_v, D, D), dtype=torch.float32, device=device)

        # Launch state update kernel
        grid_state = (total_seq_len, H_v)
        _state_update_kernel[grid_state](
            k_exp, v_exp, state_klast, g, beta, new_state,
            D=D, total_seq_len=total_seq_len, H_v=H_v
        )

        # Launch output kernel
        q_exp = q.float().contiguous()  # [T, H_q, D]
        out = torch.empty((total_seq_len, H_v, D), dtype=torch.float32, device=device)
        grid_out = (total_seq_len, H_v)
        _output_kernel[grid_out](
            q_exp, new_state, (float(scale) if scale is not None else 1.0 / math.sqrt(D)),
            out, D=D, H_q=H_q, H_v=H_v, total_seq_len=total_seq_len
        )

        # Match original output dtype and state layout
        output = out.to(torch.bfloat16)  # [T, H_v, D], bfloat16
        new_state_klast = new_state  # [H_v, D, D], float32

        return output, new_state_klast


def run(*args):
    return ModelNew()(*args)
