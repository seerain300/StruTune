import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    H,
    stride_A, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
    stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars
    A = tl.load(A_log_ptr + h_idx * stride_A).to(tl.float32)
    a = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h).to(tl.float32)
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt).to(tl.float32)
    bb = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32)

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a + dt))
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))

    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    H, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_t_b, stride_t_h,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    acc = 0.0
    # Reduce over K using vectorized offsets for kk
    kk = tl.arange(0, K)
    for kk_idx in range(0, K):
        # k[b, h, kk_idx]
        k_elem = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk_idx * stride_k_k).to(tl.float32)
        # state[b, h, :, kk_idx] reduced over V using vectorized v offsets
        v_offsets = tl.arange(0, V)
        s_row = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v_offsets * stride_s_v + kk_idx * stride_s_k)
        s_sum = tl.sum(s_row, axis=0)
        acc += k_elem * s_sum
    tl.store(tmp_ptr + b_idx * stride_t_b + h_idx * stride_t_h, acc)


@triton.jit
def kernel_update_and_output(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, tmp_ptr,
    out_ptr, new_state_ptr,
    B, H, V, K,
    scale,  # scalar float32, e.g., 1.0 / sqrt(K), passed from host
    stride_q_b, stride_q_h, stride_q_k,
    stride_k_b, stride_k_h, stride_k_k,
    stride_v_b, stride_v_h, stride_v_v,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_g_b, stride_g_h,
    stride_beta_b, stride_beta_h,
    stride_out_b, stride_out_h,
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars
    g_val = tl.load(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h).to(tl.float32)
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)

    # Initialize new_state[b, h, :, :] = g * state[b, h, :, :]
    # Load entire state slice and multiply by g
    v_offsets = tl.arange(0, V)
    k_offsets = tl.arange(0, K)
    for v in v_offsets:
        row_ptr = state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v * stride_s_v
        state_vals = tl.load(row_ptr + k_offsets * stride_s_k).to(tl.float32)  # [K]
        new_row = g_val * state_vals
        tl.store(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v * stride_ns_v + k_offsets * stride_ns_k, new_row)

    # Compute state_remove = sum_k k[k] * old_state[k] (old_state is new_state initialized above)
    k_vals = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_offsets * stride_k_k).to(tl.float32)  # [K]
    # Load corresponding old_state rows
    old_state_rows = tl.load(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v_offsets * stride_ns_v + k_offsets * stride_ns_k)  # [V, K]
    state_remove = tl.sum(k_vals[None, :] * old_state_rows, axis=1)  # [V]
    state_remove = tl.sum(state_remove)  # scalar

    # Compute new_v_scalar = beta * mean(v) + (1 - beta) * tmp_old_v
    v_mean = tl.sum(tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + v_offsets * stride_v_v).to(tl.float32)) / V
    tmp_val = tl.load(tmp_ptr + b_idx * stride_t_b + h_idx * stride_t_h).to(tl.float32)
    new_v_scalar = beta_val * v_mean + (1.0 - beta_val) * tmp_val

    # state_update = sum_k k[k] * new_v_scalar
    state_update = tl.sum(k_vals * new_v_scalar)

    # Update new_state: new_state = new_state - state_remove + state_update (broadcasted)
    # Subtract state_remove from each [v, k] element:
    # First load, modify, then store
    for v in v_offsets:
        row_ptr = new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v * stride_ns_v
        row = tl.load(row_ptr + k_offsets * stride_ns_k).to(tl.float32)
        # state_remove and state_update are scalars; subtract state_remove, add state_update
        row = row - state_remove + state_update
        tl.store(row_ptr + k_offsets * stride_ns_k, row)

    # Compute output[b, h] = scale * (q @ new_state)
    q_vals = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + k_offsets * stride_q_k).to(tl.float32)  # [K]
    out_val = tl.sum(q_vals * tl.load(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v_offsets * stride_ns_v + k_offsets * stride_ns_k).to(tl.float32), axis=0)  # reduce over K -> [V], then sum all
    # Reduce over V to scalar
    out_val = tl.sum(out_val)
    out_val = scale * out_val
    tl.store(out_ptr + b_idx * stride_out_b + h_idx * stride_out_h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version using only Triton kernels for computation.
        Returns (output [B, 1, H], new_state [B, H, V, K]).
        """
        # Shapes from inputs: q [B, 1, QH, K], k [B, 1, KH, K], v [B, 1, VH, V], state [B, H, V, K]
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and state.dim() == 4
        B, _, QH, K = q.shape
        _, _, KH, _ = k.shape
        _, _, VH, V = v.shape
        H = state.shape[1]

        device = q.device

        # Prepare inputs as float32 and contiguous, reshape to [B, H, ...]
        q32 = q.float().contiguous().view(B, H, K)
        k32 = k.float().contiguous().view(B, H, K)
        v32 = v.float().contiguous().view(B, H, V)
        state32 = state.float().contiguous().view(B, H, V, K)

        # Parameters
        A_log32 = A_log.float().contiguous()  # [H]
        a32 = a.float().contiguous().view(B, H)  # [B,H]
        dt_bias32 = dt_bias.float().contiguous()  # [H]
        b32 = b.float().contiguous().view(B, H)  # [B,H]

        # If scale is None or 0, default to 1/sqrt(K). Host must not do .sqrt(); pass as scalar.
        if scale is None or scale == 0.0:
            scale_val = 1.0 / math.sqrt(K)
        else:
            scale_val = float(scale)

        # Allocate outputs
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)
        tmp = torch.empty((B, H), dtype=torch.float32, device=device)
        out = torch.empty((B, H), dtype=torch.float32, device=device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Strides for kernels
        stride_q_b, stride_q_h, stride_q_k = q32.stride()
        stride_k_b, stride_k_h, stride_k_k = k32.stride()
        stride_v_b, stride_v_h, stride_v_v = v32.stride()
        stride_s_b, stride_s_h, stride_s_v, stride_s_k = state32.stride()
        stride_g_b, stride_g_h = g.stride()
        stride_beta_b, stride_beta_h = beta.stride()
        stride_t_b, stride_t_h = tmp.stride()
        stride_out_b, stride_out_h = out.stride()
        stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k = new_state.stride()

        # Launch Triton kernels
        kernel_g_beta[(B, H)](
            A_log32, a32, dt_bias32, b32,
            g, beta,
            H,
            A_log32.stride()[0], a32.stride()[0], a32.stride()[1], dt_bias32.stride()[0], b32.stride()[0], b32.stride()[1],
            stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
            num_warps=1,
        )

        kernel_tmp_old_v[(B, H)](
            k32, state32, tmp,
            H, V, K,
            stride_k_b, stride_k_h, stride_k_k,
            stride_s_b, stride_s_h, stride_s_v, stride_s_k,
            stride_t_b, stride_t_h,
            num_warps=1,
        )

        kernel_update_and_output[(B, H)](
            q32, k32, v32, state32, g, beta, tmp,
            out, new_state,
            B, H, V, K,
            scale_val,
            stride_q_b, stride_q_h, stride_q_k,
            stride_k_b, stride_k_h, stride_k_k,
            stride_v_b, stride_v_h, stride_v_v,
            stride_s_b, stride_s_h, stride_s_v, stride_s_k,
            stride_g_b, stride_g_h,
            stride_beta_b, stride_beta_h,
            stride_out_b, stride_out_h,
            stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
            num_warps=1,
        )

        # Return results with original desired shapes
        output = out.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H]
        new_state_out = new_state  # [B, H, V, K], float32
        return output, new_state_out


def run(*args):
    return ModelNew()(*args)
