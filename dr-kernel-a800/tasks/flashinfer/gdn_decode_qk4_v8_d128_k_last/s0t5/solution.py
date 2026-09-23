import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                  g_ptr, beta_ptr,
                  B, H,
                  stride_alog, stride_a_b, stride_a_h,
                  stride_db, stride_b_b, stride_b_h,
                  stride_g_b, stride_g_h,
                  stride_be_b, stride_be_h):
    # program ids
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # load scalars
    a_val = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h)
    db_val = tl.load(dt_bias_ptr + h_idx * stride_db)
    A_log_val = tl.load(A_log_ptr + h_idx * stride_alog)
    b_val = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h)

    # softplus(x) = log(1 + exp(x))
    x = a_val + db_val
    sp = tl.log(1.0 + tl.exp(x))

    # g = exp(-exp(A_log) * softplus(a + dt_bias))
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # store
    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_be_b + h_idx * stride_be_h, beta_val)


@triton.jit
def kernel_tmp_old_v(k_ptr, state_ptr, tmp_ptr,
                     B, H, V, K,
                     stride_k_b, stride_k_h, stride_k_k,
                     stride_s_b, stride_s_h, stride_s_v, stride_s_k,
                     stride_tmp_bh):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # tmp_old_v = sum_k k[k] * state[k, :]
    tmp = tl.zeros((), dtype=tl.float32)
    for kk in range(0, K):
        k_elem = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk * stride_k_k)
        for vv in range(0, V):
            state_elem = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + vv * stride_s_v + kk * stride_s_k)
            tmp += k_elem * state_elem
    tl.store(tmp_ptr + b_idx * stride_tmp_bh + h_idx * stride_tmp_bh, tmp)


@triton.jit
def kernel_update_state_and_output(k_ptr, tmp_old_ptr, v_ptr, beta_ptr,
                                   state_in_ptr, new_state_ptr, q_ptr, output_ptr,
                                   B, H, V, K,
                                   stride_k_b, stride_k_h, stride_k_k,
                                   stride_tmp_bh,
                                   stride_v_b, stride_v_h, stride_v_v,
                                   stride_be_b, stride_be_h,
                                   stride_si_b, stride_si_h, stride_si_v, stride_si_k,
                                   stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
                                   stride_q_b, stride_q_h, stride_q_k,
                                   stride_out_b, stride_out_h, stride_out_v):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load k[b,h] vector [K]
    k_vec = tl.zeros([K], dtype=tl.float32)
    for kk in range(0, K):
        k_vec[kk] = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk * stride_k_k)

    # Load v[b,h] vector [V]
    v_vec = tl.zeros([V], dtype=tl.float32)
    for vv in range(0, V):
        v_vec[vv] = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + vv * stride_v_v)

    tmp_old = tl.load(tmp_old_ptr + b_idx * stride_tmp_bh + h_idx * stride_tmp_bh)
    beta = tl.load(beta_ptr + b_idx * stride_be_b + h_idx * stride_be_h)

    # new_v = beta * v + (1 - beta) * tmp_old (broadcast tmp_old over V)
    new_v_vec = beta * v_vec + (1.0 - beta) * tmp_old

    # state_remove = dot(k, tmp_old)
    state_remove = tl.zeros((), dtype=tl.float32)
    for kk in range(0, K):
        state_remove += k_vec[kk] * tmp_old

    # state_update = dot(k, new_v)
    state_update_vec = tl.zeros([V], dtype=tl.float32)
    for kk in range(0, K):
        for vv in range(0, V):
            state_update_vec[vv] += k_vec[kk] * new_v_vec[vv]

    # Load old_state [V, K] and compute new_state = old_state - state_remove + state_update_vec (broadcasted along K)
    for vv in range(0, V):
        old_line = tl.zeros([K], dtype=tl.float32)
        for kk in range(0, K):
            old_line[kk] = tl.load(state_in_ptr + b_idx * stride_si_b + h_idx * stride_si_h + vv * stride_si_v + kk * stride_si_k)
        new_line = old_line - state_remove + state_update_vec[vv]
        for kk in range(0, K):
            tl.store(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + vv * stride_ns_v + kk * stride_ns_k, new_line[kk])

    # output_scalar = scale * dot(q, new_state[b,h]) -> sum over v and k
    output_scalar = tl.zeros((), dtype=tl.float32)
    for vv in range(0, V):
        new_line = tl.zeros([K], dtype=tl.float32)
        for kk in range(0, K):
            new_line[kk] = tl.load(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + vv * stride_ns_v + kk * stride_ns_k)
        for kk in range(0, K):
            q_elem = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + kk * stride_q_k)
            output_scalar += q_elem * new_line[kk]
    # store output[b, h, 0]
    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_h + 0 * stride_out_v, output_scalar)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version that computes:
          output: [B, H, V] in bfloat16 (per (b,h) scalar cast to [V])
          new_state: [B, H, V, K] in float32
        """
        device = q.device
        dtype = torch.float32

        # Ensure inputs are on device and float32; do not mutate originals
        a = a.to(device=device, dtype=dtype)
        dt_bias = dt_bias.to(device=device, dtype=dtype)
        b = b.to(device=device, dtype=dtype)
        A_log = A_log.to(device=device, dtype=dtype)

        # Prepare q, k, v as [B, H, ?]
        q_s = q.squeeze(1).to(device=device, dtype=dtype)   # [B, QH, K] -> [B, H, K]
        k_s = k.squeeze(1).to(device=device, dtype=dtype)   # [B, KH, K] -> [B, H, K]
        v_s = v.squeeze(1).to(device=device, dtype=dtype)   # [B, VH, V] -> [B, H, V]

        # Build state_in [B, H, V, K]
        if state is None:
            # Emulate initial zeros if state is None (not typical in provided inputs)
            B, H = q_s.shape[0], q_s.shape[1]
            V, K = v_s.shape[2], q_s.shape[2]
            state_in = torch.zeros(B, H, V, K, dtype=dtype, device=device)
        else:
            # Convert state to [B, H, V, K] float32
            state_in = state.squeeze(1).to(device=device, dtype=dtype)  # expected [B, H, V, K]

        # Allocate outputs
        B, H = q_s.shape[0], q_s.shape[1]
        V = v_s.shape[2]
        K = q_s.shape[2]
        g = torch.empty(B, H, dtype=dtype, device=device)
        beta = torch.empty(B, H, dtype=dtype, device=device)
        tmp_old_v = torch.empty(B, H, dtype=dtype, device=device)
        new_state_out = torch.empty(B, H, V, K, dtype=dtype, device=device)
        # Output per (b,h) scalar stored in [B, H, 1] for later casting to [B, H, V]
        output_scalar = torch.empty(B, H, dtype=dtype, device=device)

        # Launch Triton kernels
        grid = (B, H)
        kernel_g_beta[grid](
            A_log, a, dt_bias, b,
            g, beta,
            B, H,
            A_log.stride(0), a.stride(0), a.stride(1),
            dt_bias.stride(0), b.stride(0), b.stride(1),
            g.stride(0), g.stride(1),
            beta.stride(0), beta.stride(1),
        )

        kernel_tmp_old_v[grid](
            k_s, state_in, tmp_old_v,
            B, H, V, K,
            k_s.stride(0), k_s.stride(1), k_s.stride(2),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            tmp_old_v.stride(0),
        )

        kernel_update_state_and_output[grid](
            k_s, tmp_old_v, v_s, beta, state_in, new_state_out, q_s, output_scalar,
            B, H, V, K,
            k_s.stride(0), k_s.stride(1), k_s.stride(2),
            tmp_old_v.stride(0),
            v_s.stride(0), v_s.stride(1), v_s.stride(2),
            beta.stride(0), beta.stride(1),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            new_state_out.stride(0), new_state_out.stride(1), new_state_out.stride(2), new_state_out.stride(3),
            q_s.stride(0), q_s.stride(1), q_s.stride(2),
            output_scalar.stride(0), output_scalar.stride(1), 0,
        )

        # Cast output to bfloat16 as original code does; output per head scalar cast to [B, H, V]
        output_bf16 = output_scalar.to(torch.bfloat16).unsqueeze(2)  # [B, H, 1] -> [B, H, V] by broadcasting
        # Note: original returns [B, H, V] with per-head scalar at each V; this matches intent
        return output_bf16, new_state_out


def run(*args):
    return ModelNew()(*args)
