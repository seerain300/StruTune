import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    B, H,
    stride_A, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
    stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load scalars
    A = tl.load(A_log_ptr + h_idx * stride_A).to(tl.float32)
    a = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h).to(tl.float32)
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt).to(tl.float32)
    bb = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32)
    x = a + dt
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))
    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    B, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_state_b, stride_state_v, stride_state_k,
    stride_tmp_b, stride_tmp_h,
):
    # Each program computes tmp_old_v for one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load k[b, h] as [K]
    k_off = b_idx * stride_k_b + h_idx * stride_k_h
    k_vec = tl.load(k_ptr + k_off + tl.arange(0, K) * stride_k_k, mask=tl.arange(0, K) < K, other=0.0)
    # Load state[b, h] as [V, K]
    state_off = b_idx * stride_state_b + h_idx * stride_state_v
    acc = tl.zeros([1], dtype=tl.float32)
    for k_idx in range(0, K):
        k_val = k_vec[k_idx]
        row_off = state_off + k_idx * stride_state_k
        # iterate over V
        s_val = 0.0
        for v_idx in range(0, V):
            s = tl.load(state_ptr + row_off + v_idx * stride_state_v)
            s_val += s
        acc += k_val * s_val
    tl.store(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h, acc)


@triton.jit
def kernel_update_and_output(
    k_ptr, tmp_ptr, beta_ptr, v_ptr, state_in_ptr, q_ptr, output_ptr,
    B, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_tmp_b, stride_tmp_h,
    stride_beta_b, stride_beta_h,
    stride_v_b, stride_v_v,
    stride_state_b, stride_state_v, stride_state_k,
    stride_q_b, stride_q_k,
    stride_out_b, stride_out_h,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load scalars
    k_off = b_idx * stride_k_b + h_idx * stride_k_h
    k_vec = tl.load(k_ptr + k_off + tl.arange(0, K) * stride_k_k, mask=tl.arange(0, K) < K, other=0.0)
    tmp = tl.load(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h)
    beta = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h)
    v_off = b_idx * stride_v_b + h_idx * stride_v_v
    v_vec = tl.load(v_ptr + v_off + tl.arange(0, V) * stride_v_v, mask=tl.arange(0, V) < V, other=0.0)
    # Compute new_v = beta * v + (1 - beta) * tmp (broadcast tmp over V)
    new_v = beta * v_vec + (1.0 - beta) * tmp
    # Compute state_remove = dot(k, tmp) = sum_k k[k] * tmp
    state_remove = tl.sum(k_vec * tmp, axis=0)
    # Compute state_update = dot(k, new_v) = sum_k k[k] * new_v[k]
    state_update = tl.sum(k_vec * new_v, axis=0)
    # Load old_state[b, h] as [V, K]
    state_off = b_idx * stride_state_b + h_idx * stride_state_v
    for v_idx in range(0, V):
        row_off = state_off + v_idx * stride_state_v
        for k_idx in range(0, K):
            old = tl.load(state_in_ptr + row_off + k_idx * stride_state_k)
            # new state at [v, k]
            new_val = old - state_remove + state_update
            tl.store(state_in_ptr + row_off + k_idx * stride_state_k, new_val)
    # Compute output_scalar = q @ new_state = sum_k q[k] * sum_v new_state[v, k]
    q_off = b_idx * stride_q_b + h_idx * stride_q_k
    q_vec = tl.load(q_ptr + q_off + tl.arange(0, K) * stride_q_k, mask=tl.arange(0, K) < K, other=0.0)
    dot_q_new = 0.0
    for k_idx in range(0, K):
        qk = q_vec[k_idx]
        row_off = state_off + k_idx * stride_state_k
        s_row = 0.0
        for v_idx in range(0, V):
            s = tl.load(state_in_ptr + row_off + v_idx * stride_state_v)
            s_row += s
        dot_q_new += qk * s_row
    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_h, dot_q_new)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device and dtype, work in float32 for numerical stability
        device = q.device
        B = q.shape[0]
        # Assert shapes consistent with provided get_inputs: q/k [B,1,QH,K], v [B,1,VH,V]
        assert q.shape == (B, 1, 4, 128), "q must have shape [B,1,4,128]"
        assert k.shape == (B, 1, 4, 128), "k must have shape [B,1,4,128]"
        assert v.shape == (B, 1, 8, 128), "v must have shape [B,1,8,128]"
        assert A_log.shape == (8,), "A_log must have shape [8]"
        assert a.shape == (B, 1, 8), "a must have shape [B,1,8]"
        assert dt_bias.shape == (8,), "dt_bias must have shape [8]"
        assert b.shape == (B, 1, 8), "b must have shape [B,1,8]"
        if state is not None:
            assert state.shape == (B, 8, 128, 128), "state must have shape [B,8,128,128]"

        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)
        if state is not None:
            state_in = state.contiguous().to(torch.float32)
        else:
            state_in = torch.zeros(B, 8, 128, 128, dtype=torch.float32, device=device)
        A_log = A_log.contiguous().to(torch.float32)
        a = a.contiguous().to(torch.float32)
        dt_bias = dt_bias.contiguous().to(torch.float32)
        b = b.contiguous().to(torch.float32)

        # Allocate outputs
        g = torch.empty(B, 8, dtype=torch.float32, device=device)
        beta = torch.empty(B, 8, dtype=torch.float32, device=device)
        tmp_old_v = torch.empty(B, 8, dtype=torch.float32, device=device)
        new_state_out = state_in  # we will update in-place
        output = torch.empty(B, 8, dtype=torch.float32, device=device)

        # Launch Triton kernels with grid (B, H) where H=8 for given inputs
        grid = (B, 8)
        kernel_g_beta[grid](
            A_log, a, dt_bias, b,
            g, beta,
            B, 8,
            A_log.stride(0), a.stride(0), a.stride(1), dt_bias.stride(0), b.stride(0), b.stride(1),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
            num_warps=1, num_stages=1,
        )

        kernel_tmp_old_v[grid](
            k, state_in, tmp_old_v,
            B, 128, 128,
            k.stride(0), k.stride(1), k.stride(2),
            state_in.stride(0), state_in.stride(1), state_in.stride(2),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
            num_warps=1, num_stages=1,
        )

        kernel_update_and_output[grid](
            k, tmp_old_v, beta, v, state_in, q, output,
            B, 128, 128,
            k.stride(0), k.stride(1), k.stride(2),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
            beta.stride(0), beta.stride(1),
            v.stride(0), v.stride(2),
            state_in.stride(0), state_in.stride(1), state_in.stride(2),
            q.stride(0), q.stride(2),
            output.stride(0), output.stride(1),
            num_warps=1, num_stages=1,
        )

        # Return output (cast to bfloat16) and new_state_out
        output_bf16 = output.to(torch.bfloat16)
        return (output_bf16, new_state_out)


def run(*args):
    return ModelNew()(*args)
