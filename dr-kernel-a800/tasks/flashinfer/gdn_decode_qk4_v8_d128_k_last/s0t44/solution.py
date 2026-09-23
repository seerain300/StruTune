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
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load scalars
    A = tl.load(A_log_ptr + h_idx * stride_A).to(tl.float32)
    a_val = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h).to(tl.float32)
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt).to(tl.float32)
    bb = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32)
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt))
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))
    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_dot_k_state(
    k_ptr, state_ptr, tmp_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_state_b, stride_state_h, stride_state_v, stride_state_k,
    stride_tmp_b, stride_tmp_h,
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Accumulator for dot product over K
    acc = 0.0
    # Vectorize over K
    for k_off in range(0, K):
        k_val = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_off * stride_k_k).to(tl.float32)
        for v_off in range(0, V):
            state_val = tl.load(state_ptr + b_idx * stride_state_b + h_idx * stride_state_h + v_off * stride_state_v + k_off * stride_state_k).to(tl.float32)
            acc += k_val * state_val
    tl.store(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h, acc)


@triton.jit
def kernel_update_and_output(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, tmp_ptr,
    output_ptr, new_state_ptr,
    B, H, V, K, scale,
    stride_q_b, stride_q_h, stride_q_k,
    stride_k_b, stride_k_h, stride_k_k,
    stride_v_b, stride_v_h, stride_v_v,
    stride_state_b, stride_state_h, stride_state_v, stride_state_k,
    stride_g_b, stride_g_h,
    stride_beta_b, stride_beta_h,
    stride_output_b, stride_output_h,
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    g_val = tl.load(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h).to(tl.float32)
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)
    tmp_val = tl.load(tmp_ptr + b_idx * stride_output_b + h_idx * stride_output_h).to(tl.float32)  # tmp_old_v

    # Compute new_state_bh: [V, K] elementwise update
    # old_state = g * state; old_v = k @ old_state
    # new_v = beta * v + (1 - beta) * old_v
    # state_remove = k @ old_state
    # state_update = k @ new_v
    # new_state = g * old_state - state_remove + state_update
    # We'll store new_state to new_state_ptr[b, h, :, :]
    for v_off in range(0, V):
        for k_off in range(0, K):
            # Load state_old element
            state_old = tl.load(state_ptr + b_idx * stride_state_b + h_idx * stride_state_h + v_off * stride_state_v + k_off * stride_state_k).to(tl.float32)
            old_state_vec = state_old * g_val

            # Compute old_v for this head: sum over V of state_old_vec
            # But we need k @ old_state. Since k is [K], old_v = sum_v old_state_vec
            # Implement scalar reduction over V for this k_off? We need vectorized approach:
            # Instead, compute k @ old_state by looping over K (but that would recompute); better is to compute vector k and store new_state directly via elementwise formula.
            # Simplify: we can compute new_state element by element using v_ptr and k_ptr, avoiding re-computing old_v here.
            # Compute v element
            v_elem = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + v_off * stride_v_v).to(tl.float32)
            # Compute k_elem
            k_elem = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_off * stride_k_k).to(tl.float32)
            # old_v_total = sum over V of state_old_vec
            old_v_total = 0.0
            for v2_off in range(0, V):
                s = tl.load(state_ptr + b_idx * stride_state_b + h_idx * stride_state_h + v2_off * stride_state_v + k_off * stride_state_k).to(tl.float32)
                old_v_total += s * g_val
            # new_v scalar for this v_off
            new_v_elem = beta_val * v_elem + (1.0 - beta_val) * old_v_total
            # Now compute new_state_bh[v_off, k_off]
            # We need k @ new_v scalar: sum_k k_elem * new_v_elem
            # But we are at fixed k_off and v_off; elementwise update uses the current k_elem and new_v_elem, but overall new_state at (v_off, k_off) depends on all k; hence we must compute k @ new_v via sum over K.
            # Instead, we'll compute new_state element as:
            # new_state_bh[v_off, k_off] = g * state_old - (k_elem * old_v_total) + (k_elem * new_v_elem) but since new_v_elem is scalar, we need to multiply by appropriate term. This requires a consistent vector approach.
            # To keep correctness: since we can't easily reference across K here, we will fill new_state per v_off by computing contributions properly: the elementwise update formula per (v, k) is:
            # new_state[v, k] = g * state[v, k] - k[k] * (sum_v state_old[v,k]) + k[k] * (beta * v[v] + (1-beta) * (sum_v state_old[v,k]))
            # Our approach: we'll recompute for each (v, k). We can do this since loop structure allows.
            # Compute sum_v state_old: total_sum_v = sum_v (state_old * g)
            total_sum_v = 0.0
            for v2_off in range(0, V):
                s = tl.load(state_ptr + b_idx * stride_state_b + h_idx * stride_state_h + v2_off * stride_state_v + k_off * stride_state_k).to(tl.float32)
                total_sum_v += s * g_val
            new_state_elem = g_val * state_old - (k_elem * total_sum_v) + (k_elem * (beta_val * v_elem + (1.0 - beta_val) * total_sum_v))
            # Store new_state_bh[v_off, k_off]
            tl.store(new_state_ptr + b_idx * stride_state_b + h_idx * stride_state_h + v_off * stride_state_v + k_off * stride_state_k, new_state_elem)

    # Compute output_scalar[b, h] = scale * (q[b, h] @ new_state[b, h])
    out_sum = 0.0
    for v_off in range(0, V):
        for k_off in range(0, K):
            q_elem = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + k_off * stride_q_k).to(tl.float32)
            new_elem = tl.load(new_state_ptr + b_idx * stride_state_b + h_idx * stride_state_h + v_off * stride_state_v + k_off * stride_state_k).to(tl.float32)
            out_sum += q_elem * new_elem
    output_scalar = scale * out_sum
    tl.store(output_ptr + b_idx * stride_output_b + h_idx * stride_output_h, output_scalar)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure tensors are on same device and contiguous
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "Inputs must be CUDA tensors for Triton."
        B = q.shape[0]
        H = state.shape[1]
        V = v.shape[-1]
        K = q.shape[-1]

        # Prepare pointers and strides
        # q: [B, 1, QH, K] -> we ignore the size-1 dim in kernels
        # k: [B, 1, KH, K]
        # v: [B, 1, VH, V]
        # state: [B, H, V, K]
        # A_log: [H]
        # a: [B, 1, H]
        # dt_bias: [H]
        # b: [B, 1, H]

        # Allocate outputs
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)
        tmp_old_v = torch.empty((B, H), dtype=torch.float32, device=device)
        output_scalar = torch.empty((B, H), dtype=torch.float32, device=device)
        new_state = state.clone()  # will be updated in-place by kernel

        # Launch Triton kernels: grid = (B, H)
        # Kernel 1: g and beta
        # Strides: scalar for [H] tensors; for a_ptr, use b*stride_a_b + h*stride_a_h
        # We'll pass strides in units of elements; Triton pointers have element strides already.
        kernel_g_beta[(B, H)](
            A_log, a.squeeze(1), dt_bias, b.squeeze(1),
            g, beta,
            H,
            1,  # stride_A: A_log is [H], contiguous
            1, 1,  # stride_a_b, stride_a_h
            1,  # stride_dt: dt_bias is [H], contiguous
            1, 1,  # stride_b_b, stride_b_h
            1, 1,  # stride_g_b, stride_g_h
            1, 1,  # stride_beta_b, stride_beta_h
            num_warps=4,
        )

        # Kernel 2: tmp_old_v = dot(k, state)
        kernel_dot_k_state[(B, H)](
            k, state, tmp_old_v,
            B, H, V, K,
            k.stride(0), k.stride(1), k.stride(2),  # stride_k_b, stride_k_h, stride_k_k
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),  # stride_state_b, stride_state_h, stride_state_v, stride_state_k
            1, 1,  # stride_tmp_b, stride_tmp_h
            num_warps=4,
        )

        # Kernel 3: update new_state and compute output_scalar per (b,h)
        kernel_update_and_output[(B, H)](
            q, k, v, state, g, beta, tmp_old_v,
            output_scalar, new_state,
            B, H, V, K, float(scale),
            q.stride(0), q.stride(1), q.stride(2),  # stride_q_b, stride_q_h, stride_q_k
            k.stride(0), k.stride(1), k.stride(2),  # stride_k_b, stride_k_h, stride_k_k
            v.stride(0), v.stride(1), v.stride(2),  # stride_v_b, stride_v_h, stride_v_v
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),  # stride_state_b, stride_state_h, stride_state_v, stride_state_k
            g.stride(0), g.stride(1),  # stride_g_b, stride_g_h
            beta.stride(0), beta.stride(1),  # stride_beta_b, stride_beta_h
            output_scalar.stride(0), output_scalar.stride(1),  # stride_output_b, stride_output_h
            num_warps=4,
        )

        # Return output as [B, 1, H] in bfloat16 and new_state as [B, H, V, K] in float32
        output = output_scalar.unsqueeze(1).to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
