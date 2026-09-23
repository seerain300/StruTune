import triton
import triton.language as tl
import math

# Kernel 1: compute g[b,h] = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h])) and beta[b,h] = sigmoid(b[b,h])
@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    B, H,
    stride_al_h,               # A_log stride over H
    stride_a_b, stride_a_h,    # a strides [B,H]
    stride_db_h,               # dt_bias stride over H
    stride_b_b, stride_b_h,    # b strides [B,H]
    stride_g_b, stride_g_h,    # g strides [B,H]
    stride_be_b, stride_be_h,  # beta strides [B,H]
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load parameters
    a_val = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h)
    dt_bias_val = tl.load(dt_bias_ptr + h_idx * stride_db_h)
    b_val = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h)
    A_log_val = tl.load(A_log_ptr + h_idx * stride_al_h)
    x = a_val + dt_bias_val
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    g = tl.exp(-tl.exp(A_log_val) * sp)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    # store
    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g)
    tl.store(beta_ptr + b_idx * stride_be_b + h_idx * stride_be_h, beta)

# Kernel 2: tmp_old_v[b,h] = dot(k[b,h], state[b,h]) where k: [B,H,K], state: [B,H,V,K]
@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_si_b, stride_si_h, stride_si_v, stride_si_k,
    stride_tmp_bh,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Initialize accumulator
    acc = tl.zeros([1], dtype=tl.float32)
    # Reduce over K
    for kk in range(0, K):
        k_val = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk * stride_k_k)
        # Load state[:, kk] for all V
        # We need state[b,h, :, kk] vector of size V
        # For Triton, we loop over v indices and load each element.
        for v_idx in range(0, V):
            s = tl.load(state_ptr + b_idx * stride_si_b + h_idx * stride_si_h + v_idx * stride_si_v + kk * stride_si_k)
            acc += k_val * s
    tl.store(tmp_ptr + b_idx * stride_tmp_bh + h_idx * stride_tmp_bh, acc)

# Kernel 3: update new_state and compute output_scalar[b,h]
@triton.jit
def kernel_update_state_and_output(
    k_ptr, tmp_old_ptr, v_ptr, beta_ptr, state_ptr, new_state_ptr, q_ptr, output_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_tmp_b, stride_tmp_h,  # tmp_old strides [B,H]
    stride_v_b, stride_v_h, stride_v_v,                 # v strides [B,H,V]
    stride_be_b, stride_be_h,                              # beta strides [B,H]
    stride_si_b, stride_si_h, stride_si_v, stride_si_k,   # state strides [B,H,V,K]
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,   # new_state strides [B,H,V,K]
    stride_q_b, stride_q_h, stride_q_k,                   # q strides [B,H,K]
    stride_out_b, stride_out_h, stride_out_v,             # output strides [B,H,V]
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    # Load scalars
    tmp_old = tl.load(tmp_old_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h)
    beta = tl.load(beta_ptr + b_idx * stride_be_b + h_idx * stride_be_h)
    # Load k[b,h] as [K]
    k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + tl.arange(0, K) * stride_k_k)
    # Load v[b,h] as [V]
    v_vec = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + tl.arange(0, V) * stride_v_v)
    # new_v = beta * v + (1 - beta) * tmp_old
    new_v = beta * v_vec + (1.0 - beta) * tmp_old
    # state_remove = dot(k, tmp_old) (scalar)
    state_remove = tl.sum(k_vec * tmp_old)  # broadcast tmp_old to vector
    # state_update = dot(k, new_v) (vector [V])
    state_update = tl.sum(k_vec[:, None] * new_v[None, :])  # multiply [K] with [V], reduce over K -> [V]

    # Update new_state[b,h,:,:] = old_state - state_remove + state_update
    # We need to read old_state[b,h,:,:] and write new_state[b,h,:,:]
    for v_idx in range(0, V):
        for kk in range(0, K):
            old = tl.load(state_ptr + b_idx * stride_si_b + h_idx * stride_si_h + v_idx * stride_si_v + kk * stride_si_k)
            new_elem = old - state_remove + state_update[v_idx]
            tl.store(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v_idx * stride_ns_v + kk * stride_ns_k, new_elem)

    # Compute output_scalar = scale * (q[b,h] @ new_state[b,h])
    q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + tl.arange(0, K) * stride_q_k)
    # q @ new_state == sum_k q[k] * sum_v new_state[v,k] (note: here new_state is updated in-place above)
    # We need to reload new_state to compute q @ new_state. This is acceptable; we already wrote it.
    # But we don't have new_state readily as a vector; we reconstruct via reading the tensor again.
    # Alternatively, compute q @ (old - state_remove + state_update) without reloading old via recomputing; simpler is to read updated tensor.
    # We'll reconstruct scalar by summing q[k] * sum_v(new_state[v,k]). We can compute sum_v(new_state[v,k]) for each k by iterating.
    sum_v_new = tl.zeros([K], dtype=tl.float32)
    for v_idx in range(0, V):
        for kk in range(0, K):
            new_elem = tl.load(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v_idx * stride_ns_v + kk * stride_ns_k)
            sum_v_new[kk] += new_elem
    output_scalar = tl.sum(q_vec * sum_v_new)
    # Store to output[b,h,0]
    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_h + 0 * stride_out_v, output_scalar)

# Host ModelNew: entry point
class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device and dtype; compute in FP32
        device = q.device
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape

        # Move and cast inputs to float32, ensure contiguity
        a = a.squeeze(1).to(device=device, dtype=torch.float32).contiguous()       # [B,H]
        dt_bias = dt_bias.to(device=device, dtype=torch.float32).contiguous()      # [H]
        b = b.squeeze(1).to(device=device, dtype=torch.float32).contiguous()       # [B,H]
        A_log = A_log.to(device=device, dtype=torch.float32).contiguous()          # [H]
        q = q.squeeze(1).to(device=device, dtype=torch.float32).contiguous()       # [B,QH,K]
        k = k.squeeze(1).to(device=device, dtype=torch.float32).contiguous()       # [B,KH,K]
        v = v.squeeze(1).to(device=device, dtype=torch.float32).contiguous()       # [B,VH,V]

        # state handling: if provided, it is [B,H,V,K]; else create zeros
        if state is not None:
            state_in = state.squeeze(1).to(device=device, dtype=torch.float32).contiguous()  # [B,H,V,K]
        else:
            state_in = torch.zeros(B, num_v_heads, V, K, dtype=torch.float32, device=device)

        # Allocate outputs
        g = torch.empty(B, num_v_heads, dtype=torch.float32, device=device)
        beta = torch.empty(B, num_v_heads, dtype=torch.float32, device=device)
        tmp_old_v = torch.empty(B, num_v_heads, dtype=torch.float32, device=device)
        new_state_out = torch.empty(B, num_v_heads, V, K, dtype=torch.float32, device=device)
        output = torch.empty(B, num_v_heads, V, dtype=torch.float32, device=device)

        # Launch kernels
        grid = (B, num_v_heads)
        # 1) g and beta
        kernel_g_beta[grid](
            A_log, a, dt_bias, b,
            g, beta,
            B, num_v_heads,
            A_log.stride(0),
            a.stride(0), a.stride(1),
            dt_bias.stride(0),
            b.stride(0), b.stride(1),
            g.stride(0), g.stride(1),
            beta.stride(0), beta.stride(1),
            num_warps=1
        )
        # 2) tmp_old_v
        kernel_tmp_old_v[grid](
            k, state_in, tmp_old_v,
            B, num_v_heads, V, K,
            k.stride(0), k.stride(1), k.stride(2),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
            num_warps=1
        )
        # 3) update new_state and compute output
        kernel_update_state_and_output[grid](
            k, tmp_old_v, v, beta, state_in, new_state_out, q, output,
            B, num_v_heads, V, K,
            k.stride(0), k.stride(1), k.stride(2),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
            v.stride(0), v.stride(1), v.stride(2),
            beta.stride(0), beta.stride(1),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            new_state_out.stride(0), new_state_out.stride(1), new_state_out.stride(2), new_state_out.stride(3),
            q.stride(0), q.stride(1), q.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=1
        )

        # Cast output to bfloat16 as per original (return [B,H,V])
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, new_state_out


def run(*args):
    return ModelNew()(*args)
