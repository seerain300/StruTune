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
    # Sum over K: k[b, h, kk] * sum_v state[b, h, v, kk]
    for kk in range(0, K):
        k_elem = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk * stride_k_k).to(tl.float32)
        s_sum = 0.0
        for v in range(0, V):
            s_elem = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v * stride_s_v + kk * stride_s_k).to(tl.float32)
            s_sum += s_elem
        acc += k_elem * s_sum
    tl.store(tmp_ptr + b_idx * stride_t_b + h_idx * stride_t_h, acc)


@triton.jit
def kernel_update_and_output(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, tmp_ptr,
    out_ptr, new_state_ptr,
    B, H, V, K,
    scale,
    stride_q_b, stride_q_h, stride_q_k,
    stride_k_b, stride_k_h, stride_k_k,
    stride_v_b, stride_v_h, stride_v_v,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_g_b, stride_g_h,
    stride_beta_b, stride_beta_h,
    stride_out_b, stride_out_h,
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars
    g_val = tl.load(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h).to(tl.float32)
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)
    tmp_val = tl.load(tmp_ptr + b_idx * stride_t_b + h_idx * stride_t_h).to(tl.float32)

    # Load v[b, h, :]
    v_vec = tl.zeros((V,), dtype=tl.float32)
    for v in range(0, V):
        v_elem = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + v * stride_v_v).to(tl.float32)
        v_vec[v] = v_elem

    # Compute new_v = beta * v + (1 - beta) * (k @ state) where k @ state = tmp_val
    new_v = beta_val * v_vec + (1.0 - beta_val) * tmp_val

    # Write new_state[b, h, :, :] = state[b, h, :, :] - tmp_val + (k * new_v)
    # We'll iterate over K in tiles and update per element.
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # For each k in tile, compute contribution (k[k] * new_v) and add to new_state
        for kk in range(0, BLOCK_K):
            k_k = k_offsets[kk]
            if k_mask[kk]:
                k_elem = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_k * stride_k_k).to(tl.float32)
                contrib = k_elem * new_v  # scalar
                # Iterate over V and update new_state elements
                for v in range(0, V):
                    old = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v * stride_s_v + k_k * stride_s_k).to(tl.float32)
                    new_elem = old - tmp_val + contrib
                    tl.store(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v * stride_ns_v + k_k * stride_ns_k, new_elem)

    # Compute output[b, h] = scale * (q[b, h] @ new_state[b, h])
    q_dot = 0.0
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        for kk in range(0, BLOCK_K):
            k_k = k_offsets[kk]
            if k_mask[kk]:
                q_elem = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + k_k * stride_q_k).to(tl.float32)
                # sum over V of q_elem * new_state[b, h, v, k_k]
                s_sum = 0.0
                for v in range(0, V):
                    ns_elem = tl.load(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v * stride_ns_v + k_k * stride_ns_k).to(tl.float32)
                    s_sum += ns_elem
                q_dot += q_elem * s_sum
    out_val = scale * q_dot
    tl.store(out_ptr + b_idx * stride_out_b + h_idx * stride_out_h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version of the given run() function.
        - All computation is done in Triton kernels.
        - Returns output with shape [B, 1, H] cast to bfloat16 and new_state with shape [B, H, V, K] in float32.
        """
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and state.dim() == 4
        B = q.shape[0]
        # num_q_heads, num_k_heads, num_v_heads = 4, 4, 8 (assertions in original)
        H = state.shape[1]
        V = state.shape[2]
        K = state.shape[3]
        device = q.device

        # Ensure inputs are float32 and contiguous; keep original shapes
        q32 = q.contiguous().to(torch.float32)           # [B, 1, 4, K]
        k32 = k.contiguous().to(torch.float32)           # [B, 1, 4, K]
        v32 = v.contiguous().to(torch.float32)           # [B, 1, 8, V]
        state32 = state.contiguous().to(torch.float32)   # [B, H, V, K]
        A_log = A_log.contiguous().to(torch.float32)     # [H]
        a32 = a.contiguous().to(torch.float32)           # [B, 1, H]
        dt_bias = dt_bias.contiguous().to(torch.float32) # [H]
        b32 = b.contiguous().to(torch.float32)           # [B, 1, H]

        # Precompute strides for Triton kernels
        # For q,k,v:
        # We pass them as [B, H, ...] by reshaping/viewing. But to compute strides, we can just use original strides:
        # Triton requires pointers; we'll view as [B, H, dim] for q/k/v to access (b,h) directly.
        # Create views:
        q_view = q32.view(B, H, -1)           # [B, H, K]
        k_view = k32.view(B, H, -1)           # [B, H, K]
        v_view = v32.view(B, H, -1)           # [B, H, V]

        # Allocate intermediate tensors for g, beta, tmp_old_v
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)
        tmp = torch.empty((B, H), dtype=torch.float32, device=device)

        # Kernel 1: compute g and beta
        kernel_g_beta[(B, H)](
            A_log, a32, dt_bias, b32,
            g, beta,
            H,
            A_log.stride(0), a32.stride(0), a32.stride(2), dt_bias.stride(0), b32.stride(0), b32.stride(2),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
            num_warps=1,
        )

        # Kernel 2: tmp_old_v = dot(k[b,h], state[b,h])
        tmp_old_v = kernel_tmp_old_v[(B, H)](
            k_view, state32,
            tmp,
            H, V, K,
            k_view.stride(0), k_view.stride(1), k_view.stride(2),
            state32.stride(0), state32.stride(1), state32.stride(2), state32.stride(3),
            tmp.stride(0), tmp.stride(1),
            num_warps=1,
        )  # tmp is [B, H] scalar per (b,h)

        # Kernel 3: update new_state and compute output[b,h]
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)
        out = torch.empty((B, H), dtype=torch.float32, device=device)

        # We need q_view as [B, H, K]; already done above
        q_view = q32.view(B, H, K)

        # Launch main kernel
        kernel_update_and_output[(B, H)](
            q_view, k_view, v_view, state32, g, beta, tmp,
            out, new_state,
            B, H, V, K,
            float(scale),  # pass scale as scalar
            q_view.stride(0), q_view.stride(1), q_view.stride(2),
            k_view.stride(0), k_view.stride(1), k_view.stride(2),
            v_view.stride(0), v_view.stride(1), v_view.stride(2),
            state32.stride(0), state32.stride(1), state32.stride(2), state32.stride(3),
            g.stride(0), g.stride(1),
            beta.stride(0), beta.stride(1),
            out.stride(0), out.stride(1),
            new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
            BLOCK_K=64,
            num_warps=2,
        )

        # Return outputs with original shapes
        output = out.unsqueeze(1).to(torch.bfloat16)   # [B, 1, H]
        new_state_out = new_state                      # [B, H, V, K], float32
        return output, new_state_out


def run(*args):
    return ModelNew()(*args)
