import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    H,  # number of heads (second dim of state)
    stride_A, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
    stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load per-head parameters
    A = tl.load(A_log_ptr + h_idx * stride_A).to(tl.float32)         # A_log[h]
    a = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h).to(tl.float32)  # a[b,h]
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt).to(tl.float32)     # dt_bias[h]
    bb = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32) # b[b,h]

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a + dt))
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))

    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,  # k layout: [B, H, K]
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,  # state layout: [B, H, V, K]
    stride_tmp_b, stride_tmp_h,
    BLOCK_K: tl.constexpr,
):
    # One program per (b,h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Accumulate dot(k[b,h], state[b,h]) across K in blocks
    acc = 0.0
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_offsets * stride_k_k, mask=k_mask, other=0.0)

        # Sum over V: for each i in [0, V), load state[b,h,i,k_offsets] and reduce
        for i in range(0, V):
            s_vec = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + i * stride_s_v + k_offsets * stride_s_k, mask=k_mask, other=0.0)
            acc += tl.sum(k_vec * s_vec, axis=0)

    tl.store(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h, acc)


@triton.jit
def kernel_elementwise_update(
    k_ptr, beta_ptr, v_row_ptr, state_in_ptr, new_state_out_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,  # k layout: [B, H, K]
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,  # state_in layout: [B, H, V, K]
    stride_v_b, stride_v_h, stride_v_v,  # v_row layout: [B, H, V]
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,  # new_state_out layout: [B, H, V, K]
    stride_beta_b, stride_beta_h,
    BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: (B*H, ceil_div(V, BLOCK_V))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    b_idx = pid0 // H
    h_idx = pid0 % H

    v_block_start = pid1 * BLOCK_V
    # We'll process one i per program inside this function by looping, but Triton doesn't support dynamic loops per program in the way we need; instead, we implement a block of i's and loop inside.
    # To keep code simple and correct, we compute tmp_old_v first and then perform a separate elementwise update over all i using a different approach. Given V=128, we can use a single kernel launch with grid = (B*H, 1) and inner loops.

    # Load beta[b,h]
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)

    # Compute tmp_old_v[b,h] = sum_j k[b,h,j] * sum_i state[b,h,i,j]
    tmp_old = 0.0
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_offsets * stride_k_k, mask=k_mask, other=0.0)
        sum_state = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for i in range(0, V):
            s_vec = tl.load(state_in_ptr + b_idx * stride_s_b + h_idx * stride_s_h + i * stride_s_v + k_offsets * stride_s_k, mask=k_mask, other=0.0)
            sum_state += s_vec
        tmp_old += tl.sum(k_vec * sum_state, axis=0)

    # Now perform elementwise update for all i
    # For each i, new_state[b,h,i,j] = state_in[b,h,i,j] - tmp_old + k[b,h] · (beta * v_row[b,h,i] + (1 - beta) * tmp_old)
    # We need v_row[b,h,i]; v_row_ptr is [B,H,V]
    for i in range(0, V):
        # Load v_row[b,h,i]
        v_i = tl.load(v_row_ptr + b_idx * stride_v_b + h_idx * stride_v_h + i * stride_v_v)
        # Compute state_update_scalar = beta * v_i + (1 - beta) * tmp_old
        state_update_scalar = beta_val * v_i + (1.0 - beta_val) * tmp_old

        # Load state_in row i and update
        s_in = tl.load(state_in_ptr + b_idx * stride_s_b + h_idx * stride_s_h + i * stride_s_v + tl.arange(0, K) * stride_s_k)
        new_row = s_in - tmp_old + state_update_scalar

        # Store to new_state_out
        tl.store(new_state_out_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + i * stride_ns_v + tl.arange(0, K) * stride_ns_k, new_row)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # All inputs must be on CUDA and contiguous for Triton
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "All inputs must be CUDA tensors"

        # Shapes
        B = q.shape[0]
        H = state.shape[1]  # number of heads (8 in provided inputs)
        V = state.shape[2]
        K = state.shape[3]

        # Prepare inputs for kernels
        # a: [B,1,H] -> [B,H]
        a_exp = a.squeeze(1)
        # dt_bias: [H]
        dt_bias_exp = dt_bias
        # b: [B,1,H] -> [B,H]
        b_exp = b.squeeze(1)

        # Allocate intermediates
        g = torch.empty((B, H), device=device, dtype=torch.float32)
        beta = torch.empty((B, H), device=device, dtype=torch.float32)
        tmp_old_v = torch.empty((B, H), device=device, dtype=torch.float32)

        # Launch kernel_g_beta
        kernel_g_beta[(B, H)](
            A_log, a_exp, dt_bias_exp, b_exp,
            g, beta,
            H,
            A_log.stride(0), a_exp.stride(0), a_exp.stride(1), dt_bias_exp.stride(0), b_exp.stride(0), b_exp.stride(1),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
            num_warps=1,
        )

        # Compute tmp_old_v via Triton
        # k: [B, 1, K] -> squeeze to [B, K]
        k_bk = k.squeeze(1)  # [B,K]
        # state: [B,H,V,K], ensure contiguous
        state_c = state.contiguous()  # [B,H,V,K]

        kernel_tmp_old_v[(B, H)](
            k_bk, state_c, tmp_old_v,
            B, H, V, K,
            k_bk.stride(0), k_bk.stride(1), k_bk.stride(2),  # k layout: [B,K]
            state_c.stride(0), state_c.stride(1), state_c.stride(2), state_c.stride(3),
            tmp_old_v.stride(0), tmp_old_v.stride(1),
            BLOCK_K=128,
        )

        # Prepare v_row as [B,H,V] for Triton: v: [B,1,V] -> squeeze, then expand to [B,H,V]
        v_b = v.squeeze(1).contiguous()  # [B,V]
        v_row = v_b.unsqueeze(1).expand(B, H, V).contiguous()  # [B,H,V]

        # Allocate new_state_out: [B,H,V,K]
        new_state_out = torch.empty((B, H, V, K), device=device, dtype=torch.float32)

        # Launch elementwise update kernel: grid over (B*H, 1) since we loop over V and K inside
        kernel_elementwise_update[(B * H, 1)](
            k_bk, beta, v_row, state_c, new_state_out,
            B, H, V, K,
            k_bk.stride(0), k_bk.stride(1), k_bk.stride(2),  # k layout: [B,K]
            state_c.stride(0), state_c.stride(1), state_c.stride(2), state_c.stride(3),
            v_row.stride(0), v_row.stride(1), v_row.stride(2),
            new_state_out.stride(0), new_state_out.stride(1), new_state_out.stride(2), new_state_out.stride(3),
            beta.stride(0), beta.stride(1),
            BLOCK_V=128, BLOCK_K=128,
        )

        # Compute final output with torch: output[b,h] = scale * (q[b,h] @ new_state[b,h])
        # q: [B,1,4,128] -> per head q[b,h,:] is q[b,0,h,:]
        q_perh = q[:, 0, :, :]  # [B,4,128]
        output = torch.empty((B, 1, H), device=device, dtype=torch.float32)
        for b_idx in range(B):
            for h_idx in range(H):
                q_h = q_perh[b_idx, h_idx, :]  # [128]
                new_state_h = new_state_out[b_idx, h_idx]  # [V,K] = [128,128]
                out_val = (q_h @ new_state_h).to(torch.float32)
                # apply scale if needed
                if scale is None or scale == 0.0:
                    out_val = out_val * (1.0 / math.sqrt(K))
                else:
                    out_val = out_val * float(scale)
                output[b_idx, 0, h_idx] = out_val

        # Return output cast to bfloat16 as [B,1,H], new_state_out as [B,H,V,K]
        output = output.to(torch.bfloat16)
        return [output, new_state_out]


def run(*args):
    return ModelNew()(*args)
