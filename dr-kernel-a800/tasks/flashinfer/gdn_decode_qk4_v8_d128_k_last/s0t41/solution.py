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
def kernel_tmp_old_v(
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
    # k[b, h] is length K
    k = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + tl.arange(0, K), mask=tl.arange(0, K) < K)
    # state[b, h] is [V, K]
    state_mat = tl.load(state_ptr + b_idx * stride_state_b + h_idx * stride_state_h
                        + tl.arange(0, V)[:, None] * stride_state_v + tl.arange(0, K)[None, :] * stride_state_k,
                        mask=(tl.arange(0, V)[:, None] < V) & (tl.arange(0, K)[None, :] < K),
                        other=0.0)
    # tmp_old_v[b, h] = sum over K of k[k] * state_mat[k, :]
    tmp_scalar = tl.sum(state_mat * k[None, :], axis=1)  # [V]
    tmp_scalar = tl.sum(tmp_scalar, axis=0)  # sum over rows
    tl.store(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h, tmp_scalar)


@triton.jit
def kernel_update_and_output(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, tmp_ptr,
    output_ptr, new_state_ptr,
    B, H, V, K,
    scale,
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
    # Scalars
    g_val = tl.load(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h)
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h)
    old_v_scalar = tl.load(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h)  # tmp_old_v[b, h]

    # Load vectors
    k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + tl.arange(0, K), mask=tl.arange(0, K) < K)
    q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + tl.arange(0, K), mask=tl.arange(0, K) < K)
    v_vec = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + tl.arange(0, V), mask=tl.arange(0, V) < V)

    # Load state matrix [V, K]
    state_mat = tl.load(state_ptr + b_idx * stride_state_b + h_idx * stride_state_h
                        + tl.arange(0, V)[:, None] * stride_state_v + tl.arange(0, K)[None, :] * stride_state_k,
                        mask=(tl.arange(0, V)[:, None] < V) & (tl.arange(0, K)[None, :] < K),
                        other=0.0)  # [V, K]

    # Compute new_v per row: beta * v + (1 - beta) * old_v_scalar
    beta_v = beta_val * v_vec  # [V]
    new_v_vec = beta_v + (1.0 - beta_val) * old_v_scalar  # [V]
    # Compute state_remove = dot(k, state_mat) = sum over K
    state_remove = tl.sum(state_mat * k_vec[None, :], axis=1)  # [V]
    # Compute dot of k_vec with new_v_vec (scalar)
    k_new_v = tl.sum(k_vec * new_v_vec, axis=0)
    state_update = k_new_v  # scalar

    # Update new_state_mat elementwise across [V, K]
    new_state_mat = g_val * state_mat - state_remove[:, None] + state_update

    # Store new_state_mat to new_state_ptr
    tl.store(new_state_ptr + b_idx * stride_state_b + h_idx * stride_state_h
             + tl.arange(0, V)[:, None] * stride_state_v + tl.arange(0, K)[None, :] * stride_state_k,
             new_state_mat,
             mask=(tl.arange(0, V)[:, None] < V) & (tl.arange(0, K)[None, :] < K))

    # Compute output_scalar = scale * (q_vec @ new_state_mat)
    dot_per_row = tl.sum(new_state_mat * q_vec[None, :], axis=1)  # [V]
    output_scalar = scale * tl.sum(dot_per_row, axis=0)
    tl.store(output_ptr + b_idx * stride_output_b + h_idx * stride_output_h, output_scalar)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward. Inputs are as in the original run() function.
        Returns:
          output: [B, 1, H] in bfloat16
          new_state: [B, H, V, K] in float32
        """
        device = q.device
        B = q.shape[0]  # batch
        H = state.shape[1]  # number of heads (QH, typically 4 per provided inputs)
        V = v.shape[-1]  # vector length (8 per provided inputs)
        K = q.shape[-1]  # key length (128 per provided inputs)

        # Ensure tensors are contiguous and on device
        k = k.squeeze(1).contiguous()  # [B, KH, K], KH is assumed to match H as in provided inputs (4)
        q = q.squeeze(1).contiguous()  # [B, QH, K], QH is 4 in provided inputs
        v = v.squeeze(1).contiguous()  # [B, VH, V], VH is 8 in provided inputs
        state = state.contiguous()     # [B, H, V, K]

        # Allocate Triton scalar outputs [B, H]
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)
        tmp_old_v = torch.empty((B, H), dtype=torch.float32, device=device)
        output_scalar = torch.empty((B, H), dtype=torch.float32, device=device)

        # Allocate new state output tensor [B, H, V, K] in float32
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Kernel 1: compute g and beta
        stride_A = 1
        stride_a_b = 1
        stride_a_h = 1
        stride_dt = 1
        stride_b_b = 1
        stride_b_h = 1
        stride_g_b = 1
        stride_g_h = 1
        stride_beta_b = 1
        stride_beta_h = 1
        kernel_g_beta[(B, H)](
            A_log, a.squeeze(1), dt_bias, b.squeeze(1),
            g, beta,
            H,
            stride_A, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
            stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
            num_warps=4,
        )

        # Kernel 2: tmp_old_v = dot(k, state) over K for each (b,h)
        stride_k_b = k.stride(0)
        stride_k_h = k.stride(1)
        stride_k_k = k.stride(2)
        stride_state_b = state.stride(0)
        stride_state_h = state.stride(1)
        stride_state_v = state.stride(2)
        stride_state_k = state.stride(3)
        stride_tmp_b = 1
        stride_tmp_h = 1
        kernel_tmp_old_v[(B, H)](
            k, state, tmp_old_v,
            B, H, V, K,
            stride_k_b, stride_k_h, stride_k_k,
            stride_state_b, stride_state_h, stride_state_v, stride_state_k,
            stride_tmp_b, stride_tmp_h,
            num_warps=4,
        )

        # Kernel 3: update new_state and compute output_scalar per (b,h)
        stride_q_b = q.stride(0)
        stride_q_h = q.stride(1)
        stride_q_k = q.stride(2)
        stride_k_b = k.stride(0)
        stride_k_h = k.stride(1)
        stride_k_k = k.stride(2)
        stride_v_b = v.stride(0)
        stride_v_h = v.stride(1)
        stride_v_v = v.stride(2)
        stride_state_b = state.stride(0)
        stride_state_h = state.stride(1)
        stride_state_v = state.stride(2)
        stride_state_k = state.stride(3)
        stride_g_b = g.stride(0)
        stride_g_h = g.stride(1)
        stride_beta_b = beta.stride(0)
        stride_beta_h = beta.stride(1)
        stride_output_b = output_scalar.stride(0)
        stride_output_h = output_scalar.stride(1)
        kernel_update_and_output[(B, H)](
            q, k, v, state, g, beta, tmp_old_v,
            output_scalar, new_state,
            B, H, V, K,
            float(scale),
            stride_q_b, stride_q_h, stride_q_k,
            stride_k_b, stride_k_h, stride_k_k,
            stride_v_b, stride_v_h, stride_v_v,
            stride_state_b, stride_state_h, stride_state_v, stride_state_k,
            stride_g_b, stride_g_h,
            stride_beta_b, stride_beta_h,
            stride_output_b, stride_output_h,
            num_warps=4,
        )

        # Return output cast to bfloat16 in shape [B, 1, H], and new_state in float32 [B, H, V, K]
        output = output_scalar.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H]
        return output, new_state


def run(*args):
    return ModelNew()(*args)
