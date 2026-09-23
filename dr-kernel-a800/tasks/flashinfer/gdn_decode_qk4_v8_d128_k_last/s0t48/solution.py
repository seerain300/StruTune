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
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
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
    B, H, K, V,
    stride_k_b, stride_k_h, stride_k_k,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_tmp_b, stride_tmp_h,
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    k_offs = tl.arange(0, K)
    k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_offs * stride_k_k)
    v = tl.arange(0, V)
    k_dim = tl.arange(0, K)
    s_ptrs = state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v[:, None] * stride_s_v + k_dim[None, :] * stride_s_k
    mask = (v[:, None] < V) & (k_dim[None, :] < K)
    state_block = tl.load(s_ptrs, mask=mask, other=0.0)
    tmp_vec = state_block * k_vec[None, :]
    tmp_sum = tl.sum(tmp_vec, axis=1)  # sum over K for each v
    tmp_val = tl.sum(tmp_sum, axis=0)  # sum over V to scalar
    tl.store(tmp_ptr + b_idx * stride_tmp_b + h_idx * stride_tmp_h, tmp_val)


@triton.jit
def kernel_update_and_output(
    q_ptr, k_ptr, beta_ptr, v_ptr, state_ptr, new_state_ptr, output_ptr, scale,
    B, H, K, V,
    stride_q_b, stride_q_h, stride_q_k,
    stride_k_b, stride_k_h, stride_k_k,
    stride_beta_b, stride_beta_h,
    stride_v_b, stride_v_h, stride_v_v,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
    stride_out_b, stride_out_h,
    num_warps: tl.constexpr,
):
    # Each program handles one (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load k and beta scalars
    k_offs = tl.arange(0, K)
    k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_offs * stride_k_k)
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)

    # Load q vector
    q_offs = tl.arange(0, K)
    q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + q_offs * stride_q_k)

    # Prepare state block [V, K]
    v = tl.arange(0, V)
    k_dim = tl.arange(0, K)
    s_ptrs = state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v[:, None] * stride_s_v + k_dim[None, :] * stride_s_k
    mask = (v[:, None] < V) & (k_dim[None, :] < K)
    state_block = tl.load(s_ptrs, mask=mask, other=0.0)

    # Compute old_v = k @ state
    old_v = tl.sum(state_block * k_vec[None, :], axis=1)  # sum over K for each v, returns [V]
    old_v_scalar = tl.sum(old_v, axis=0)  # sum over V to scalar

    # Compute new_v = beta * v + (1 - beta) * old_v
    # v_ptr[b, h, v] where v varies, but we can load per v and form vector
    v_vec_ptrs = v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + v * stride_v_v
    v_vec = tl.load(v_vec_ptrs, mask=(v < V), other=0.0)
    new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # broadcasting beta_val to [V]

    # Compute state_remove = k @ old_state (old_state = beta * state)
    old_state = beta_val * state_block  # [V, K]
    state_remove_scalar = tl.sum(old_state * k_vec[None, :], axis=1)  # [V]
    state_remove_scalar = tl.sum(state_remove_scalar, axis=0)  # scalar

    # Compute state_update = k @ new_v
    state_update_vec = tl.sum(new_v[None, :] * k_vec[None, :], axis=1)  # [K]
    # h_state = old_state - state_remove + state_update
    h_state = old_state - state_remove_scalar + state_update_vec[None, :]  # broadcasting state_update across rows

    # Store new_state
    ns_ptrs = new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v[:, None] * stride_ns_v + k_dim[None, :] * stride_ns_k
    tl.store(ns_ptrs, h_state, mask=mask)

    # Compute output = scale * (q @ h_state)
    prod = h_state * q_vec[None, :]  # [V, K] * [K] -> [V, K]
    prod_sum = tl.sum(prod, axis=1)  # [V]
    out_val = tl.sum(prod_sum, axis=0)  # scalar
    out_val = scale * out_val
    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_h, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure inputs are contiguous and float32
        device = q.device
        dtype_in = q.dtype
        B, T_q, num_q_heads, K = q.shape
        _, T_k, num_k_heads, _ = k.shape
        _, T_v, num_v_heads, V = v.shape
        # Original code asserts T_q == 1, num_q_heads == 4, num_k_heads == 4, num_v_heads == 8, K == 128, V == 128.
        # We will use these as fixed values in the computation to match the provided inputs.
        # However, we will still allocate outputs using actual B, H, V, K derived from inputs.

        # Cast inputs to float32 for compute
        q32 = q.reshape(B, 1, num_q_heads, K).to(torch.float32).contiguous()  # [B, 1, 4, 128]
        k32 = k.reshape(B, 1, num_k_heads, K).to(torch.float32).contiguous()  # [B, 1, 4, 128]
        v32 = v.reshape(B, 1, num_v_heads, V).to(torch.float32).contiguous()  # [B, 1, 8, 128]
        state32 = state.to(torch.float32).contiguous()                        # [B, 8, 128, 128]
        A_log32 = A_log.to(torch.float32).contiguous()                        # [8]
        a32 = a.reshape(B, 1, num_v_heads).to(torch.float32).contiguous()     # [B, 1, 8]
        dt_bias32 = dt_bias.to(torch.float32).contiguous()                    # [8]
        b32 = b.reshape(B, 1, num_v_heads).to(torch.float32).contiguous()     # [B, 1, 8]
        scale_f32 = float(scale)

        # We operate with H=num_v_heads=8 (per original code logic)
        H = num_v_heads

        # Allocate outputs with runtime shapes
        # g and beta are [B, H]
        g = torch.empty((B, H), dtype=torch.float32, device=device)
        beta = torch.empty((B, H), dtype=torch.float32, device=device)

        # tmp_old_v is [B, H]
        tmp = torch.empty((B, H), dtype=torch.float32, device=device)

        # new_state is [B, H, V, K] float32
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # output is [B, H] float32; we'll cast to bfloat16 at the end as [B, 1, H]
        output = torch.empty((B, H), dtype=torch.float32, device=device)

        # Compute strides for kernels
        stride_q_b, stride_q_h, stride_q_k = q32.stride(0), q32.stride(1), q32.stride(2)
        stride_k_b, stride_k_h, stride_k_k = k32.stride(0), k32.stride(1), k32.stride(2)
        stride_beta_b, stride_beta_h = beta.stride(0), beta.stride(1)
        stride_v_b, stride_v_h, stride_v_v = v32.stride(0), v32.stride(1), v32.stride(2)
        stride_s_b, stride_s_h, stride_s_v, stride_s_k = state32.stride(0), state32.stride(1), state32.stride(2), state32.stride(3)
        stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k = new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3)
        stride_out_b, stride_out_h = output.stride(0), output.stride(1)

        # Launch kernels: grid=(B, H)
        kernel_g_beta[(B, H)](
            A_log32, a32, dt_bias32, b32, g, beta,
            B, H,
            A_log32.stride(0), a32.stride(0), a32.stride(1), dt_bias32.stride(0), b32.stride(0), b32.stride(1),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
            num_warps=4,
        )

        kernel_tmp_old_v[(B, H)](
            k32, state32, tmp,
            B, H, K, V,
            k32.stride(0), k32.stride(1), k32.stride(2),
            state32.stride(0), state32.stride(1), state32.stride(2), state32.stride(3),
            tmp.stride(0), tmp.stride(1),
            num_warps=4,
        )

        kernel_update_and_output[(B, H)](
            q32, k32, beta, v32, state32, new_state, output, scale_f32,
            B, H, K, V,
            stride_q_b, stride_q_h, stride_q_k,
            k32.stride(0), k32.stride(1), k32.stride(2),
            beta.stride(0), beta.stride(1),
            v32.stride(0), v32.stride(1), v32.stride(2),
            state32.stride(0), state32.stride(1), state32.stride(2), state32.stride(3),
            new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
            output.stride(0), output.stride(1),
            num_warps=4,
        )

        # Return: output cast to bfloat16 as [B, 1, H], new_state float32 as [B, H, V, K]
        output_out = output.view(B, 1, H).to(torch.bfloat16)
        return output_out, new_state


def run(*args):
    return ModelNew()(*args)
