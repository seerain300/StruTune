import math
import torch
import triton
import triton.language as tl


# Kernel: compute g = exp(-exp(A_log[h]) * softplus(x)), where x = a[b,h] + dt_bias[h]
# softplus(x) = log(1 + exp(x))
@triton.jit
def softplus_and_exp_kernel(
    dt_bias_ptr,    # float32 [H]
    a_ptr,          # float32 [B*H]
    A_log_ptr,      # float32 [H]
    g_out_ptr,      # float32 [B*H]
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H-1
    h = pid % H
    a_val = tl.load(a_ptr + pid)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val
    s = tl.log(1.0 + tl.exp(x))
    e = tl.exp(tl.load(A_log_ptr + h))
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Kernel: compute beta = sigmoid(b[b,h]) where index pid = b*H + h
@triton.jit
def sigmoid_kernel(
    b_ptr,          # float32 [B*H]
    beta_out_ptr,   # float32 [B*H]
    H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Fused kernel: per (b,h), compute:
#   old_v = dot(k, state)
#   g_state = g * state; state_remove = dot(k, g_state)
#   new_v = beta * v + (1 - beta) * old_v; state_update = dot(k, new_v)
#   h_state_vec[i] = sum_j state[i,j] * g - state_remove + state_update
#   output_scalar = scale * dot(q, h_state_vec)
#   write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
@triton.jit
def fused_bh_kernel(
    # inputs
    q_ptr,          # float32 [K]
    k_ptr,          # float32 [K]
    v_ptr,          # float32 [V]
    state_ptr,      # float32 [V*K] flattened
    g_val,          # float32 scalar g[b,h]
    beta_val,       # float32 scalar beta[b,h]
    scale,          # float32 scalar
    # outputs
    out_scalar_ptr, # float32 [1]
    new_state_ptr,  # float32 [V*K] (we write this via pointer arithmetic)
    B, H, V, K,     # runtime ints (not needed but can be used if needed)
    H_const: tl.constexpr,
    V_const: tl.constexpr,
    K_const: tl.constexpr,
):
    # We assume B,H,V,K are consistent with the caller; here we operate on a single (b,h) pair
    # The grid is set to 1 program per (b,h), so B,H are not needed beyond launching.
    # Compute dot products
    # old_v = dot(k, state)
    old_v = 0.0
    for j in range(K_const):
        k_j = tl.load(k_ptr + j)
        for i in range(V_const):
            state_ij = tl.load(state_ptr + i * K_const + j)
            old_v += state_ij * k_j

    # g_state = g * state; state_remove = dot(k, g_state)
    g_state_prod = 0.0
    for j in range(K_const):
        k_j = tl.load(k_ptr + j)
        for i in range(V_const):
            state_ij = tl.load(state_ptr + i * K_const + j)
            g_state_prod += state_ij * g_val
    state_remove = 0.0
    for j in range(K_const):
        k_j = tl.load(k_ptr + j)
        state_remove += g_state_prod * k_j

    # new_v = beta * v + (1 - beta) * old_v; state_update = dot(k, new_v)
    new_v = 0.0
    for i in range(V_const):
        v_i = tl.load(v_ptr + i)
        new_v += v_i * beta_val + (1.0 - beta_val) * old_v
    state_update = 0.0
    for j in range(K_const):
        k_j = tl.load(k_ptr + j)
        state_update += new_v * k_j

    # h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update
    h_state_vec = [0.0 for _ in range(V_const)]
    for i in range(V_const):
        dot_g = 0.0
        for j in range(K_const):
            state_ij = tl.load(state_ptr + i * K_const + j)
            dot_g += state_ij * g_val
        h_state_vec[i] = dot_g - state_remove + state_update

    # output_scalar = scale * dot(q, h_state_vec)
    out_sum = 0.0
    for i in range(V_const):
        out_sum += h_state_vec[i]
    out_scalar = scale * out_sum
    tl.store(out_scalar_ptr, out_scalar)

    # Write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
    base = 0
    for i in range(V_const):
        val = h_state_vec[i]
        for j in range(K_const):
            tl.store(new_state_ptr + base + j, val)
        base += K_const


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only forward: compute output and new_state with Triton kernels.
        Returns (output: [B, 1, H], bfloat16), (new_state: [B, H, V, K], float32)
        """
        device = q.device
        # Convert inputs to float32 for compute stability
        q_f = q.contiguous().float()       # [B, 1, QH, K] but we only use q[b,h], so we can squeeze T=1 -> [B, QH, K]
        k_f = k.contiguous().float()       # [B, 1, KH, K] -> [B, KH, K]
        v_f = v.contiguous().float()       # [B, 1, VH, V] -> [B, VH, V]
        a_f = a.contiguous().float()       # [B, 1, VH] -> [B*VH]
        b_f = b.contiguous().float()       # [B, 1, VH] -> [B*VH]
        dt_bias_f = dt_bias.contiguous().float()  # [VH]
        state_f = state.contiguous().float()      # [B, VH, V, K] -> [B, VH, V*K] flatten for kernel

        # Compute g and beta in Triton
        B = q.shape[0]
        H = a_f.shape[0]  # equals b.shape[0] * b.shape[1], but here assert B*H == a_f.numel()
        assert a_f.numel() == b_f.numel(), "a and b must have same number of elements"
        assert dt_bias_f.numel() == B, "A_log length must match H dimension"
        assert state_f.shape[0] == B, "state batch must match q/k/v batch"
        assert state_f.shape[1] == dt_bias_f.numel(), "state heads must match A_log length"
        assert state_f.shape[2] == dt_bias_f.numel(), "state last dim V must match heads"
        assert state_f.shape[3] == 128, "K must be 128"
        V = state_f.shape[2]
        K = state_f.shape[3]
        # Ensure constants
        assert V == 128 and K == 128, "V and K must be 128"

        g_out = torch.empty(B * dt_bias_f.numel(), dtype=torch.float32, device=device)
        beta_out = torch.empty(B * dt_bias_f.numel(), dtype=torch.float32, device=device)

        # Launch kernels to compute g and beta
        grid_g = (B * dt_bias_f.numel(),)
        softplus_and_exp_kernel[grid_g](
            dt_bias_f, a_f, dt_bias_f, g_out, H_const=dt_bias_f.numel(),
        )
        grid_beta = (B * dt_bias_f.numel(),)
        sigmoid_kernel[grid_beta](
            b_f, beta_out, H_const=dt_bias_f.numel(),
        )

        # Now, for each (b,h), launch fused_bh_kernel
        output_list = []
        new_state = torch.empty((B, dt_bias_f.numel(), V, K), dtype=torch.float32, device=device)

        # Prepare q_ptr, k_ptr, v_ptr for each (b,h)
        # We loop over b and h; q, k, v have dimensions [B,1,*,*]. Since T=1, we can index as (b,0,*,*).
        for b_idx in range(B):
            for h_idx in range(dt_bias_f.numel()):
                # Compute pointers for q[b,h], k[b,h], v[b,h] (shape [K], [K], [V])
                q_bh = q_f[b_idx, 0, h_idx].contiguous().view(-1)  # [K]
                k_bh = k_f[b_idx, 0, h_idx].contiguous().view(-1)  # [K]
                v_bh = v_f[b_idx, 0, h_idx].contiguous().view(-1)  # [V]
                # state[b,h] is [V*K] flattened
                state_bh = state_f[b_idx, h_idx].contiguous().view(-1)  # [V*K]

                # Output scalar buffer
                out_scalar = torch.empty(1, dtype=torch.float32, device=device)

                # Launch fused_bh_kernel
                grid_bh = (1,)
                fused_bh_kernel[grid_bh](
                    q_bh, k_bh, v_bh, state_bh,
                    g_out[b_idx * dt_bias_f.numel() + h_idx],
                    beta_out[b_idx * dt_bias_f.numel() + h_idx],
                    float(scale),
                    out_scalar,
                    new_state[b_idx, h_idx].contiguous().view(-1),  # write [V*K] contiguous
                    B, dt_bias_f.numel(), V, K,
                    H_const=dt_bias_f.numel(), V_const=V, K_const=K,
                )

                # output is [B, 1, H], bfloat16 (as in original helper)
                output_list.append(out_scalar[0].to(torch.bfloat16))

        # Stack outputs into [B, 1, H]
        output = torch.stack(output_list, dim=1)  # shape [B, 1, H]

        return output, new_state


def run(*args):
    return ModelNew()(*args)
