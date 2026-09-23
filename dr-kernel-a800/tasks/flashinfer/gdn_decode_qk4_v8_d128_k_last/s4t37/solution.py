import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute g[b,h] = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
# softplus(x) = log(1 + exp(x))
@triton.jit
def softplus_and_exp_kernel(
    dt_bias_ptr,    # *f32, [H]
    a_ptr,          # *f32, [B*H]
    A_log_ptr,      # *f32, [H]
    g_out_ptr,      # *f32, [B*H]
    H: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H - 1
    h = pid % H
    b = pid // H
    a_val = tl.load(a_ptr + pid)
    dt_val = tl.load(dt_bias_ptr + h)
    x = a_val + dt_val
    s = tl.log(1.0 + tl.exp(x))
    e = tl.exp(tl.load(A_log_ptr + h))
    g = tl.exp(-e * s)
    tl.store(g_out_ptr + pid, g)


# Triton kernel: compute beta[b,h] = 1 / (1 + exp(-b[b,h]))
@triton.jit
def sigmoid_kernel(
    b_ptr,          # *f32, [B*H]
    beta_out_ptr,   # *f32, [B*H]
    H: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*H - 1
    h = pid % H
    b_val = tl.load(b_ptr + pid)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    tl.store(beta_out_ptr + pid, beta)


# Triton kernel: compute scalar old_v = dot(k_vec[KH], state_mat_flat[V*K]) -> out_ptr[1]
@triton.jit
def dot_k_state_kernel(
    k_ptr,          # *f32, [KH]
    state_ptr,      # *f32, [V*K]
    out_ptr,        # *f32, [1]
    K: tl.constexpr,
    V: tl.constexpr,
    KH: tl.constexpr,
):
    acc = 0.0
    for j in range(KH):  # k_vec index
        k_j = tl.load(k_ptr + j)
        for i in range(V * K):
            state_ij = tl.load(state_ptr + i)
            acc += state_ij * k_j
    tl.store(out_ptr, acc)


# Triton kernel: compute scalar state_remove = dot(k_vec[KH], g * state_mat_flat[V*K]) -> out_ptr[1]
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # *f32, [KH]
    state_ptr,      # *f32, [V*K]
    g_val,          # f32 scalar
    out_ptr,        # *f32, [1]
    K: tl.constexpr,
    V: tl.constexpr,
    KH: tl.constexpr,
):
    acc = 0.0
    for j in range(KH):
        k_j = tl.load(k_ptr + j)
        for i in range(V * K):
            state_ij = tl.load(state_ptr + i)
            acc += state_ij * k_j
    acc = acc * g_val
    tl.store(out_ptr, acc)


# Triton kernel: compute scalar state_update = dot(k_vec[KH], beta * v_vec[V] + (1-beta) * old_v)
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # *f32, [KH]
    v_ptr,          # *f32, [V]
    old_v,          # f32 scalar
    beta_val,       # f32 scalar
    out_ptr,        # *f32, [1]
    K: tl.constexpr,
    V: tl.constexpr,
    KH: tl.constexpr,
):
    acc = 0.0
    for j in range(KH):
        k_j = tl.load(k_ptr + j)
        for i in range(V):
            v_i = tl.load(v_ptr + i)
            acc += v_i * k_j
    acc = acc * beta_val + (1.0 - beta_val) * old_v
    tl.store(out_ptr, acc)


# Triton kernel: compute h_state_vec[B*V] elementwise: h_state_vec[i] = sum_j state[b,h,i,j] * g - state_remove + state_update
# We write directly into h_state_vec_ptr [B*V]. Inputs: state_flat [V*K], g_scalar, state_remove_scalar, state_update_scalar.
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # *f32, [V*K]
    g_val,          # f32 scalar
    state_remove,   # f32 scalar
    state_update,   # f32 scalar
    h_state_ptr,    # *f32, [B*V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*V - 1
    acc = 0.0
    # Compute dot over K: sum_j state[i,j]
    for j in range(K):
        base = pid * K + j
        acc += tl.load(state_ptr + base)
    h_val = acc * g_val - state_remove + state_update
    tl.store(h_state_ptr + pid, h_val)


# Triton kernel: compute output_scalar[b,h] = scale * dot(q_vec[V], h_state_vec[V]) -> out_ptr[1]
@triton.jit
def dot_q_hstate_kernel_write(
    q_ptr,          # *f32, [V]
    h_state_ptr,    # *f32, [V]
    scale,          # f32 scalar
    out_ptr,        # *f32, [1]
    V: tl.constexpr,
):
    acc = 0.0
    for i in range(V):
        q_i = tl.load(q_ptr + i)
        hs_i = tl.load(h_state_ptr + i)
        acc += q_i * hs_i
    acc = acc * scale
    tl.store(out_ptr, acc)


# Triton kernel: write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # *f32, [B*V]
    new_state_ptr,  # *f32, [B*V*K]
    V: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0 .. B*V - 1
    val = tl.load(h_state_ptr + pid)
    # write val across K for this i
    for j in range(K):
        base = pid * K + j
        tl.store(new_state_ptr + base, val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation that matches original signature:
        q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128], state: [B, 8, 128, 128]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        Returns:
          output: [B, 1, 8], dtype bfloat16
          new_state: [B, 8, 128, 128], dtype float32
        """
        # Shapes derived
        Bq, Tq, QH, K = q.shape  # e.g., [B, 1, 4, 128]
        _, Tk, KH, _ = k.shape   # e.g., [B, 1, 4, 128]
        _, Tv, VH, V = v.shape   # e.g., [B, 1, 8, 128]
        B, H, V2, K2 = state.shape
        # Asserts to match original logic
        assert QH == 4 and KH == 4 and VH == 8 and K == 128 and V == 128 and H == V2 and K2 == 128 and Tq == 1 and Tk == 1 and Tv == 1
        assert B == Bq

        # Ensure all inputs are on CUDA and contiguous; compute in float32
        device = q.device
        q32 = q.float().contiguous()
        k32 = k.float().contiguous()
        v32 = v.float().contiguous()
        state32 = state.float().contiguous()
        A_log32 = A_log.float().contiguous()
        a32 = a.float().contiguous()
        dt_bias32 = dt_bias.float().contiguous()
        b32 = b.float().contiguous()

        # Compute g and beta using Triton
        g_out = torch.empty(B * H, dtype=torch.float32, device=device)
        beta_out = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch softplus_and_exp_kernel
        grid_g = (B * H,)
        softplus_and_exp_kernel[grid_g](
            dt_bias32, a32, A_log32, g_out, H=H, B=B
        )

        # Launch sigmoid_kernel
        grid_b = (B * H,)
        beta_out = beta_out.view(B * H)
        sigmoid_kernel[grid_b](
            b32, beta_out, H=H, B=B
        )

        # Prepare outputs
        output = torch.empty((B, 1, H), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Process each (b,h) using Triton kernels
        for b_idx in range(B):
            for h_idx in range(H):
                base = b_idx * H + h_idx

                # Extract per-(b,h) slices
                # k_vec: [KH], state_mat_flat: [V*K], v_vec: [V], q_vec: [V]
                k_vec = k32[b_idx, 0, h_idx]            # [KH]
                state_mat = state32[b_idx, h_idx]       # [V, K], but pass flat [V*K]
                v_vec = v32[b_idx, 0, h_idx]            # [V]
                q_vec = q32[b_idx, 0, h_idx]            # [V]

                # old_v = dot(k_vec, state_mat_flat)
                old_v = torch.empty(1, dtype=torch.float32, device=device)
                dot_k_state_kernel[(1,)](
                    k_vec, state_mat.reshape(-1), old_v, K=K, V=V, KH=KH
                )
                old_v = old_v[0]

                # state_remove = dot(k_vec, g * state_mat_flat)
                state_remove = torch.empty(1, dtype=torch.float32, device=device)
                g_val = g_out[base]
                dot_k_gstate_kernel[(1,)](
                    k_vec, state_mat.reshape(-1), g_val, state_remove, K=K, V=V, KH=KH
                )
                state_remove = state_remove[0]

                # state_update = dot(k_vec, beta * v_vec + (1-beta) * old_v)
                beta_val = beta_out[base]
                state_update = torch.empty(1, dtype=torch.float32, device=device)
                dot_k_newv_kernel[(1,)](
                    k_vec, v_vec, old_v, beta_val, state_update, K=K, V=V, KH=KH
                )
                state_update = state_update[0]

                # Compute h_state_vec[B*V]: sum_j state[b,h,i,j] * g - state_remove + state_update
                h_state_vec = torch.empty(B * V, dtype=torch.float32, device=device)
                h_state_vec_kernel[(B * V,)](
                    state_mat.reshape(-1), g_val, state_remove, state_update, h_state_vec, V=V, K=K
                )

                # output[b,h] = scale * dot(q_vec, h_state_vec)
                out_scalar = torch.empty(1, dtype=torch.float32, device=device)
                dot_q_hstate_kernel_write[(1,)](
                    q_vec, h_state_vec, scale, out_scalar, V=V
                )
                output[b_idx, 0, h_idx] = out_scalar[0].to(torch.bfloat16)

                # Write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
                base_new = b_idx * (H * V * K) + h_idx * (V * K)
                new_state[b_idx, h_idx] = write_new_state_kernel[(1,)](
                    h_state_vec, new_state.view(B * H * V * K), V=V, K=K
                )  # Note: This write is implicit via the kernel; we pass pointer to flattened tensor.

        return output, new_state


def run(*args):
    return ModelNew()(*args)
