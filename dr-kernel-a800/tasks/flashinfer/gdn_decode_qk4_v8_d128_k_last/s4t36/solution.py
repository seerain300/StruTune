import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute g = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
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


# Triton kernel: compute beta = sigmoid(b[b,h]) -> [B*H] float32
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


# Triton kernel: compute dot(k_vec, state_mat) -> scalar [1]
# k_ptr: [K], state_ptr: [V*K], out_ptr: [1]
@triton.jit
def dot_k_state_kernel(
    k_ptr,          # *f32, [K]
    state_ptr,      # *f32, [V*K]
    out_ptr,        # *f32, [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    j = 0
    while j < K:
        k_j = tl.load(k_ptr + j)
        i = 0
        while i < V:
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j
            i += 1
        j += 1
    tl.store(out_ptr, acc)


# Triton kernel: compute dot(k_vec, g * state_mat) -> scalar [1]
@triton.jit
def dot_k_gstate_kernel(
    k_ptr,          # *f32, [K]
    state_ptr,      # *f32, [V*K]
    g_val,          # f32 scalar
    out_ptr,        # *f32, [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    j = 0
    while j < K:
        k_j = tl.load(k_ptr + j)
        i = 0
        while i < V:
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * k_j * g_val
            i += 1
        j += 1
    tl.store(out_ptr, acc)


# Triton kernel: compute dot(k_vec, beta * v_vec + (1-beta) * old_v) -> scalar [1]
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # *f32, [K]
    v_ptr,          # *f32, [V]
    old_v_ptr,      # *f32, [1] scalar
    beta_val,       # f32 scalar
    out_ptr,        # *f32, [1]
    V: tl.constexpr,
    K: tl.constexpr,
):
    acc = 0.0
    j = 0
    while j < K:
        k_j = tl.load(k_ptr + j)
        i = 0
        while i < V:
            v_i = tl.load(v_ptr + i)
            old_v = tl.load(old_v_ptr)
            acc += k_j * (beta_val * v_i + (1.0 - beta_val) * old_v)
            i += 1
        j += 1
    tl.store(out_ptr, acc)


# Triton kernel: compute h_state_vec[i] = sum_j state[b,h,i,j] * g - state_remove + state_update for i in [0..V-1]
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # *f32, [V*K]
    g_val,          # f32 scalar
    state_remove,   # f32 scalar
    state_update,   # f32 scalar
    h_state_ptr,    # *f32, [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    i = 0
    while i < V:
        acc = 0.0
        j = 0
        while j < K:
            state_ij = tl.load(state_ptr + i * K + j)
            acc += state_ij * g_val
            j += 1
        acc = acc - state_remove + state_update
        tl.store(h_state_ptr + i, acc)
        i += 1


# Triton kernel: compute output_scalar[b,h] = scale * dot(q_vec, h_state_vec) -> store to [1]
@triton.jit
def dot_q_hstate_kernel_write(
    q_ptr,          # *f32, [V]
    h_state_ptr,    # *f32, [V]
    scale,          # f32 scalar
    out_ptr,        # *f32, [1]
    V: tl.constexpr,
):
    acc = 0.0
    i = 0
    while i < V:
        q_i = tl.load(q_ptr + i)
        hs_i = tl.load(h_state_ptr + i)
        acc += q_i * hs_i
        i += 1
    acc = acc * scale
    tl.store(out_ptr, acc)


# Triton kernel: write new_state[b,h] as [V,K] from h_state_vec[i], broadcasting i over K
@triton.jit
def write_new_state_kernel(
    h_state_ptr,    # *f32, [V]
    new_state_ptr,  # *f32, [V*K]
    V: tl.constexpr,
    K: tl.constexpr,
):
    i = 0
    while i < V:
        hs_i = tl.load(h_state_ptr + i)
        j = 0
        while j < K:
            tl.store(new_state_ptr + i * K + j, hs_i)
            j += 1
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation: compute and return output and new_state.
        Returns:
          output: [B, 1, H], dtype bfloat16
          new_state: [B, H, V, K], dtype float32
        """
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All tensors must be CUDA tensors."
        device = q.device

        # Shapes (from original code assumptions)
        Bq, Tq, QH, K = q.shape
        _, Tk, KH, _ = k.shape
        _, Tv, VH, V = v.shape
        B, H, V2, K2 = state.shape
        assert QH == 4 and KH == 4 and VH == 8 and K == 128 and V == 128 and Bq == B and H == V2 and K == K2 and Tq == 1 and Tk == 1 and Tv == 1

        # Prepare inputs in float32
        a_f = a.to(torch.float32).contiguous()
        b_f = b.to(torch.float32).contiguous()
        dt_bias_f = dt_bias.to(torch.float32).contiguous()
        A_log_f = A_log.to(torch.float32).contiguous()

        # Compute g and beta using Triton
        g_out = torch.empty(B * H, dtype=torch.float32, device=device)
        beta_out = torch.empty(B * H, dtype=torch.float32, device=device)

        grid_g = (B * H,)
        softplus_and_exp_kernel[grid_g](
            dt_bias_ptr=dt_bias_f,
            a_ptr=a_f,
            A_log_ptr=A_log_f,
            g_out_ptr=g_out,
            H=H, B=B,
        )

        grid_beta = (B * H,)
        sigmoid_kernel[grid_beta](
            b_ptr=b_f,
            beta_out_ptr=beta_out,
            H=H, B=B,
        )

        # Output and new_state
        output = torch.empty((B, 1, H), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Per-(b,h) updates
        for b_idx in range(B):
            for h_idx in range(H):
                idx = b_idx * H + h_idx

                # Extract vectors and matrix
                q_vec = q[b_idx, 0, h_idx].contiguous().to(torch.float32)          # [V]
                k_vec = k[b_idx, 0, h_idx].contiguous().to(torch.float32)          # [K]
                v_vec = v[b_idx, 0, h_idx].contiguous().to(torch.float32)          # [V]
                state_mat = state[b_idx, h_idx].contiguous().to(torch.float32)     # [V,K]

                # Scalars and vectors for update
                old_v = torch.empty(1, dtype=torch.float32, device=device)
                state_remove = torch.empty(1, dtype=torch.float32, device=device)
                state_update = torch.empty(1, dtype=torch.float32, device=device)
                h_state_vec = torch.empty(V, dtype=torch.float32, device=device)   # [V]

                # Compute old_v = k @ state
                dot_k_state_kernel[(1,)](
                    k_ptr=k_vec, state_ptr=state_mat, out_ptr=old_v, V=V, K=K
                )

                # state_remove = k @ (g * state)
                g_val = g_out[idx]
                dot_k_gstate_kernel[(1,)](
                    k_ptr=k_vec, state_ptr=state_mat, g_val=g_val, out_ptr=state_remove, V=V, K=K
                )

                # state_update = k @ (beta * v + (1 - beta) * old_v_scalar)
                beta_val = beta_out[idx]
                dot_k_newv_kernel[(1,)](
                    k_ptr=k_vec, v_ptr=v_vec, old_v_ptr=old_v, beta_val=beta_val, out_ptr=state_update, V=V, K=K
                )

                # h_state_vec[i] = sum_j state_mat[i,j] * g - state_remove + state_update
                h_state_vec_kernel[(1,)](
                    state_ptr=state_mat,
                    g_val=g_val,
                    state_remove=state_remove[0],
                    state_update=state_update[0],
                    h_state_ptr=h_state_vec,
                    V=V, K=K,
                )

                # output_scalar[b,h] = scale * dot(q_vec, h_state_vec)
                out_scalar = torch.empty(1, dtype=torch.float32, device=device)
                dot_q_hstate_kernel_write[(1,)](
                    q_ptr=q_vec,
                    h_state_ptr=h_state_vec,
                    scale=float(scale),
                    out_ptr=out_scalar,
                    V=V,
                )
                # Store to output [B,1,H] as bfloat16
                output[b_idx, 0, h_idx] = out_scalar[0].to(torch.bfloat16)

                # Write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
                new_state_row = new_state[b_idx, h_idx]  # [V,K]
                write_new_state_kernel[(1,)](
                    h_state_ptr=h_state_vec,
                    new_state_ptr=new_state_row,    # linear indexing over [V*K]
                    V=V, K=K,
                )

        return output, new_state


def run(*args):
    return ModelNew()(*args)
