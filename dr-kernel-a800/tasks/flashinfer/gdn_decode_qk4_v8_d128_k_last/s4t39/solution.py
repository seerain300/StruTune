import math
import torch
import triton
import triton.language as tl


# Triton: compute g = exp(-exp(A_log[h]) * softplus(x)), x = a[b,h] + dt_bias[h], softplus(x) = log(1 + exp(x))
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


# Triton: compute beta = sigmoid(b), b is [B*H]
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


# Triton: compute dot(k_vec, state_mat) -> scalar [1]
# k_ptr: *f32, [K]
# state_ptr: *f32, [V*K]
# out_ptr: *f32, [1]
# V: tl.constexpr, K: tl.constexpr
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
            s_ij = tl.load(state_ptr + i * K + j)
            acc += s_ij * k_j
            i += 1
        j += 1
    tl.store(out_ptr, acc)


# Triton: compute dot(k_vec, beta * v_vec + (1 - beta) * old_v) -> scalar [1]
# k_ptr: *f32, [K]
# v_ptr: *f32, [V]
# out_ptr: *f32, [1]
# V: tl.constexpr, K: tl.constexpr
# beta_scalar: f32 scalar (1 - beta) not used here; handled on host
@triton.jit
def dot_k_newv_kernel(
    k_ptr,          # *f32, [K]
    v_ptr,          # *f32, [V]
    beta_scalar,    # f32 scalar
    old_v_scalar,   # f32 scalar
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
            # beta_scalar is passed; we can multiply directly in-kernel
            acc += v_i * k_j * beta_scalar
            i += 1
        j += 1
    acc += old_v_scalar
    tl.store(out_ptr, acc)


# Triton: compute h_state_vec[i] = (sum_j state[i,j] * g) - state_remove + state_update, for i in [0..V-1]
# state_ptr: *f32, [V*K]
# g_scalar: f32 scalar (g_out[b,h])
# state_remove_scalar: f32 scalar
# state_update_scalar: f32 scalar
# h_state_ptr: *f32, [V]
# V: tl.constexpr, K: tl.constexpr
@triton.jit
def h_state_vec_kernel(
    state_ptr,      # *f32, [V*K]
    g_scalar,       # f32 scalar
    state_remove_scalar,  # f32 scalar
    state_update_scalar,  # f32 scalar
    h_state_ptr,    # *f32, [V]
    V: tl.constexpr,
    K: tl.constexpr,
):
    i = 0
    while i < V:
        acc = 0.0
        j = 0
        while j < K:
            s_ij = tl.load(state_ptr + i * K + j)
            acc += s_ij * g_scalar
            j += 1
        h = acc - state_remove_scalar + state_update_scalar
        tl.store(h_state_ptr + i, h)
        i += 1


# Triton: compute output_scalar = scale * dot(q_vec, h_state_vec)
# q_ptr: *f32, [V]
# h_state_ptr: *f32, [V]
# scale: f32 scalar
# out_ptr: *f32, [1]
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


# Triton: write new_state[b,h] as [V,K] from h_state_vec[i], broadcasting across K
# h_state_ptr: *f32, [V]
# new_state_ptr: *f32, [V*K], row-major [V,K]
# V: tl.constexpr, K: tl.constexpr
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
            # write row i across K columns
            tl.store(new_state_ptr + i * K + j, hs_i)
            j += 1
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only forward:
        Accepts: q, k, v, state, A_log, a, dt_bias, b, scale
        Returns: (output, new_state)
          output: [B, 1, H], bfloat16
          new_state: [B, H, V, K], float32
        """
        # Shapes: q: [B,1,4,128], k: [B,1,4,128], v: [B,1,8,128], state: [B,H,V,K], A_log: [H], a: [B,1,H], dt_bias: [H], b: [B,1,H], scale: float
        Bq, Tq, QH, Kq = q.shape
        Bk, Tk, KH, Kk = k.shape
        Bv, Tv, VH, V = v.shape
        Bs, H, V2, K2 = state.shape
        assert QH == 4 and KH == 4 and VH == 8 and Kq == 128 and V == 128 and K2 == 128 and H == 8 and Tq == 1 and Tk == 1 and Tv == 1
        assert Bq == Bk == Bv == Bs
        B = Bq

        # Prepare inputs as float32 on device
        a_f = a.to(torch.float32).contiguous()
        dt_bias_f = dt_bias.to(torch.float32).contiguous()
        A_log_f = A_log.to(torch.float32).contiguous()
        b_f = b.to(torch.float32).contiguous()

        # Compute g and beta (B*H vectors)
        g_out = torch.empty(B * H, dtype=torch.float32, device=q.device)
        beta_out = torch.empty(B * H, dtype=torch.float32, device=q.device)

        # Launch Triton kernels
        softplus_and_exp_kernel[(B * H,)](dt_bias_f, a_f, A_log_f, g_out, H=H, B=B)
        sigmoid_kernel[(B * H,)](b_f, beta_out, H=H, B=B)

        # Prepare outputs
        output = torch.empty(B, H, dtype=torch.float32, device=q.device)  # per (b,h)
        new_state = torch.empty(B, H, V, K, dtype=torch.float32, device=q.device)

        # Pre-allocate scalars for per-(b,h) computations
        old_v_buf = torch.empty(1, dtype=torch.float32, device=q.device)
        state_remove_buf = torch.empty(1, dtype=torch.float32, device=q.device)
        state_update_buf = torch.empty(1, dtype=torch.float32, device=q.device)

        # Pre-allocate h_state vector
        h_state_vec = torch.empty(V, dtype=torch.float32, device=q.device)

        # Per (b,h) loop to compute everything
        for b_idx in range(B):
            for h_idx in range(H):
                base = b_idx * H + h_idx

                # Flatten k[b,h] and state[b,h]
                k_vec = k[b_idx, 0, h_idx].to(torch.float32).contiguous().view(-1)  # [K]
                state_mat = state[b_idx, h_idx].to(torch.float32).contiguous().view(-1)  # [V*K]
                q_vec = q[b_idx, 0, h_idx].to(torch.float32).contiguous().view(-1)  # [V]

                # 1) old_v = dot(k, state)
                dot_k_state_kernel[(1,)](k_vec, state_mat, old_v_buf, V=V, K=K)
                old_v = old_v_buf[0]

                # 2) state_remove = dot(k, g * state) -> compute dot_k_state and multiply by g on host
                dot_k_state_kernel[(1,)](k_vec, state_mat, state_remove_buf, V=V, K=K)
                g_val = g_out[base]
                state_remove = state_remove_buf[0] * g_val

                # 3) state_update = dot(k, beta * v + (1 - beta) * old_v)
                v_vec = v[b_idx, 0, h_idx].to(torch.float32).contiguous().view(-1)  # [V]
                beta_val = beta_out[base]
                one_minus_beta = 1.0 - beta_val
                dot_k_newv_kernel[(1,)](k_vec, v_vec, one_minus_beta, old_v, state_update_buf, V=V, K=K)
                state_update = state_update_buf[0]

                # 4) h_state_vec = sum_j state[i,j] * g - state_remove + state_update
                h_state_vec_kernel[(1,)](state_mat, g_val, state_remove, state_update, h_state_vec, V=V, K=K)

                # 5) output_scalar = scale * dot(q, h_state_vec)
                # scale may be provided as float; ensure it's a float32 scalar
                scale_val = float(scale)  # host scalar, Triton will receive as f32
                dot_q_hstate_kernel_write[(1,)](q_vec, h_state_vec, scale_val, output[b_idx, h_idx], V=V)

                # 6) write new_state[b,h] as [V,K] by broadcasting h_state_vec across K
                new_state[b_idx, h_idx] = write_new_state_kernel[(1,)](h_state_vec, new_state[b_idx, h_idx], V=V, K=K)

        # Cast output to bfloat16 and reshape to [B,1,H]
        output_bf16 = output.view(B, 1, H).to(torch.bfloat16)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
