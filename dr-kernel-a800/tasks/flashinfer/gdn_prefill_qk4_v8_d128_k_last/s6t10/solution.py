import torch
import math
import triton
import triton.language as tl


# Triton kernels

@triton.jit
def compute_g_beta_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, beta_ptr, T: tl.int32, H: tl.int32):
    """
    Compute per (t, h):
      g = exp(-exp(A_log[h]) * softplus(a[t,h] + dt_bias[h]))
      beta[t,h] = sigmoid(b[t,h])
    a_ptr: [T*H] bfloat16 (flattened)
    dt_bias_ptr: [H] float32
    A_log_ptr: [H] float32
    g_ptr: [T*H] float32
    beta_ptr: [T*H] float32 (we store precomputed beta from host)
    """
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    if t >= T:
        return
    a_val = tl.load(a_ptr + pid).to(tl.float32)          # bfloat16 -> float32
    db_val = tl.load(dt_bias_ptr + h)                   # float32
    A_val = tl.load(A_log_ptr + h)                      # float32
    x = a_val + db_val
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    g_val = tl.exp(-tl.exp(A_val) * sp)               # g
    # beta = sigmoid(b); beta_ptr precomputed by host via sigmoid_triton
    beta_val = tl.load(beta_ptr + pid)
    tl.store(g_ptr + pid, g_val)
    tl.store(beta_ptr + pid, beta_val)


@triton.jit
def softplus_triton(x_ptr, out_ptr, N: tl.int32):
    """
    Elementwise softplus(x) = log(1 + exp(x)) into out_ptr
    x_ptr: [N] float32
    out_ptr: [N] float32
    """
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + pid, y)


@triton.jit
def sigmoid_triton(x_ptr, out_ptr, N: tl.int32):
    """
    Elementwise sigmoid(x) = 1 / (1 + exp(-x)) into out_ptr
    x_ptr: [N] float32
    out_ptr: [N] float32
    """
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + pid, y)


@triton.jit
def mm_k_state_single(a_ptr, B_ptr, out_ptr, N: tl.int32, K: tl.int32):
    """
    Vector matmul: out[n] = sum_k a[k] * B[k, n], where
      a_ptr: [K] float32 (k vector for this head at time t)
      B_ptr: [K, N] float32 (state_old[h] as [K, N] flattened)
      out_ptr: [N] float32
    """
    n = tl.program_id(0)
    if n >= N:
        return
    sum_val = 0.0
    for kk in range(0, K):
        a_k = tl.load(a_ptr + kk)
        # B[kk, n] index in flattened is kk*N + n
        B_kn = tl.load(B_ptr + kk * N + n)
        sum_val += a_k * B_kn
    tl.store(out_ptr + n, sum_val)


@triton.jit
def mm_kT_vec_single(k_ptr, v_ptr, out_ptr, N: tl.int32, K: tl.int32):
    """
    Scalar per kk: out[kk] = sum_n k[kk] * v[n]
    k_ptr: [K] float32
    v_ptr: [N] float32
    out_ptr: [K] float32
    """
    kk = tl.program_id(0)
    if kk >= K:
        return
    sum_val = 0.0
    for n in range(0, N):
        k_k = tl.load(k_ptr + kk)
        v_n = tl.load(v_ptr + n)
        sum_val += k_k * v_n
    tl.store(out_ptr + kk, sum_val)


@triton.jit
def update_state_single(state_ptr, delta_ptr, g_ptr, new_ptr, K: tl.int32, N: tl.int32):
    """
    Elementwise update per matrix element (kk, n):
      new[kk, n] = g * state[kk, n] + delta[kk, n]
    state_ptr: [K, N] float32 flattened
    delta_ptr: [K, N] float32 flattened
    g_ptr: scalar float32
    new_ptr: [K, N] float32 flattened
    """
    kk = tl.program_id(0)
    n = tl.program_id(1)
    if (kk >= K) or (n >= N):
        return
    g = tl.load(g_ptr)  # scalar
    state_val = tl.load(state_ptr + kk * N + n)
    delta_val = tl.load(delta_ptr + kk * N + n)
    new_val = g * state_val + delta_val
    tl.store(new_ptr + kk * N + n, new_val)


@triton.jit
def matmul_row_single(A_ptr, B_ptr, out_ptr, N: tl.int32):
    """
    Compute out[:] = A[:] @ B[,:] where:
      A_ptr: [N] float32 (row vector q_exp[t, h])
      B_ptr: [N, N] float32 (new_state[h] as [N, N] flattened row-major)
      out_ptr: [N] float32 (result vector)
    """
    pid = tl.program_id(0)  # one program per output column
    col = pid
    if col >= N:
        return
    sum_val = 0.0
    for j in range(0, N):
        # B row-major: index = j*N + col
        B_j_col = tl.load(B_ptr + j * N + col)
        A_j = tl.load(A_ptr + j)
        sum_val += A_j * B_j_col
    tl.store(out_ptr + col, sum_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-only implementation of the original run function.
        - Compute g and beta using Triton kernels.
        - Update state per sequence and per time step using Triton matmuls.
        - Produce output per t using Triton matmul.
        Returns:
          output: [T, Hv, N] bfloat16
          new_state: [num_seqs, Hv, N, N] float32
        """
        device = q.device

        # Shapes
        T, Hq, K = q.shape
        Hk, Hv = k.shape[1], v.shape[1]
        N = K  # head_size 128 (as in original)

        # Expand q/k to v heads
        q_exp = q.repeat_interleave(Hv // Hq, dim=1).contiguous()  # [T, Hv, K]
        k_exp = k.repeat_interleave(Hv // Hk, dim=1).contiguous()  # [T, Hv, K]
        v_exp = v.contiguous()                                     # [T, Hv, K]

        # Prepare a_exp and b_exp
        a_exp = a.repeat_interleave(Hv // Hq, dim=1).contiguous() # [T, Hv]
        b_exp = b.repeat_interleave(Hv // Hk, dim=1).contiguous() # [T, Hv]

        # Dtypes for Triton
        a_exp_bf16 = a_exp.to(torch.bfloat16)       # [T, Hv] bfloat16
        dt_bias_f32 = dt_bias.to(torch.float32)     # [Hv] float32
        A_log_f32 = A_log.to(torch.float32)         # [Hv] float32
        b_exp_f32 = b_exp.to(torch.float32)         # [T, Hv] float32

        # Allocate g and beta (float32)
        g = torch.empty((T, Hv), dtype=torch.float32, device=device)
        beta = torch.empty((T, Hv), dtype=torch.float32, device=device)

        # Launch Triton compute_g_beta_kernel
        grid_g = (T * Hv,)
        compute_g_beta_kernel[grid_g](a_exp_bf16.view(-1), dt_bias_f32, A_log_f32, g, beta, T=T, H=Hv)

        # Recompute beta via sigmoid in Triton (elementwise) if needed; g is already computed in-kernel above with softplus
        # Note: compute_g_beta_kernel re-stores beta_ptr; so beta should be correct.
        # Ensure beta is sigmoid(b), if not already:
        b_flat = b_exp_f32.view(-1)
        beta_flat = beta.view(-1)
        sigmoid_triton[b_flat.numel()](b_flat, beta_flat, N=b_flat.numel())
        beta = beta_flat.view(T, Hv)

        # Prepare output and new_state
        output = torch.empty((T, Hv, N), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((cu_seqlens.numel() - 1, Hv, N, N), dtype=torch.float32, device=device)

        # Process each sequence range
        for seq_idx in range(cu_seqlens.numel() - 1):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Slice expanded q/k/v for this sequence
            q_exp_s = q_exp[seq_start:seq_end]   #


def run(*args):
    return ModelNew()(*args)
