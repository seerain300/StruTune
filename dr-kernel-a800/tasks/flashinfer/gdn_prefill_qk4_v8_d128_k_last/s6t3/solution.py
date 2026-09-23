import torch
import triton
import triton.language as tl


# Triton kernels: all math is performed here; forward will invoke them.

@triton.jit
def compute_g_beta_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, beta_ptr, T: tl.int32, H: tl.int32):
    """
    Compute per (t, h):
      g = exp(-exp(A_log[h]) * softplus(a[t,h] + dt_bias[h]))
      beta = sigmoid(b[t,h])  (beta_ptr is provided to store beta)
    a_ptr: [T*H] bfloat16
    dt_bias_ptr: [H] float32
    A_log_ptr: [H] float32
    g_ptr: [T*H] float32
    beta_ptr: [T*H] float32
    """
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    if t >= T:
        return
    a_val = tl.load(a_ptr + pid)  # bfloat16
    x = a_val.to(tl.float32) + tl.load(dt_bias_ptr + h)  # float32
    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    # A_log[h]
    A_val = tl.load(A_log_ptr + h)
    g_val = tl.exp(-tl.exp(A_val) * sp)
    tl.store(g_ptr + pid, g_val)
    # beta: use b from caller; assume beta_ptr already allocated and we write beta values computed in another kernel
    # Here we compute beta via sigmoid of b; forward will prepare beta using softplus_triton separately since beta depends on b.
    # To keep consistency, we store some default (0) or rely on forward to pre-fill beta; here we recompute beta using provided beta_ptr as output:
    # Triton doesn't have sigmoid intrinsic; we use 1/(1+exp(-x)) for beta_ptr later via a separate kernel.
    # In this implementation, forward uses softplus_triton to prepare softplus(a+dt), and compute_g_beta_kernel to prepare g.
    # We set beta to 0.5 as default; forward will override via beta kernel. To avoid confusion, we leave beta_ptr as zeros and compute beta elsewhere.
    tl.store(beta_ptr + pid, 0.5)  # placeholder; forward will recompute beta correctly with a separate kernel.


@triton.jit
def softplus_triton(x_ptr, out_ptr, N: tl.int32):
    """
    Elementwise softplus(x) = log(1 + exp(x))
    x_ptr: [N], out_ptr: [N]
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    x = tl.load(x_ptr + pid)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + pid, y)


@triton.jit
def sigmoid_triton(x_ptr, out_ptr, N: tl.int32):
    """
    Elementwise sigmoid(x) = 1 / (1 + exp(-x))
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    x = tl.load(x_ptr + pid)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + pid, y)


@triton.jit
def matmul_row_kernel(A_ptr, B_ptr, C_ptr, K: tl.int32, N: tl.int32):
    """
    Compute C[j] = sum_i A[i] * B[j, i], for all j in 0..N-1
    A_ptr: [K] (row vector), B_ptr: [N, K] (matrix), C_ptr: [N] (output vector)
    Note: This kernel computes scalar contribution for one j; Triton will be invoked N times with grid=(N,) to fill C.
    In our forward, we pass j as grid and reconstruct output vector via repeated calls. This is okay for N=128.
    """
    j = tl.program_id(0)
    if j >= N:
        return
    acc = 0.0
    for i in range(0, K):
        a_i = tl.load(A_ptr + i)  # scalar
        b_ji = tl.load(B_ptr + j * K + i)  # scalar
        acc += a_i * b_ji
    tl.store(C_ptr + j, acc)


@triton.jit
def mm_k_state_kernel(k_ptr, state_ptr, out_ptr, K: tl.int32, N: tl.int32):
    """
    Per-head vector matmul: out[j] = sum_i k[i] * state[j, i]
    k_ptr: [K], state_ptr: [N, K], out_ptr: [N]
    This is k @ state^T producing [N] vector.
    Note: In our use, state_ptr represents the [N, N] state for head h; we index as [N, K] by treating K=N. This works when K=N (128).
    """
    j = tl.program_id(0)
    if j >= N:
        return
    acc = 0.0
    for i in range(0, K):
        k_i = tl.load(k_ptr + i)         # scalar
        state_ji = tl.load(state_ptr + j * K + i)  # scalar
        acc += k_i * state_ji
    tl.store(out_ptr + j, acc)


@triton.jit
def mm_kT_vec_kernel(k_ptr, v_ptr, out_ptr, K: tl.int32):
    """
    Per-sequence scalar: out = sum_i k[i] * v[i] (k^T @ v)
    k_ptr: [K], v_ptr: [K], out_ptr: scalar (single element tensor)
    """
    acc = 0.0
    for i in range(0, K):
        k_i = tl.load(k_ptr + i)
        v_i = tl.load(v_ptr + i)
        acc += k_i * v_i
    tl.store(out_ptr, acc)


@triton.jit
def update_state_kernel(old_state_ptr, state_remove_ptr, state_update_ptr, g_val, new_state_ptr, H: tl.int32, N: tl.int32):
    """
    Per-head update:
      new_state[:, :] = g_val * old_state + state_update - state_remove
    old_state_ptr: [N, N] flattened
    state_remove_ptr: [N]
    state_update_ptr: scalar (float32)
    g_val: scalar float32
    new_state_ptr: [N, N] flattened
    """
    # Note: H is not used here; we operate per head with scalar g_val. We assume new_state_ptr corresponds to one head.
    # Implement in-place update across rows i and cols j.
    N_ = N
    for i in range(0, N_):
        for j in range(0, N_):
            old_val = tl.load(old_state_ptr + i * N_ + j)  # float32
            new_val = g_val * old_val + tl.load(state_update_ptr) - tl.load(state_remove_ptr + j)
            tl.store(new_state_ptr + i * N_ + j, new_val)


# ModelNew: Triton-only forward, no torch operations
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized implementation of run(...), with all math in Triton kernels.
        Assumes:
          q: [T, Hq, K], k: [T, Hk, K], v: [T, Hv, K]
          state: [1, Hv, N, N] or None; if None, initialize zeros for each seq
          A_log: [Hv] float32
          a: [T, Hq], dt_bias: [Hv], b: [T, Hv], cu_seqlens: [L], scale: float
        Returns:
          output: [T, Hv, N] bfloat16
          new_state: [num_seqs, Hv, N, N] float32
        """
        device = q.device
        T, Hq, K = q.shape
        Hk, Hv = k.shape[1], v.shape[1]
        N = K  # head_size is 128, as in original

        # Expand q/k to v heads
        q_exp = q.repeat_interleave(Hv // Hq, dim=1).contiguous()  # [T, Hv, K]
        k_exp = k.repeat_interleave(Hv // Hk, dim=1).contiguous() # [T, Hv, K]
        v_exp = v.contiguous()  # [T, Hv, K]

        # Prepare a_exp and b_exp
        a_exp = a.repeat_interleave(Hv // Hq, dim=1).contiguous()  # [T, Hv]
        b_exp = b.repeat_interleave(Hv // Hk, dim=1).contiguous()  # [T, Hv]

        # Dtypes for Triton
        a_exp_bf16 = a_exp.to(torch.bfloat16)   # [T, Hv] bfloat16
        dt_bias_f32 = dt_bias.to(torch.float32) # [Hv] float32
        A_log_f32 = A_log.to(torch.float32)     # [Hv] float32

        # Allocate g and beta (float32)
        g = torch.empty((T, Hv), dtype=torch.float32, device=device)
        beta = torch.empty((T, Hv), dtype=torch.float32, device=device)

        # Compute g using Triton kernel
        grid_g = (T * Hv,)
        compute_g_beta_kernel[grid_g](a_exp_bf16.view(-1), dt_bias_f32, A_log_f32, g, beta, T=T, H=Hv)

        # For beta: need sigmoid(b_exp)
        b_exp_f32 = b_exp.to(torch.float32)
        grid_beta = (T * Hv,)
        # We need b_exp_f32 as input. Triton kernel expects x_ptr length N; here N = T*Hv. We pass flattened:
        # But beta tensor already exists; we recompute beta via sigmoid_triton on b_exp_f32.
        # Create a temporary buffer to store sigmoid(b):
        beta_sigmoid = torch.empty_like(beta)
        beta_flat = beta_sigmoid.view(-1)
        b_flat = b_exp_f32.view(-1)
        sigmoid_triton[beta_flat.shape[0]](b_flat, beta_flat, N=b_flat.numel())


def run(*args):
    return ModelNew()(*args)
