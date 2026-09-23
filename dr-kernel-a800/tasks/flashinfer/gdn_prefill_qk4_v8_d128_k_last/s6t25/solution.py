import torch
import math
import triton
import triton.language as tl


# Triton kernels (proper syntax, no extra commas or '@' in kernel definitions)

@triton.jit
def compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, T: tl.int32, H: tl.int32):
    """
    Compute g per (t, h):
      g = exp(-exp(A_log[h]) * softplus(a[t, h] + dt_bias[h]))
    a_ptr: [T*H], bfloat16
    dt_bias_ptr: [H], float32
    A_log_ptr: [H], float32
    g_ptr: [T*H], float32
    """
    pid = tl.program_id(0)
    if pid >= T * H:
        return
    t = pid // H
    h = pid % H
    if t >= T:
        return
    a_val = tl.load(a_ptr + pid)
    db_val = tl.load(dt_bias_ptr + h)
    A_val = tl.load(A_log_ptr + h)
    x = a_val.to(tl.float32) + db_val
    sp = tl.log(1.0 + tl.exp(x))  # softplus
    g_val = tl.exp(-tl.exp(A_val) * sp)
    tl.store(g_ptr + pid, g_val)


@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, N: tl.int32):
    """
    Elementwise sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    x = tl.load(x_ptr + pid)
    sig = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + pid, sig)


@triton.jit
def matmul_row_kernel(A_row_ptr, B_ptr, out_ptr, K: tl.int32, N: tl.int32):
    """
    Compute output = dot(A_row[K], B[N, N] flattened) and store to out[0].
    A_row_ptr: [K]
    B_ptr: [N*N] (row-major)
    out_ptr: [1]
    """
    acc = 0.0
    k = 0
    while k < K:
        a = tl.load(A_row_ptr + k)
        n = 0
        while n < N:
            b = tl.load(B_ptr + n * N + k)
            acc += a * b
            n += 1
        k += 1
    tl.store(out_ptr, acc)


@triton.jit
def mm_k_state_kernel(k_vec_ptr, state_ptr, out_ptr, N: tl.int32, K: tl.int32):
    """
    Compute old_v = k_vec[K] @ state[K, N] -> out[N].
    state_ptr: [K, N] flattened row-major
    out_ptr: [N]
    """
    pid = tl.program_id(0)  # one program per n
    n = pid
    if n >= N:
        return
    acc = 0.0
    k = 0
    while k < K:
        kval = tl.load(k_vec_ptr + k)
        val = tl.load(state_ptr + k * N + n)
        acc += kval * val
        k += 1
    tl.store(out_ptr + n, acc)


@triton.jit
def mm_kT_vec_kernel(k_vec_ptr, vec_ptr, out_ptr, N: tl.int32, K: tl.int32):
    """
    Compute rr = sum_k k_vec[k] * vec[k].
    vec_ptr: [K]
    out_ptr: [1]
    """
    acc = 0.0
    k = 0
    while k < K:
        kval = tl.load(k_vec_ptr + k)
        vval = tl.load(vec_ptr + k)
        acc += kval * vval
        k += 1
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale
    ):
        """
        q: [T, Hq, K] bfloat16
        k: [T, Hk, K] bfloat16
        v: [T, Hv, K] bfloat16
        state: optional [1, Hv, N, N] float32 (k-last); we transpose to [Hv, N, N] per seq
        A_log: [Hv] float32
        a: [T, Hq] bfloat16
        dt_bias: [Hv] float32
        b: [T, Hv] bfloat16
        cu_seqlens: [L] int64, L=num_seqs+1
        scale: float
        Returns:
        output: [T, Hv, N] bfloat16
        new_state: [num_seqs, Hv, N, N] float32
        """
        device = q.device
        T, Hq, K = q.shape
        Hk = k.shape[1]
        Hv = v.shape[1]
        N = K  # head_size, per original

        # Assertions consistent with original code
        assert Hq == 4
        assert Hk == 4
        assert Hv == 8
        assert N == 128

        # Expand q/k to Hv heads
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
        b_exp_f32 = b_exp.to(torch.float32)     # [T, Hv] float32

        # Allocate g and beta (float32)
        g = torch.empty((T, Hv), dtype=torch.float32, device=device)
        beta = torch.empty((T, Hv), dtype=torch.float32, device=device)

        # Launch Triton compute_g_kernel (elementwise g)
        N_pairs = T * Hv
        compute_g_kernel[(N_pairs,)](a_exp_bf16.view(-1), dt_bias_f32, A_log_f32, g, T=T, H=Hv)

        # Launch sigmoid kernel for beta
        beta_flat = beta.view(-1)
        b_flat = b_exp_f32.view(-1)
        sigmoid_kernel[(b_flat.numel(),)](b_flat, beta_flat, N=b_flat.numel())
        beta = beta_flat.view(T, Hv)

        # Prepare output and new_state
        output = torch.empty((T, Hv, N), dtype=torch.bfloat16, device=device)
        num_seqs = cu_seqlens.numel() - 1
        new_state = torch.zeros((num_seqs, Hv, N, N), dtype=torch.float32, device=device)

        # Process each sequence
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Slice expanded q/k/v for this sequence


def run(*args):
    return ModelNew()(*args)
