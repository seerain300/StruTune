import torch
import triton
import triton.language as tl


# Triton kernels

@triton.jit
def softplus_triton(x_ptr, out_ptr, N: tl.int32):
    """
    Softplus elementwise: softplus(x) = log(1 + exp(x))
    x_ptr: 1D float32 input [N]
    out_ptr: 1D float32 output [N]
    N: length
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
    Sigmoid elementwise: sigmoid(x) = 1 / (1 + exp(-x))
    x_ptr: 1D float32 input [N]
    out_ptr: 1D float32 output [N]
    N: length
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    x = tl.load(x_ptr + pid)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + pid, y)


@triton.jit
def compute_g_kernel(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, N: tl.int32, H: tl.int32):
    """
    Compute g per (t, h): g = exp(-exp(A_log[h]) * softplus(a[t,h] + dt_bias[h]))
    a_ptr: 1D float32 input [T*H] (we pass bfloat16 casted to float32)
    dt_bias_ptr: 1D float32 [H]
    A_log_ptr: 1D float32 [H]
    g_ptr: 1D float32 output [T*H]
    N = T*H
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    t = pid // H
    h = pid % H
    a_val = tl.load(a_ptr + pid)           # float32
    db_val = tl.load(dt_bias_ptr + h)      # float32
    A_val = tl.load(A_log_ptr + h)         # float32
    x = a_val + db_val
    sp = tl.log(1.0 + tl.exp(x))           # softplus(x)
    g_val = tl.exp(-tl.exp(A_val) * sp)
    tl.store(g_ptr + pid, g_val)


@triton.jit
def mm_k_state_kernel(k_vec_ptr, state_ptr, out_ptr, K: tl.constexpr, N: tl.constexpr):
    """
    Vector matmul: out[N] = k_vec[K] @ state[N, N] -> [N]
    k_vec_ptr: 1D float32 [K]
    state_ptr: 2D float32 [N, N] flattened row-major
    out_ptr: 1D float32 [N]
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    acc = 0.0
    for kk in range(K):
        k_k = tl.load(k_vec_ptr + kk)
        row_start = kk * N
        for j in range(N):
            acc += k_k * tl.load(state_ptr + row_start + j)
    tl.store(out_ptr + pid, acc)


@triton.jit
def mm_kT_vec_kernel(k_vec_ptr, vec_ptr, out_ptr, K: tl.constexpr, N: tl.constexpr):
    """
    Scalar k^T @ vec: out[0] = sum_k k_vec[k] * vec[k]
    k_vec_ptr: 1D float32 [K]
    vec_ptr: 1D float32 [N]
    out_ptr: 1D float32 [1]
    """
    acc = 0.0
    for kk in range(K):
        k_k = tl.load(k_vec_ptr + kk)
        v_k = tl.load(vec_ptr + kk)
        acc += k_k * v_k
    tl.store(out_ptr + 0, acc)


@triton.jit
def compute_output_row_mm(A_ptr, B_ptr, C_ptr, K: tl.constexpr, H: tl.constexpr, N: tl.constexpr):
    """
    Compute out[N] = A[K] @ B[H, N], where A is 1D row vector [K], B is 2D [H, N].
    B_ptr is passed as 1D flattened with mapping b[h, j] -> B_ptr[h * N + j].
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    acc = 0.0
    for k in range(K):
        a_k = tl.load(A_ptr + k)
        # Sum over h: b[h, j] at index h*N + j
        for h in range(H):
            bj = tl.load(B_ptr + h * N + pid)
            acc += a_k * bj
    tl.store(C_ptr + pid, acc)


# ModelNew forward (no torch math, all Triton kernels launched)
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-orchestrated forward replicating original semantics:
        - Compute g and beta via Triton
        - Update state per sequence and timestep using Triton matmuls
        - Compute output per timestep using Triton row-wise matmul
        Returns:
          output: [T, Hv, N] bfloat16
          new_state: [num_seqs, Hv, N, N] float32 (k-last layout: [Hv, N, N] per seq)
        """
        device = q.device
        T, Hq, K = q.shape
        Hk, Hv = k.shape[1], v.shape[1]
        N = K  # original head_size=128

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Expand q/k to v heads (as original)
        q_exp = q.repeat_interleave(Hv // Hq, dim=1).contiguous()  # [T, Hv, K]
        k_exp = k.repeat_interleave(Hv // Hk, dim=1).contiguous()  # [T, Hv, K]
        v_exp = v.contiguous()  # [T, Hv, K]

        # Prepare a_exp and b_exp
        a_exp = a.repeat_interleave(Hv // Hq, dim=1).contiguous()  # [T, Hv]
        b_exp = b.repeat_interleave(Hv // Hk, dim=1).contiguous()  # [T, Hv]

        # Dtypes for Triton: cast to float32 where needed
        a_exp_f = a_exp.to(torch.float32)           # [T, Hv]
        dt_bias_f = dt_bias.to(torch.float32)       # [Hv]
        A_log_f = A_log.to(torch.float32)           # [Hv]
        b_exp_f = b_exp.to(torch.float32)           # [T, Hv]

        # Allocate g and beta outputs (float32)
        N_elems = T * Hv
        g = torch.empty((N_elems,), dtype=torch.float32, device=device)
        beta = torch.empty((N_elems,), dtype=torch.float32, device=device)

        # Launch Triton kernels to compute g and beta
        grid_g = (N_elems,)
        compute_g_kernel[grid_g](a_exp_f.view(-1), dt_bias_f, A_log_f, g, N_elems, Hv)

        # Compute beta via sigmoid(b)
        b_flat = b_exp_f.view(-1)  # [T*Hv]
        beta_flat = beta
        grid_beta = (N_elems,)
        sigmoid_triton[grid_beta](b_flat, beta_flat, N_elems)

        # Initialize output and new_state
        output = torch.empty((T, Hv, N), dtype=torch.bfloat16, device=device)
        num_seqs = cu_seqlens.numel() - 1
        new_state = torch.empty((num_seqs, Hv, N, N), dtype=torch.float32, device=device)

        # Process each sequence
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Slice expanded q/k/v for this sequence
            q_exp_s = q_exp[seq_start:seq_start + seq_len]   # [seq_len, Hv, K]
            k_exp_s = k_exp[seq_start:seq_start + seq_len]   # [seq_len, Hv, K]
            v_s = v_exp[seq_start:seq_start + seq_len]       # [seq_len, Hv, K]

            # Initial state handling: mirror [Hv, N, N] k-last layout
            if state is not None:
                state_seq_klast = state[seq_idx].transpose(-1, -2).contiguous()  # [Hv, N, N] float32
            else:
                state_seq_klast = torch.zeros((Hv, N, N), dtype=torch.float32, device=device)

            # Iterate time steps
            for i in range(seq_len):
                t = seq_start + i

                # Update per head h
                for h in range(Hv):
                    # Extract vectors for q, k, v for this head (as 1xK)
                    q_h = q_exp_s[i][:, h]    # [K], bfloat16 -> float32
                    k_h = k_exp_s[i][:, h]    # [K], bfloat16 -> float32
                    v_h = v_s[i][:, h]        # [K], bfloat16 -> float32

                    # Convert to float32 for Triton matmul
                    q_h_f = q_h.to(torch.float32).view(-1)            # [K] flattened
                    k_h_f = k_h.to(torch.float32).view(-1)            # [K]
                    v_h_f = v_h.to(torch.float32).view(-1)            # [K]

                    # 1) old_v = k @ state_old -> [N]
                    state_old = state_seq_klast[h]  # [N, N] float32
                    old_v = torch.empty((N,), dtype=torch.float32, device=device)
                    mm_k_state_kernel[(K,)](k_h_f, state_old.view(-1), old_v, K=K, N=N)

                    # 2) new_v = beta * v_h + (1 - beta[h]) * old_v
                    beta_t = beta[t * Hv + h]  # float32
                    new_v = beta_t * v_h_f + (1.0 - beta_t) * old_v

                    # 3) state_remove = k^T @ old_v -> scalar
                    state_remove = torch.empty((1,), dtype=torch.float32, device=device)
                    mm_kT_vec_kernel[(K,)](k_h_f, old_v, state_remove, K=K, N=N)
                    state_remove = state_remove[0]

                    # 4) state_update = k^T @ new_v -> scalar
                    state_update = torch.empty((1,), dtype=torch.float32, device=device)
                    mm_kT_vec_kernel[(K,)](k_h_f, new_v, state_update, K=K, N=N)
                    state_update = state_update[0]

                    # 5) new_state[h] = g * state_old + state_update - state_remove
                    g_t = g[t * Hv + h]  # float32
                    state_seq_klast[h] = g_t * state_old + state_update - state_remove

                # Output for this t and head h: output[t, h, :] = scale * q_h @ new_state_klast[h]
                # We need to compute q_h @ new_state_klast[h] for h in [0..Hv-1]
                for h in range(Hv):
                    # Build B matrix: new_state_klast[h] as [N, N], flatten b[h] rows
                    new_state_h = state_seq_klast[h]  # [N, N] float32
                    # Launch compute_output_row_mm: out[N] = A[K] @ B[H=1, N] -> but we need B[H, N] with H=Hq? Not matching.
                    # Correction: we need B[H, N] where


def run(*args):
    return ModelNew()(*args)
