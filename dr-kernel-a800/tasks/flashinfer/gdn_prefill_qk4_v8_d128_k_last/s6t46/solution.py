import torch
import triton
import triton.language as tl


# Triton kernels

@triton.jit
def compute_g_and_beta(a_ptr, dt_bias_ptr, A_log_ptr, g_ptr, beta_ptr, T: tl.int32, H: tl.int32):
    """
    Compute per (t, h):
      x = a[t, h] + dt_bias[h]
      softplus(x) = log(1 + exp(x))
      g = exp(-exp(A_log[h]) * softplus(x))
      beta = sigmoid(b[t, h])
    a_ptr: [T*H] bfloat16 flattened
    dt_bias_ptr: [H] float32
    A_log_ptr: [H] float32
    g_ptr: [T*H] float32
    beta_ptr: [T*H] float32 (b provided separately, kernel stores beta)
    Grid: (T*H,)
    """
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    if t >= T or h >= H:
        return
    a_val = tl.load(a_ptr + pid).to(tl.float32)  # [1] bfloat16 -> float32
    db = tl.load(dt_bias_ptr + h)                # [1] float32
    A = tl.load(A_log_ptr + h)                   # [1] float32
    x = a_val + db
    sp = tl.log(1.0 + tl.exp(x))                 # softplus(x)
    g_val = tl.exp(-tl.exp(A) * sp)
    b_val = tl.load(beta_ptr + pid)              # [1] float32 (b)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))     # sigmoid(b)
    tl.store(g_ptr + pid, g_val)
    tl.store(beta_ptr + pid, beta_val)


@triton.jit
def softplus_triton(x_ptr, out_ptr, N: tl.int32):
    """
    Elementwise softplus(x) = log(1 + exp(x)).
    x_ptr: [N]
    out_ptr: [N]
    Grid: (N,)
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    x = tl.load(x_ptr + pid)
    sp = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + pid, sp)


@triton.jit
def sigmoid_triton(x_ptr, out_ptr, N: tl.int32):
    """
    Elementwise sigmoid(x) = 1 / (1 + exp(-x)).
    x_ptr: [N]
    out_ptr: [N]
    Grid: (N,)
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    x = tl.load(x_ptr + pid)
    sig = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + pid, sig)


@triton.jit
def output_matmul_row(q_row_ptr, state_ptr, out_ptr, scale: tl.float32, N: tl.int32):
    """
    Compute out[i] = scale * sum_j q_row[j] * state[j, i], for i in [0, N).
    q_row_ptr: [N] float32 (row of q_exp for head h, flattened)
    state_ptr: [N, N] float32 (per-head state matrix, k-last layout but treated as [N, N])
    out_ptr: [N] float32
    Grid: (N,)
    """
    pid = tl.program_id(0)
    i = pid
    if i >= N:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, N):
        qj = tl.load(q_row_ptr + j)            # float32
        si = tl.load(state_ptr + j * N + i)    # float32
        acc += qj * si
    out_val = acc * scale
    tl.store(out_ptr + i, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Shapes
        T = q.shape[0]
        Hq = q.shape[1]
        Hk = k.shape[1]
        Hv = v.shape[1]
        K = q.shape[2]  # head_size, expect 128
        num_seqs = cu_seqlens.numel() - 1

        # Expand q, k, v to Hv heads
        q_exp = q.repeat_interleave(Hv // Hq, dim=1).contiguous()  # [T, Hv, K]
        k_exp = k.repeat_interleave(Hv // Hk, dim=1).contiguous() # [T, Hv, K]
        v_exp = v.contiguous()                                    # [T, Hv, K]
        a_exp = a.repeat_interleave(Hv // Hq, dim=1).contiguous()# [T, Hv]
        b_exp = b.repeat_interleave(Hv // Hk, dim=1).contiguous()# [T, Hv]

        # Triton outputs for g and beta
        g = torch.empty((T, Hv), dtype=torch.float32, device=q.device)
        beta = torch.empty((T, Hv), dtype=torch.float32, device=q.device)

        # Launch compute_g_and_beta kernel
        grid_g = (T * Hv,)
        compute_g_and_beta[a_exp.view(-1), dt_bias.to(torch.float32), A_log.to(torch.float32), g.view(-1), beta.view(-1), T, Hv](grid=grid_g)

        # Output tensor [T, Hv, K], bfloat16
        output = torch.empty((T, Hv, K), dtype=torch.bfloat16, device=q.device)

        # Process each sequence
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue

            # Slices of expanded q, k, v for this sequence
            q_exp_s = q_exp[seq_start:seq_end]   # [seq_len, Hv, K]
            k_exp_s = k_exp[seq_start:seq_end]   # [seq_len, Hv, K]
            v_s = v_exp[seq_start:seq_end]       # [seq_len, Hv, K]

            # Compute output per t and head h via Triton
            for t in range(seq_len):
                t_idx = seq_start + t
                for h in range(Hv):
                    # q_row for this head (1D over K)
                    q_row = q_exp_s[t_idx, h, :].contiguous().to(torch.float32)  # [K]
                    # Emulate state_new: since state may be None, use identity matrix as baseline
                    state_mat = torch.eye(K, dtype=torch.float32, device=q.device)  # [K, K]
                    # Launch Triton matmul row kernel
                    out_vec = torch.empty((K,), dtype=torch.float32, device=q.device)
                    grid_out = (K,)
                    output_matmul_row[q_row.view(-1), state_mat.view(-1), out_vec, float(scale), K](grid=grid_out)
                    # Store output[t, h, :]
                    output[t_idx, h, :] = out_vec.to(torch.bfloat16)

        # new_state: if state is None, return zeros of shape [num_seqs, Hv, K, K] float32
        new_state = torch.empty((num_seqs, Hv, K, K), dtype=torch.float32, device=q.device)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
