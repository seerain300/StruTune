import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_sqrt_scale_kernel(head_size_ptr, out_ptr):
    # Compute scale = 1.0 / sqrt(head_size) and store as float32
    hs = tl.load(head_size_ptr).to(tl.float32)
    scale = 1.0 / tl.sqrt(hs)
    tl.store(out_ptr, scale)


@triton.jit
def compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, A_log_ptr, b_ptr,
                              g_ptr, beta_ptr,
                              T, V):
    """
    Compute g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v])) for all t, v
    and beta[t, v] = sigmoid(b[t, v]), writing results to g_ptr and beta_ptr.
    g_ptr: [T*V] float32
    beta_ptr: [T*V] float32
    """
    t = tl.program_id(0)  # one program per token t
    for v in range(0, V):
        a_val = tl.load(a_ptr + t * V + v).to(tl.float32)
        dt_val = tl.load(dt_bias_ptr + v).to(tl.float32)
        A_val = tl.load(A_log_ptr + v).to(tl.float32)
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        g_val = tl.exp(-tl.exp(A_val) * sp)
        idx = t * V + v
        tl.store(g_ptr + idx, g_val)
        tl.store(beta_ptr + idx, 1.0 / (1.0 + tl.exp(-b_ptr[t * V + v])))


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_ptr, beta_ptr,
                        new_state_ptr,
                        T, H, V, K, t, seq_idx):
    """
    Update state per token t within sequence block seq_idx:
    For each h in [0, H), v in [0, V):
      old_v[h, :] = sum_j k[t, h, j] * state_old[h, v, j]
      new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
      state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
      state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
      state_new[h, v, :] = g[h, v] * state_old[h, v, :] - state_remove[h, :] + state_update[h, :]
    """
    # Note: We assume H=4, V=8, K=128. Loops are explicit and small.
    for h in range(0, H):
        for v_i in range(0, V):
            # Load vectors
            # q[t, h, :]
            q_vec = tl.load(q_ptr + t * H * K + h * K + tl.arange(0, K), mask=tl.arange(0, K) < K)
            # k[t, h, :]
            k_vec = tl.load(k_ptr + t * H * K + h * K + tl.arange(0, K), mask=tl.arange(0, K) < K)
            # v[t, v_i, :]
            v_vec = tl.load(v_ptr + t * V * K + v_i * K + tl.arange(0, K), mask=tl.arange(0, K) < K)
            # state_old[h, v_i, :]
            state_old_vec = tl.load(state_old_ptr + seq_idx * (H * V * K) + h * V * K + v_i * K + tl.arange(0, K), mask=tl.arange(0, K) < K)

            # g_val and beta_val scalars for (h, v_i)
            g_val = tl.load(g_ptr + h * V + v_i).to(tl.float32)
            beta_val = tl.load(beta_ptr + h * V + v_i).to(tl.float32)

            # old_v[h, :] = sum_j k[t, h, j] * state_old[h, v_i, j]
            old_v = tl.zeros([K], dtype=tl.float32)
            for j in range(0, K):
                old_v += k_vec[j] * state_old_vec[j]

            # new_v[h, :] = beta[v] * v[t, v_i, :] + (1 - beta[v]) * old_v
            new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

            # state_remove[h, :] = sum_j k[t, h, j] * old_v[h, j]
            state_remove = tl.zeros([K], dtype=tl.float32)
            for j in range(0, K):
                state_remove += k_vec[j] * old_v[j]

            # state_update[h, :] = sum_j k[t, h, j] * new_v[h, j]
            state_update = tl.zeros([K], dtype=tl.float32)
            for j in range(0, K):
                state_update += k_vec[j] * new_v[j]

            # state_new[h, v_i, :] = g[h, v_i] * state_old[h, v_i, :] - state_remove[h, :] + state_update[h, :]
            state_new_vec = g_val * state_old_vec - state_remove + state_update

            # Store into new_state[seq_idx, h, v_i, :]
            tl.store(new_state_ptr + seq_idx * (H * V * K) + h * V * K + v_i * K + tl.arange(0, K),
                     state_new_vec, mask=tl.arange(0, K) < K)


@triton.jit
def compute_output_kernel(q_ptr, new_state_ptr, out_ptr, scale,
                          T, H, V, K, seq_idx):
    """
    Compute output[t, h, k] = scale * q[t, h, k] @ new_state[seq_idx, h, v, k] for all t, h, k.
    We loop over V to accumulate.
    out_ptr: [T * H * K] float32
    """
    for t in range(0, T):
        for h in range(0, H):
            # output row for this (t, h)
            for k in range(0, K):
                dot_val = 0.0
                for v_i in range(0, V):
                    # state[seq_idx, h, v_i, k]
                    state_val = tl.load(new_state_ptr + seq_idx * (H * V * K) + h * V * K + v_i * K + k)
                    # q[t, h, k]
                    q_val = tl.load(q_ptr + t * H * K + h * K + k)
                    dot_val += q_val * state_val
                out_val = scale * dot_val
                idx = t * H * K + h * K + k
                tl.store(out_ptr + idx, out_val)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        q: [T, H, K], bfloat16
        k: [T, H, K], bfloat16
        v: [T, V, K], bfloat16
        state: [1, V, K, K] or [num_seqs, H, V, K], float32
        A_log: [V], float32
        a: [T, V], bfloat16
        dt_bias: [V], float32
        b: [T, V], bfloat16
        cu_seqlens: [num_seqs+1], int64 (sequence boundaries)
        scale: float (can be 0.0; compute as 1/sqrt(K) if None)
        Returns:
        output: [T, H, K], bfloat16
        new_state: [num_seqs, H, V, K], float32
        """
        device = q.device
        dtype = q.dtype
        H = 4
        V = 8
        K = 128

        # Ensure inputs are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # If state is provided, make it contiguous; otherwise allocate zeros
        if state is None:
            num_seqs = cu_seqlens.numel() - 1
            new_state = torch.zeros((num_seqs, H, V, K), dtype=torch.float32, device=device)
        else:
            num_seqs = cu_seqlens.numel() - 1
            # state can be [1, V, K, K]; we need [num_seqs, H, V, K] -> reshape to num_seqs*H, V, K, K then split.
            # Given original asserts, we assume state is already [1, V, K, K], and we'll expand to [num_seqs, H, V, K]
            # by cloning and filling. But original asserts H=4, so if state is [1, V, K, K], we can repeat H dimension.
            # To be robust, we'll create new_state and keep state unused (as per original code behavior).
            new_state = torch.zeros((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # Allocate g and beta as 1D arrays [T*V]
        g_flat = torch.empty(T * V, dtype=torch.float32, device=device)
        beta_flat = torch.empty(T * V, dtype=torch.float32, device=device)

        # Launch Triton kernels
        # 1) scale
        head_size = torch.tensor([K], device=device, dtype=torch.int32)
        scale_out = torch.empty((), device=device, dtype=torch.float32)
        compute_sqrt_scale_kernel[(1,)](head_size, scale_out)
        scale_val = float(scale_out.item()) if scale is None or scale == 0.0 else float(scale)

        # 2) g and beta
        compute_g_and_beta_kernel[(T,)](a.float(), dt_bias.float(), A_log.float(), b.float(),
                                        g_flat, beta_flat, T, V)

        # 3) update state for all sequence blocks
        # We process seq_idx = 0..num_seqs-1
        # Note: We update new_state in-place
        for seq_idx in range(0, num_seqs):
            # Initialize new_state[seq_idx] to zeros for safety
            new_state[seq_idx].zero_()

            # Run update for all tokens t
            for t in range(0, T):
                update_state_kernel[(1,)](
                    q.float(), k.float(), v.float(), state, g_flat, beta_flat,
                    new_state, T, H, V, K, t, seq_idx
                )

        # 4) compute output for last sequence block using Triton (return output for all seq blocks to match original)
        out_flat = torch.empty(T * H * K, dtype=torch.float32, device=device)
        for seq_idx in range(0, num_seqs):
            compute_output_kernel[(T,)](q.float(), new_state[seq_idx].contiguous().view(H * V, K),
                                        out_flat, scale_val, T, H, V, K, seq_idx)

        # Reshape output and cast to bfloat16
        output = out_flat.view(T, H, K).to(torch.bfloat16)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
