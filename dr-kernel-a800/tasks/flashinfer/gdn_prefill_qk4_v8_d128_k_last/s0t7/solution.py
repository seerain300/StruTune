import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_g_beta_kernel(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    H, V, T
):
    """
    Compute g and beta:
    - g[i] = exp(-exp(A_log[i // V]) * softplus(a[t, i % V] + dt_bias[i % V])) for i in [0, H*V)
    - beta[i] = sigmoid(b[t, i]) for i in [0, V)
    Write g to g_ptr[H*V], beta to beta_ptr[V], both float32.
    """
    for i in range(0, H * V):
        v_i = i % V
        t_i = i // V
        # Load parameters
        A_log_val = tl.load(A_log_ptr + v_i).to(tl.float32)
        a_val = tl.load(a_ptr + t_i * V + v_i).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + v_i).to(tl.float32)
        b_val = tl.load(b_ptr + t_i * V + v_i).to(tl.float32)
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(a_val + dt_bias_val))
        g_val = tl.exp(-tl.exp(A_log_val) * sp)
        tl.store(g_ptr + i, g_val)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(beta_ptr + v_i, beta_val)


@triton.jit
def update_state_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, new_state_ptr, g_ptr, beta_ptr, scale_ptr,
    H, V, K,
    t, seq_idx
):
    """
    Update state for all (h, v) at token t and sequence block seq_idx.
    Compute:
      old_v[h, :] = k[t, h, :] @ state[seq_idx, h, v, :]
      new_v[h, :] = beta[v] * v[t, v, :] + (1 - beta[v]) * old_v[h, :]
      state_remove[h, :] = k[t, h, :] @ old_v[h, :]
      state_update[h, :] = k[t, h, :] @ new_v[h, :]
      state_new[h, v, :] = g[h, v] * state[seq_idx, h, v, :] - state_remove[h, :] + state_update[h, :]
    Store state_new into new_state[seq_idx, h, v, :].
    """
    for h in range(0, H):
        for v_i in range(0, V):
            # Load g[h, v_i] and beta[v_i]
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_val = tl.load(beta_ptr + v_i).to(tl.float32)
            scale_val = tl.load(scale_ptr).to(tl.float32)

            # Load k_vec[t, h, K]
            k_vec = tl.zeros([K], dtype=tl.float32)
            for j in range(0, K):
                k_off = t * (H * K) + h * K + j
                k_elem = tl.load(k_ptr + k_off).to(tl.float32)
                k_vec[j] = k_elem

            # Load state_old[h, v_i, K]
            state_old_vec = tl.zeros([K], dtype=tl.float32)
            for j in range(0, K):
                state_off = seq_idx * (H * V * K) + h * (V * K) + v_i * K + j
                state_elem = tl.load(state_ptr + state_off).to(tl.float32)
                state_old_vec[j] = state_elem

            # old_v = k_vec @ state_old_vec
            old_v = tl.zeros([K], dtype=tl.float32)
            for k_j in range(0, K):
                dot_val = 0.0
                for kk in range(0, K):
                    dot_val += k_vec[kk] * state_old_vec[kk]
                old_v[k_j] = dot_val

            # v_vec[t, v_i, K]
            v_vec = tl.zeros([K], dtype=tl.float32)
            for j in range(0, K):
                v_off = t * (V * K) + v_i * K + j
                v_elem = tl.load(v_ptr + v_off).to(tl.float32)
                v_vec[j] = v_elem

            # new_v = beta[v_i] * v_vec + (1 - beta[v_i]) * old_v
            new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

            # state_remove = k_vec @ old_v
            state_remove = tl.zeros([K], dtype=tl.float32)
            for k_j in range(0, K):
                dot_j = 0.0
                for kk in range(0, K):
                    dot_j += k_vec[kk] * old_v[k_j]
                state_remove[k_j] = dot_j

            # state_update = k_vec @ new_v
            state_update = tl.zeros([K], dtype=tl.float32)
            for k_j in range(0, K):
                dot_j = 0.0
                for kk in range(0, K):
                    dot_j += k_vec[kk] * new_v[k_j]
                state_update[k_j] = dot_j

            # state_new_vec = g * state_old - state_remove + state_update
            state_new_vec = g_val * state_old_vec - state_remove + state_update

            # Store into new_state[seq_idx, h, v_i, :]
            for j in range(0, K):
                new_state_off = seq_idx * (H * V * K) + h * (V * K) + v_i * K + j
                tl.store(new_state_ptr + new_state_off, state_new_vec[j])


@triton.jit
def output_kernel(
    q_ptr, new_state_ptr, out_ptr,
    H, V, K,
    t, scale_ptr
):
    """
    Compute output[t, h, :] = scale * q[t, h, :] @ new_state[t, h, v, :]
    new_state is [H, V, K] (interpreted as per-(h,v) vectors of length K).
    Output is [H, K], we'll store into out[t, h, k].
    """
    for h in range(0, H):
        q_vec = tl.zeros([K], dtype=tl.float32)
        for j in range(0, K):
            q_off = t * (H * K) + h * K + j
            q_elem = tl.load(q_ptr + q_off).to(tl.float32)
            q_vec[j] = q_elem

        scale_val = tl.load(scale_ptr).to(tl.float32)
        # For each v, compute q @ new_state[h, v, :]
        for v_i in range(0, V):
            state_vec = tl.zeros([K], dtype=tl.float32)
            for j in range(0, K):
                state_off = t * (H * V * K) + h * (V * K) + v_i * K + j
                state_elem = tl.load(new_state_ptr + state_off).to(tl.float32)
                state_vec[j] = state_elem
            out_vec = scale_val * q_vec @ state_vec  # per-K element
            for j in range(0, K):
                out_off = t * (H * K) + h * K + j
                tl.store(out_ptr + out_off, out_vec[j])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure all tensors are on same device and contiguous
        device = q.device
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        T = q.shape[0]
        H = q.shape[1]
        V = v.shape[1]
        K = q.shape[2]
        num_seqs = cu_seqlens.numel() - 1

        # Prepare g and beta (float32)
        g_flat = torch.empty(H * V, dtype=torch.float32, device=device)
        beta_flat = torch.empty(V, dtype=torch.float32, device=device)

        # Create scale tensor on device (float32) to avoid host-side elementwise ops
        # Original code sets scale = 1.0 / sqrt(head_size) if not provided.
        if scale is None:
            scale_tensor = torch.tensor(1.0 / (K ** 0.5), dtype=torch.float32, device=device)
        else:
            # Pass scale as float32 tensor
            scale_tensor = torch.tensor(float(scale), dtype=torch.float32, device=device)

        # Launch compute_g_beta_kernel
        compute_g_beta_kernel[(1,)](
            A_log, a, dt_bias, b,
            g_flat, beta_flat,
            H, V, T,
        )

        # Allocate new_state as float32 [num_seqs, H, V, K]
        new_state = torch.empty((num_seqs, H, V, K), dtype=torch.float32, device=device)

        # Initialize state (original signature: [num_seqs, V, K, K])
        if state is None:
            # We need to initialize state for update. Since original code uses state if provided,
            # and the benchmark often provides None, we initialize zeros as [num_seqs, V, K, K] to be safe.
            state = torch.zeros((num_seqs, V, K, K), dtype=torch.float32, device=device)
        state = state.contiguous()

        # Update state for each sequence block and each token
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue
            for t in range(seq_len):
                update_state_kernel[(1,)](
                    q, k, v, state, new_state, g_flat, beta_flat, scale_tensor,
                    H, V, K,
                    seq_start + t, seq_idx
                )

        # Compute output per token using Triton kernel: out[t, h, k] = scale * q[t, h, k] @ new_state[t, h, v, k]
        out = torch.empty((T, H, K), dtype=torch.float32, device=device)
        for t in range(T):
            output_kernel[(1,)](
                q, new_state, out,
                H, V, K,
                t, scale_tensor
            )

        # Return output and new_state; match original signature (output bfloat16, new_state float32)
        # Cast output to bfloat16 to mimic original behavior.
        out = out.to(torch.bfloat16)
        return out, new_state


def run(*args):
    return ModelNew()(*args)
