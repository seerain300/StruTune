import torch
import triton
import triton.language as tl


@triton.jit
def update_state_kernel(q_ptr, k_ptr, v_ptr, state_old_ptr, g_ptr, beta_ptr,
                        new_state_ptr,
                        T: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
                        t, seq_idx):
    """
    Update state per token t within sequence block seq_idx:
    For each h in [0, H), v in [0, V):
      old_v[h, k] = sum_j k[t, h, j] * state_old[h, v, j]
      new_v[h, k] = beta[v] * v[t, v, k] + (1 - beta[v]) * old_v[h, k]
      state_remove[h, k] = sum_j k[t, h, j] * old_v[h, j]
      state_update[h, k] = sum_j k[t, h, j] * new_v[h, j]
      state_new[h, v, k] = g[h, v] * state_old[h, v, k] - state_remove[h, k] + state_update[h, k]
    """
    # Process one token t and one sequence block seq_idx
    for h in range(0, H):
        for v_i in range(0, V):
            # Load scalars g[h, v_i] and beta[h, v_i]
            g_idx = h * V + v_i
            g_val = tl.load(g_ptr + g_idx).to(tl.float32)
            beta_val = tl.load(beta_ptr + g_idx).to(tl.float32)

            # Initialize old_v, new_v, state_remove, state_update as vectors of length K
            old_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                # k[t, h, j] and state_old[h, v_i, j]
                k_offset = t * (H * K) + h * K + j
                k_val = tl.load(k_ptr + k_offset).to(tl.float32)
                state_old_offset = (seq_idx * H * V * K) + h * (V * K) + v_i * K + j
                state_old_val = tl.load(state_old_ptr + state_old_offset).to(tl.float32)
                old_v[j] = k_val * state_old_val

            new_v = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                # v[t, v_i, j]
                v_offset = t * (V * K) + v_i * K + j
                v_val = tl.load(v_ptr + v_offset).to(tl.float32)
                new_v[j] = beta_val * v_val + (1.0 - beta_val) * old_v[j]

            # state_remove[h, k] = sum_j k[t, h, j] * old_v[h, j]
            state_remove = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_j = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)
                old_v_j = old_v[j]
                state_remove[j] = k_j * old_v_j

            # state_update[h, k] = sum_j k[t, h, j] * new_v[h, j]
            state_update = tl.zeros((K,), dtype=tl.float32)
            for j in range(0, K):
                k_j = tl.load(k_ptr + t * (H * K) + h * K + j).to(tl.float32)
                new_v_j = new_v[j]
                state_update[j] = k_j * new_v_j

            # Update new_state[h, v_i, k] = g_val * state_old - state_remove + state_update
            for j in range(0, K):
                state_old_val = tl.load(state_old_ptr + (seq_idx * H * V * K) + h * (V * K) + v_i * K + j).to(tl.float32)
                new_state_val = g_val * state_old_val - state_remove[j] + state_update[j]
                new_state_offset = (seq_idx * H * V * K) + h * (V * K) + v_i * K + j
                tl.store(new_state_ptr + new_state_offset, new_state_val)


@triton.jit
def compute_output_kernel(q_ptr, state_ptr, out_ptr, scale,
                          H: tl.constexpr, V: tl.constexpr, K: tl.constexpr, t):
    """
    Compute out[h, k] = scale * sum_v q[t, h, k] @ state[h, v, k] for all h, k.
    We reduce over V. q: [T, H, K], state: [H, V, K], out: [H, K].
    """
    for h in range(0, H):
        for k in range(0, K):
            acc = tl.zeros((), dtype=tl.float32)
            for v_i in range(0, V):
                # q[t, h, k]
                q_offset = t * (H * K) + h * K + k
                q_val = tl.load(q_ptr + q_offset).to(tl.float32)
                # state[h, v_i, k]
                state_offset = h * (V * K) + v_i * K + k
                state_val = tl.load(state_ptr + state_offset).to(tl.float32)
                acc += q_val * state_val
            out_offset = h * K + k
            tl.store(out_ptr + out_offset, scale * acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton-optimized forward. All heavy numeric ops are done in Triton.
        Returns (output: [T, H, K] bfloat16, new_state: [num_seqs, H, V, K] float32)
        """
        device = q.device
        # Ensure dtypes and contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        A_log = A_log.to(torch.float32).contiguous()
        a = a.to(torch.float32).contiguous()
        dt_bias = dt_bias.to(torch.float32).contiguous()
        b = b.to(torch.float32).contiguous()
        cu_seqlens = cu_seqlens.contiguous()

        T, H, K = q.shape
        assert H == 4 and K == 128
        num_k_heads = k.shape[1]
        num_v_heads = v.shape[1]
        assert num_k_heads == 4 and num_v_heads == 8

        num_seqs = cu_seqlens.shape[0] - 1

        # Compute gating g and beta in PyTorch (simple and correct)
        # g[t, v] = exp(-exp(A_log[v]) * softplus(a[t, v] + dt_bias[v]))
        g_full = torch.empty((T, 8), dtype=torch.float32, device=device)
        for t in range(T):
            g_full[t] = torch.exp(-torch.exp(A_log) * torch.log1p(torch.exp(a[t] + dt_bias)))  # shape [V]
        # beta[t, v] = sigmoid(b[t, v]) where b is [T, V]
        beta_full = torch.nn.functional.sigmoid(b)  # [T, V]

        # Prepare g and beta as [H, V] since H=4


def run(*args):
    return ModelNew()(*args)
