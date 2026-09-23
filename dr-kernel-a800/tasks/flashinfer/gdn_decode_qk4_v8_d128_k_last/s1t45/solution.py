import math
import torch
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = softplus(x[i]) = log(1 + exp(x[i])) for i in [0, N).
    """
    pid = tl.program_id(axis=0)
    x = tl.load(x_ptr + pid, mask=pid < N, other=0.0)
    # softplus: log(1 + exp(x)) (x is float32 here)
    y = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + pid, y, mask=pid < N)


@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = sigmoid(x[i]) = 1 / (1 + exp(-x[i])) for i in [0, N).
    """
    pid = tl.program_id(axis=0)
    x = tl.load(x_ptr + pid, mask=pid < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + pid, y, mask=pid < N)


@triton.jit
def exp_kernel(inp_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = exp(inp[i]) for i in [0, N).
    """
    pid = tl.program_id(axis=0)
    x = tl.load(inp_ptr + pid, mask=pid < N, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + pid, y, mask=pid < N)


@triton.jit
def matvec_kernel(x_ptr, k_ptr, y_ptr, K: tl.constexpr, V: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    """
    Compute y = k @ x, where:
      - x is a 2D matrix of shape [K, V] (passed as a contiguous 1D pointer of length K*V)
      - k is a 1D vector of length K
      - y is a 1D vector of length V
    Each program instance handles a block of V outputs and loops over K in chunks of BLOCK_K.
    """
    pid = tl.program_id(axis=0)
    v_start = pid * BLOCK_V
    v_offsets = v_start + tl.arange(0, BLOCK_V)
    y_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_chunk = tl.load(k_ptr + k_offsets, mask=k_offsets < K, other=0.0)  # [BLOCK_K]
        for kk in range(0, BLOCK_K):
            k_val = k_chunk[kk]  # scalar
            k_idx = k_start + kk
            x_vals = tl.load(x_ptr + k_idx * V + v_offsets, mask=v_offsets < V, other=0.0)
            y_acc += k_val * x_vals

    tl.store(y_ptr + v_offsets, y_acc, mask=v_offsets < V)


@triton.jit
def dot_kernel(q_ptr, x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute out = sum_i q[i] * x[i], where q and x are 1D vectors of length N.
    Each program handles a block of N elements and writes a partial sum to out_ptr.
    Grid should be (1,), so no atomic contention.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    prod = q * x
    partial = tl.sum(prod, axis=0)  # sum over this block
    tl.store(out_ptr + pid, partial)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward: elementwise gates and matvecs in Triton.
        Returns (output: [B, 1, H, V], bfloat16), (new_state: [B, H, V, K], float32).
        """
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        num_heads = num_v_heads
        device = q.device

        # Repeat q, k along head dimension to match v’s heads (as in original code).
        q_exp = q.squeeze(1).repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [B, H, K]
        k_exp = k.squeeze(1).repeat_interleave(num_k_heads // num_k_heads, dim=1)  # [B, H, K]

        # Compute g and beta in Triton:
        # g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
        A_log = A_log.to(torch.float32).contiguous()
        a = a.to(torch.float32).contiguous()
        dt_bias = dt_bias.to(torch.float32).contiguous()
        b = b.to(torch.float32).contiguous()

        # A = a + dt_bias (torch is okay for small vector)
        A = a + dt_bias  # [H]

        # Triton softplus and sigmoid
        A_softplus = torch.empty_like(A)
        sigmoid_b = torch.empty_like(b)  # b has shape [1, 1, H]; we treat last dim as H

        N_A = A.numel()
        softplus_kernel[(N_A,)](A, A_softplus, N_A)

        N_b = b.numel()
        sigmoid_b = torch.empty(N_b, dtype=torch.float32, device=device)
        sigmoid_kernel[(N_b,)](b.view(-1), sigmoid_b, N_b)
        sigmoid_b = sigmoid_b.view(1, 1, -1)  # reshape to [1,1,H]

        # g = exp(-exp(A_softplus) * A_softplus)
        exp_A_soft = torch.empty_like(A_softplus)
        exp_kernel[(N_A,)](A_softplus, exp_A_soft, N_A)
        g = torch.exp(-exp_A_soft * A_softplus)  # [H], float32

        # Prepare output and new_state
        output = torch.empty((B, 1, num_heads, V), dtype=torch.bfloat16, device=device)
        new_state = torch.zeros((B, num_heads, V, K), dtype=torch.float32, device=device)

        # Process each (b,h)
        for b_idx in range(B):
            for h_idx in range(num_heads):
                q_h = q_exp[b_idx, h_idx]            # [K], float32
                k_h = k_exp[b_idx, h_idx]            # [K], float32
                v_h = v[b_idx, 0, h_idx]             # [V], float32
                old_state = state[b_idx, h_idx].contiguous()  # [V, K], float32

                # Compute old_v = k_h @ old_state
                old_state_t = old_state.t().contiguous()  # [K, V]
                old_v = torch.empty(V, dtype=torch.float32, device=device)
                matvec_kernel[(triton.cdiv(V, 128),)](old_state_t, k_h, old_v, K, V, 128, 128)

                # Compute new_v = beta * v_h + (1 - beta) * old_v
                beta_val = sigmoid_b[0, 0, h_idx].item()  # scalar float32
                new_v = beta_val * v_h + (1.0 - beta_val) * old_v  # [V], float32

                # Compute state_remove = k_h @ old_v, state_update = k_h @ new_v
                state_remove = torch.empty(V, dtype=torch.float32, device=device)
                matvec_kernel[(1,)](old_v, k_h, state_remove, K, V, 128, 128)

                state_update = torch.empty(V, dtype=torch.float32, device=device)
                matvec_kernel[(1,)](new_v, k_h, state_update, K, V, 128, 128)

                # Update new_state elementwise:
                g_val = float(g[h_idx])
                updated_state = torch.zeros((V, K), dtype=torch.float32, device=device)
                for i in range(V):
                    updated_state[i, :] = g_val * old_state[i, :] - state_remove[i] + state_update[i]
                new_state[b_idx, h_idx] = updated_state  # [V, K], float32

                # Compute output scalar: scale * (q_h @ new_state_vec), where new_state_vec[i] = sum_j new_state[i, j]
                new_state_t = new_state[b_idx, h_idx].t()  # [K, V]
                new_state_vec = torch.sum(new_state_t, dim=1)  # [V]
                out_scalar = torch.dot(q_h, new_state_vec)  # scalar float32

                # Apply scale
                if scale is None or scale == 0.0:
                    # Use 1/sqrt(K) without using sqrt in host
                    scale_val = 1.0 / 128.0
                else:
                    scale_val = float(scale)
                out_scalar = out_scalar * scale_val

                # Store into output[b, 0, h, 0] as bfloat16 (no torch.tensor creation)
                output[b_idx, 0, h_idx, 0] = out_scalar.to(torch.bfloat16)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
