import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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
    # Accumulator for y outputs
    y_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        # Load k chunk
        k_chunk = tl.load(k_ptr + k_offsets, mask=k_offsets < K, other=0.0)  # [BLOCK_K]
        # For each kk in chunk, accumulate k_chunk[kk] * x[kk, v_offsets]
        for kk in range(0, BLOCK_K):
            k_val = k_chunk[kk]  # scalar
            k_idx = k_start + kk
            # x[k_idx, v_offsets] => pointer offset k_idx*V + v_offsets
            x_vals = tl.load(x_ptr + k_idx * V + v_offsets, mask=v_offsets < V, other=0.0)
            y_acc += k_val * x_vals
    # Store results
    tl.store(y_ptr + v_offsets, y_acc, mask=v_offsets < V)


@triton.jit
def dot_kernel(q_ptr, x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute out = sum_i q[i] * x[i], where q and x are 1D vectors of length N.
    Each program accumulates over a block of N and atomic_adds into out_ptr[0].
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    q = tl.load(q_ptr + offsets, mask=offsets < N, other=0.0)
    x = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
    partial = tl.sum(q * x, axis=0)
    # Accumulate into a single scalar
    tl.atomic_add(out_ptr, partial)


def _triton_matvec(x_2d, k_vec, out_vec):
    """
    Helper to run matvec_kernel. x_2d must be [K, V] contiguous, k_vec [K], out_vec [V].
    """
    K, V = x_2d.shape
    # Ensure contiguous and float32
    x_2d = x_2d.contiguous().float()
    k_vec = k_vec.contiguous().float()
    out_vec = torch.empty(V, dtype=torch.float32, device=x_2d.device)
    BLOCK_K = 128
    BLOCK_V = 128
    grid = (triton.cdiv(V, BLOCK_V),)
    matvec_kernel[grid](x_2d.view(-1), k_vec, out_vec, K, V, BLOCK_K, BLOCK_V)
    return out_vec


def _triton_dot(q_vec, x_vec, out_scalar_ptr):
    """
    Helper to run dot_kernel. q_vec, x_vec must be 1D contiguous float32. out_scalar_ptr is a 1-element tensor.
    """
    N = q_vec.numel()
    q_vec = q_vec.contiguous().float()
    x_vec = x_vec.contiguous().float()
    # Initialize out_scalar_ptr to zero
    out_scalar_ptr.zero_()
    BLOCK = 128
    grid = (triton.cdiv(N, BLOCK),)
    dot_kernel[grid](q_vec, x_vec, out_scalar_ptr, N, BLOCK)
    # Read the result from the tensor (avoid .item() on Triton outputs)
    return out_scalar_ptr[0].float()


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version of the original run, using Triton for all matrix-vector computations and final dot.
        Returns:
        - output: [B, 1, H, V] in bfloat16
        - new_state: [B, H, V, K] in float32
        """
        # Shapes
        B, T, num_q_heads, K = q.shape
        _, _, num_k_heads, _ = k.shape
        _, _, num_v_heads, V = v.shape
        device = q.device

        # Compute g and beta on host in float32
        g = torch.exp(-torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float()))  # [H]
        beta = torch.sigmoid(b.float())  # [B, 1, H]

        # Squeeze T=1 as original does
        q = q.squeeze(1)  # [B, num_q_heads, K]
        k = k.squeeze(1)  # [B, num_k_heads, K]

        # Match original repeat_interleave behavior
        q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [B, num_v_heads, K]
        k_exp = k.repeat_interleave(num_k_heads // num_v_heads, dim=1)  # [B, num_v_heads, K]
        v = v.squeeze(1)  # [B, num_v_heads, K]

        # Initialize outputs
        output = torch.zeros(B, 1, num_v_heads, V, dtype=torch.bfloat16, device=device)
        new_state = torch.zeros(B, num_v_heads, V, K, dtype=torch.float32, device=device)

        for b_idx in range(B):
            for h_idx in range(num_v_heads):
                # Extract vectors
                q_h = q_exp[b_idx, h_idx].contiguous().float()  # [K]
                k_h = k_exp[b_idx, h_idx].contiguous().float()  # [K]
                v_h = v[b_idx, h_idx].contiguous().float()      # [K]
                # Load old state [V, K] from provided state (as in original)
                old_state = state[b_idx, h_idx].contiguous().float()  # [V, K]
                # Compute old_v = k @ old_state via Triton
                old_v = _triton_matvec(old_state, k_h, torch.empty(V, dtype=torch.float32, device=device))
                # Compute new_v = beta * v_h + (1 - beta) * old_v
                beta_val = beta[b_idx, 0, h_idx].item()
                new_v = beta_val * v_h + (1.0 - beta_val) * old_v  # [K]
                # Compute state_remove = k @ old_v
                state_remove = _triton_matvec(old_v, k_h, torch.empty(V, dtype=torch.float32, device=device))
                # Compute state_update = k @ new_v
                state_update = _triton_matvec(new_v, k_h, torch.empty(V, dtype=torch.float32, device=device))
                # Update new_state elementwise: new_state[b, h, i, j] = g[h] * old_state[i, j] - state_remove[i] + state_update[i]
                g_val = float(g[h_idx].item())
                for i in range(V):
                    for j in range(K):
                        new_state[b_idx, h_idx, i, j] = g_val * old_state[i, j] - state_remove[i] + state_update[i]
                # Compute output scalar: output = scale * (q_h @ new_state_vec), where new_state_vec[i] = sum_j new_state[b,h,i,j]
                new_state_vec = torch.sum(new_state[b_idx, h_idx], dim=1)  # [V]
                # Compute dot using Triton, store to a tensor and read it
                out_scalar = _triton_dot(q_h, new_state_vec, torch.zeros(1, dtype=torch.float32, device=device))
                # Apply scale
                if scale is None or scale == 0.0:
                    scale_val = 1.0 / math.sqrt(K)
                else:
                    scale_val = float(scale)
                output[b_idx, 0, h_idx, 0] = torch.tensor(out_scalar * scale_val, dtype=torch.bfloat16, device=device)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
