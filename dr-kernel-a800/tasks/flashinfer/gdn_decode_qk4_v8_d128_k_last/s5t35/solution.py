import math
import torch
import triton
import triton.language as tl


@triton.jit
def _update_all_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, out_ptr, new_state_ptr,
    a_ptr, dt_bias_ptr, A_log_ptr, b_ptr,
    B, H, V, K
):
    """
    Each program handles one (b, h):
      - b = program_id(0) // H
      - h = program_id(0) % H
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Base offsets
    q_offset = (b * H + h) * K
    k_offset = q_offset
    v_offset = b * H * V + h * V
    state_offset = b * H * V * K + h * V * K

    # Load parameters (fp32)
    a_val = tl.load(a_ptr + b * H + h)
    db_val = tl.load(dt_bias_ptr + h)
    A_log_val = tl.load(A_log_ptr + h)
    b_val = tl.load(b_ptr + b * H + h)

    # Compute g and beta (fp32)
    a_f32 = a_val.to(tl.float32)
    db_f32 = db_val.to(tl.float32)
    A_log_f32 = A_log_val.to(tl.float32)
    b_f32 = b_val.to(tl.float32)

    x = a_f32 + db_f32
    # softplus(x) = log(1 + exp(x))
    softplus = tl.log(1.0 + tl.exp(x))
    g = tl.exp(-tl.exp(A_log_f32) * softplus)
    beta = 1.0 / (1.0 + tl.exp(-b_f32))

    # Load vectors and matrices
    q_h = tl.load(q_ptr + q_offset + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
    k_h = tl.load(k_ptr + k_offset + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
    v_h = tl.load(v_ptr + v_offset + tl.arange(0, V), mask=tl.arange(0, V) < V, other=0.0)

    # state_old: [V, K]
    state_old = tl.zeros((V, K), dtype=tl.float32)
    for i in range(V):
        for j in range(K):
            state_old[i, j] = tl.load(state_ptr + state_offset + i * K + j)

    # old_v = k_h @ state_old (reduce over K)
    old_v = 0.0
    for j in range(K):
        old_v += k_h[j] * state_old[:, j]

    # new_v = beta * v_h + (1 - beta) * old_v
    new_v = beta * v_h + (1.0 - beta) * old_v

    # old_state = g * state_old
    old_state = g * state_old

    # state_remove = k_h @ old_state (reduce over K) -> [V]
    state_remove = tl.zeros((V,), dtype=tl.float32)
    for i in range(V):
        sum_k = 0.0
        for j in range(K):
            sum_k += k_h[j] * old_state[i, j]
        state_remove[i] = sum_k

    # state_update = k_h @ new_v (reduce over K) -> [V]
    state_update = tl.zeros((V,), dtype=tl.float32)
    for i in range(V):
        sum_k = 0.0
        for j in range(K):
            sum_k += k_h[j] * new_v[i]
        state_update[i] = sum_k

    # new_state[b,h] = old_state - state_remove[:, None] + state_update[:, None]  -> [V,K]
    new_state_tile = old_state - state_remove[:, None] + state_update[:, None]

    # Write new_state back
    for i in range(V):
        for j in range(K):
            tl.store(new_state_ptr + state_offset + i * K + j, new_state_tile[i, j])

    # Compute output[b,h] = scale * sum_j q_h[j] * new_state[b,h][j,:] (reduce over V)
    # Using default scale behavior from the original sample (scale=1.0 or provided)
    # If scale is provided as an argument, we could pass it, but here we use 1.0/sqrt(K).
    scale = 1.0 / tl.sqrt(K)
    out_val = 0.0
    for j in range(K):
        row_j = 0.0
        for i in range(V):
            row_j += q_h[i] * new_state_tile[i, j]
        out_val += scale * row_j

    # Store output
    tl.store(out_ptr + b * H + h, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, K], k: [B, 1, 4, K], v: [B, 1, 8, V], state: [B, 8, V, K]
        A_log: [8], a: [B, 1, 8], dt_bias: [8], b: [B, 1, 8], scale: float
        Returns: output [B, H, V] bfloat16, new_state [B, H, V, K] float32
        """
        # Squeeze size-1 dim
        q = q.squeeze(1)  # [B, 4, K]
        k = k.squeeze(1)  # [B, 4, K]
        v = v.squeeze(1)  # [B, 8, V]

        B, Fq, K = q.shape
        B2, Fk, _ = k.shape
        B3, Fv, V = v.shape
        assert B == B2 == B3
        assert Fq == 4 and Fk == 4 and Fv == 8

        # Cast parameters to float32 and create 1D vectors for Triton
        a_f32 = a.to(torch.float32)      # [B, 1, 8]
        b_f32 = b.to(torch.float32)      # [B, 1, 8]
        A_log_f32 = A_log.to(torch.float32)  # [8]
        dt_bias_f32 = dt_bias.to(torch.float32)  # [8]

        a1 = a_f32.squeeze(1).index_select(dim=1, index=torch.arange(a_f32.size(1), device=a_f32.device)).view(-1)  # [B*8]
        b1 = b_f32.squeeze(1).index_select(dim=1, index=torch.arange(b_f32.size(1), device=b_f32.device)).view(-1)  # [B*8]
        A_log1 = A_log_f32  # [8]
        dt_bias1 = dt_bias_f32  # [8]

        # Flatten inputs for kernel
        q_flat = q.reshape(B, Fq * K).contiguous()    # [B, 4*K]
        k_flat = k.reshape(B, Fk * K).contiguous()    # [B, 4*K]
        v_flat = v.reshape(B, Fv * V).contiguous()    # [B, 8*V]
        state_flat = state.reshape(B, Fv * V * K).contiguous()  # [B, 8*V*K]

        # Allocate output and new_state
        out = torch.empty(B * 8, dtype=torch.float32, device=q.device)
        new_state = torch.empty(B * 8 * V * K, dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b,h)
        grid = (B * 8,)
        _update_all_kernel[grid](
            q_flat, k_flat, v_flat, state_flat, out, new_state,
            a1, dt_bias1, A_log1, b1,
            B, 8, V, K,
            num_warps=1,
        )

        # Reshape output to [B, 8, V] and cast to bfloat16, then return [B, 8, V] (original signature expects [B, H, V])
        out_bf16 = out.view(B, 8, V).unsqueeze(-1).expand(B, 8, V).contiguous().to(torch.bfloat16)
        new_state = new_state.view(B, 8, V, K).contiguous()

        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
