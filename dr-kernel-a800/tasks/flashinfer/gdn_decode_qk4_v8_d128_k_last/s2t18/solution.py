import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gate_beta_kernel(
    A_log_ptr,         # *float32, shape [H]
    a_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h)
    dt_bias_ptr,       # *float32, shape [H]
    b_ptr,             # *bfloat16, shape [B, 1, H] (we index by b,h)
    g_ptr,             # *float32, shape [B, 1, H]
    beta_ptr,          # *float32, shape [B, 1, H]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # number of heads (num_v_heads)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Load a and dt_bias to compute x = a + dt_bias
    a_val = tl.cast(tl.load(a_ptr + b * a_ptr.stride(0) + h * a_ptr.stride(1)), tl.float32)
    dt_val = tl.load(dt_bias_ptr + h)  # dt_bias is [H]
    x = a_val + dt_val

    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # g = exp(-exp(A_log) * softplus(x))
    A_log_val = tl.load(A_log_ptr + h)  # A_log is [H]
    g_val = tl.exp(-tl.exp(A_log_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    b_val = tl.cast(tl.load(b_ptr + b * b_ptr.stride(0) + h * b_ptr.stride(1)), tl.float32)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results to [B, 1, H] layout (linear indexing; 1D)
    tl.store(g_ptr + pid, g_val)
    tl.store(beta_ptr + pid, beta_val)


@triton.jit
def triton_update_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,     # input tensors
    g_ptr, beta_ptr,                    # gating tensors
    output_ptr, new_state_ptr,          # output tensors
    B: tl.constexpr, H: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
    scale: tl.constexpr,                # scalar, e.g., 1.0 / sqrt(K)
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Offsets for this (b, h)
    # q, k, v have shape [B, H, K] -> contiguous offset = b*H*K + h*K + k_offset
    # state, new_state have shape [B, H, V, K] -> contiguous offset = b*(H*V*K) + h*(V*K) + v*K + k_offset
    base_q = b * H * K + h * K
    base_k = b * H * K + h * K
    base_v = b * H * V + h * V
    base_state = b * (H * V * K) + h * (V * K)

    # Load q_h, k_h, v_h
    q_vec = tl.zeros([K], dtype=tl.float32)
    k_vec = tl.zeros([K], dtype=tl.float32)
    v_vec = tl.zeros([V], dtype=tl.float32)

    for j in range(0, K):
        q_elem = tl.cast(tl.load(q_ptr + base_q + j), tl.float32)
        k_elem = tl.cast(tl.load(k_ptr + base_k + j), tl.float32)
        q_vec[j] = q_elem
        k_vec[j] = k_elem

    for v_idx in range(0, V):
        v_vec[v_idx] = tl.cast(tl.load(v_ptr + base_v + v_idx), tl.float32)

    # Load old_state [V, K]
    h_state = tl.zeros([V, K], dtype=tl.float32)
    for v_idx in range(0, V):
        row_base = state_ptr + base_state + v_idx * K
        for k_idx in range(0, K):
            h_state[v_idx, k_idx] = tl.cast(tl.load(row_base + k_idx), tl.float32)

    # Compute g and beta
    g_val = tl.load(g_ptr + pid)      # scalar
    beta_val = tl.load(beta_ptr + pid)  # scalar

    # Compute old_v = k @ (g * old_state) -> (K,)
    old_v_vec = tl.zeros([K], dtype=tl.float32)
    for j in range(0, K):
        sum_val = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_val += h_state[v_idx, j] * g_val
        old_v_vec[j] = sum_val  # scalar product with k_vec[j], implicit scalar

    # Compute new_v = beta * v + (1 - beta) * old_v
    new_v_vec = tl.zeros([V], dtype=tl.float32)
    for v_idx in range(0, V):
        new_v_vec[v_idx] = beta_val * v_vec[v_idx] + (1.0 - beta_val) * old_v_vec[v_idx]

    # Compute state_remove and state_update: scalar
    state_remove = tl.zeros((), dtype=tl.float32)
    state_update = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        state_remove += old_v_vec[j] * k_vec[j]
        state_update += new_v_vec[j] * k_vec[j]

    # Update h_state_new elementwise: h_state_new = (g * h_state) - state_remove + state_update
    h_state_new = h_state * g_val - state_remove + state_update  # [V, K]

    # Compute output scalar: output = scale * q_h @ h_state_new
    output_scalar = tl.zeros((), dtype=tl.float32)
    for j in range(0, K):
        qj = q_vec[j]
        col_j = h_state_new[:, j]  # [V] vector, sum over V
        sum_j = tl.zeros((), dtype=tl.float32)
        for v_idx in range(0, V):
            sum_j += col_j[v_idx]
        output_scalar += qj * sum_j

    # Store output scalar to output[b, h] (linear layout [B*H])
    tl.store(output_ptr + pid, output_scalar)

    # Store updated state: new_state[b, h, :, :] = h_state_new (float32)
    new_state_base = b * (H * V * K) + h * (V * K)
    for v_idx in range(0, V):
        row_base_new = new_state_ptr + new_state_base + v_idx * K
        for k_idx in range(0, K):
            tl.store(row_base_new + k_idx, h_state_new[v_idx, k_idx])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, Nq, K]
        k: [B, 1, Nk, K]
        v: [B, 1, Hv, V]
        state: [B, Hv, V, K]
        A_log: [Hv]
        a: [B, 1, Hv] bfloat16
        dt_bias: [Hv] float32
        b: [B, 1, Hv] bfloat16
        scale: float or None
        Returns: (output [B, 1, Hv] bfloat16), new_state [B, Hv, V, K] float32
        """
        B, T_q, Nq, K = q.shape
        B2, T_k, Nk, K2 = k.shape
        B3, T_v, Hv, V = v.shape
        B4, T_s, Hv2, V2, K3 = state.shape
        assert B == B2 == B3 == B4, "Batch size mismatch"
        assert T_q == T_k == T_v == 1, "Only T=1 supported"
        assert Nq == 4 and Nk == 4, "num_q_heads must be 4 and num_k_heads must be 4"
        assert Hv == Hv2 and V == V2 and K == K3, "v and state must match in shape"
        assert Hv == dt_bias.shape[0], "dt_bias shape mismatch"
        assert A_log.shape[0] == Hv, "A_log shape mismatch"
        assert a.shape[0] == B and a.shape[2] == Hv, "a shape mismatch"
        assert b.shape[0] == B and b.shape[2] == Hv, "b shape mismatch"

        device = q.device
        q_c = q.contiguous().to(torch.float32)
        k_c = k.contiguous().to(torch.float32)
        v_c = v.contiguous().to(torch.float32)
        state_c = state.contiguous().to(torch.float32)
        a_c = a.contiguous().to(torch.float32)  # [B, 1, Hv]
        dt_bias_c = dt_bias.contiguous().to(torch.float32)  # [Hv]
        b_c = b.contiguous().to(torch.float32)  # [B, 1, Hv]

        # Allocate outputs
        g = torch.empty((B, 1, Hv), dtype=torch.float32, device=device)
        beta = torch.empty((B, 1, Hv), dtype=torch.float32, device=device)
        output = torch.empty((B, Hv), dtype=torch.float32, device=device)  # will be [B,1,Hv] after unsqueeze

        # Launch gate+beta kernel
        grid_g = (B * Hv,)
        triton_gate_beta_kernel[grid_g](
            A_log, a_c, dt_bias_c, b_c, g, beta, B, Hv
        )

        # Launch update kernel
        grid_u = (B * Hv,)
        # Compute scale in host, pass as constexpr-like scalar (Triton treats it as float)
        if scale is None or float(scale) == 0.0:
            scale_val = 1.0 / math.sqrt(K)
        else:
            scale_val = float(scale)

        triton_update_kernel[grid_u](
            q_c, k_c, v_c, state_c, g, beta, output, None,  # new_state we'll allocate and write in-kernel
            B=B, H=Hv, V=V, K=K, scale=scale_val
        )

        # Prepare new_state tensor and store updated values inside kernel (we already computed and stored)
        # We need to create a new_state tensor and have the kernel write into it; since Triton cannot return, we store via output_ptr? Wait—kernel wrote into new_state_ptr we passed. However, we didn't pass new_state_ptr above. Fix: allocate new_state and pass it in launch.

        # Correction: re-launch update with proper new_state_ptr
        new_state = torch.empty((B, Hv, V, K), dtype=torch.float32, device=device)
        triton_update_kernel[grid_u](
            q_c, k_c, v_c, state_c, g, beta, output, new_state,
            B=B, H=Hv, V=V, K=K, scale=scale_val
        )

        # Return outputs: output as bfloat16, unsqueezed to (B,1,Hv), new_state as float32
        output_bf16 = output.unsqueeze(1).to(torch.bfloat16)  # shape [B,1,Hv]
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
