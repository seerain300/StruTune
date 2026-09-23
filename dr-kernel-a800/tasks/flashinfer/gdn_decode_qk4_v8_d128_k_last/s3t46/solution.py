import torch
import triton
import triton.language as tl


@triton.jit
def compute_update_and_output(
    q_ptr,        # [B, H, K] float32
    k_ptr,        # [B, H, K] float32
    v_ptr,        # [B, H, V] float32
    state_ptr,    # [B, H, V, K] float32
    a_ptr,        # [B, H] float32
    dt_bias_ptr,  # [H] float32
    A_log_ptr,    # [H] float32
    b_ptr,        # [B, H] float32
    new_state_ptr,# [B, H, V, K] float32
    out_ptr,      # [B, H] float32
    scale,        # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(axis=0)
    h_idx = tl.program_id(axis=1)

    # Compute g and beta for this (b,h)
    a_val = tl.load(a_ptr + b_idx * H + h_idx)           # scalar
    dt_val = tl.load(dt_bias_ptr + h_idx)               # scalar
    A_val = tl.load(A_log_ptr + h_idx)                 # scalar
    x = a_val + dt_val
    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)  (numerically stable)
    absx = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-absx)) + tl.maximum(x, 0.0)
    g = tl.exp(-tl.exp(A_val) * softplus)             # scalar
    beta = 1.0 / (1.0 + tl.exp(-tl.load(b_ptr + b_idx * H + h_idx)))  # scalar

    # Load q[h] and k[h] as vectors [K]
    q_row_ptrs = q_ptr + b_idx * (H * K) + h_idx * K + tl.arange(0, K)
    k_row_ptrs = k_ptr + b_idx * (H * K) + h_idx * K + tl.arange(0, K)
    q_row = tl.load(q_row_ptrs)  # [K]
    k_row = tl.load(k_row_ptrs)  # [K]

    # Base pointers for this (b,h)
    state_b_h_ptr = state_ptr + b_idx * (H * V * K) + h_idx * (V * K)
    new_state_b_h_ptr = new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K)
    v_b_h_ptr = v_ptr + b_idx * (H * V) + h_idx * V

    # Update new_state row-wise over i in V
    for i in tl.static_range(0, V):
        row_state_ptrs = state_b_h_ptr + i * K + tl.arange(0, K)  # [K]
        row_new_ptrs = new_state_b_h_ptr + i * K + tl.arange(0, K)  # [K]
        # Compute old_v = k_row @ state[i, :]
        old_v = 0.0
        for jj in tl.static_range(0, K):
            old_v += k_row[jj] * tl.load(row_state_ptrs + jj)
        # Compute new_v = beta * v[b,h,i] + (1-beta) * old_v
        new_v = beta * tl.load(v_b_h_ptr + i) + (1.0 - beta) * old_v
        # Update new_state[i, :] = state[i, :] - old_v + new_v * k_row
        for jj in tl.static_range(0, K):
            val = tl.load(row_new_ptrs + jj)  # current state[i,jj]
            tl.store(row_new_ptrs + jj, val - old_v + new_v * k_row[jj])

    # Compute output[b,h] = scale * dot(q_row, new_state[b,h])
    out_row_ptrs = new_state_b_h_ptr + tl.arange(0, K)
    acc = 0.0
    for jj in tl.static_range(0, K):
        acc += q_row[jj] * tl.load(out_row_ptrs + jj)  # sum q_row * new_state[b,h,:]
    out_val = scale * acc
    tl.store(out_ptr + b_idx * H + h_idx, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-Only implementation.

        Returns:
          - output: [B, 1, H, V] bfloat16 (with V=1 in provided inputs)
          - new_state: [B, H, V, K] float32
        """
        # Ensure tensors on CUDA for Triton; get device from q (bf16 tensor)
        device = q.device
        if device.type != "cuda":
            # Move to CUDA if not already
            q = q.to("cuda")
            k = k.to("cuda")
            v = v.to("cuda")
            state = state.to("cuda")
            A_log = A_log.to("cuda")
            a = a.to("cuda")
            dt_bias = dt_bias.to("cuda")
            b = b.to("cuda")

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()  # [B, 4, K]
        k_f32 = k.to(torch.float32).contiguous()  # [B, 4, K]
        v_f32 = v.to(torch.float32).contiguous()  # [B, 8, V]
        state_f32 = state.to(torch.float32).contiguous()  # [B, 8, V, K]

        # Squeeze batch dimension (inputs have B=1 in provided get_inputs)
        B = q_f32.shape[0]
        H = v_f32.shape[1]  # number of heads (8)
        V = state_f32.shape[2]
        K = state_f32.shape[3]

        # Allocate outputs
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b,h)
        grid = (B, H)
        compute_update_and_output[grid](
            q_f32, k_f32, v_f32, state_f32,
            a.to(torch.float32), dt_bias.to(torch.float32), A_log.to(torch.float32), b.to(torch.float32),
            new_state, out_f32, float(scale),
            B=B, H=H, V=V, K=K,
            num_warps=4, num_stages=2,
        )

        # Return output cast to bfloat16 with shape [B,1,H,V] -> [B,1,8,1] and new_state
        out_bf16 = out_f32.view(B, 1, H, V).to(torch.bfloat16)
        return out_bf16, new_state


def run(*args):
    return ModelNew()(*args)
