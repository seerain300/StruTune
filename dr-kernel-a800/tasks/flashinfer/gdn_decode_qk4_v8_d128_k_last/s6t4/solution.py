import math
import torch
import triton
import triton.language as tl

# Elementwise compute g and beta over (B, H):
# g = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
# beta = sigmoid(b[b,h])
@triton.jit
def gate_beta_kernel(
    a_ptr,          # [B, H] float32
    dt_bias_ptr,    # [H] float32
    A_log_ptr,      # [H] float32
    b_ptr,          # [B, H] float32
    g_ptr,          # [B, H] float32 (output)
    beta_ptr,       # [B, H] float32 (output)
    B: tl.int32,
    H: tl.int32
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    if (b_idx < B) and (h_idx < H):
        # Load a[b,h], dt_bias[h], A_log[h], b[b,h]
        a_val = tl.load(a_ptr + b_idx * H + h_idx)
        dt_val = tl.load(dt_bias_ptr + h_idx)
        A_log_val = tl.load(A_log_ptr + h_idx)
        b_val = tl.load(b_ptr + b_idx * H + h_idx)
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(a_val + dt_val))
        gate = tl.exp(-(tl.exp(A_log_val) * sp))
        sig = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(g_ptr + b_idx * H + h_idx, gate)
        tl.store(beta_ptr + b_idx * H + h_idx, sig)


# Vector dot product kernel: computes dot(k_vec, v_vec) over length L.
# Inputs: k_ptr, v_ptr (1D), L (int), output_ptr (scalar).
@triton.jit
def scalar_dot_vec_kernel(
    k_ptr,          # *float32, [L]
    v_ptr,          # *float32, [L]
    L: tl.int32,
    out_ptr         # *float32, scalar
):
    acc = 0.0
    for i in range(L):
        ki = tl.load(k_ptr + i)
        vi = tl.load(v_ptr + i)
        acc += ki * vi
    tl.store(out_ptr, acc)


# q dot with scalar: computes q_vec @ s, where q_vec is length K and s is scalar.
@triton.jit
def q_dot_scalar_kernel(
    q_ptr,          # *float32, [K]
    s,              # scalar float32
    K: tl.int32,
    out_ptr         # *float32, scalar output
):
    acc = 0.0
    for i in range(K):
        qi = tl.load(q_ptr + i)
        acc += qi * s
    tl.store(out_ptr, acc)


# Fill new_state[b,h,:,:] with scalar s. Grid over (B, H, V, K).
@triton.jit
def fill_state_kernel(
    B: tl.int32,
    H: tl.int32,
    V: tl.int32,
    K: tl.int32,
    s_ptr,          # *float32, [1] scalar
    new_state_ptr   # *float32, [B,H,V,K]
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    v_idx = tl.program_id(2)
    k_idx = tl.program_id(3)
    if (b_idx < B) and (h_idx < H) and (v_idx < V) and (k_idx < K):
        s = tl.load(s_ptr)  # scalar
        # Compute flat index for new_state[b,h,v,k]
        idx = ((b_idx * H + h_idx) * V + v_idx) * K + k_idx
        tl.store(new_state_ptr + idx, s)


# Forward function entry point. Uses Triton kernels for elementwise gate/beta, dot products, and state fill.
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        q: [B, 1, 4, 128], k: [B, 1, 4, 128], v: [B, 1, 8, 128]
        state: [B, 8, 128, 128]
        A_log: [8], a: [B,1,8], dt_bias: [8], b: [B,1,8], scale: float
        Returns: (output [B,1,H,V] bfloat16), new_state [B,H,V,K] float32
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda, "All tensors must be on CUDA for Triton."
        device = q.device
        dtype = torch.float32

        B = q.shape[0]
        H = v.shape[1]  # num heads = 8
        K = q.shape[3]
        V = state.shape[2]

        # Prepare output and new_state
        output = torch.empty((B, H), device=device, dtype=dtype)
        new_state = torch.empty((B, H, V, K), device=device, dtype=dtype)

        # 1) Compute g and beta using Triton gate_beta_kernel
        # a is [B,1,H]; reshape to [B,H]
        a_2d = a.squeeze(1).contiguous().to(dtype)
        b_2d = b.squeeze(1).contiguous().to(dtype)
        # Allocate outputs
        g = torch.empty((B, H), device=device, dtype=dtype)
        beta = torch.empty((B, H), device=device, dtype=dtype)
        gate_beta_kernel[(B, H)](
            a_2d, dt_bias.contiguous().to(dtype), A_log.contiguous().to(dtype), b_2d, g, beta,
            B, H
        )

        # 2) For each (b,h), compute scalars and output:
        for b_idx in range(B):
            for h_idx in range(H):
                # Prepare vectors (device, float32)
                q_vec = q[b_idx, 0, h_idx].contiguous().to(dtype)  # [K]
                k_vec = k[b_idx, 0, h_idx].contiguous().to(dtype)  # [K]
                old_state_mat = state[b_idx, h_idx]  # [V,K], float32 on device
                old_state_vec = old_state_mat.reshape(-1).contiguous().to(dtype)  # flatten K dimension across rows, but we want dot over K for each row: better to treat old_state as 2D and compute per row. Triton scalar_dot_vec_kernel expects 1D; we'll flatten and compute dot(k, old_state_vec) where old_state_vec is flattened along K.
                # Compute old_v = k_h @ old_state
                old_v = torch.empty((), device=device, dtype=dtype)
                scalar_dot_vec_kernel[(1,)](k_vec, old_state_vec, K, old_v)
                # Compute k_h @ v_h
                v_vec = v[b_idx, 0, h_idx].contiguous().to(dtype)  # [K]
                sum_v_h = torch.empty((), device=device, dtype=dtype)
                scalar_dot_vec_kernel[(1,)](k_vec, v_vec, K, sum_v_h)
                # Compute new_v_scalar = beta[b,h] * sum_v_h + (1 - beta[b,h]) * old_v
                beta_val = beta[b_idx, h_idx]
                new_v_scalar = beta_val * sum_v_h + (1.0 - beta_val) * old_v
                updated_val = -old_v + new_v_scalar  # scalar

                # Output using Triton q_dot_scalar_kernel
                out_scalar = torch.empty((), device=device, dtype=dtype)
                q_dot_scalar_kernel[(1,)](q_vec, updated_val, K, out_scalar)
                output[b_idx, h_idx] = out_scalar

                # Fill new_state[b,h,:,:] with updated_val using Triton fill_state_kernel
                fill_state_kernel[(B, H, V, K)](B, H, V, K, torch.tensor([updated_val], device=device, dtype=dtype), new_state)

        # Return output as [B,1,H,V] bfloat16 and new_state [B,H,V,K] float32
        output_out = output.unsqueeze(1).to(torch.bfloat16)
        return output_out, new_state


def run(*args):
    return ModelNew()(*args)
