import torch
import triton
import triton.language as tl


@triton.jit
def output_dot_kernel(
    q_exp_ptr,          # [H*K] float32
    k_ptr,              # [B*H*K] float32
    v_ptr,              # [B*H*V] float32
    state_ptr,          # [B*H*V*K] float32
    new_state_ptr,      # [B*H*V*K] float32
    out_ptr,            # [B*H] float32
    A_log_ptr,          # [H] float32
    a_ptr,              # [B*H] float32
    dt_bias_ptr,        # [H] float32
    b_ptr,              # [B*H] float32
    scale,              # float32 scalar
    B: tl.constexpr,    # int (not used in kernel but can be useful)
    H: tl.constexpr,    # number of heads (used for grid)
    V: tl.constexpr,    # rows in state (128)
    K: tl.constexpr,    # columns in state (128)
):
    pid = tl.program_id(axis=0)  # 0 .. (B*H - 1)
    b_idx = pid // H
    h_idx = pid % H

    # Load scalars for this (b, h)
    a_val = tl.load(a_ptr + b_idx * H + h_idx)
    dt_val = tl.load(dt_bias_ptr + h_idx)
    A_val = tl.load(A_log_ptr + h_idx)
    b_val = tl.load(b_ptr + b_idx * H + h_idx)

    # Compute g and beta
    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    x = a_val + dt_val
    absx = tl.abs(x)
    softplus = tl.log(1.0 + tl.exp(-absx)) + tl.maximum(x, 0.0)
    g = tl.exp(-tl.exp(A_val) * softplus)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # Initialize new_state[b,h] to zeros
    # We'll update it per i in V
    # Note: We cannot form a "row pointer" due to Triton's pointer semantics,
    # but we can index scalar positions and store. Since K and V are tl.constexpr,
    # we can rely on scalar indexing and tl.store per element.

    # Prepare base offsets for state and new_state at (b,h)
    # state indexing: [b, h, i, j] -> offset = b*(H*V*K) + h*(V*K) + i*K + j
    # new_state indexing: [b, h, i, j] -> offset = b*(H*V*K) + h*(V*K) + i*K + j
    # We'll compute per (i,j) using scalar loops and tl.store.

    # For each i in V: compute old_v, new_v, then update new_state[b,h,i,j]
    # old_v = sum_j k[h,j] * state[b,h,i,j]
    for i in tl.static_range(0, V):
        # Compute old_v
        old_v = 0.0
        for j in tl.static_range(0, K):
            s_old = tl.load(state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j)
            # k[h,j] is the same for all b when addressing k_ptr; we need k_exp[b,h,j]
            # k_exp was created in host, so k_ptr + (b_idx*H + h_idx)*K + j
            k_idx = (b_idx * H + h_idx) * K + j
            k_j = tl.load(k_ptr + k_idx)
            old_v += k_j * s_old
        # Load v[b,h,i]
        v_val = tl.load(v_ptr + b_idx * (H * V) + h_idx * V + i)
        new_v = beta * v_val + (1.0 - beta) * old_v
        # Update new_state[b,h,i,j] = state - old_v + k[h,j]*new_v
        for j in tl.static_range(0, K):
            s_old = tl.load(state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j)
            k_j = tl.load(k_ptr + (b_idx * H + h_idx) * K + j)
            n = s_old - old_v + k_j * new_v
            tl.store(new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j, n)

    # Compute output[b,h] = scale * (q_exp[h] @ new_state[b,h])
    # new_state[b,h] has shape [V,K]; with V=1 in given inputs, this is [1,K].
    # If V>1, we sum across all rows i.
    acc = 0.0
    for i in tl.static_range(0, V):
        for j in tl.static_range(0, K):
            s_new = tl.load(new_state_ptr + b_idx * (H * V * K) + h_idx * (V * K) + i * K + j)
            # Address q_exp[h,:] element j:
            q_j = tl.load(q_exp_ptr + h_idx * K + j)
            acc += q_j * s_new
    out_val = acc * scale
    tl.store(out_ptr + pid, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized forward. Computes:
          - output: [B, 1, H, 1], bfloat16
          - new_state: [B, H, V, K], float32
        """
        device = q.device
        # Cast inputs to float32 for computation (matching original matmul to float)
        q_f32 = q.to(torch.float32).contiguous()           # [B, 1, Q, K]
        k_f32 = k.to(torch.float32).contiguous()           # [B, 1, K, K]
        v_f32 = v.to(torch.float32).contiguous()           # [B, 1, V, V]
        state_f32 = state.to(torch.float32).contiguous()   # [B, H, V, K]
        A_log_f32 = A_log.to(torch.float32).contiguous()   # [H]
        a_f32 = a.to(torch.float32).contiguous()           # [B, 1, H]
        dt_bias_f32 = dt_bias.to(torch.float32).contiguous()  # [H]
        b_f32 = b.to(torch.float32).contiguous()           # [B, 1, H]

        # Expand q and k along heads by repeat_interleave (ratio=2 for provided inputs).
        # In general, ratio = v.shape[1] // q.shape[1] = 8 // 4 = 2.
        ratio = v_f32.shape[1] // q_f32.shape[1]
        q_exp = q_f32.repeat_interleave(ratio, dim=1)      # [B, H, K]
        k_exp = k_f32.repeat_interleave(ratio, dim=1)      # [B, H, K]

        B = q_f32.shape[0]
        H = q_f32.shape[1]
        V = state_f32.shape[2]
        K = state_f32.shape[3]

        # Allocate new_state
        new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

        # Allocate output buffer [B*H]
        out = torch.empty((B * H,), dtype=torch.float32, device=device)

        # Launch Triton kernel: grid over (B*H)
        output_dot_kernel[(B * H,)](
            q_exp.view(H * K),                      # [H*K]
            k_exp.view(B * H * K),                 # [B*H*K]
            v_f32.view(B * H * V),                 # [B*H*V]
            state_f32,                             # [B*H*V*K]
            new_state,                             # [B*H*V*K]
            out,                                   # [B*H]
            A_log_f32,                             # [H]
            a_f32.view(B * H),                     # [B*H]
            dt_bias_f32,                           # [H]
            b_f32.view(B * H),                     # [B*H]
            float(scale),                          # scalar
            B=B, H=H, V=V, K=K,
        )

        # Return outputs: output [B,1,H,1] bfloat16 and new_state [B,H,V,K] float32
        output = out.view(B, 1, H, 1).to(torch.bfloat16)
        return output, new_state


def run(*args):
    return ModelNew()(*args)
