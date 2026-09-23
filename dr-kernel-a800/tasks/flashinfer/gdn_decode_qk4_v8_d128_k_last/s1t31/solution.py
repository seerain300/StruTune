import math
import torch
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = softplus(x[i]) = max(x[i], 0) + log(1 + exp(-|x[i]|)) (numerically stable).
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    absx = tl.abs(x)
    maxx = tl.maximum(x, 0.0)
    y = maxx + tl.log(1.0 + tl.exp(-absx))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = sigmoid(x[i]) = 1 / (1 + exp(-x[i])).
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def exp_kernel(inp_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = exp(inp[i]).
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(inp_ptr + i, mask=i < N, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def sqrt_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = sqrt(x[i]).
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.sqrt(x)
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def matvec_kernel(x_ptr, k_ptr, y_ptr, K: tl.constexpr, V: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    """
    Compute y = k @ x, where:
      - x is a 2D matrix of shape [K, V] (passed as a contiguous 1D pointer of length K*V)
      - k is a 1D vector of length K
      - y is a 1D vector of length V
    Each program instance handles a block of V outputs, looping over K in chunks of BLOCK_K.
    """
    pid = tl.program_id(axis=0)
    v_start = pid * BLOCK_V
    v_offsets = v_start + tl.arange(0, BLOCK_V)
    y_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_chunk = tl.load(k_ptr + k_offsets, mask=k_offsets < K, other=0.0)  # [BLOCK_K]
        for kk in range(0, BLOCK_K):
            k_val = k_chunk[kk]
            k_idx = k_start + kk
            x_vals = tl.load(x_ptr + k_idx * V + v_offsets, mask=v_offsets < V, other=0.0)
            y_acc += k_val * x_vals
    tl.store(y_ptr + v_offsets, y_acc, mask=v_offsets < V)


@triton.jit
def dot_kernel(q_ptr, x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute out[0] = sum_i q[i] * x[i], where q and x are 1D vectors of length N.
    Grid is 1 program (N handled via loop).
    """
    offsets = tl.arange(0, BLOCK)
    q = tl.load(q_ptr + offsets, mask=offsets < N, other=0.0)
    x = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
    acc = tl.sum(q * x, axis=0)
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: torch.Tensor,
                A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor, scale):
        """
        q: [B, 1, 4, 128] bfloat16
        k: [B, 1, 4, 128] bfloat16
        v: [B, 1, 8, 128] bfloat16
        state: [B, 8, 128, 128] float32 (k-last: [B, H, V, K])
        A_log: [8] float32
        a: [B, 1, 8] bfloat16
        dt_bias: [8] float32
        b: [B, 1, 8] bfloat16
        scale: float or None
        Returns:
        output: [B, 1, 8, 1] bfloat16
        new_state: [B, 8, 128, 128] float32
        """
        B = q.shape[0]
        K = q.shape[-1]  # 128
        num_q_heads = 4
        num_k_heads = 4
        num_v_heads = 8
        V = 128
        device = q.device

        # Squeeze T=1 and prepare
        q_s = q.squeeze(1)  # [B, 4, 128]
        k_s = k.squeeze(1)  # [B, 4, 128]
        v_s = v.squeeze(1)  # [B, 8, 128]

        # Repeat q, k along head dimension to match v’s heads
        q_exp = q_s.repeat_interleave(8 // 4, dim=1)    # [B, 8, 128]
        k_exp = k_s.repeat_interleave(8 // 4, dim=1)    # [B, 8, 128]
        v_exp = v_s                                        # [B, 8, 128]

        # Prepare output and new_state (we'll fill new_state via torch update)
        output = torch.empty((B, 1, 8, 1), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((B, 8, 128, 128), dtype=torch.float32, device=device)

        # Precompute g and beta via Triton elementwise kernels to avoid torch ops in forward
        # For each (b, h), we need:
        # g[h] = exp(-exp(A_log[h]) * softplus(a[b, h] + dt_bias[h]))
        # beta[h] = sigmoid(b[b, h])

        for b_idx in range(B):
            # Flatten a[:, h] and dt_bias[h], compute per-head
            for h_idx in range(8):
                # Extract scalars a[b, h], dt_bias[h]
                a_scalar = a[b_idx, 0, h_idx].float().item()    # scalar float
                dt_bias_scalar = dt_bias[h_idx].float().item()  # scalar float

                # Compute softplus(a + dt_bias) with Triton
                x_in = torch.tensor([a_scalar + dt_bias_scalar], dtype=torch.float32, device=device)
                out_sp = torch.empty(1, dtype=torch.float32, device=device)
                softplus_kernel[(1,)](x_in, out_sp, N=1)
                softplus_val = out_sp[0]  # scalar

                # Compute exp(A_log[h])
                A_log_h = A_log[h_idx].float().item()
                exp_A = torch.empty(1, dtype=torch.float32, device=device)
                exp_kernel[(1,)](torch.tensor([A_log_h], dtype=torch.float32, device=device), exp_A, N=1)
                exp_A_val = exp_A[0]  # scalar

                # g = exp(-exp(A_log) * softplus(a + dt_bias))
                g_val = torch.exp(-exp_A_val * softplus_val).item()  # scalar float

                # beta = sigmoid(b[b, h])
                b_scalar = b[b_idx, 0, h_idx].float().item()
                out_sig = torch.empty(1, dtype=torch.float32, device=device)
                sigmoid_kernel[(1,)](torch.tensor([b_scalar], dtype=torch.float32, device=device), out_sig, N=1)
                beta_val = out_sig[0]  # scalar float

                # Vectors for this (b, h)
                q_h = q_exp[b_idx, h_idx].float()       # [128]
                k_h = k_exp[b_idx, h_idx].float()       # [128]
                v_h = v_exp[b_idx, h_idx].float()       # [128]

                # state_old[b, h] is [V, K], float32, contiguous
                old_state = state[b_idx, h_idx].contiguous()  # [128, 128] float32

                # Compute old_v = k_h @ old_state using Triton matvec
                K_int = 128
                V_int = 128
                old_v = torch.empty((V_int,), dtype=torch.float32, device=device)
                x_ptr = old_state.view(K_int, V_int)          # [K,V]
                x_flat = x_ptr.reshape(-1)                    # [K*V]
                k_flat = k_h.reshape(-1)                     # [K]
                matvec_kernel[(1,)](x_flat, k_flat, old_v, K_int, V_int, 32, 64)

                # new_v = beta * v_h + (1 - beta) * old_v
                new_v = (beta_val * v_h) + (1.0 - beta_val) * old_v  # [128]

                # state_remove = k_h @ old_v
                state_remove = torch.empty((V_int,), dtype=torch.float32, device=device)
                matvec_kernel[(1,)](old_v, k_flat, state_remove, K_int, V_int, 32, 64)

                # state_update = k_h @ new_v
                state_update = torch.empty((V_int,), dtype=torch.float32, device=device)
                matvec_kernel[(1,)](new_v, k_flat, state_update, K_int, V_int, 32, 64)

                # Update new_state[b, h] = old_state * g - state_remove + state_update (broadcast along K)
                new_state[b_idx, h_idx] = (old_state * g_val) - state_remove.unsqueeze(1) + state_update.unsqueeze(1)

                # Compute output scalar: scale * (q_h @ (state_update - state_remove + g * old_state))
                combined = state_update - state_remove + (old_state * g_val).reshape(128)
                out_scalar_buf = torch.empty(1, dtype=torch.float32, device=device)
                dot_kernel[(128,)](q_h.reshape(-1), combined.reshape(-1), out_scalar_buf)
                # Apply scale
                if scale is None or scale == 0.0:
                    scale_val = 1.0 / math.sqrt(K)
                else:
                    scale_val = float(scale)
                out_scalar = out_scalar_buf[0] * scale_val

                # Store into output[b, 0, h, 0] as bfloat16 (no torch.dot or scalar creation in host beyond write)
                output[b_idx, 0, h_idx, 0] = torch.tensor(out_scalar, dtype=torch.bfloat16, device=device)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
