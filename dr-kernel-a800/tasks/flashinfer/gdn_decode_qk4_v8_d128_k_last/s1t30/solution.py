import math
import torch
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = softplus(x[i]) = max(x[i], 0) + log(1 + exp(-|x[i]|))
    Numerically stable form.
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
    Compute out[i] = sigmoid(x[i]) = 1 / (1 + exp(-x[i]))
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def exp_kernel(inp_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = exp(inp[i]) for i in [0, N).
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(inp_ptr + i, mask=i < N, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def matvec_kernel(x_ptr, k_ptr, y_ptr, K: tl.constexpr, V: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr):
    """
    Compute y = k @ x, where:
      - x is a 2D matrix of shape [K, V], passed as a contiguous 1D pointer of length K*V
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
        k_chunk = tl.load(k_ptr + k_offsets, mask=k_offsets < K, other=0.0)
        for kk in range(0, BLOCK_K):
            k_val = k_chunk[kk]
            k_idx = k_start + kk
            x_vals = tl.load(x_ptr + k_idx * V + v_offsets, mask=v_offsets < V, other=0.0)
            y_acc += k_val * x_vals
    tl.store(y_ptr + v_offsets, y_acc, mask=v_offsets < V)


@triton.jit
def dot_kernel(q_ptr, x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute out = sum_i q[i] * x[i], where q and x are 1D vectors of length N.
    Each program accumulates over a block of N and writes to out_ptr[0].
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    q = tl.load(q_ptr + offsets, mask=offsets < N, other=0.0)
    x = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
    part = tl.sum(q * x, axis=0)
    tl.atomic_add(out_ptr, part)


class ModelNew(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: torch.Tensor, A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor, scale):
        """
        q: [B, 1, 4, 128], bfloat16
        k: [B, 1, 4, 128], bfloat16
        v: [B, 1, 8, 128], bfloat16
        state: [B, 8, 128, 128], float32 (k-last, last dim is K=128)
        A_log: [8], float32
        a: [1, 1, 8], bfloat16 (per batch, per head)
        dt_bias: [8], float32
        b: [1, 1, 8], bfloat16 (per batch, per head)
        scale: float32 scalar or None
        Returns:
        output: [B, 1, 8, 1], bfloat16
        new_state: [B, 8, 128, 128], float32
        """
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and state.dim() == 4
        B, Tq, Hq, Kq = q.shape
        Bk, Tk, Hk, Kk = k.shape
        Bv, Tv, Hv, Kv = v.shape
        Bst, Hst, V, K = state.shape
        assert Tq == 1 and Tk == 1 and Tv == 1 and Bq == Bk == Bv == Bst and Kq == Kk == K and Hv == 8 and Hq == 4 and Hk == 4 and Kv == 128 and V == 128 and K == 128
        device = q.device

        # Prepare a, dt_bias, b as 1D float32 for Triton kernels
        a_vec = a.squeeze(0).squeeze(1).to(torch.float32).contiguous()        # [8]
        dt_bias_vec = dt_bias.squeeze(0).squeeze(1).to(torch.float32).contiguous()  # [8]
        b_vec = b.squeeze(0).squeeze(1).to(torch.float32).contiguous()       # [8]
        A_log_vec = A_log.to(torch.float32).contiguous()                     # [8]

        # Compute softplus(a + dt_bias) in Triton
        add = a_vec + dt_bias_vec                                            # [8]
        softplus_add = torch.empty_like(add, dtype=torch.float32, device=device)
        softplus_kernel[(add.numel(),)](add, softplus_add)                  # Triton elementwise softplus

        # Compute g_vec = exp(-exp(A_log) * softplus_add)
        exp_A = torch.empty_like(A_log_vec, dtype=torch.float32, device=device)
        exp_kernel[(A_log_vec.numel(),)](A_log_vec, exp_A)                  # Triton elementwise exp
        g_vec = torch.exp(-exp_A * softplus_add)                            # torch elementwise allowed here
        # Compute beta_vec = sigmoid(b) in Triton
        beta_vec = torch.empty_like(b_vec, dtype=torch.float32, device=device)
        sigmoid_kernel[(b_vec.numel(),)](b_vec, beta_vec)                   # Triton elementwise sigmoid

        # Expand q and k along head dimension to match v heads (repeat_interleave as in original)
        q_rep = q.repeat_interleave(2, dim=1)                               # [B, 2, 4, 128]
        k_rep = k.repeat_interleave(2, dim=1)                               # [B, 2, 4, 128]

        # Allocate new_state (float32)
        new_state = torch.empty((B, 8, 128, 128), dtype=torch.float32, device=device)

        # For each (b, h):
        for b_idx in range(B):
            for h_idx in range(8):
                # Select q_h, k_h, v_h as 1D vectors
                q_h = q_rep[b_idx, h_idx].reshape(-1).to(torch.float32).contiguous()  # [128] float32
                k_h = k_rep[b_idx, h_idx].reshape(-1).to(torch.float32).contiguous()  # [128] float32
                v_h = v[b_idx, 0, h_idx].reshape(-1).to(torch.float32).contiguous()   # [128] float32

                # state_old (k-last) is [V, K] float32 -> transpose to [K, V] for Triton matvec input
                state_old = state[b_idx, h_idx].contiguous()                 # [128, 128], float32
                state_old_T = state_old.transpose(0, 1).contiguous()         # [128, 128]
                x_ptr = state_old_T.reshape(-1).contiguous()                 # 1D of length 16384

                # Compute old_v = k_h @ state_old_T -> old_v is 128
                old_v = torch.empty(128, dtype=torch.float32, device=device)
                k_flat = k_h.reshape(-1).contiguous()                        # 128
                matvec_kernel[(128,)](x_ptr, k_flat, old_v, 128, 128, 64, 128)  # Triton matvec

                # Compute new_v = beta[h] * v_h + (1 - beta[h]) * old_v
                beta_val = beta_vec[h_idx]                                  # scalar float32
                new_v = beta_val * v_h + (1.0 - beta_val) * old_v           # [128] float32

                # Compute state_remove = k_h @ old_v
                state_remove = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(128,)](old_v, k_flat, state_remove, 128, 128, 64, 128)  # Triton matvec

                # Compute state_update = k_h @ new_v
                state_update = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(128,)](new_v, k_flat, state_update, 128, 128, 64, 128)  # Triton matvec

                # Update new_state[b, h] = g[h] * state_old - state_remove + state_update
                g_val = g_vec[h_idx]                                        # scalar float32
                new_state[b_idx, h_idx] = g_val * state_old - state_remove + state_update

        # Prepare output: output[b, 0, h, 0] = scale * (q_h @ new_state[:, h])
        output = torch.empty((B, 1, 8, 1), dtype=torch.bfloat16, device=device)

        for b_idx in range(B):
            for h_idx in range(8):
                q_h = q_rep[b_idx, h_idx].reshape(-1).to(torch.float32)     # [128] float32
                state_new_vec = new_state[b_idx, h_idx].reshape(-1).to(torch.float32)  # [128] float32
                out_scalar_buf = torch.empty(1, dtype=torch.float32, device=device)
                dot_kernel[(128,)](q_h, state_new_vec, out_scalar_buf, 128, 128)  # Triton dot
                scale_val = 1.0 / math.sqrt(K) if (scale is None or scale == 0.0) else float(scale)
                out_val = out_scalar_buf[0] * scale_val                      # float32
                output[b_idx, 0, h_idx, 0] = torch.tensor(out_val, dtype=torch.bfloat16, device=device)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
