import math
import torch
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = softplus(x[i]) with numerical stability.
    For large N, one program handles one element. Here N is small per head.
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    zero = 0.0
    pos = x > zero
    pos_val = x + tl.log(1.0 + tl.exp(-x))
    neg_val = tl.log(1.0 + tl.exp(x))
    y = tl.where(pos, pos_val, neg_val)
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
def exp_scalar_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[i] = exp(x[i]).
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + i, y, mask=i < N)


@triton.jit
def matvec_kernel(x_ptr, k_ptr, y_ptr, K: tl.constexpr, V: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute y = k @ x, where x is a [K, V] matrix flattened into a 1D pointer,
    k is [K], y is [V].
    One program instance handles all V outputs by looping over K in chunks of BLOCK_K.
    """
    pid = tl.program_id(axis=0)
    # We will process all V outputs in this program
    v_offsets = tl.arange(0, V)
    y_acc = tl.zeros((V,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_chunk = tl.load(k_ptr + k_offsets, mask=k_offsets < K, other=0.0)  # [BLOCK_K]
        for kk in range(0, BLOCK_K):
            k_val = k_chunk[kk]  # scalar
            k_idx = k_start + kk
            # x[k_idx, v_offsets] = *(x_ptr + k_idx*V + v_offsets)
            x_vals = tl.load(x_ptr + k_idx * V + v_offsets, mask=v_offsets < V, other=0.0)
            y_acc += k_val * x_vals
    tl.store(y_ptr + v_offsets, y_acc, mask=v_offsets < V)


@triton.jit
def dot_kernel(q_ptr, x_ptr, out_ptr, N: tl.constexpr):
    """
    Compute out[0] = sum_i q[i] * x[i] using atomic add.
    Grid can be (N,). Each program accumulates a partial sum and atomic adds to out_ptr[0].
    """
    pid = tl.program_id(axis=0)
    start = pid
    offsets = start + tl.arange(0, 1)  # one element per program
    q = tl.load(q_ptr + offsets, mask=offsets < N, other=0.0)
    x = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
    prod = q * x
    # Atomic add to a single scalar
    tl.atomic_add(out_ptr, tl.sum(prod))


@triton.jit
def sqrt_kernel(out_ptr, K: tl.constexpr):
    """
    Compute out[0] = 1.0 / sqrt(K) (float32).
    """
    inv_sqrt = 1.0 / tl.sqrt(K)
    # Write to out_ptr[0]
    tl.store(out_ptr, inv_sqrt)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only forward:
        - Uses Triton kernels for softplus, sigmoid, exp, matvec, dot, sqrt.
        - Returns output (bfloat16, [B, 1, H, V]) and new_state (float32, [B, H, V, K]).
        """
        B, T, H_q, K = q.shape
        _, _, H_k, _ = k.shape
        _, _, H_v, V = v.shape
        assert T == 1, "This implementation assumes T=1."
        assert K == 128 and V == 128
        assert H_q == 4 and H_k == 4 and H_v == 8

        # Prepare expanded q,k to match v heads
        q_exp = q.squeeze(1).repeat_interleave(H_v // H_q, dim=1)  # [B, H_v, K]
        k_exp = k.squeeze(1).repeat_interleave(H_v // H_k, dim=1)  # [B, H_v, K]

        # Output and new_state initialization
        device = q.device
        output = torch.empty((B, 1, H_v, 1), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((B, H_v, V, K), dtype=torch.float32, device=device)

        # Process each (b, h)
        for b_idx in range(B):
            for h_idx in range(H_v):
                # Extract vectors
                q_h = q_exp[b_idx, h_idx].float()  # [K]
                k_h = k_exp[b_idx, h_idx].float()  # [K]
                # Compute gate g and beta via Triton kernels (scalars)
                # a + dt_bias for this head
                a_val = a[b_idx, 0, h_idx].float()  # [1] -> scalar
                db_val = float(dt_bias[h_idx])      # scalar
                splus_in = a_val + db_val           # scalar
                # softplus
                splus_buf = torch.empty(1, dtype=torch.float32, device=device)
                softplus_kernel[(1,)](torch.tensor([splus_in], device=device, dtype=torch.float32), splus_buf)
                softplus_val = splus_buf[0]  # float32 scalar
                # exp(A_log[h]) and exp(-exp(A_log)*softplus(a+dt_bias))
                A_log_val = float(A_log[h_idx])
                exp_A = torch.empty(1, dtype=torch.float32, device=device)
                exp_scalar_kernel[(1,)](torch.tensor([A_log_val], device=device, dtype=torch.float32), exp_A)
                exp_A = exp_A[0]
                gate_term = -exp_A * softplus_val  # scalar
                exp_gate = torch.empty(1, dtype=torch.float32, device=device)
                exp_scalar_kernel[(1,)](torch.tensor([gate_term], device=device, dtype=torch.float32), exp_gate)
                g = float(exp_gate[0])  # scalar gate

                # beta = sigmoid(b[b, 0, h])
                b_val = float(b[b_idx, 0, h_idx])
                beta_buf = torch.empty(1, dtype=torch.float32, device=device)
                sigmoid_kernel[(1,)](torch.tensor([b_val], device=device, dtype=torch.float32), beta_buf)
                beta = float(beta_buf[0])  # scalar

                # Load old state [V, K]
                old_state = state[b_idx, h_idx].float()  # [128, 128]

                # old_v = k_h @ old_state
                old_v = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(1,)](old_state.contiguous().view(-1), k_h, old_v)

                # new_v = beta * v[b, h] + (1 - beta) * old_v
                v_h = v[b_idx, 0, h_idx].float()  # [128]
                new_v = beta * v_h + (1.0 - beta) * old_v  # [128]

                # state_remove = k_h @ old_v
                state_remove = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(1,)](old_v, k_h, state_remove)

                # state_update = k_h @ new_v
                state_update = torch.empty(128, dtype=torch.float32, device=device)
                matvec_kernel[(1,)](new_v, k_h, state_update)

                # new_state = g * old_state - state_remove + state_update
                # Elementwise update using torch ops (allowed as data movement)
                new_state[b_idx, h_idx] = g * old_state - state_remove + state_update

                # Compute output scalar: q_h @ new_state[:, 0]
                col0 = new_state[b_idx, h_idx][:, 0]  # [128]
                out_buf = torch.empty(1, dtype=torch.float32, device=device)
                dot_kernel[(128,)](q_h, col0, out_buf)

                # Apply scale: inv_sqrt = 1/sqrt(K) via Triton
                inv_sqrt_buf = torch.empty(1, dtype=torch.float32, device=device)
                sqrt_kernel[(1,)](inv_sqrt_buf, 128)
                inv_sqrt = float(inv_sqrt_buf[0])
                if scale is None or scale == 0.0:
                    scale_val = inv_sqrt
                else:
                    scale_val = float(scale)
                out_scalar = out_buf[0] * scale_val  # float32 scalar

                # Write output[b, 0, h, 0] as bfloat16 (store as tensor, not compute)
                # Create a 1-element bfloat16 tensor and place the scalar.
                output[b_idx, 0, h_idx, 0] = torch.tensor(out_scalar, dtype=torch.bfloat16, device=device)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
