import math
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    """
    out[i] = a[i] + b[i]
    """
    pid = tl.program_id(axis=0)
    i = pid
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, a + b)


@triton.jit
def softplus_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    out[i] = softplus(x[i]) = log(1 + exp(x[i]))
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i)
    out = tl.log(1.0 + tl.exp(x))
    tl.store(out_ptr + i, out)


@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, N: tl.constexpr):
    """
    out[i] = sigmoid(x[i]) = 1 / (1 + exp(-x[i]))
    """
    pid = tl.program_id(axis=0)
    i = pid
    x = tl.load(x_ptr + i)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, out)


@triton.jit
def gate_softplus_exp_kernel(A_ptr, add_ptr, softplus_ptr, out_ptr, N: tl.constexpr):
    """
    Compute g[i] = exp(-exp(A[i]) * softplus(add[i]))
    Where add[i] = a[i] + dt[i]
    """
    pid = tl.program_id(axis=0)
    i = pid
    eA = tl.exp(tl.load(A_ptr + i))
    sp = tl.load(softplus_ptr + i)
    g = tl.exp(-eA * sp)
    tl.store(out_ptr + i, g)


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
            k_val = k_chunk[kk]
            k_idx = k_start + kk
            x_vals = tl.load(x_ptr + k_idx * V + v_offsets, mask=v_offsets < V, other=0.0)
            y_acc += k_val * x_vals

    tl.store(y_ptr + v_offsets, y_acc, mask=v_offsets < V)


@triton.jit
def elementwise_update_kernel(old_ptr, g_ptr, rm_ptr, up_ptr, new_ptr,
                              K: tl.constexpr, V: tl.constexpr,
                              BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    For each (i in [0,V), j in [0,K)):
      new[i*K + j] = g * old[i*K + j] - rm[i] + up[i]
    old_ptr: [V*K] contiguous
    rm_ptr: [V] contiguous
    up_ptr: [V] contiguous
    g: scalar, loaded from g_ptr[0]
    new_ptr: [V*K] contiguous
    """
    pid = tl.program_id(axis=0)
    v_start = pid * BLOCK_V
    v_offsets = v_start + tl.arange(0, BLOCK_V)  # [BLOCK_V]
    g_val = tl.load(g_ptr + 0)

    for i in range(0, BLOCK_V):
        vi = v_offsets[i]
        if vi >= V:
            break
        # Load rm[i], up[i]
        rm_i = tl.load(rm_ptr + vi)
        up_i = tl.load(up_ptr + vi)
        # Compute new row for this vi across K
        for j in range(0, K):
            old_val = tl.load(old_ptr + vi * K + j)
            new_val = g_val * old_val - rm_i + up_i
            tl.store(new_ptr + vi * K + j, new_val)


@triton.jit
def dot_kernel(x_ptr, w_ptr, out_ptr, N: tl.constexpr):
    """
    out[0] = sum_i x[i] * w[i]
    Single element output buffer; grid size 1. Iterates over N scalar elements.
    """
    pid = tl.program_id(axis=0)
    acc = 0.0
    for i in range(0, N):
        acc += tl.load(x_ptr + i) * tl.load(w_ptr + i)
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-only implementation:
        - Compute gates g and beta
        - Compute matvecs in Triton
        - Update state in Triton
        - Compute output scalar via Triton dot
        Return: (output [B, 1, H, 1] bfloat16), new_state [B, H, V, K] float32
        """
        # Extract shapes
        B, T, num_q_heads, K = q.shape
        assert T == 1, "T must be 1"
        assert num_q_heads == 4
        _, _, num_k_heads, _ = k.shape
        assert num_k_heads == 4
        _, _, num_v_heads, V = v.shape
        assert num_v_heads == 8
        assert K == 128 and V == 128

        # Squeeze T dimension and repeat q/k along head dimension as original code does
        q_exp = q.squeeze(1)  # [B, 4, 128]
        k_exp = k.squeeze(1)  # [B, 4, 128]
        v_exp = v.squeeze(1)  # [B, 8, 128]

        # Prepare inputs for Triton
        device = q.device
        a_f32 = a.squeeze(1).squeeze(1).to(torch.float32).contiguous()       # [8]
        dt_f32 = dt_bias.to(torch.float32).contiguous()                      # [8]
        b_f32 = b.squeeze(1).squeeze(1).to(torch.float32).contiguous()       # [8]
        A_log_f32 = A_log.to(torch.float32).contiguous()                     # [8]
        state_f32 = state.to(torch.float32).contiguous()                     # [B, 8, 128, 128]
        K_int = K
        V_int = V

        # Compute g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
        add_res = torch.empty(8, dtype=torch.float32, device=device)         # [8]
        softplus_buf = torch.empty(8, dtype=torch.float32, device=device)    # [8]
        beta_buf = torch.empty(8, dtype=torch.float32, device=device)        # [8]
        g_buf = torch.empty(8, dtype=torch.float32, device=device)           # [8]

        # Triton kernels: add, softplus, sigmoid, gate
        add_kernel[(8,)](a_f32, dt_f32, add_res, 8)
        softplus_kernel[(8,)](add_res, softplus_buf, 8)
        sigmoid_kernel[(8,)](b_f32, beta_buf, 8)
        # gate = exp(-exp(A_log) * softplus(a+dt))
        # Note: Triton has tl.exp, but to keep kernels simple, use torch for scalar operations:
        eA = torch.exp(A_log_f32)  # [8]
        # Launch Triton kernel gate_softplus_exp_kernel
        gate_softplus_exp_kernel[(8,)](A_log_f32, add_res, softplus_buf, g_buf, 8)

        # Compute q_h for each head h: q_exp[b, h] -> [128]
        # We need B and H loops: B is batch size; H = 8
        # Prepare output and new_state
        B_size = q_exp.shape[0]
        H = 8
        output = torch.empty((B_size, 1, H, 1), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((B_size, H, V, K), dtype=torch.float32, device=device)

        # For each batch b, heads h in [0..7], compute everything in Triton
        for b_idx in range(B_size):
            # Load q_exp[b], k_exp[b], v_exp[b]
            q_h = q_exp[b_idx]  # [4, 128]
            k_h = k_exp[b_idx]  # [4, 128]
            v_h = v_exp[b_idx]  # [8, 128]
            # old state: [8, 128, 128] -> we need state[b, h] per h
            # Loop h to compute matvecs and update
            for h_idx in range(H):
                # Load k vector for this h from k_h: k_h[h] -> [128]
                k_vec = k_h[h_idx]  # [128], float32
                # Load q_h vector: q_h[h] -> [128] for our expanded logic, we use q_h[0] since original code repeats:
                # Original code repeats q,k to match v heads (num_v_heads=8 from num_q_heads=4), but here num_q_heads==num_k_heads==4.
                # To match original semantics: use q_h[0] (head 0) for head 0, q_h[1] for head 1, etc.
                q_vec = q_h[h_idx]  # [128], float32

                # Load old state: state[b, h] as [V, K]
                old_state = state_f32[b_idx, h_idx]  # [128, 128], contiguous
                # Cast k_vec, q_vec, v_h[h] to float32 1D
                k_vec_f32 = k_vec.to(torch.float32).contiguous()               # [128]
                q_vec_f32 = q_vec.to(torch.float32).contiguous()               # [128]
                v_vec_f32 = v_h[h_idx].to(torch.float32).contiguous()          # [128]

                # Compute old_v = k @ old_state (x is [K, V])
                old_state_flat = old_state.view(K_int * V_int).contiguous()    # [K*V], but we want x as [K, V]: make it [K, V] then flatten not needed; we pass [K, V] view
                # We need to pass [K, V] as a 1D contiguous: create a contiguous view
                x_ptr = old_state.contiguous().view(K_int, V_int).contiguous().reshape(K_int * V_int)
                # Allocate y_old_v
                y_old_v = torch.empty(V_int, dtype=torch.float32, device=device)
                matvec_kernel[(1,)](x_ptr, k_vec_f32, y_old_v, K_int, V_int, 128, 128)  # launch 1 program; N is constexpr so this is fine

                # Compute new_v = beta[h] * v_vec + (1 - beta[h]) * old_v
                beta_val = beta_buf[h_idx]
                g_val = g_buf[h_idx]
                new_v = (beta_val * v_vec_f32) + ((1.0 - beta_val) * y_old_v)

                # Compute state_remove = k @ old_v (matvec)
                rm_vec = torch.empty(V_int, dtype=torch.float32, device=device)
                matvec_kernel[(1,)](y_old_v, k_vec_f32, rm_vec, 128, 128, 128, 128)  # old_v is [V]

                # Compute state_update = k @ new_v (matvec)
                up_vec = torch.empty(V_int, dtype=torch.float32, device=device)
                matvec_kernel[(1,)](new_v, k_vec_f32, up_vec, 128, 128, 128, 128)  # new_v is [V]

                # Update new_state[b, h, :, :] = g * old_state - rm + up
                old_flat = old_state.contiguous().view(V_int * K_int)        # [V*K]
                new_flat = torch.empty(V_int * K_int, dtype=torch.float32, device=device)
                # elementwise_update_kernel: old_ptr points to old_flat; g is scalar g_val; rm_ptr, up_ptr, new_ptr as allocated
                elementwise_update_kernel[(V_int,)](
                    old_flat, torch.tensor([g_val], dtype=torch.float32, device=device), rm_vec, up_vec, new_flat,
                    K_int, V_int, 128, 128
                )
                # Reshape new_flat back to [V, K]
                new_state[b_idx, h_idx] = new_flat.view(V_int, K_int)  # in-place: torch.copy_ to assign

                # Compute output scalar: scale * (q_h @ sum_j new_state_col0)
                # Sum over K to get column0 vector
                col0 = torch.empty(V_int, dtype=torch.float32, device=device)
                # We need to extract column 0 from new_state[b, h]; since we just updated new_state[b,h], read it:
                col0 = new_state[b_idx, h_idx].index_select(1, 0)  # select j=0 across K
                # Now dot q_vec with col0
                out_scalar_buf = torch.empty(1, dtype=torch.float32, device=device)
                dot_kernel[(V_int,)](q_vec_f32, col0, out_scalar_buf)

                # Apply scale
                if scale is None or scale == 0.0:
                    scale_val = 1.0 / math.sqrt(K_int)
                else:
                    scale_val = float(scale)
                out_val = out_scalar_buf[0] * scale_val

                # Store output[b, 0, h, 0] as bfloat16
                output[b_idx, 0, h_idx, 0] = torch.tensor(out_val, dtype=torch.bfloat16, device=device)

        return output, new_state


def run(*args):
    return ModelNew()(*args)
