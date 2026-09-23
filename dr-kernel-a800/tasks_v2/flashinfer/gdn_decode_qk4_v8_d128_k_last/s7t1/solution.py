import torch
import math
import triton
import triton.language as tl


# Kernel: compute g and beta for each (b,h) using Triton
@triton.jit
def _gate_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr, g_out_ptr, beta_out_ptr,
                            NUM_HEADS: tl.constexpr):
    b_idx = tl.program_id(0)  # one program per batch
    # Loop over heads h=0..NUM_HEADS-1
    for h in range(NUM_HEADS):
        # Load scalars
        a_val = tl.load(a_ptr + b_idx * NUM_HEADS + h)                  # [B, 1, H] -> [B,H], here H=8
        dt_val = tl.load(dt_bias_ptr + h)                               # [H]
        A_val = tl.load(A_log_ptr + h)                                  # [H]
        b_val = tl.load(b_ptr + b_idx * NUM_HEADS + h)                  # [B, 1, H]

        # Compute softplus(a + dt) and sigmoid(b)
        x = a_val + dt_val
        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(x))
        g = tl.exp(-tl.exp(A_val) * sp)                                 # g = exp(-exp(A) * softplus(a+dt))
        sig = 1.0 / (1.0 + tl.exp(-b_val))                              # sigmoid(b)
        # Store results
        tl.store(g_out_ptr + b_idx * NUM_HEADS + h, g)
        tl.store(beta_out_ptr + b_idx * NUM_HEADS + h, sig)


# Kernel: compute old_v = k_h @ state_h (reduction over K)
# Inputs: k_h_ptr [K], state_h_ptr [V,K], out_oldv_ptr [K]
@triton.jit
def _vec_matmul_kernel(k_ptr, state_ptr, out_ptr,
                       V: tl.constexpr, K: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    # One program handles one reduction for a vector k against matrix state
    # We iterate over K in chunks and accumulate into a vector out
    out = tl.zeros((K,), dtype=tl.float32)
    for off in range(0, K, BLOCK_K):
        k_chunk = off + tl.arange(0, BLOCK_K)
        mask = k_chunk < K
        # k_chunk values
        k_vals = tl.load(k_ptr + k_chunk, mask=mask, other=0.0)         # [BLOCK_K]
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        # Reduce over V: for each v in [0..V), load state[:, k_chunk] and dot
        for v in range(0, V):
            # state[v, k_chunk] -> vector of length BLOCK_K
            state_vec = tl.load(state_ptr + v * K + k_chunk, mask=mask, other=0.0)  # [BLOCK_K]
            acc += k_vals * state_vec
        # Sum acc across BLOCK_K to scalar and add to out
        out += tl.sum(acc, axis=0)
    # Store out
    for i in range(K):
        tl.store(out_ptr + i, out[i])


# Kernel: elementwise new_v = beta * v_h + (1 - beta) * old_v
@triton.jit
def _elementwise_newv_kernel(beta, oldv_ptr, v_ptr, newv_ptr,
                             K: tl.constexpr, BLOCK_K: tl.constexpr):
    for off in range(0, K, BLOCK_K):
        idx = off + tl.arange(0, BLOCK_K)
        mask = idx < K
        oldv = tl.load(oldv_ptr + idx, mask=mask, other=0.0)
        v = tl.load(v_ptr + idx, mask=mask, other=0.0)
        newv = beta * v + (1.0 - beta) * oldv
        tl.store(newv_ptr + idx, newv, mask=mask)


# Kernel: compute scalar k_h @ oldv (reduction over K)
@triton.jit
def _scalar_dot_kernel(k_ptr, vec_ptr, out_ptr,
                       K: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, K, BLOCK_K):
        idx = off + tl.arange(0, BLOCK_K)
        mask = idx < K
        k_vals = tl.load(k_ptr + idx, mask=mask, other=0.0)             # [BLOCK_K]
        vec_vals = tl.load(vec_ptr + idx, mask=mask, other=0.0)         # [BLOCK_K]
        acc += tl.sum(k_vals * vec_vals, axis=0)
    tl.store(out_ptr, acc)


# Kernel: elementwise update new_state = g * state - remove + update
# Inputs: state_ptr [V,K], new_state_ptr [V,K], g_val (scalar), beta_val (unused, but passed), remove (scalar), update (scalar)
@triton.jit
def _elementwise_update_kernel(state_ptr, new_state_ptr, g_val, remove, update,
                                V: tl.constexpr, K: tl.constexpr,
                                BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr):
    # Loop over tiles of V and K
    for v_off in range(0, V, BLOCK_V):
        for k_off in range(0, K, BLOCK_K):
            v_idx = v_off + tl.arange(0, BLOCK_V)
            k_idx = k_off + tl.arange(0, BLOCK_K)
            vmask = v_idx < V
            kmask = k_idx < K
            # Load state tile [BLOCK_V, BLOCK_K]
            state_tile = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
            for i in range(BLOCK_V):
                row_v = v_idx[i]
                if vmask[i]:
                    # Load row row_v across k_idx
                    state_row = tl.load(state_ptr + row_v * K + k_idx, mask=kmask, other=0.0)  # [BLOCK_K]
                    state_tile[i, :] = state_row
            # Compute new_state tile: g * state - remove + update, broadcast scalars
            new_state_tile = g_val * state_tile - remove + update
            # Store
            for i in range(BLOCK_V):
                row_v = v_idx[i]
                if vmask[i]:
                    for j in range(BLOCK_K):
                        k_pos = k_idx[j]
                        if kmask[j]:
                            # Store to new_state_ptr[row_v, k_pos]
                            tl.store(new_state_ptr + row_v * K + k_pos, new_state_tile[i, j])


# Kernel: compute output scalar = scale * sum_v sum_k q[k] * new_state[v,k]
@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr, scale,
                          V: tl.constexpr, K: tl.constexpr,
                          BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros((), dtype=tl.float32)
    for v_off in range(0, V, BLOCK_V):
        for k_off in range(0, K, BLOCK_K):
            v_idx = v_off + tl.arange(0, BLOCK_V)
            k_idx = k_off + tl.arange(0, BLOCK_K)
            vmask = v_idx < V
            kmask = k_idx < K
            # Load q[k_idx]
            q_vec = tl.load(q_ptr + k_idx, mask=kmask, other=0.0)        # [BLOCK_K]
            # Load new_state[v_idx, k_idx] tile -> [BLOCK_V, BLOCK_K]
            new_state_tile = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
            for i in range(BLOCK_V):
                row_v = v_idx[i]
                if vmask[i]:
                    ns_row = tl.load(new_state_ptr + row_v * K + k_idx, mask=kmask, other=0.0)  # [BLOCK_K]
                    new_state_tile[i, :] = ns_row
            # Accumulate: sum over k of q[k] * new_state[v,k] for each v
            for i in range(BLOCK_V):
                for j in range(BLOCK_K):
                    if vmask[i] and kmask[j]:
                        acc += q_vec[j] * new_state_tile[i, j]
    total = acc
    # Apply scale
    total = total * scale
    tl.store(out_ptr, total)


class ModelNew(torch.nn.Module):
    def __init__(self, K=128, V=128, NUM_HEADS=8, NUM_Q_HEADS=4, NUM_K_HEADS=4, NUM_V_HEADS=8):
        super().__init__()
        self.K = K
        self.V = V
        self.NUM_HEADS = NUM_HEADS
        self.NUM_Q_HEADS = NUM_Q_HEADS
        self.NUM_K_HEADS = NUM_K_HEADS
        self.NUM_V_HEADS = NUM_V_HEADS

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Shapes as per original:
        # q: [B, 1, num_q_heads, K] -> [B, 1, 4, 128]
        # k: [B, 1, num_k_heads, K] -> [B, 1, 4, 128]
        # v: [B, 1, num_v_heads, V] -> [B, 1, 8, 128]
        # state: [B, num_v_heads, V, K] -> [B, 8, 128, 128]
        B = q.shape[0]
        device = q.device

        # Compute g and beta via Triton kernels
        # Prepare pointers: a, dt_bias, b are [B,1,H]; we pass flattened [B*H] and reconstruct index
        # But we need a[:, h] where h varies. We'll create per-(b,h) scalars by launching grid=(B,)
        # Define output buffers
        g_out = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)
        beta_out = torch.empty((B, self.NUM_HEADS), dtype=torch.float32, device=device)

        # We need a_ptr of shape [B, H]; original a is [B,1,H], we can unsqueeze to [B,H] by duplicating dim1
        # But a is [B,1,H]; we will pass a_ptr as a.view(B,H)
        # Similarly for b.
        # Create a contiguous [B,H] views:
        # a is [B,1,H] -> we can unsqueeze dim1 to [B,1,H], but we need [B,H]. We can use a.squeeze(1) to [B,H] by using original shape.
        # In PyTorch, a is [B,1,H], so to get [B,H], we can index as a[:, 0, :]. But we'll keep original a[b,0,h].
        # We'll create a_per_bh tensor by slicing: a_per_bh[b,h] = a[b,0,h].
        a_per_bh = a[:, 0, :]                              # [B, H]
        b_per_bh = b[:, 0, :]                              # [B, H]

        # Launch gate kernel: grid over batch
        _gate_g_and_beta_kernel[(B,)](
            a_per_bh, dt_bias, b_per_bh, A_log, g_out, beta_out,
            NUM_HEADS=self.NUM_HEADS
        )

        # Now, for each batch b, compute matmuls and updates per head h
        output_bf16 = torch.empty((B, 1, self.NUM_HEADS, self.V), dtype=torch.bfloat16, device=device)
        new_state = torch.empty((B, self.NUM_HEADS, self.V, self.K), dtype=torch.float32, device=device)

        # Loop over batch, each Triton program handles all heads sequentially
        for b_idx in range(B):
            # Compute output and new_state for each head
            for h_idx in range(self.NUM_HEADS):
                # Load tensors for this (b,h)
                # q_h, k_h, v_h, state_h
                # q: [B,1,4,128] -> q[b,0,:] = [4,128]
                q_h = q[b_idx, 0, :].float().contiguous()            # [4, 128]
                k_h = k[b_idx, 0, :].float().contiguous()            # [4, 128]
                v_h = v[b_idx, 0, :].float().contiguous()            # [8, 128]
                state_h = state[b_idx, h_idx].float().contiguous()   # [128, 128]

                # 1) old_v = k_h @ state_h (reduction over K)
                oldv = torch.empty((self.K,), dtype=torch.float32, device=device)
                _vec_matmul_kernel[(1,)](
                    k_h, state_h, oldv,
                    V=self.V, K=self.K, BLOCK_K=128
                )

                # 2) new_v = beta[h] * v_h + (1 - beta[h]) * old_v
                beta_val = beta_out[b_idx, h_idx]                    # scalar
                newv = torch.empty((self.K,), dtype=torch.float32, device=device)
                _elementwise_newv_kernel[(1,)](
                    beta_val, oldv, v_h, newv,
                    K=self.K, BLOCK_K=128
                )

                # 3) state_remove and state_update scalars
                remove = torch.empty((), dtype=torch.float32, device=device)
                update = torch.empty((), dtype=torch.float32, device=device)
                _scalar_dot_kernel[(1,)](
                    k_h, oldv, remove,
                    K=self.K, BLOCK_K=128
                )
                _scalar_dot_kernel[(1,)](
                    k_h, newv, update,
                    K=self.K, BLOCK_K=128
                )

                # 4) Update new_state = g * state - remove + update
                g_val = g_out[b_idx, h_idx]
                new_state[b_idx, h_idx] = torch.empty_like(state_h, dtype=torch.float32, device=device)
                _elementwise_update_kernel[(1,)](
                    state_h, new_state[b_idx, h_idx], g_val, remove.item(), update.item(),
                    V=self.V, K=self.K, BLOCK_V=128, BLOCK_K=128
                )

                # 5) Output scalar = scale * (q_h @ new_state_h)
                # We need q_h as 1D [K]
                q_h_vec = q_h[:, 0].contiguous()                     # [128]
                out_scalar = torch.empty((), dtype=torch.float32, device=device)
                _output_scalar_kernel[(1,)](
                    q_h_vec, new_state[b_idx, h_idx], out_scalar, scale,
                    V=self.V, K=self.K, BLOCK_V=128, BLOCK_K=128
                )
                # Store output
                output_bf16[b_idx, 0, h_idx] = out_scalar.to(torch.bfloat16)

        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
