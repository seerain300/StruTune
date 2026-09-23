import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_old_v_vec_kernel(
    q_ptr, k_ptr, state_ptr, tmp_ptr,
    B, H, V, K,
    stride_q_bh, stride_q_k,
    stride_k_bh, stride_k_k,
    stride_state_bh_v, stride_state_bh_k,
    stride_tmp_bh_v,
    BLOCK_V: tl.constexpr,
):
    # One program per (b,h)
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Load k[b,h]
    k_off = h * stride_k_bh
    k_vec = tl.load(k_ptr + k_off, mask=tl.arange(0, K) < K, other=0.0)
    # Initialize tmp_old_v_vec
    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_idx < V
        # old_v_vec_tile[K] = sum over K of k[k] * state[k, v]
        old_v_vec_tile = tl.zeros([BLOCK_V], dtype=tl.float32)
        for k_start in range(0, K, 128):
            k_sub = k_start + tl.arange(0, 128)
            k_mask = k_sub < K
            # Load state[k_sub, v_idx] -> shape [128, BLOCK_V]
            state_ptrs = state_ptr + b * stride_state_bh_v + h * stride_state_bh_k + k_sub[:, None] * stride_state_bh_k + v_idx[None, :] * stride_state_bh_v
            state_vals = tl.load(state_ptrs, mask=k_mask[:, None] & v_mask[None, :], other=0.0)
            # sum over k_sub: dot(k_sub, state_vals)
            dot_row = tl.sum(state_vals * k_vec[k_sub][:, None], axis=0)
            old_v_vec_tile += dot_row
        # Store tmp_old_v_vec[b,h,v]
        tmp_ptrs = tmp_ptr + b * stride_tmp_bh_v + h * stride_tmp_bh_v + v_idx * stride_tmp_bh_v
        tl.store(tmp_ptrs, old_v_vec_tile, mask=v_mask)


@triton.jit
def compute_scalars_kernel(
    k_ptr, tmp_ptr, v_ptr, b_ptr, state_in_ptr, state_out_ptr,
    B, H, V, K,
    stride_k_bh, stride_k_k,
    stride_tmp_bh_v,
    stride_v_bh,
    stride_state_in_bh_v, stride_state_in_bh_k,
    stride_state_out_bh_v, stride_state_out_bh_k,
    BLOCK_V: tl.constexpr,
):
    # One program per (b,h)
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Load k[b,h]
    k_off = h * stride_k_bh
    k_vec = tl.load(k_ptr + k_off, mask=tl.arange(0, K) < K, other=0.0)
    # Load beta[b,h]
    beta = tl.load(b_ptr + b * stride_tmp_bh_v + h * stride_tmp_bh_v)
    # Compute state_remove = sum_v k @ tmp_old_v_vec
    state_remove = tl.zeros([1], dtype=tl.float32)
    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_idx < V
        tmp_vals = tl.load(tmp_ptr + b * stride_tmp_bh_v + h * stride_tmp_bh_v + v_idx * stride_tmp_bh_v, mask=v_mask, other=0.0)
        # Compute dot over K for each v_idx: sum_k k[k] * tmp_vals[v]
        # tmp_vals is [BLOCK_V]; k_vec is [K]
        # We need to align shapes for multiplication; since k_vec is [K], we cannot directly multiply; instead, we loop over K tiles to accumulate.
        # To implement, we recompute dot per v (outer) loop; since BLOCK_V is small, it's acceptable.
        # For simplicity, since tmp_vals is [BLOCK_V] per iteration, we need to multiply with k_vec elementwise for each v in the tile.
        # Instead, we sum over K using k_vec and tmp_vals for each v in tile. We'll do it by computing dot = sum_k k[k] * tmp_vals[v] via another loop.
        # However, Triton requires static shapes; better approach: compute per v separately in small loops (BLOCK_V=128 here).
        for vv in range(0, BLOCK_V):
            v_valid = vv < V
            if v_valid:
                dot_k = tl.zeros([1], dtype=tl.float32)
                for k_start in range(0, K, 128):
                    k_sub = k_start + tl.arange(0, 128)
                    k_mask = k_sub < K
                    # Load tmp_old_v_vec[v]
                    # But tmp_vals has the whole tile; we can load tmp_old_v_vec[v] directly as tmp_vals[vv] when v_valid.
                    # Compute dot over K using k_vec and tmp_vals[vv]
                    # We need k_vec[k] * tmp_vals[vv], sum over K. Implement as: for kk in 0..K-1
                    # Since Triton doesn't support Python if on runtime, we can approximate by using masks: we'll recompute by loading tmp_vals[vv].
                    # tmp_scalar = tmp_vals[vv] (guarded by v_valid). We'll set tmp_scalar = 0 if not v_valid.
                    tmp_scalar = tl.zeros([1], dtype=tl.float32)
                    if v_valid:
                        # tmp_scalar = tmp_vals[vv]
                        tmp_scalar = tmp_vals[vv]
                    # dot_k += sum_k k[k] * tmp_scalar
                    # We need to multiply k_vec (length K) by tmp_scalar and sum. Use a tiled sum:
                    for k_sub_start in range(0, K, 128):
                        k_sub = k_sub_start + tl.arange(0, 128)
                        k_mask = k_sub < K
                        # Multiply k_vec with tmp_scalar (scalar) and sum
                        # tl.sum(k_vec * tmp_scalar, axis=0) doesn't exist; multiply each element then sum:
                        # We can broadcast tmp_scalar to [128] and multiply
                        k_sub_vals = tl.load(k_ptr + h * stride_k_bh + k_sub * stride_k_k, mask=k_mask, other=0.0)
                        # But k_ptr points to k[b,h], already loaded above as k_vec. Here we need k_sub_vals of k_vec. We can just use k_vec loaded earlier.
                        # We need to form a vector filled with tmp_scalar. Create it explicitly.
                        # Since k_vec is loaded for this program, we can multiply k_vec segment by tmp_scalar and sum.
                        # However, Triton requires static loops; we'll use a simple loop for K=128.
                        # For clarity, we'll implement a K loop (BLOCK_K=128) to sum. We can do it by loading k segment and multiplying by tmp_scalar.
                        for kk in range(0, 128):
                            # tmp_scalar is scalar; add k_vec[k_sub_start + kk] * tmp_scalar
                            k_idx = k_sub_start + kk
                            k_valid = k_idx < K
                            if k_valid:
                                dot_k += k_vec[k_idx] * tmp_scalar
                    state_remove += dot_k
    # Load v[b,h]
    v_val = tl.load(v_ptr + b * stride_v_bh + h * stride_v_bh)
    # Compute state_update = sum_v k @ (beta * v + (1 - beta) * tmp_old_v_vec)
    state_update = tl.zeros([1], dtype=tl.float32)
    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_idx < V
        tmp_vals = tl.load(tmp_ptr + b * stride_tmp_bh_v + h * stride_tmp_bh_v + v_idx * stride_tmp_bh_v, mask=v_mask, other=0.0)
        # Compute new_v_vec = beta * v + (1 - beta) * tmp_old_v_vec
        new_v_vals = beta * v_val + (1.0 - beta) * tmp_vals
        # state_update += sum_v sum_k k[k] * new_v_vals[v]
        for vv in range(0, BLOCK_V):
            v_valid = vv < V
            if v_valid:
                tmp_scalar_new = new_v_vals[vv]
                dot_k = tl.zeros([1], dtype=tl.float32)
                for k_sub_start in range(0, K, 128):
                    k_sub = k_sub_start + tl.arange(0, 128)
                    k_mask = k_sub < K
                    for kk in range(0, 128):
                        k_idx = k_sub_start + kk
                        k_valid = k_idx < K
                        if k_valid:
                            dot_k += k_vec[k_idx] * tmp_scalar_new
                state_update += dot_k
    # Update new_state = state_in - state_remove + state_update
    # Load old_state[b,h] as [K,V] (we have state_in_ptr)
    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_idx < V
        for k_start in range(0, K, 128):
            k_sub = k_start + tl.arange(0, 128)
            k_mask = k_sub < K
            state_in_ptrs = state_in_ptr + b * stride_state_in_bh_v + h * stride_state_in_bh_k + k_sub[:, None] * stride_state_in_bh_k + v_idx[None, :] * stride_state_in_bh_v
            state_in_vals = tl.load(state_in_ptrs, mask=k_mask[:, None] & v_mask[None, :], other=0.0)
            # new_state is elementwise: state_in_vals - state_remove + state_update
            # state_remove and state_update are scalars; broadcast
            new_state_vals = state_in_vals - state_remove + state_update
            state_out_ptrs = state_out_ptr + b * stride_state_out_bh_v + h * stride_state_out_bh_k + k_sub[:, None] * stride_state_out_bh_k + v_idx[None, :] * stride_state_out_bh_v
            tl.store(state_out_ptrs, new_state_vals, mask=k_mask[:, None] & v_mask[None, :])


@triton.jit
def compute_output_scalar_kernel(
    q_ptr, state_out_ptr, output_ptr,
    B, H, V, K,
    stride_q_bh, stride_q_k,
    stride_state_out_bh_v, stride_state_out_bh_k,
    stride_output_bh_v,
    BLOCK_V: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Load q[b,h]
    q_off = h * stride_q_bh
    q_vec = tl.load(q_ptr + q_off, mask=tl.arange(0, K) < K, other=0.0)
    # Compute output_scalar = scale * (q @ new_state) where new_state is [V,K]
    output_scalar = tl.zeros([1], dtype=tl.float32)
    scale = 1.0 / math.sqrt(K)  # consistent with original code
    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_idx < V
        for k_start in range(0, K, 128):
            k_sub = k_start + tl.arange(0, 128)
            k_mask = k_sub < K
            state_ptrs = state_out_ptr + b * stride_state_out_bh_v + h * stride_state_out_bh_k + k_sub[:, None] * stride_state_out_bh_k + v_idx[None, :] * stride_state_out_bh_v
            state_vals = tl.load(state_ptrs, mask=k_mask[:, None] & v_mask[None, :], other=0.0)
            # q_vec is [K]; state_vals is [128, BLOCK_V]; reduce over K and V
            # Multiply q_vec with state_vals and sum
            prod = state_vals * q_vec[:, None, None]
            # Sum over K and V dims
            # Reduce over axis 0 (K) then axis 1 (V)
            sum_k = tl.sum(prod, axis=0)  # shape [BLOCK_V]
            output_scalar += tl.sum(sum_k, axis=0)
    output_scalar = output_scalar * scale
    # Store output[b,h,V] = output_scalar
    out_ptr = output_ptr + b * stride_output_bh_v + h * stride_output_bh_v + V * stride_output_bh_v
    tl.store(out_ptr, output_scalar)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        """
        Triton-optimized version of the run function. We compute all heavy operations in Triton.
        Returns:
        - output: [B,H,V] in bfloat16
        - new_state: [B,H,V,K] in float32
        """
        device = q.device
        B = q.shape[0]
        H = v.shape[1]  # num_v_heads
        V = v.shape[3]
        K = q.shape[3]

        # Compute g and beta on host in FP32
        # A_log: [H], a: [B,1,H], dt_bias: [H], b: [B,1,H]
        A_log_f = A_log.to(device=device, dtype=torch.float32)
        dt_bias_f = dt_bias.to(device=device, dtype=torch.float32)
        # Compute x = a + dt_bias
        # a is [B,1,H], dt_bias is [H]; sum over dim=1 of a to get [B,H]
        a_f = a.to(device=device, dtype=torch.float32)
        x = a_f.squeeze(1) + dt_bias_f  # [B,H]
        g = torch.exp(-torch.exp(A_log_f) * torch.nn.functional.softplus(x))  # [B,H]
        beta = torch.sigmoid(b.to(device=device, dtype=torch.float32).squeeze(1))  # [B,H]

        # Squeeze dim 1 for q,k,v
        q_s = q.squeeze(1)  # [B,QH,K]
        k_s = k.squeeze(1)  # [B,KH,K]
        v_s = v.squeeze(1)  # [B,VH,V]

        # Prepare tensors
        # state_in is input state [B,H,V,K] float32
        if state is None:
            state_in = torch.zeros(B, H, V, K, dtype=torch.float32, device=device)
        else:
            state_in = state.to(device=device, dtype=torch.float32)

        # Allocate tmp_old_v [B,H,V] float32
        tmp_old_v = torch.empty(B, H, V, dtype=torch.float32, device=device)

        # Allocate new_state_out [B,H,V,K] float32
        new_state_out = torch.empty(B, H, V, K, dtype=torch.float32, device=device)

        # Allocate output [B,H,V] float32, we will cast to bfloat16 later
        output = torch.empty(B, H, V, dtype=torch.float32, device=device)

        # Launch Triton kernels per (b,h)
        grid = (B, H)
        # 1) compute_old_v_vec: tmp_old_v
        compute_old_v_vec_kernel[grid](
            q_s, k_s, state_in, tmp_old_v,
            B, H, V, K,
            q_s.stride(0), q_s.stride(2),
            k_s.stride(0), k_s.stride(2),
            state_in.stride(1), state_in.stride(3),
            tmp_old_v.stride(0),  # stride for b is V, but here we only need stride for pointer arithmetic within Triton; we pass stride along V as 1*V, but Triton expects element strides. Using tmp_old_v.stride(0) which is 1 for contiguous [B,H,V]
            tmp_old_v.stride(0),
        )

        # 2) compute_scalars and update new_state_out
        # We need b_ptr for beta; beta is [B,H], we can pass its pointer. Triton kernel expects beta as [B,H].
        b_ptr = beta  # [B,H]
        compute_scalars_kernel[grid](
            k_s, tmp_old_v, v_s, b_ptr, state_in, new_state_out,
            B, H, V, K,
            k_s.stride(0), k_s.stride(2),
            tmp_old_v.stride(0),
            v_s.stride(0), v_s.stride(2),
            state_in.stride(1), state_in.stride(3),
            new_state_out.stride(1), new_state_out.stride(3),
            V,  # BLOCK_V
        )

        # 3) compute_output_scalar per (b,h) and store into output[b,h,V]
        # We need scale; original code uses 1/sqrt(K) if scale is None
        if scale is None or scale == 0.0:
            scale_val = 1.0 / math.sqrt(K)
        else:
            scale_val = float(scale)
        compute_output_scalar_kernel[grid](
            q_s, new_state_out, output,
            B, H, V, K,
            q_s.stride(0), q_s.stride(2),
            new_state_out.stride(1), new_state_out.stride(3),
            output.stride(0),
            V,  # BLOCK_V
            scale=scale_val,
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, new_state_out


def run(*args):
    return ModelNew()(*args)
