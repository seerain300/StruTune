import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    B, H,
    stride_A, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
    stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars and compute g, beta
    A = tl.load(A_log_ptr + h_idx * stride_A).to(tl.float32)
    a = tl.load(a_ptr + b_idx * stride_a_b + h_idx * stride_a_h).to(tl.float32)
    dt = tl.load(dt_bias_ptr + h_idx * stride_dt).to(tl.float32)
    bb = tl.load(b_ptr + b_idx * stride_b_b + h_idx * stride_b_h).to(tl.float32)

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a + dt))
    g_val = tl.exp(-tl.exp(A) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-bb))

    tl.store(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h, g_val)
    tl.store(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h, beta_val)


@triton.jit
def kernel_tmp_old_v(
    k_ptr, state_ptr, tmp_ptr,
    B, H, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_t_b, stride_t_h,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Accumulate dot(k[b, h], state[b, h]) across V, K
    acc = tl.zeros((), dtype=tl.float32)
    # Iterate over K in blocks
    BLOCK = 128
    for k_off in range(0, K, BLOCK):
        k_idx = k_off + tl.arange(0, BLOCK)
        mask_k = k_idx < K
        # k[b, h, k_idx]
        k_vec = tl.load(
            k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_idx * stride_k_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        # For each v in [0, V), load state[b, h, v, k_idx] and reduce
        for v in range(0, V):
            s = tl.load(
                state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v * stride_s_v + k_idx * stride_s_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(k_vec * s, axis=0)

    tl.store(tmp_ptr + b_idx * stride_t_b + h_idx * stride_t_h, acc)


@triton.jit
def kernel_update_and_output(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, tmp_ptr,
    out_ptr, new_state_ptr,
    B, H, V, K,
    stride_q_b, stride_q_h, stride_q_k,
    stride_k_b, stride_k_h, stride_k_k,
    stride_v_b, stride_v_h, stride_v_v,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_g_b, stride_g_h,
    stride_beta_b, stride_beta_h,
    stride_out_b, stride_out_h,
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
    scale,  # float32 scalar
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars
    g_val = tl.load(g_ptr + b_idx * stride_g_b + h_idx * stride_g_h).to(tl.float32)
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)
    tmp_val = tl.load(tmp_ptr + b_idx * stride_out_b + h_idx * stride_out_h).to(tl.float32)  # tmp_old_v[b, h]

    # Compute new_state[b, h] elementwise: shape [V, K]
    for v in range(0, V):
        old_v = 0.0
        for k_off in range(0, K, 128):
            kk = k_off + tl.arange(0, 128)
            mask_k = kk < K
            # old_state[b, h, v, kk] = dot(k[b, h], state[b, h, v, :])
            # We need k @ state[:, v] across K
            # But state is [V, K], so compute directly via k dot state reduced by V index
            # Instead, load k[b, h, kk], and compute v_old = dot(kk, state[v, :]) by iterating kk in chunks
            v_old_chunk = 0.0
            for kk_off in range(0, K, 128):
                kk2 = kk_off + tl.arange(0, 128)
                mask_k2 = kk2 < K
                k_vec2 = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk2 * stride_k_k, mask=mask_k2, other=0.0).to(tl.float32)
                # state[b, h, v, kk2]
                s_vec = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v * stride_s_v + kk2 * stride_s_k, mask=mask_k2, other=0.0).to(tl.float32)
                v_old_chunk += tl.sum(k_vec2 * s_vec, axis=0)
            old_v = v_old_chunk

        # Update new_state for this v across K
        new_v_vec = tl.zeros((128,), dtype=tl.float32)
        for k_off in range(0, K, 128):
            kk = k_off + tl.arange(0, 128)
            mask_k = kk < K
            # Load q[b, h, kk] and k[b, h, kk]
            q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + kk * stride_q_k, mask=mask_k, other=0.0).to(tl.float32)
            k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk * stride_k_k, mask=mask_k, other=0.0).to(tl.float32)

            # Load v[b, h, v] (scalar), and old_v (scalar), beta (scalar), g (scalar), tmp (scalar)
            v_elem = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + v * stride_v_v).to(tl.float32)
            # new_v = beta * v + (1 - beta) * (tmp + old_v)
            new_v = beta_val * v_elem + (1.0 - beta_val) * (tmp_val + old_v)
            # new_state[b, h, v, kk] = g * old_state[b, h, v, kk] - (k @ old_state[:, v]) + (k @ new_v)
            # We need to compute (k @ old_state[:, v]) and (k @ new_v)
            # old_state[:, v] is loaded per kk chunk via k @ state reduced over V, but here we only have v. We'll compute it by loading state and reducing.
            old_dot = 0.0
            for kk_off in range(0, K, 128):
                kk2 = kk_off + tl.arange(0, 128)
                mask_k2 = kk2 < K
                k_vec2 = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk2 * stride_k_k, mask=mask_k2, other=0.0).to(tl.float32)
                # state[b, h, v, kk2]
                s_vec = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v * stride_s_v + kk2 * stride_s_k, mask=mask_k2, other=0.0).to(tl.float32)
                old_dot += tl.sum(k_vec2 * s_vec, axis=0)

            new_dot = 0.0
            for kk_off in range(0, K, 128):
                kk2 = kk_off + tl.arange(0, 128)
                mask_k2 = kk2 < K
                k_vec2 = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk2 * stride_k_k, mask=mask_k2, other=0.0).to(tl.float32)
                new_dot += tl.sum(k_vec2 * new_v, axis=0)

            # new_state = g * old_state - old_dot + new_dot
            # We don't have old_state, but we can reconstruct elementwise: new_state[b, h, v, kk] is computed from v_old and v_elem.
            # However, the previous loop already computed old_v scalar. Since state is updated by the kernel, we compute the updated values for each kk.
            # Here we recompute state contributions per kk by reading current state (after update), but Triton kernel doesn't have visibility of updated state; we need to store new_state out.
            # Approach: directly compute new_state as above using old_v and new_v, and store into new_state_ptr. We'll re-load q, k, v, and state to do that.

            # To simplify and avoid confusion, we compute new_state for each kk chunk by re-deriving contributions:
            # new_state[kk] = g * old_state[kk] - old_dot + new_dot
            # We don't have old_state, so we use the algebra: new_state = g * (state - (k @ state_old)) + (k @ new_v) - (k @ old_v)
            # But since we don't have state_old, we can implement by directly loading state and re-deriving contributions. Given complexity, we instead compute the algebra using loaded vectors.

            # Compute per-kk contribution: derive from loaded q,k,v, and beta,g,tmp
            # We will compute new_state for each kk chunk by reconstructing contributions. However, it's clearer to compute per kk:
            # For each kk in the chunk, new_state[b,h,v,kk] = g * (state_old[b,h,v,kk] - (k @ state_old[:,v])) + (k @ new_v)
            # Since we don't have state_old, we instead compute new_state via the update rule:
            # state_new = g * state_old + (k @ new_v) - (k @ old_v)
            # We can compute k @ new_v and k @ old_v as scalars; however, we need per-kk state_old. Triton kernel cannot access past updates.

            # Therefore, we will compute new_state for each kk by loading state[b,h,v,kk] and then:
            # new_state[b,h,v,kk] = g * state[b,h,v,kk] - (k @ state[:,v]) + (k @ new_v)
            # But we don't know k @ state[:,v] without knowing state_old. Given the update logic, the correct expression is:
            # state_new = g * state_old + (k @ new_v) - (k @ old_v)
            # We cannot derive state_old; thus we need to store new_state and update in a way that respects state history. Triton kernel cannot carry state across.

            # To resolve, we will compute new_state per kk by re-deriving from q,k,v, and beta,g,tmp. However, the expression still depends on state_old. The correct approach is to implement the update by:
            # 1) Compute old_v = dot(k, state[v, :])
            # 2) Compute new_v = beta*v + (1-beta)*(tmp + old_v)
            # 3) Compute new_state[v, kk] = g * state[v, kk] - (k @ state[v, :]) + (k @ new_v)
            # Note: (k @ state[v, :]) is the scalar old_v; (k @ new_v) is a scalar. Thus:
            # new_state[v, kk] = g * state[v, kk] - old_v + (k @ new_v)
            # We can compute (k @ new_v) once per (b,h).
            # However, Triton scalar operations are limited. To keep it simple and correct, we compute new_state for each kk chunk by loading state and using above formula.

            # Compute (k @ new_v) scalar for this (b,h)
            k_new_v = 0.0
            for kk_off2 in range(0, K, 128):
                kk2 = kk_off2 + tl.arange(0, 128)
                mask_k2 = kk2 < K
                k_vec2 = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk2 * stride_k_k, mask=mask_k2, other=0.0).to(tl.float32)
                k_new_v += tl.sum(k_vec2 * new_v, axis=0)

            # old_v is already computed above: old_v = dot(k, state[v, :]) scalar for this (b,h)
            # Compute new_state for each kk in the chunk
            for kk_off2 in range(0, K, 128):
                kk2 = kk_off2 + tl.arange(0, 128)
                mask_k2 = kk2 < K
                k_vec2 = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + kk2 * stride_k_k, mask=mask_k2, other=0.0).to(tl.float32)
                s_vec = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v * stride_s_v + kk2 * stride_s_k, mask=mask_k2, other=0.0).to(tl.float32)
                new_s_vec = g_val * s_vec - old_v + k_new_v
                # Store new_state[b,h,v,kk]
                tl.store(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v * stride_ns_v + kk2 * stride_ns_k, new_s_vec, mask=mask_k2)

    # Compute output[b, h] = scale * dot(q[b, h, :], new_state[b, h, :, :])
    out_acc = 0.0
    for k_off in range(0, K, 128):
        kk = k_off + tl.arange(0, 128)
        mask_k = kk < K
        q_vec = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + kk * stride_q_k, mask=mask_k, other=0.0).to(tl.float32)
        # Load new_state[b, h, :, kk] for all v
        # We'll compute the dot by looping v
        for v in range(0, V):
            s_vec = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v * stride_s_v + kk * stride_s_k, mask=mask_k, other=0.0).to(tl.float32)
            out_acc += tl.sum(q_vec * s_vec, axis=0)

    out_val = scale * out_acc
    tl.store(out_ptr + b_idx * stride_out_b + h_idx * stride_out_h, out_val)


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Triton-optimized reference implementation for Gated Delta Net decode.
    Operates on:
      - q: [B, 1, 4, 128] bfloat16
      - k: [B, 1, 4, 128] bfloat16
      - v: [B, 1, 8, 128] bfloat16
      - state: [B, 8, 128, 128] float32 (k-last)
      - A_log: [8] float32
      - a: [1, 1, 8] bfloat16 (we use a[b, h] where b is provided via a_ptr)
      - dt_bias: [8] float32
      - b: [1, 1, 8] bfloat16 (we use b[b, h] where b is provided via b_ptr)
      - scale: float32 scalar or None
    Returns:
      - output: [B, 1, 8] bfloat16
      - new_state: [B, 8, 128, 128] float32
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "Triton requires CUDA tensors"
    B, _, QH, K = q.shape
    _, _, KH, _ = k.shape
    _, _, VH, V = v.shape
    B_state, H, _, K_state = state.shape
    assert B_state == B
    assert K == K_state == 128
    assert V == 128
    assert H == 8
    assert QH == 4 and KH == 4 and VH == 8

    device = q.device

    # 1) Compute g and beta in Triton: g[b,h], beta[b,h]
    A_log_f32 = A_log.float()
    a_f32 = a.squeeze(1).float()  # [B, 8]
    dt_bias_f32 = dt_bias.float()
    b_f32 = b.squeeze(1).float()  # [B, 8]

    g = torch.empty((B, H), dtype=torch.float32, device=device)
    beta = torch.empty((B, H), dtype=torch.float32, device=device)

    # Strides for A_log, a, dt_bias, b
    stride_A = A_log_f32.stride(0) if A_log_f32.dim() > 0 else 1
    stride_a_b, stride_a_h = a_f32.stride(0), a_f32.stride(1)
    stride_dt = dt_bias_f32.stride(0) if dt_bias_f32.dim() > 0 else 1
    stride_b_b, stride_b_h = b_f32.stride(0), b_f32.stride(1)
    stride_g_b, stride_g_h = g.stride(0), g.stride(1)
    stride_beta_b, stride_beta_h = beta.stride(0), beta.stride(1)

    # Launch kernel_g_beta
    grid = (B, H)
    triton.run(
        kernel_g_beta,
        grid=grid,
        num_warps=4,
        A_log_ptr=A_log_f32, a_ptr=a_f32, dt_bias_ptr=dt_bias_f32, b_ptr=b_f32,
        g_ptr=g, beta_ptr=beta,
        B=B, H=H,
        stride_A=stride_A, stride_a_b=stride_a_b, stride_a_h=stride_a_h, stride_dt=stride_dt,
        stride_b_b=stride_b_b, stride_b_h=stride_b_h,
        stride_g_b=stride_g_b, stride_g_h=stride_g_h, stride_beta_b=stride_beta_b, stride_beta_h=stride_beta_h,
    )

    # 2) tmp_old_v[b, h] = dot(k[b, h], state[b, h])
    k_f32 = k.squeeze(1).float()  # [B, 4, 128]
    state_f32 = state.float()     # [B, 8, 128, 128]
    tmp = torch.empty((B, H), dtype=torch.float32, device=device)

    stride_k_b, stride_k_h, stride_k_k = k_f32.stride(0), k_f32.stride(1), k_f32.stride(2)
    stride_s_b, stride_s_h, stride_s_v, stride_s_k = state_f32.stride(0), state_f32.stride(1), state_f32.stride(2), state_f32.stride(3)
    stride_t_b, stride_t_h = tmp.stride(0), tmp.stride(1)

    triton.run(
        kernel_tmp_old_v,
        grid=grid,
        num_warps=4,
        k_ptr=k_f32, state_ptr=state_f32, tmp_ptr=tmp,
        B=B, H=H, V=V, K=K,
        stride_k_b=stride_k_b, stride_k_h=stride_k_h, stride_k_k=stride_k_k,
        stride_s_b=stride_s_b, stride_s_h=stride_s_h, stride_s_v=stride_s_v, stride_s_k=stride_s_k,
        stride_t_b=stride_t_b, stride_t_h=stride_t_h,
    )

    # 3) Compute new_state and output in Triton
    q_f32 = q.squeeze(1).float()  # [B, 4, 128]
    new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

    stride_q_b, stride_q_h, stride_q_k = q_f32.stride(0), q_f32.stride(1), q_f32.stride(2)
    stride_k_b, stride_k_h, stride_k_k = k_f32.stride(0), k_f32.stride(1), k_f32.stride(2)
    stride_v_b, stride_v_h, stride_v_v = v.float().stride(0), v.float().stride(1), v.float().stride(2)  # [B, 8, 128]
    stride_s_b, stride_s_h, stride_s_v, stride_s_k = state_f32.stride(0), state_f32.stride(1), state_f32.stride(2), state_f32.stride(3)
    stride_out_b, stride_out_h = tmp.stride(0), tmp.stride(1)  # reuse tmp strides for out_ptr
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k = new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3)

    scale_val = float(scale if scale is not None else 1.0 / math.sqrt(K))

    triton.run(
        kernel_update_and_output,
        grid=grid,
        num_warps=4,
        q_ptr=q_f32, k_ptr=k_f32, v_ptr=v.float(), state_ptr=state_f32, g_ptr=g, beta_ptr=beta, tmp_ptr=tmp,
        out_ptr=tmp, new_state_ptr=new_state,
        B=B, H=H, V=V, K=K,
        stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_k=stride_q_k,
        stride_k_b=stride_k_b, stride_k_h=stride_k_h, stride_k_k=stride_k_k,
        stride_v_b=stride_v_b, stride_v_h=stride_v_h, stride_v_v=stride_v_v,
        stride_s_b=stride_s_b, stride_s_h=stride_s_h, stride_s_v=stride_s_v, stride_s_k=stride_s_k,
        stride_g_b=g.stride(0), stride_g_h=g.stride(1),
        stride_beta_b=beta.stride(0), stride_beta_h=beta.stride(1),
        stride_out_b=tmp.stride(0), stride_out_h=tmp.stride(1),
        stride_ns_b=stride_ns_b, stride_ns_h=stride_ns_h, stride_ns_v=stride_ns_v, stride_ns_k=stride_ns_k,
        scale=scale_val,
    )

    # 4) Return outputs: output [B, 1, H] cast to bfloat16, new_state [B, H, V, K]
    output = tmp.unsqueeze(1)  # [B, 1, H]
    output = output.to(torch.bfloat16)
    return output, new_state


def get_inputs():
    q = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 1, 4, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    state = torch.randn([1, 8, 128, 128], dtype=torch.float32, device='cuda')
    A_log = torch.randn([8], dtype=torch.float32, device='cuda')
    a = torch.randn([1, 1, 8], dtype=torch.bfloat16, device='cuda')
    dt_bias = torch.randn([8], dtype=torch.float32, device='cuda')
    b = torch.randn([1, 1, 8], dtype=torch.bfloat16, device='cuda')
    scale = 1.0
    return [q, k, v, state, A_log, a, dt_bias, b, scale]


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        return run(q, k, v, state, A_log, a, dt_bias, b, scale)


def run(*args):
    return ModelNew()(*args)
