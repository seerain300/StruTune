import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_ptr, beta_ptr,
    H,
    stride_A, stride_a_b, stride_a_h, stride_dt, stride_b_b, stride_b_h,
    stride_g_b, stride_g_h, stride_beta_b, stride_beta_h,
    num_warps: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars
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
    H, V, K,
    stride_k_b, stride_k_h, stride_k_k,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_t_b, stride_t_h,
    BLOCK_K: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    acc = 0.0
    for kk in range(0, K, BLOCK_K):
        k_off = kk + tl.arange(0, BLOCK_K)
        mask_k = k_off < K
        k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_off * stride_k_k,
                        mask=mask_k, other=0.0)
        # Reduce over V for each kk: s[b,h, :, kk]
        s_sum = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for v in range(0, V):
            s_val = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v * stride_s_v + kk * stride_s_k,
                            mask=mask_k, other=0.0)
            s_sum += s_val
        acc += tl.sum(k_vec * s_sum, axis=0)
    tl.store(tmp_ptr + b_idx * stride_t_b + h_idx * stride_t_h, acc)


@triton.jit
def kernel_update_and_output(
    q_ptr, k_ptr, v_ptr, state_ptr, g_ptr, beta_ptr, tmp_ptr,
    out_ptr, new_state_ptr,
    B, H, V, K,
    scale,
    stride_q_b, stride_q_h, stride_q_k,
    stride_k_b, stride_k_h, stride_k_k,
    stride_v_b, stride_v_h, stride_v_v,
    stride_s_b, stride_s_h, stride_s_v, stride_s_k,
    stride_out_b, stride_out_h,
    stride_ns_b, stride_ns_h, stride_ns_v, stride_ns_k,
    BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load scalars
    g_val = tl.load(g_ptr + b_idx * stride_out_b + h_idx * stride_out_h).to(tl.float32)  # Note: strides below are for out_ptr, but here we load scalar? We'll compute out via q @ new_state.
    # Correction: We need beta and tmp to compute new_state update. Let's load them.
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b + h_idx * stride_beta_h).to(tl.float32)
    tmp_old = tl.load(tmp_ptr + b_idx * stride_out_b + h_idx * stride_out_h).to(tl.float32)  # tmp_old_v[b,h]

    # Compute new_v vector across V
    new_v = tl.zeros((V,), dtype=tl.float32)
    for v in range(0, V):
        vv = tl.full((), v, tl.int32)
        vv_f = vv.to(tl.float32)
        v_elem = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_h + vv * stride_v_v).to(tl.float32)
        new_v[v] = beta_val * v_elem + (1.0 - beta_val) * tmp_old

    # Initialize output accumulator
    out_acc = 0.0

    # Update new_state and accumulate q @ new_state
    for kk in range(0, K, BLOCK_K):
        k_off = kk + tl.arange(0, BLOCK_K)
        mask_k = k_off < K
        k_vec = tl.load(k_ptr + b_idx * stride_k_b + h_idx * stride_k_h + k_off * stride_k_k,
                        mask=mask_k, other=0.0)

        # For each tile of V, compute add term and update new_state
        for vv in range(0, V, BLOCK_V):
            v_off = vv + tl.arange(0, BLOCK_V)
            mask_v = v_off < V

            # Load q elements for this kk (scalar for each kk)
            q_k = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + kk * stride_q_k,
                          mask=mask_k, other=0.0)  # scalar, but we loop over kk

            # Load state tile [BLOCK_V, BLOCK_K]
            s_tile = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
            for vi in range(0, BLOCK_V):
                v_idx = v_off[vi]
                valid_v = v_idx < V
                if valid_v:
                    # Load state[b,h,v_idx, kk:kk+BLOCK_K]
                    for kj in range(0, BLOCK_K):
                        kkj = kk + kj
                        valid_kj = kkj < K
                        s_val = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v_idx * stride_s_v + kkj * stride_s_k,
                                        mask=valid_kj, other=0.0)
                        s_tile[vi, kj] = s_val

            # Load k_vec for add computation
            k_k = k_vec  # [BLOCK_K]

            # Compute add = dot(k_k, new_v[v_off]) over valid v indices
            add_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for vi in range(0, BLOCK_V):
                v_idx = v_off[vi]
                valid_v = v_idx < V
                if valid_v:
                    add_vec += s_tile[vi, :] * new_v[v_idx]

            # Update new_state for all valid (v, k)
            for vi in range(0, BLOCK_V):
                v_idx = v_off[vi]
                valid_v = v_idx < V
                if valid_v:
                    for kj in range(0, BLOCK_K):
                        kkj = kk + kj
                        valid_kj = kkj < K
                        if valid_kj:
                            s_old = tl.load(state_ptr + b_idx * stride_s_b + h_idx * stride_s_h + v_idx * stride_s_v + kkj * stride_s_k,
                                            mask=valid_kj, other=0.0)
                            new_s = s_old - tmp_old + add_vec[kj]
                            tl.store(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v_idx * stride_ns_v + kkj * stride_ns_k,
                                     new_s)

            # Accumulate output: q_k * new_state[b,h,:,kk:kk+BLOCK_K]
            for vi in range(0, BLOCK_V):
                v_idx = v_off[vi]
                valid_v = v_idx < V
                if valid_v:
                    for kj in range(0, BLOCK_K):
                        kkj = kk + kj
                        valid_kj = kkj < K
                        if valid_kj:
                            s_new = tl.load(new_state_ptr + b_idx * stride_ns_b + h_idx * stride_ns_h + v_idx * stride_ns_v + kkj * stride_ns_k,
                                            mask=valid_kj, other=0.0)
                            q_elem = tl.load(q_ptr + b_idx * stride_q_b + h_idx * stride_q_h + kkj * stride_q_k,
                                             mask=valid_kj, other=0.0)  # scalar per kk
                            out_acc += q_elem * s_new

    # Scale final output
    out_acc = out_acc * scale
    tl.store(out_ptr + b_idx * stride_out_b + h_idx * stride_out_h, out_acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure inputs are on same device and dtype; we will use float32 for computation
        device = q.device
        dtype = torch.float32

        # q: [B, 1, QH, K], k: [B, 1, KH, K], v: [B, 1, VH, V], state: [B, H, V, K]
        B, _, QH, K = q.shape
        _, _, KH, _ = k.shape
        _, _, VH, V = v.shape
        H = state.shape[1]  # number of heads for state

        # Ensure shapes match expectations: QH=4, KH=4, VH=8, K=128, V=128, H fixed from state
        assert QH == 4 and KH == 4 and VH == 8, "Head counts must match expected shapes."
        assert K == 128 and V == 128, "K and V must be 128."

        # Make tensors contiguous
        q32 = q.contiguous().view(B, H, K).to(dtype)
        k32 = k.contiguous().view(B, H, K).to(dtype)
        v32 = v.contiguous().view(B, H, V).to(dtype)
        state32 = state.contiguous().view(B, H, V, K).to(dtype)
        A_log32 = A_log.contiguous().to(dtype)  # [H]
        a32 = a.contiguous().view(B, H).to(dtype)  # [B, H]
        dt_bias32 = dt_bias.contiguous().to(dtype)  # [H]
        b32 = b.contiguous().view(B, H).to(dtype)   # [B, H]

        # Allocate outputs
        g = torch.empty((B, H), dtype=dtype, device=device)
        beta = torch.empty((B, H), dtype=dtype, device=device)
        tmp = torch.empty((B, H), dtype=dtype, device=device)
        out = torch.empty((B, H), dtype=dtype, device=device)
        new_state = torch.empty((B, H, V, K), dtype=dtype, device=device)

        # Launch Triton kernels
        # Kernel 1: g and beta
        # Strides: A_log has 1 dim (H), a has strides (B, H), dt_bias has 1 dim (H), b has strides (B, H), g,beta have strides (B,H)
        kernel_g_beta[(B, H)](
            A_log32, a32, dt_bias32, b32,
            g, beta,
            H,
            A_log32.stride(0), a32.stride(0), a32.stride(1), dt_bias32.stride(0), b32.stride(0), b32.stride(1),
            g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
            num_warps=1,
        )

        # Kernel 2: tmp_old_v = sum_k k[b,h,k] * sum_v state[b,h,v,k]
        BLOCK_K = 64
        kernel_tmp_old_v[(B, H)](
            k32, state32, tmp,
            H, V, K,
            k32.stride(0), k32.stride(1), k32.stride(2),
            state32.stride(0), state32.stride(1), state32.stride(2), state32.stride(3),
            tmp.stride(0), tmp.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=1,
        )

        # Kernel 3: update new_state and compute out[b,h]
        BLOCK_V = 64
        kernel_update_and_output[(B, H)](
            q32, k32, v32, state32, g, beta, tmp,
            out, new_state,
            B, H, V, K,
            float(scale),  # pass scale as float
            q32.stride(0), q32.stride(1), q32.stride(2),
            k32.stride(0), k32.stride(1), k32.stride(2),
            v32.stride(0), v32.stride(1), v32.stride(2),
            state32.stride(0), state32.stride(1), state32.stride(2), state32.stride(3),
            out.stride(0), out.stride(1),
            new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
            BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K,
            num_warps=1,
        )

        # Return results: output [B, 1, H] bfloat16, new_state [B, H, V, K] float32
        output = out.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H]
        return output, new_state


def run(*args):
    return ModelNew()(*args)
