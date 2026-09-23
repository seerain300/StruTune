import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr,
                                g_ptr, beta_ptr,
                                B: tl.constexpr, H: tl.constexpr):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Load scalars
    a_val = tl.load(a_ptr + b * H + h)          # a[b, h]
    dt_val = tl.load(dt_bias_ptr + h)           # dt_bias[h]
    A_val = tl.load(A_log_ptr + h)              # A_log[h]
    b_val = tl.load(b_ptr + b * H + h)          # b[b, h]

    # Compute x = a + dt
    x = a_val + dt_val

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(x))

    # g = exp(-exp(A) * softplus(x))
    g_val = tl.exp(-tl.exp(A_val) * sp)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b * H + h, g_val)
    tl.store(beta_ptr + b * H + h, beta_val)


@triton.jit
def _vec_matmul_tile_vec(k_ptr, state_ptr, out_ptr,
                          K: tl.constexpr, V: tl.constexpr):
    # Compute out[k] = sum_v k[v] * state[v, k] for k in 0..K-1
    for k in tl.static_range(0, K):
        acc = 0.0
        for v in tl.static_range(0, V):
            k_val = tl.load(k_ptr + v)            # k[v]
            state_val = tl.load(state_ptr + v * K + k)  # state[v, k]
            acc += float(k_val) * float(state_val)
        tl.store(out_ptr + k, acc)


@triton.jit
def _vec_matmul_scalar(k_ptr, vec_ptr, out_ptr,
                        K: tl.constexpr):
    # Compute scalar = sum_k k[k] * vec[k]
    acc = 0.0
    for k in tl.static_range(0, K):
        k_val = tl.load(k_ptr + k)
        vec_val = tl.load(vec_ptr + k)
        acc += float(k_val) * float(vec_val)
    tl.store(out_ptr, acc)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr,
                           scale_ptr,
                           K: tl.constexpr, V: tl.constexpr):
    # One program per (b, h): compute output = scale * (q @ new_state)
    pid = tl.program_id(0)
    # q_ptr points to q_vec for this (b,h); new_state_ptr points to new_state_row flattened as [V*K] for this (b,h)
    scale_val = tl.load(scale_ptr)
    acc = 0.0
    for v in tl.static_range(0, V):
        dot = 0.0
        for k in tl.static_range(0, K):
            qk = tl.load(q_ptr + k)
            state_elem = tl.load(new_state_ptr + v * K + k)
            dot += float(qk) * float(state_elem)
        acc += scale_val * float(dot)
    tl.store(out_ptr + pid, acc)


@triton.jit
def _sqrt_scale_kernel(scale_ptr, K: tl.constexpr):
    s = 1.0 / tl.sqrt(float(K))
    tl.store(scale_ptr, s)


class ModelNew(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state, A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor, scale):
        # Cast to float32 for numerical stability and make contiguous
        q_f32 = q.contiguous().float()   # [B,1,Hq,K]
        k_f32 = k.contiguous().float()   # [B,1,Hk,K]
        v_f32 = v.contiguous().float()   # [B,1,Hv,V]
        state_f32 = state.contiguous().float()  # [B,Hv,V,K]
        A_log_f32 = A_log.contiguous().float()  # [H]
        a_f32 = a.contiguous().float()   # [B,1,H]
        dt_bias_f32 = dt_bias.contiguous().float()  # [H]
        b_f32 = b.contiguous().float()   # [B,1,H]

        B, _, Hq, K = q_f32.shape
        _, _, Hv, V = v_f32.shape

        # Output buffers
        g_out = torch.empty(B * Hv, device=q.device, dtype=torch.float32)
        beta_out = torch.empty(B * Hv, device=q.device, dtype=torch.float32)
        output_f = torch.empty(B * Hv, device=q.device, dtype=torch.float32)
        new_state_f = torch.empty((B, Hv, V, K), device=q.device, dtype=torch.float32)

        # Launch compute g and beta
        _compute_g_and_beta_kernel[(B * Hv,)](a_f32.view(-1), dt_bias_f32, b_f32.view(-1), A_log_f32,
                                              g_out, beta_out,
                                              B=B, H=Hv)  # meta-params

        # Compute scale in Triton (1/sqrt(K))
        scale_buf = torch.empty(1, device=q.device, dtype=torch.float32)
        _sqrt_scale_kernel[(1,)](scale_buf, K=K)  # meta-param

        # Now perform per-(b,h) updates
        for b_idx in range(B):
            for h_idx in range(Hv):
                # Prepare vectors/pointers
                q_vec = q_f32[b_idx, 0, h_idx]  # [K]
                k_vec = k_f32[b_idx, 0, h_idx]  # [K]
                state_mat = state_f32[b_idx, h_idx]  # [V, K], flattened to [V*K]
                v_vec = v_f32[b_idx, 0, h_idx]      # [V]

                # 1) old_v = k @ state_old
                old_v = torch.empty(K, device=q.device, dtype=torch.float32)
                _vec_matmul_tile_vec[(K,)](k_vec, state_mat.reshape(-1), old_v,
                                            K=K, V=V)  # meta-params

                # 2) beta = sigmoid(b[b,h]) already in beta_out; not needed to reload

                # 3) new_v = beta * v + (1 - beta) * old_v
                beta_val = beta_out[b_idx * Hv + h_idx]
                new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [V]

                # 4) state_remove = k @ old_v (scalar)
                state_remove = torch.empty((), device=q.device, dtype=torch.float32)
                _vec_matmul_scalar[(K,)](k_vec, old_v, state_remove, K=K)  # meta-param

                # 5) state_update = k @ new_v (scalar)
                state_update = torch.empty((), device=q.device, dtype=torch.float32)
                _vec_matmul_scalar[(K,)](k_vec, new_v, state_update, K=K)  # meta-param

                # 6) new_state = g * state_old - state_remove + state_update
                g_val = g_out[b_idx * Hv + h_idx]
                new_state_row = (g_val * state_mat.reshape(-1)) - state_remove + state_update  # [V*K]
                new_state_f[b_idx, h_idx] = new_state_row.reshape(V, K)

                # 7) output = scale * (q @ new_state)
                out_val = torch.empty((), device=q.device, dtype=torch.float32)
                _output_scalar_kernel[(1,)](q_vec, new_state_row, out_val, scale_buf,
                                             K=K, V=V)  # meta-params
                output_f[b_idx * Hv + h_idx] = out_val

        # Return output in bfloat16 and new_state (float32 as per original)
        output_bf16 = output_f.view(B, Hv).unsqueeze(1).to(torch.bfloat16)  # [B,1,Hv]
        return output_bf16, new_state_f


def run(*args):
    return ModelNew()(*args)
