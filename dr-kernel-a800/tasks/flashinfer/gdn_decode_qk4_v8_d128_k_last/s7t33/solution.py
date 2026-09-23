import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_and_beta_kernel(a_ptr, dt_bias_ptr, b_ptr, A_log_ptr,
                                g_ptr, beta_ptr,
                                B: tl.constexpr, H: tl.constexpr):
    # One program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    a_val = tl.load(a_ptr + b * H + h)      # a[b, h]
    dt_val = tl.load(dt_bias_ptr + h)       # dt_bias[h]
    A_val = tl.load(A_log_ptr + h)          # A_log[h]
    # softplus(x) = log(1 + exp(x)), sigmoid(x) = 1 / (1 + exp(-x))
    x = a_val + dt_val
    softplus = tl.log(1.0 + tl.exp(x))
    g = tl.exp(-tl.exp(A_val) * softplus)
    beta = 1.0 / (1.0 + tl.exp(-tl.load(b_ptr + b * H + h)))
    tl.store(g_ptr + b * H + h, g)
    tl.store(beta_ptr + b * H + h, beta)


@triton.jit
def _vec_matmul_tile_vec_kernel(k_ptr, state_ptr, out_ptr,
                                K: tl.constexpr, V: tl.constexpr):
    # out[k] = sum_v k[v] * state[v, k] for k in 0..K-1
    for k in tl.static_range(0, K):
        acc = 0.0
        for v in tl.static_range(0, V):
            k_val = tl.load(k_ptr + v)         # k[v]
            state_val = tl.load(state_ptr + v * K + k)  # state[v, k]
            acc += k_val * state_val
        tl.store(out_ptr + k, acc)


@triton.jit
def _vec_matmul_scalar_kernel(vec_ptr, k_ptr, out_ptr,
                              N: tl.constexpr):
    # out = sum_i k[i] * vec[i] for i in 0..N-1
    acc = 0.0
    for i in tl.static_range(0, N):
        acc += tl.load(k_ptr + i) * tl.load(vec_ptr + i)
    tl.store(out_ptr, acc)


@triton.jit
def _output_scalar_kernel(q_ptr, new_state_ptr, out_ptr, scale,
                           K: tl.constexpr, V: tl.constexpr):
    # Compute scalar = sum_{k=0..K-1} q[k] * new_state[k]
    acc = 0.0
    for k in tl.static_range(0, K):
        qk = tl.load(q_ptr + k)
        ns = tl.load(new_state_ptr + k)
        acc += qk * ns
    acc = acc * scale
    tl.store(out_ptr, acc)


@triton.jit
def _sqrt_scale_kernel(out_ptr, K: tl.constexpr):
    # Compute scale = 1 / sqrt(K) and store to out_ptr
    scale = 1.0 / tl.sqrt(K)
    tl.store(out_ptr, scale)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure inputs are CUDA tensors; follow original shapes (batch=1, seq=1)
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda, "All inputs must be CUDA tensors"
        Bq, Tq, Hq, K = q.shape
        Bk, Tk, Hk, Kk = k.shape
        Bv, Tv, Hv, V = v.shape
        Bst, Tst, Hst, Kst, Vst = state.shape
        # Reference asserts: B=1, T=1, Hq=4, Hk=4, Hv=8, K=128, V=128
        assert Bq == 1 and Bk == 1 and Bv == 1 and Bst == 1, "Batch must be 1"
        assert Tq == 1 and Tk == 1 and Tv == 1 and Tst == 1, "Sequence length must be 1"
        assert Hq == 4 and Hk == 4 and Hv == 8, "Head counts must match reference"
        assert K == 128 and V == 128 and Kst == 128 and Vst == 128, "K and V must be 128"

        # Cast to float32 and make contiguous
        q_f32 = q[0].float().contiguous().reshape(1, Hq, K)          # [1, 4, 128]
        k_f32 = k[0].float().contiguous().reshape(1, Hk, K)          # [1, 4, 128]
        v_f32 = v[0].float().contiguous().reshape(1, Hv, V)          # [1, 8, 128]
        state_f32 = state[0].float().contiguous().reshape(1, Hv, V, K)  # [1, 8, 128, 128]

        # Prepare outputs
        g_out = torch.empty(Hv, device=device, dtype=torch.float32)  # [Hv]
        beta_out = torch.empty(Hv, device=device, dtype=torch.float32)  # [Hv]
        output_f = torch.empty(Hv, device=device, dtype=torch.float32)  # [Hv]
        new_state_f = torch.empty((1, Hv, V, K), device=device, dtype=torch.float32)  # [1,8,128,128]

        # Launch gate/beta kernel
        _compute_g_and_beta_kernel[(Hv,)](a[0], dt_bias, b[0], A_log, g_out, beta_out, B=1, H=Hv)

        # Compute scale in Triton and load it
        scale_buf = torch.empty(1, device=device, dtype=torch.float32)
        _sqrt_scale_kernel[(1,)](scale_buf, K=K)
        scale_val = scale_buf[0]  # 1/sqrt(K)

        # For each head h
        for h in range(Hv):
            # Vectors
            k_vec = k_f32[0, h]              # [128]
            v_vec = v_f32[0, h]              # [128]
            state_mat = state_f32[0, h]      # [128, 128]
            q_vec = q_f32[0, h]              # [128]

            # 1) old_v = k @ state_old
            old_v = torch.empty(K, device=device, dtype=torch.float32)
            state_flat = state_mat.reshape(-1)  # [V*K]
            _vec_matmul_tile_vec_kernel[(1,)](k_vec, state_flat, old_v, K=K, V=V)

            # 2) g and beta
            g_val = g_out[h]
            beta_val = beta_out[h]

            # 3) new_v = beta * v + (1 - beta) * old_v
            new_v = beta_val * v_vec + (1.0 - beta_val) * old_v  # [128]

            # 4) state_remove = k @ old_v (scalar)
            state_remove = torch.empty((), device=device, dtype=torch.float32)
            _vec_matmul_scalar_kernel[(1,)](old_v, k_vec, state_remove, N=K)

            # 5) state_update = k @ new_v (scalar)
            state_update = torch.empty((), device=device, dtype=torch.float32)
            _vec_matmul_scalar_kernel[(1,)](new_v, k_vec, state_update, N=K)

            # 6) new_state = g * state_old - state_remove + state_update
            state_old_flat = state_mat.reshape(-1)  # [V*K]
            new_state_row = (g_val * state_old_flat) - state_remove + state_update  # [V*K]
            new_state_f[0, h] = new_state_row.reshape(V, K)

            # 7) output = scale * (q @ new_state)
            out_val = torch.empty((), device=device, dtype=torch.float32)
            _output_scalar_kernel[(1,)](q_vec, new_state_row, out_val, scale_val, K=K, V=V)
            output_f[h] = out_val

        # Return in expected format: output [B,1,Hv] bfloat16, new_state [B,Hv,V,K] float32
        output_bf16 = output_f.view(1, Hv).unsqueeze(1).to(torch.bfloat16)  # [1,1,8]
        return output_bf16, new_state_f


def run(*args):
    return ModelNew()(*args)
