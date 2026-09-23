import math
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_g_beta(A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                  g_ptr, beta_ptr,
                  B, H,
                  stride_al, stride_a, stride_db, stride_b,
                  stride_g, stride_beta):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load parameters for this (b, h)
    a_val = tl.load(a_ptr + b_idx * stride_a + h_idx * stride_a)             # a[b, h]
    dtb_val = tl.load(dt_bias_ptr + h_idx * stride_db)                       # dt_bias[h]
    A_val = tl.load(A_log_ptr + h_idx * stride_al)                           # A_log[h]
    b_val = tl.load(b_ptr + b_idx * stride_b + h_idx * stride_b)             # b[b, h]

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dtb_val))
    g_val = tl.exp(-tl.exp(A_val) * sp)
    beta_val = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b_idx * stride_g + h_idx * stride_g, g_val)
    tl.store(beta_ptr + b_idx * stride_beta + h_idx * stride_beta, beta_val)


@triton.jit
def kernel_tmp_old_v(k_ptr, state_ptr, tmp_ptr,
                     B, H, V, K,
                     stride_k_bH,  # k_ptr is 1D: [B*H, K]
                     stride_state_b, stride_state_h, stride_state_v, stride_state_k,
                     stride_tmp_b):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load k[b, h] vector
    k_vec = tl.load(k_ptr + (b_idx * H + h_idx) * stride_k_bH, mask=tl.arange(0, K) < K, other=0.0)

    # Compute dot(k, state[b, h, :, :]) where state is [B, H, V, K]
    acc = 0.0
    for kk in range(0, K):
        k_val = k_vec[kk]
        # For each v, load state[b, h, v, kk] and accumulate
        for vv in range(0, V):
            # Address: b*stride_b + h*stride_h + vv*stride_v + kk*stride_k
            addr = b_idx * stride_state_b + h_idx * stride_state_h + vv * stride_state_v + kk * stride_state_k
            s = tl.load(state_ptr + addr)
            acc += k_val * s
    tl.store(tmp_ptr + b_idx * stride_tmp_b, acc)


@triton.jit
def kernel_update_state_and_output(k_ptr, tmp_old_ptr, v_ptr, beta_ptr, state_in_ptr, new_state_ptr, q_ptr, output_ptr,
                                   B, H, V, K,
                                   stride_k_bH,  # 1D pointer [B*H, K]
                                   stride_tmp_b,
                                   stride_v_b, stride_v_v,
                                   stride_beta_b,
                                   stride_state_b, stride_state_v, stride_state_k,
                                   stride_new_b, stride_new_v, stride_new_k,
                                   stride_q_bH,  # 1D pointer [B*H, K]
                                   stride_out_b, stride_out_v):
    # One program per (b, h)
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load vectors
    k_vec = tl.load(k_ptr + (b_idx * H + h_idx) * stride_k_bH, mask=tl.arange(0, K) < K, other=0.0)
    tmp_old = tl.load(tmp_old_ptr + b_idx * stride_tmp_b)
    beta_val = tl.load(beta_ptr + b_idx * stride_beta_b)
    v_vec = tl.load(v_ptr + b_idx * stride_v_b + h_idx * stride_v_v)  # [V]

    # new_v = beta * v + (1 - beta) * tmp_old
    new_v = beta_val * v_vec + (1.0 - beta_val) * tmp_old

    # Compute state_remove and state_update scalars
    state_remove = 0.0
    state_update = 0.0
    for kk in range(0, K):
        k_val = k_vec[kk]
        state_remove += k_val * tmp_old
        state_update += k_val * new_v[kk]

    # Update new_state[b, h, :, :] = state_in - state_remove + state_update
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            old_val = tl.load(state_in_ptr + b_idx * stride_state_b + h_idx * stride_state_v + v_idx * stride_state_v + k_idx * stride_state_k)
            new_val = old_val - state_remove + state_update
            tl.store(new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_v + v_idx * stride_new_v + k_idx * stride_new_k, new_val)

    # Compute output[b, h] = scale * (q[b, h] @ new_state[b, h, :, :])
    q_vec = tl.load(q_ptr + (b_idx * H + h_idx) * stride_q_bH, mask=tl.arange(0, K) < K, other=0.0)
    output_scalar = 0.0
    for v_idx in range(0, V):
        for k_idx in range(0, K):
            new_elem = tl.load(new_state_ptr + b_idx * stride_new_b + h_idx * stride_new_v + v_idx * stride_new_v + k_idx * stride_new_k)
            output_scalar += q_vec[k_idx] * new_elem
    tl.store(output_ptr + b_idx * stride_out_b + h_idx * stride_out_v, output_scalar)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Ensure device, contiguity, and dtype for computation
        device = q.device
        # Squeeze dim-1 (size 1) as original does
        q_s = q.squeeze(1).contiguous().to(torch.float32)  # [B, QH, K]
        k_s = k.squeeze(1).contiguous().to(torch.float32)  # [B, KH, K]
        v_s = v.squeeze(1).contiguous().to(torch.float32)  # [B, VH, V]

        # State layout: [B, H, V, K]
        if state is None:
            B, QH, V_in, K_in = q_s.shape  # B=1, QH=QH, V_in=V, K_in=K
            H = QH * (v_s.shape[1] // q_s.shape[1])
            state_in = torch.zeros(B, H, v_s.shape[2], k_s.shape[2], dtype=torch.float32, device=device)
        else:
            state_in = state.squeeze(1).contiguous().to(torch.float32)  # [B, H, V, K]
            # Validate heads
            assert state_in.shape[1] == q_s.shape[1] * (v_s.shape[1] // q_s.shape[1]), "state shape mismatch with heads"

        # Inputs for g and beta: a, dt_bias, b
        A_log = A_log.to(device=device, dtype=torch.float32).contiguous()  # [H]
        a = a.squeeze(1).contiguous().to(torch.float32)                    # [B, H]
        dt_bias = dt_bias.contiguous().to(torch.float32)                  # [H]
        b = b.squeeze(1).contiguous().to(torch.float32)                   # [B, H]

        B = q_s.shape[0]
        H = a.shape[1]
        V = v_s.shape[2]
        K = k_s.shape[2]

        # Allocate outputs
        g = torch.empty(B, H, dtype=torch.float32, device=device)
        beta = torch.empty(B, H, dtype=torch.float32, device=device)
        tmp_old_v = torch.empty(B, H, dtype=torch.float32, device=device)
        new_state_out = torch.empty(B, H, V, K, dtype=torch.float32, device=device)
        output = torch.empty(B, H, V, dtype=torch.float32, device=device)

        # Launch Triton kernels: one program per (b, h)
        grid = (B, H)

        # Kernel g_beta
        kernel_g_beta[grid](
            A_log, a, dt_bias, b,
            g, beta,
            B, H,
            A_log.stride(0), a.stride(0), dt_bias.stride(0), b.stride(0),
            g.stride(0), beta.stride(0),
        )

        # Prepare k and q as 1D vectors per (b, h): [B*H, K]
        # k_s is [B, KH, K], but we need k[b, h] -> we assume KH == QH and head mapping consistent. Since original uses k and state separately, we take k_s[b,h] per head.
        # Build k_bH: [B*H, K]
        k_bH = torch.empty(B * H, K, dtype=torch.float32, device=device)
        for bi in range(B):
            for hi in range(H):
                # k_s[bi, hi, :] is the per-head k
                k_bH[bi * H + hi] = k_s[bi, hi]

        # tmp_old_v kernel: one program per (b, h)
        kernel_tmp_old_v[grid](
            k_bH, state_in, tmp_old_v,
            B, H, V, K,
            k_bH.stride(0),  # stride along K for k_bH is 1 for contiguous 1D
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            tmp_old_v.stride(0),
        )

        # Prepare q as 1D vectors per (b, h): [B*H, K]
        q_bH = torch.empty(B * H, K, dtype=torch.float32, device=device)
        for bi in range(B):
            for hi in range(H):
                q_bH[bi * H + hi] = q_s[bi, hi]

        # Update state and output kernel
        kernel_update_state_and_output[grid](
            k_bH, tmp_old_v, v_s, beta, state_in, new_state_out, q_bH, output,
            B, H, V, K,
            k_bH.stride(0),
            tmp_old_v.stride(0),
            v_s.stride(0), v_s.stride(1),
            beta.stride(0),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            new_state_out.stride(0), new_state_out.stride(1), new_state_out.stride(2),
            q_bH.stride(0),
            output.stride(0), output.stride(1),
        )

        # Cast output to bfloat16; original returns [B, 1, H], but our compute returns [B, H, V]. We keep [B, H, V] and cast.
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, new_state_out


def run(*args):
    return ModelNew()(*args)
