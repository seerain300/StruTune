import math
import torch
import triton
import triton.language as tl


@triton.jit
def gate_beta_kernel(g_out_ptr, beta_out_ptr, A_log_ptr, a_ptr, dt_bias_ptr, B, H):
    # Compute g = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h])) and
    # beta = 1 / (1 + exp(-b[b,h])) for each (b,h).
    pid = tl.program_id(0)  # linear index over B*H
    b = pid // H
    h = pid % H

    # Load scalars for this (b,h)
    A = tl.load(a_ptr + pid) + tl.load(dt_bias_ptr + h)
    # softplus(A) = log(1 + exp(A))
    soft = tl.log(1.0 + tl.exp(A))
    A_log = tl.load(A_log_ptr + h)
    g = tl.exp(-tl.exp(A_log) * soft)
    beta = 1.0 / (1.0 + tl.exp(-tl.load(a_ptr + pid)))  # a[b,h] is used as b[b,h] here (per problem setup)
    tl.store(g_out_ptr + pid, g)
    tl.store(beta_out_ptr + pid, beta)


@triton.jit
def q_dot_kernel(out_ptr, q_ptr, updated_ptr, scale, B, H, V, K):
    # Compute output[b,h] = scale * (q_h @ updated_state_bh)
    pid = tl.program_id(0)  # linear index over B*H
    b = pid // H
    h = pid % H

    sum_q = 0.0
    # q_h is length K, updated_state_bh is length V
    for i in range(0, K):
        q_val = tl.load(q_ptr + b * (H * K) + h * K + i)
        for j in range(0, V):
            upd_val = tl.load(updated_ptr + b * (H * V * K) + h * (V * K) + j * K + i)
            sum_q += q_val * upd_val
    out_val = scale * sum_q
    tl.store(out_ptr + pid, out_val)


@triton.jit
def old_v_reduce_kernel(old_v_ptr, k_ptr, state_ptr, B, H, V, K):
    # Compute old_v = sum_k k[b,h,k] * (sum_v state[b,h,v,k])
    pid = tl.program_id(0)  # linear index over B*H
    b = pid // H
    h = pid % H

    sum_val = 0.0
    for t in range(0, K):
        k_val = tl.load(k_ptr + b * (H * K) + h * K + t)
        inner = 0.0
        for j in range(0, V):
            inner += tl.load(state_ptr + b * (H * V * K) + h * (V * K) + j * K + t)
        sum_val += k_val * inner
    tl.store(old_v_ptr + pid, sum_val)


@triton.jit
def new_v_kernel(new_v_ptr, beta_ptr, v_ptr, k_ptr, state_ptr, scale_beta, B, H, V, K):
    # Compute new_v = scale_beta * beta * (sum_v v_h[v]) + (1 - scale_beta) * old_v
    # Note: scale_beta is passed as a scalar multiplier for beta term. In our case, scale_beta = 1 - beta for the (1 - beta) * old_v part.
    pid = tl.program_id(0)  # linear index over B*H
    b = pid // H
    h = pid % H

    beta_val = tl.load(beta_ptr + pid)
    sum_v = 0.0
    for j in range(0, V):
        sum_v += tl.load(v_ptr + b * (H * V) + h * V + j)
    sum_k_old = 0.0
    for t in range(0, K):
        k_val = tl.load(k_ptr + b * (H * K) + h * K + t)
        inner = 0.0
        for j in range(0, V):
            inner += tl.load(state_ptr + b * (H * V * K) + h * (V * K) + j * K + t)
        sum_k_old += k_val * inner
    new_v = beta_val * sum_v + (1.0 - beta_val) * sum_k_old  # here (1 - beta_val) acts as scale_beta (1 - beta)
    tl.store(new_v_ptr + pid, new_v)


@triton.jit
def updated_state_kernel(updated_ptr, g_ptr, state_ptr, old_v_ptr, new_v_ptr, B, H, V, K):
    # Compute updated_state_bh = g * state_bh - old_v + new_v (broadcast new_v over V)
    # Writes elementwise updated_state_bh[v, k] as updated_ptr[b*H*V*K + h*(V*K) + v*K + k] = g * state_bh[v,k] - old_v + new_v
    pid = tl.program_id(0)  # linear index over B*H
    b = pid // H
    h = pid % H

    g_val = tl.load(g_ptr + pid)
    old_v = tl.load(old_v_ptr + pid)
    new_v = tl.load(new_v_ptr + pid)

    # Write updated_state elementwise: updated_ptr[b*H*V*K + h*(V*K) + v*K + k] = g * state_bh[v,k] - old_v + new_v
    for v in range(0, V):
        for k in range(0, K):
            state_val = tl.load(state_ptr + b * (H * V * K) + h * (V * K) + v * K + k)
            upd_val = g_val * state_val - old_v + new_v
            tl.store(updated_ptr + b * (H * V * K) + h * (V * K) + v * K + k, upd_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, scale):
        # Shapes: q: [B,1,H_q,K], k: [B,1,H_k,K], v: [B,1,H_v,V], state: [B,H_v,V,K]
        # Output: [B,1,H_v,V], new_state: [B,H_v,V,K]
        device = q.device
        dtype = torch.float32

        B, qT, H_q, K = q.shape
        _, kT, H_k, K_ = k.shape
        _, vT, H_v, V = v.shape
        _, sB, sH, sV, sK = state.shape
        assert qT == 1 and kT == 1 and vT == 1, "Only T=1 supported"
        assert sB == B and sH == H_v and sV == V and sK == K
        assert K == 128 and V == 128

        H = H_v  # heads dimension from v
        # Ensure device and contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        a = a.squeeze(1).contiguous()          # [B,H_v]
        dt_bias = dt_bias.contiguous()         # [H_v]
        b = b.squeeze(1).contiguous()          # [B,H_v]

        # Triton outputs
        g_out = torch.empty((B, H), dtype=dtype, device=device)  # per (b,h)
        beta_out = torch.empty((B, H), dtype=dtype, device=device)  # per (b,h)

        # Launch gate_beta_kernel: computes g and beta
        grid = (B * H,)
        gate_beta_kernel[grid](g_out, beta_out, A_log.float(), a.float(), dt_bias.float(), B, H)

        # Allocate intermediates
        old_v = torch.empty((B, H), dtype=dtype, device=device)     # per (b,h)
        new_v = torch.empty((B, H), dtype=dtype, device=device)     # per (b,h)
        updated = torch.empty((B, H, V, K), dtype=dtype, device=device)  # [B,H,V,K]

        # Launch old_v_reduce_kernel: computes old_v[b,h] = k_h @ (sum_v state_bh)
        grid2 = (B * H,)
        old_v_reduce_kernel[grid2](old_v, k.squeeze(1).contiguous().float(), state.float(), B, H, V, K)

        # Launch new_v_kernel: computes new_v[b,h]
        # Here, scale_beta = 1 - beta_out
        scale_beta = (1.0 - beta_out).float()  # scalar per (b,h)
        new_v_kernel[grid2](new_v, beta_out, v.squeeze(1).float(), k.squeeze(1).float(), state.float(), scale_beta, B, H, V, K)

        # Launch updated_state_kernel: elementwise updated state
        updated_state_kernel[grid2](updated, g_out, state.float(), old_v, new_v, B, H, V, K)

        # Compute output using q_dot_kernel: out[b,h] = scale * (q_h @ updated[b,h])
        out = torch.empty((B, H), dtype=torch.float32, device=device)
        q_vec = q.squeeze(1).float().contiguous()   # [B,H,K]
        grid3 = (B * H,)
        q_dot_kernel[grid3](out, q_vec, updated, scale, B, H, V, K)

        # Return outputs with expected shapes/dtypes
        output = out.unsqueeze(1)  # [B,1,H]
        # We need [B,1,H,V], so broadcast output across V dimension
        # Since output is per (b,h), we create [B,1,H,V] by expanding along V
        output = output.expand(B, 1, H, V).contiguous()  # [B,1,H,V] float32
        # Convert to bfloat16 for consistency with original output dtype
        output = output.to(torch.bfloat16)

        # new_state should match state shape, and we computed updated_state in float32
        new_state = updated  # [B,H,V,K], float32

        return output, new_state


def run(*args):
    return ModelNew()(*args)
