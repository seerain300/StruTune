import torch
import triton
import triton.language as tl


@triton.jit
def _compute_g_beta_kernel(
    a_ptr,         # [B, H_v] float32
    dt_bias_ptr,   # [H_v] float32
    A_log_ptr,     # [H_v] float32
    b_ptr,         # [B, H_v] float32
    g_ptr,         # [B, H_v] float32 output
    beta_ptr,      # [B, H_v] float32 output
    B: tl.int32,
    H_v: tl.int32,
):
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    # Load inputs
    a_val = tl.load(a_ptr + b_idx * H_v + hv_idx)        # a[b, hv]
    dt = tl.load(dt_bias_ptr + hv_idx)                   # dt_bias[hv]
    A_log_val = tl.load(A_log_ptr + hv_idx)              # A_log[hv]
    b_val = tl.load(b_ptr + b_idx * H_v + hv_idx)        # b[b, hv]

    # softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(a_val + dt))
    g = tl.exp(-tl.exp(A_log_val) * sp)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # Store results
    tl.store(g_ptr + b_idx * H_v + hv_idx, g)
    tl.store(beta_ptr + b_idx * H_v + hv_idx, beta)


@triton.jit
def _state_update_kernel(
    k_ptr,     # [B, H_v, D] float32
    v_ptr,     # [B, H_v, D] float32
    state_ptr, # [H_v, D, D] float32
    g_ptr,     # [B, H_v] float32
    beta_ptr,  # [B, H_v] float32
    B: tl.int32,
    H_v: tl.int32,
    D: tl.int32,
):
    b_idx = tl.program_id(0)   # token index 0..B-1
    hv_idx = tl.program_id(1)  # head index 0..H_v-1

    # Load g and beta for this (b, hv)
    g_val = tl.load(g_ptr + b_idx * H_v + hv_idx)
    beta_val = tl.load(beta_ptr + b_idx * H_v + hv_idx)

    # Initialize updated state rows
    for i in range(0, D):
        # old_v = sum_k k[b, hv, k] * state[hv, i, k]
        old_v = 0.0
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for k in range(0, D):
            k_vec[k] = tl.load(k_ptr + b_idx * H_v * D + hv_idx * D + k)
        for k in range(0, D):
            state_row_k_ptr = state_ptr + hv_idx * D * D + i * D + k
            old_v += k_vec[k] * tl.load(state_row_k_ptr)

        # new_v = beta * v + (1 - beta) * old_v
        v_vec = tl.zeros((D,), dtype=tl.float32)
        for k in range(0, D):
            v_vec[k] = tl.load(v_ptr + b_idx * H_v * D + hv_idx * D + k)
        new_v = beta_val * v_vec + (1.0 - beta_val) * old_v

        # kT_old and kT_newv: dot(k, old_v) and dot(k, new_v)
        kT_old = 0.0
        for k in range(0, D):
            kT_old += k_vec[k] * old_v
        kT_newv = 0.0
        for k in range(0, D):
            kT_newv += k_vec[k] * new_v

        # Update state row: state[hv, i, :] = g * state[hv, i, :] - kT_old + kT_newv
        # Since old_v/new_v are scalars per iteration, implement per element update
        # We need to update each element j in row i: state_ptr[i*D + j]
        for j in range(0, D):
            orig_ptr = state_ptr + hv_idx * D * D + i * D + j
            val = tl.load(orig_ptr)
            val = val * g_val - kT_old + kT_newv
            tl.store(orig_ptr, val)


@triton.jit
def _output_kernel(
    q_ptr,        # [B, H_q, D] float32 (H_q=4, D=128)
    state_ptr,    # [H_v, D, D] float32
    output_ptr,   # [B, H_v, D] float32
    B: tl.int32,
    H_v: tl.int32,
    D: tl.int32,
):
    b_idx = tl.program_id(0)
    hv_idx = tl.program_id(1)

    # Form q_exp by concatenating q[t,0,:] and q[t,1,:]
    idx = hv_idx % 2  # 0 -> use q[t,0,:]; 1 -> use q[t,1,:]
    q_exp = tl.load(q_ptr + b_idx * H_q * D + idx * D + tl.arange(0, D))  # [D]

    # Compute output_vec[hv, :] = q_exp @ state[hv, :, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for j in range(0, D):
        acc = 0.0
        for i in range(0, D):
            acc += tl.load(state_ptr + hv_idx * D * D + i * D + j)
        out_vec[j] = q_exp @ tl.load(state_ptr + hv_idx * D * D + tl.arange(0, D))  # incorrectly used; correct as below
        # Note: Triton requires vectorized pattern. Correct way: use tl.sum(q_exp * state_row_vec)
        # We cannot easily access a row vector in Triton directly; implement per scalar j:
        # Compute dot(q_exp, state[hv, :, j]) via loading each row element.
        # Better approach: precompute rows outside Triton or use matrix multiply with tensor API in Python.
        # Since Triton lacks native matmul here, we implement per j by loading each row:
        # But this is error-prone. Instead, we can compute row-wise by iterating over i:
        # For correctness in this environment, we approximate by using tl.sum over q_exp with state_row_vec computed similarly.
        # However, to keep it simple and correct, we'll implement a scalar approach by loading each row element and dot with q_exp.
        # We will do it by re-deriving out_vec[j] as sum_i state[hv, i, j] * q_exp[i].
        # Thus, for each j, compute acc = sum_i state[hv, i, j] * q_exp[i]:
        acc = 0.0
        for i in range(0, D):
            acc += tl.load(state_ptr + hv_idx * D * D + i * D + j) * q_exp[i]
        out_vec[j] = acc

    # Store output vector
    out_ptr_base = output_ptr + b_idx * H_v * D + hv_idx * D
    for j in range(0, D):
        tl.store(out_ptr_base + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure constraints: total_seq_len must be 6
        assert q.shape[0] == 6, "q: total_seq_len must be 6"
        device = q.device

        # Cast inputs to float32 for compute
        q_fp32 = q.contiguous().float()
        k_fp32 = k.contiguous().float()
        v_fp32 = v.contiguous().float()
        a_fp32 = a.contiguous().float()
        dt_bias_fp32 = dt_bias.contiguous().float()
        b_fp32 = b.contiguous().float()
        A_log_fp32 = A_log.contiguous().float()

        B = q_fp32.shape[0]       # total_seq_len = 6
        H_q = q_fp32.shape[1]     # 4
        H_v = v_fp32.shape[1]     # 8
        D = q_fp32.shape[2]       # 128

        # Compute g and beta using Triton
        g = torch.empty((B, H_v), dtype=torch.float32, device=device)
        beta = torch.empty((B, H_v), dtype=torch.float32, device=device)
        grid_g = (B, H_v)
        _compute_g_beta_kernel[grid_g](
            a_fp32, dt_bias_fp32, A_log_fp32, b_fp32, g, beta, B, H_v
        )

        # Prepare state [H_v, D, D] float32
        # If state is provided, convert from [1, H_v, D, D] to [H_v, D, D]; otherwise initialize zeros
        if state is not None:
            # state provided as [1, H_v, D, D]; convert
            state_HVD = state[0].contiguous().float()  # [H_v, D, D]
        else:
            state_HVD = torch.zeros((H_v, D, D), dtype=torch.float32, device=device)

        # Update state per token and head using Triton
        # Launch grid over (B, H_v)
        _state_update_kernel[(B, H_v)](
            k_fp32, v_fp32, state_HVD, g, beta, B, H_v, D
        )

        # Prepare output [B, H_v, D] float32 using Triton
        output_fp32 = torch.empty((B, H_v, D), dtype=torch.float32, device=device)
        _output_kernel[(B, H_v)](
            q_fp32, state_HVD, output_fp32, B, H_v, D
        )

        # Return output as bfloat16 and state as [1, H_v, D, D]
        output_bf16 = output_fp32.to(torch.bfloat16)
        state_out = state_HVD.unsqueeze(0)  # [1, H_v, D, D]

        return output_bf16, state_out


def run(*args):
    return ModelNew()(*args)
