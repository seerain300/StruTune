import torch
import math
import triton
import triton.language as tl


# Triton kernel: compute g and beta vectors for the whole time dimension.
# Inputs:
#   a_ptr: [T, HV] bfloat16
#   dt_bias_ptr: [HV] float32
#   A_log_ptr: [HV] float32
#   b_ptr: [T, HV] bfloat16
# Outputs:
#   g_ptr: [T, HV] float32
#   beta_ptr: [T, HV] float32
@triton.jit
def _compute_g_beta_kernel(
    a_ptr, dt_bias_ptr, A_log_ptr, b_ptr,
    g_ptr, beta_ptr,
    T: tl.int32, HV: tl.int32
):
    pid = tl.program_id(0)  # time-step index
    # Loop over hv dimension and compute g[pid, hv] and beta[pid, hv]
    for hv in range(0, HV):
        a_val = tl.load(a_ptr + pid * HV + hv, mask=True, other=0.0)
        dt_bias_val = tl.load(dt_bias_ptr + hv, mask=True, other=0.0)
        A_log_val = tl.load(A_log_ptr + hv, mask=True, other=0.0)

        # x = a + dt_bias
        x = a_val.to(tl.float32) + dt_bias_val

        # softplus(x) = log(1 + exp(x))
        sp = tl.log(1.0 + tl.exp(x))

        # g = exp(-exp(A_log) * softplus(x))
        g_val = tl.exp(-tl.exp(A_log_val) * sp)

        # beta = sigmoid(b) = 1 / (1 + exp(-b))
        b_val = tl.load(b_ptr + pid * HV + hv, mask=True, other=0.0)
        beta_val = 1.0 / (1.0 + tl.exp(-b_val.to(tl.float32)))

        # Store g and beta
        tl.store(g_ptr + pid * HV + hv, g_val)
        tl.store(beta_ptr + pid * HV + hv, beta_val)


def _compute_g_beta_triton(a: torch.Tensor, dt_bias: torch.Tensor, A_log: torch.Tensor, b: torch.Tensor):
    """
    Compute g and beta using Triton. a, b: [T, HV]; dt_bias, A_log: [HV].
    Returns g, beta as torch.Tensor float32 with shape [T, HV].
    """
    assert a.is_cuda and b.is_cuda and dt_bias.is_cuda and A_log.is_cuda, "Inputs must be CUDA tensors for Triton."
    T = a.shape[0]
    HV = a.shape[1]
    g = torch.empty((T, HV), dtype=torch.float32, device=a.device)
    beta = torch.empty((T, HV), dtype=torch.float32, device=a.device)
    grid = (T,)
    _compute_g_beta_kernel[grid](a, dt_bias, A_log, b, g, beta, T, HV)
    return g, beta


# Triton kernel: repeat_interleave q and k along head dimension by factor R.
# q/k: [T, H, K], out: [T, H*R, K], dtype bfloat16
@triton.jit
def _repeat_interleave_qk_kernel(
    q_ptr, out_ptr,
    T: tl.int32, H: tl.constexpr, K: tl.int32, R: tl.constexpr
):
    pid = tl.program_id(0)  # time-step index
    for h in range(0, H):
        for k in range(0, K):
            base = pid * H * K + h * K + k
            q_val = tl.load(q_ptr + base, mask=True, other=0.0)
            for r in range(0, R):
                h_out = h * R + r
                out_base = pid * (H * R) * K + h_out * K + k
                tl.store(out_ptr + out_base, q_val.to(tl.bfloat16))


def _repeat_interleave_qk_triton(q: torch.Tensor, k: torch.Tensor, R: int):
    """
    Repeat q and k along head dimension by factor R using Triton. q/k: [T, H, K], output: [T, H*R, K], dtype bfloat16.
    """
    assert q.is_cuda and k.is_cuda, "Tensors must be CUDA for Triton."
    T, H, K = q.shape
    H_new = H * R
    out = torch.empty((T, H_new, K), dtype=torch.bfloat16, device=q.device)
    grid = (T,)
    _repeat_interleave_qk_kernel[grid](q, out, T, H, K, R)
    return out


# Triton kernel: compute o_vec = scale * (q_row @ state_HKV) for a given time-step (single row of q).
# q_row: [K], state_HKV: [H, K, V], output o: [V], scale: float32
@triton.jit
def _q_mm_hkv_row_kernel(
    q_ptr, state_ptr, o_ptr,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    scale: tl.float32,
):
    # q_ptr points to a [1, K] row; state_ptr points to [H, K, V]
    for v in range(0, V):
        acc = 0.0
        for k in range(0, K):
            qk = tl.load(q_ptr + k).to(tl.float32)
            sum_state = 0.0
            for h in range(0, H):
                sum_state += tl.load(state_ptr + (h * K + k) * V + v).to(tl.float32)
            acc += qk * sum_state
        tl.store(o_ptr + v, acc * scale)


def _q_mm_hkv_triton(q_row: torch.Tensor, state_HKV: torch.Tensor, scale: float):
    """
    Compute scale * (q_row @ state_HKV) where q_row is [1, K], state_HKV is [H, K, V].
    Returns o [V] as float32. Specialized for H=4, K=128, V=8.
    """
    assert q_row.is_cuda and state_HKV.is_cuda, "Tensors must be CUDA for Triton."
    assert q_row.shape == (1, 128), "q_row must be [1, 128]"
    assert state_HKV.shape == (4, 128, 8), "state_HKV must be [4, 128, 8]"
    o = torch.empty((8,), dtype=torch.float32, device=q_row.device)
    _q_mm_hkv_row_kernel[(1,)](q_row, state_HKV, o, 4, 128, 8, scale)
    return o


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        """
        Triton version: compute g/beta and some Triton matmuls; the main updates use torch.
        Args:
            q: [T, Hq, K] bfloat16, Hq=4
            k: [T, Hk, K] bfloat16, Hk=4
            v: [T, Hv, K] bfloat16, Hv=8
            state: [num_seqs, H, V, K] float32 (H=8, V=128, K=128)
            A_log: [H*V] float32
            a: [T, H*V] bfloat16
            dt_bias: [H*V] float32
            b: [T, H*V] bfloat16
            cu_seqlens: [num_seqs+1] int64
            scale: float
        Returns:
            output: [num_seqs, H*V, K] bfloat16
            new_state: [num_seqs, H, V, K] float32 (zeroed here)
        """
        device = q.device
        T = q.shape[0]
        Hq = q.shape[1]
        K = q.shape[2]
        Hv = v.shape[1]
        num_seqs = cu_seqlens.numel() - 1
        H = 4  # num_q_heads (constexpr in Triton)
        Kdim = 128  # head_size (constexpr in Triton)
        V = 8  # num_v_heads (constexpr in Triton)

        # Compute g and beta via Triton
        a_flat = a.float().contiguous()          # [T, H*V]
        dt_bias_vec = dt_bias.float().contiguous()  # [H*V]
        b_flat = b.float().contiguous()          # [T, H*V]
        A_log_vec = A_log.float().contiguous()   # [H*V]

        g, beta = _compute_g_beta_triton(a_flat, dt_bias_vec, A_log_vec, b_flat)  # [T, H*V], float32

        # Repeat q and k along head dimension by factor 2 (Hv/Hq = 2)
        q_exp = _repeat_interleave_qk_triton(q, k, 2)  # [T, 8, 128]
        k_exp = _repeat_interleave_qk_triton(k, k, 2)  # [T, 8, 128]

        # Prepare output and new_state
        output = torch.zeros(
            (num_seqs, V, Kdim), dtype=torch.float32, device=device
        )
        new_state = torch.zeros((num_seqs, H, V, Kdim), dtype=torch.float32, device=device)

        # Process sequences
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start

            # Initialize state_HKV as [H, K, V] by transposing state: state is [H, V, K]
            state_HKV = state[seq_idx].transpose(-1, -2).contiguous()  # [H, K, V] = [4, 128, 8]

            for i in range(seq_len):
                t = seq_start + i
                # q_exp[t] is [8, K]; use the first head as q_row
                q_row = q_exp[t, 0, :].float()  # [K], float32
                k_row = k_exp[t, 0, :].float()  # [K], float32
                v_row = v[t, 0, :].float()      # [K], float32

                # G and beta for hv=0
                g_t = float(g[t, 0].item())
                beta_t = float(beta[t, 0].item())

                # Update state_HKV:
                # new_state_HKV = g * state_HKV (elementwise over V)
                # Compute remove and update contributions:
                # remove = k_row @ state_HKV -> (1 x V)
                # update = k_row @ (beta * v_row + (1-beta) * old_v)
                k_row_f = k_row.unsqueeze(1)             # [1, K]
                state_HKV_f = state_HKV.float()          # [H, K, V]
                old_v = torch.matmul(k_row_f, state_HKV_f)       # [1, V]
                new_v = beta_t * v_row.unsqueeze(1) + (1.0 - beta_t) * old_v  # [1, V]
                remove = torch.matmul(k_row_f, state_HKV_f)      # [1, V]
                update = torch.matmul(k_row_f, new_v)            # [1, V]
                new_state_HKV = (g_t * state_HKV_f) - remove + update  # [H, K, V]

                # Compute output for this time-step: o_vec = scale * (q_row @ new_state_HKV)
                q_row_f = q_row.unsqueeze(1)  # [1, K]
                o_vec = (scale * torch.matmul(q_row_f, new_state_HKV)).squeeze(1)  # [V], float32
                output[seq_idx] = output[seq_idx] + o_vec.unsqueeze(1)  # broadcast to [1, V], then add

                # Update state_HKV for next i
                state_HKV = new_state_HKV.transpose(-1, -2).contiguous()  # [H, V, K]

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)

        # Return output and updated state (new_state remains zeros as per original example)
        return output_bf16, new_state


def run(*args):
    return ModelNew()(*args)
