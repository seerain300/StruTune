import math
import torch
import triton
import triton.language as tl


@triton.jit
def expand_heads_kernel(
    k_ptr, v_ptr,  # *float32, input [N, 8, 128]
    k_exp_ptr, v_exp_ptr,  # *float32, output [N*32, 32, 128]
    N,  # int32, number of rows (Nk or Nq per segment)
    NUM_HEADS_IN: tl.constexpr,  # 8
    NUM_HEADS_OUT: tl.constexpr,  # 32
    D: tl.constexpr,             # 128
):
    """
    Triton kernel to expand 8 heads to 32 heads using repeat_interleave along head dim.
    Input k_ptr/v_ptr: shape [N, 8, 128]
    Output k_exp_ptr/v_exp_ptr: shape [N*NUM_HEADS_OUT, NUM_HEADS_OUT, D]
    For each input row r, each head in_in, and each output head in_out, copy the 128 dims.
    """
    r = tl.program_id(axis=0)  # row index in [0, N)
    in_out = tl.program_id(axis=1)  # output head index in [0, NUM_HEADS_OUT)
    in_in = tl.program_id(axis=2)   # original head index in [0, NUM_HEADS_IN)

    # Compute input row base: ((r * NUM_HEADS_IN) + in_in) * D
    in_row_base = (r * NUM_HEADS_IN + in_in) * D

    # Destination row index in output: r * NUM_HEADS_OUT + in_out
    out_row_idx = r * NUM_HEADS_OUT + in_out
    out_row_base = out_row_idx * (NUM_HEADS_OUT * D) + in_out * D

    # Copy 128 dims
    for d in range(D):
        val = tl.load(k_ptr + in_row_base + d)
        tl.store(k_exp_ptr + out_row_base + d, val)

        val2 = tl.load(v_ptr + in_row_base + d)
        tl.store(v_exp_ptr + out_row_base + d, val2)


@triton.jit
def per_query_attention_gqa_kernel(
    q_ptr,          # *float32, [total_q, 32, 128]
    k_exp_ptr,      # *float32, [total_kv * 32, 32, 128]
    v_exp_ptr,      # *float32, [total_kv * 32, 32, 128]
    out_ptr,        # *float32, [total_q, 32, 128]
    lse_ptr,        # *float32, [total_q, 32]
    qo_indptr_ptr,  # *int32, [len_indptr]
    kv_indptr_ptr,  # *int32, [len_indptr]
    sm_scale,       # float32
    total_q,        # int32
    total_kv,       # int32
    NUM_SEGMENTS: tl.constexpr,
):
    """
    Triton kernel: one program per (i, h) pair. For each segment b, compute attention for that (i, h).
    k_exp_ptr and v_exp_ptr are expanded versions (32 heads) of k/v for all segments combined.
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    # Prepare base pointers for q[i, h, :]
    q_base = i * 32 * 128 + h * 128
    q_vec = tl.load(q_ptr + q_base)  # [128] float32

    # For each segment b
    for b in range(NUM_SEGMENTS):
        q_start = tl.load(qo_indptr_ptr + b)     # int32
        q_end = tl.load(qo_indptr_ptr + b + 1)   # int32
        kv_start = tl.load(kv_indptr_ptr + b)    # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1)  # int32

        Nq = q_end - q_start
        Nk = kv_end - kv_start
        delta = Nk - Nq  # per-segment delta

        # First, compute logits for j in 0..7 and apply causal mask
        logits = tl.zeros((8,), dtype=tl.float32)

        # We need to sum over k rows for each j. Load k_exp rows: shape [Nk*32, 32, 128]
        # But we don't have per-segment slicing here; we iterate rows r in [kv_start, kv_end)
        # We'll emulate by loading k_exp_ptr rows using r and head index j. Note: Triton supports while loops.
        r = kv_start
        while r < kv_end:
            # For each j, compute dot product with q_vec
            for j in range(8):
                # Destination head index in expanded k: j
                # Row index in expanded k: r * 32 + j
                # Load k_exp[(r*32 + j), j, :] as 128-vector
                row_idx_k = r * 32 + j
                # Base pointer for k_exp[row_idx_k] and v_exp[row_idx_k] at head j
                # k_exp/v_exp layout: [row, head, dim]; row_idx_k * (32 * 128) + j * 128
                # Actually k_exp_ptr is a flat array of rows (Nk*32) each 32*128 bytes; but we passed
                # k_exp_ptr as expanded for all segments; we need to restrict to this segment.
                # To avoid slicing, we set a validity flag: if r < kv_end, we proceed; otherwise skip.
                # But since we loop r < kv_end, this is already ensured.
                k_row_base = row_idx_k * (32 * 128) + j * 128
                k_vec = tl.load(k_exp_ptr + k_row_base)  # [128] float32
                dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
                # Apply forward-causal mask: if j >= (i + 1 + delta), set -inf
                if j >= (i + 1 + delta):
                    dot = -float('inf')
                logits[j] = dot
            r += 1

        # Compute base-2 logsumexp over 8 positions
        m = logits[0]
        for jj in range(1, 8):
            m = tl.maximum(m, logits[jj])
        sum_exp = 0.0
        for jj in range(8):
            sum_exp += tl.exp(logits[jj] - m)
        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

        # Softmax across 8 positions
        soft = tl.zeros((8,), dtype=tl.float32)
        for jj in range(8):
            soft[jj] = tl.exp(logits[jj] - lse_val)

        # Accumulate output[i, h, :] = sum_j soft[j] * sum_r v_exp[r, j, :]
        out_base = i * 32 * 128 + h * 128
        for d in range(128):
            acc = 0.0
            for j in range(8):
                # Sum over rows r in segment: we already computed softmax; we need v_exp rows
                # For each r in segment, load v_exp[r, j, d]
                r_inner = kv_start
                while r_inner < kv_end:
                    row_idx_v = r_inner * 32 + j
                    v_row_base = row_idx_v * (32 * 128) + j * 128 + d
                    val = tl.load(v_exp_ptr + v_row_base)  # scalar
                    acc += val  # sum over rows
                    r_inner += 1
                acc *= soft[j]
            tl.store(out_ptr + out_base + d, acc)

        # Store lse[i, h]
        lse_lin = i * 32 + h
        tl.store(lse_ptr + lse_lin, lse_val)

    # Done


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA and contiguous tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All inputs must be CUDA tensors"
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)

        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Expand k and v to 32 heads using Triton kernel
        Nk = total_kv
        k_exp = torch.empty((Nk * 32, 32, 128), dtype=torch.float32, device=q.device)
        v_exp = torch.empty((Nk * 32, 32, 128), dtype=torch.float32, device=q.device)

        NUM_HEADS_IN = 8
        NUM_HEADS_OUT = 32
        D = 128
        grid_expand = (Nk * NUM_HEADS_IN, NUM_HEADS_OUT, NUM_HEADS_IN)
        expand_heads_kernel[grid_expand](k, v, k_exp, v_exp, Nk, NUM_HEADS_IN, NUM_HEADS_OUT, D)

        # Output and lse buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

        NUM_SEGMENTS = qo_indptr.numel() - 1

        # Launch Triton kernel: one program per (i, h)
        grid = (total_q * num_qo_heads,)
        per_query_attention_gqa_kernel[grid](q, k_exp, v_exp, output, lse, qo_indptr, kv_indptr, sm_scale, total_q, total_kv, NUM_SEGMENTS)

        # Return output in bfloat16 and lse in float32 (base-2 logsumexp)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
