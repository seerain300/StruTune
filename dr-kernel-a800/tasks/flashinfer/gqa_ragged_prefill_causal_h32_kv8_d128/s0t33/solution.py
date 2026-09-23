import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_forward_segment_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    q_start, q_end, kv_start, kv_end, sm_scale, delta,
    NUM_Q, NUM_K,
):
    """
    Triton kernel: process one (i, h) for the segment.
    q_ptr: [NUM_Q, 32, 128] float32
    k_ptr: [NUM_K, 32, 128] float32 (k_expanded after repeat_interleave(4) for the segment)
    v_ptr: [NUM_K, 32, 128] float32 (v_expanded similarly)
    out_ptr: [NUM_Q, 32, 128] float32 (accumulator)
    lse_ptr: [NUM_Q, 32] float32 (base-2 logsumexp)
    q_start, q_end, kv_start, kv_end: int32 scalars (offsets)
    sm_scale: float32
    delta: int32 (Nk - Nq for this segment)
    grid: (NUM_Q * 32,) — one program per (i,h)
    """
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32

    if i >= NUM_Q:
        return

    # Load q[i, h, :]
    q_lin = (q_start + i) * 32 * 128 + h * 128
    q_vec = tl.load(q_ptr + q_lin)  # [128] float32

    # Compute logits vector for j in 0..7 (8 positions), apply causal mask
    logits = tl.zeros((8,), dtype=tl.float32)
    for j in tl.static_range(8):
        orig_h = h % 8
        # k_expanded line for this j and orig_h
        k_lin = (kv_start + j) * 32 * 128 + orig_h * 128
        k_vec = tl.load(k_ptr + k_lin)  # [128] float32

        dot = tl.sum(q_vec * k_vec, axis=0) * sm_scale
        # Apply forward-causal mask: if j >= (i + 1 + delta), set to -inf
        if j >= (i + 1 + delta):
            dot = -float("inf")
        logits[j] = dot

    # Base-2 logsumexp across 8 positions
    m = tl.max(logits, axis=0)
    sum_exp = tl.sum(tl.exp(logits - m), axis=0)
    lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

    # Softmax across 8 positions
    soft = tl.exp(logits - lse_val)  # [8] float32

    # Accumulate output[i, h, :] = sum over j of soft[j] * v_expanded[:, j, orig_h]
    # We need v_expanded[k, 32, 128]; only v_ptr with j-th column for this orig_h
    for j in tl.static_range(8):
        orig_h = h % 8
        v_lin = (kv_start + j) * 32 * 128 + orig_h * 128
        v_vec = tl.load(v_ptr + v_lin)  # [128] float32
        out_lin = (q_start + i) * 32 * 128 + h * 128
        # Atomic add to support potential grid larger than 1 in case, but here grid=(NUM_Q*32,)
        tl.atomic_add(out_ptr + out_lin, v_vec * soft[j])

    # Store lse[i, h] (base-2)
    lse_lin = i * 32 + h
    tl.store(lse_ptr + lse_lin, lse_val)

def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q = int(q.shape[0])
    num_qo_heads = int(q.shape[1])  # 32
    head_dim = int(q.shape[2])      # 128

    total_kv = int(k.shape[0])
    num_kv_heads = int(k.shape[1])  # 8

    len_indptr = int(qo_indptr.numel())

    # Checks (same as original)
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert total_q == int(qo_indptr[-1].item())
    assert total_kv == int(kv_indptr[-1].item())

    device = q.device

    # Cast to float32 for computation
    q_f32 = q.to(torch.float32).contiguous()
    k_f32 = k.to(torch.float32).contiguous()
    v_f32 = v.to(torch.float32).contiguous()

    # Output buffers
    output = torch.zeros(
        (total_q, num_qo_heads, head_dim),
        dtype=torch.float32,
        device=device,
    )
    lse = torch.full(
        (total_q, num_qo_heads),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )

    gqa_ratio = num_qo_heads // num_kv_heads  # 4

    # Iterate over segments (no recursion; Triton kernel is launched per segment)
    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        Nq = q_end - q_start
        Nk = kv_end - kv_start

        if Nq <= 0 or Nk <= 0:
            continue

        # Slice and expand for this segment
        q_batch = q_f32[q_start:q_end]   # [Nq, 32, 128]
        k_batch = k_f32[kv_start:kv_end] # [Nk, 8, 128], we will use k_expanded conceptually in kernel
        v_batch = v_f32[kv_start:kv_end] # [Nk, 8, 128]

        # k_expanded and v_expanded are conceptually handled inside kernel by indexing j and h%8.
        # We pass q_batch, k_batch, v_batch to kernel, which computes as needed.

        # Launch Triton kernel: one program per (i, h)
        grid = (Nq * num_qo_heads,)
        attention_forward_segment_kernel[grid](
            q_batch, k_batch, v_batch, output, lse,
            q_start, q_end, kv_start, kv_end, sm_scale, (Nk - Nq),
            Nq, (kv_end - kv_start),
            num_warps=1, num_stages=1,
        )

    # Return outputs in original dtype and lse in float32 (base-2)
    return output.to(torch.bfloat16), lse

def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
