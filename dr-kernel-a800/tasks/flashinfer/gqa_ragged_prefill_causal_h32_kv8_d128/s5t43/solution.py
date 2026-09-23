import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _attention_block_kernel(
    q_ptr,        # *float32, shape [M, G, D]
    k_ptr,        # *float32, shape [N, GH, D]
    v_ptr,        # *float32, shape [N, GH, D]
    out_ptr,      # *float32, shape [M, G, D]
    lse_ptr,      # *float32, shape [M, G]
    sm_scale,     # float
    kv_start,     # int
    q_start,      # int
    M,            # int (runtime)
    N,            # int (runtime)
    D: tl.constexpr,     # head dim (128)
    G: tl.constexpr,     # num query heads (32)
    GH: tl.constexpr      # num kv heads (8)
):
    # One program instance per query index q_idx
    q_idx = tl.program_id(0)  # q_idx in [0, M)

    # Base pointer for this q_idx row in Q: q_ptr layout is [M, G, D] with row stride G*D
    q_row_base = q_ptr + q_idx * G * D

    # Accumulator for LSE per head [G]
    lse_row = tl.full((G,), -float('inf'), tl.float32)

    # Process each query head g
    for g in range(0, G):
        # Load Q vector for head g: shape [D]
        qg = tl.zeros((D,), dtype=tl.float32)
        q_vec_ptr = q_row_base + g * D
        for d in range(0, D):
            qg[d] = tl.load(q_vec_ptr + d)

        # Initialize logits for this head: [N]
        logits = tl.full((N,), -float('inf'), tl.float32)

        # For each KV token j in this block
        for j in range(0, N):
            kv_j = kv_start + j
            # Load K vector across GH heads and accumulate
            k_acc = tl.zeros((D,), dtype=tl.float32)
            for gh in range(0, GH):
                k_vec_ptr = k_ptr + kv_j * GH * D + gh * D
                for d in range(0, D):
                    k_acc[d] += tl.load(k_vec_ptr + d)

            # Compute dot = Qg . K_acc
            dot = tl.zeros((), dtype=tl.float32)
            for d in range(0, D):
                dot += qg[d] * k_acc[d]
            # Apply scaling
            dot = dot * sm_scale

            # Causal mask: j < q_idx + 1 + delta
            # Note: delta is defined per-block at host side (N - M). Runtime int.
            causal = (j < (q_idx + 1 + tl.cast(delta, tl.int32)))
            if causal:
                logits[j] = dot
            else:
                logits[j] = -float('inf')

        # Compute logsumexp along N for this head (base-e)
        # First pass: max
        max_log = tl.full((), -float('inf'), tl.float32)
        for j in range(0, N):
            max_log = tl.maximum(max_log, logits[j])
        # Second pass: sum exp
        sum_exp = tl.zeros((), dtype=tl.float32)
        for j in range(0, N):
            sum_exp += tl.exp(logits[j] - max_log)
        lse_row[g] = max_log + tl.log(sum_exp)

        # Compute output: softmax(logits) * V expanded to all heads (we use first KV head)
        out_acc = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, N):
            e = tl.exp(logits[j] - lse_row[g])
            # Load V vector for j-th KV token, use first head (GH is irrelevant due to repeat)
            v_vec_ptr = v_ptr + kv_j * GH * D  # each head is contiguous D
            vj = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                vj[d] = tl.load(v_vec_ptr + 0 * D + d)
            out_acc += e * vj

        # Store out_acc to out_ptr for this q_idx and head g
        out_row_base = out_ptr + q_idx * G * D
        out_vec_ptr = out_row_base + g * D
        for d in range(0, D):
            tl.store(out_vec_ptr + d, out_acc[d])

    # Store LSE for this q_idx [G]
    lse_out_ptr = lse_ptr + q_idx * G
    for g in range(0, G):
        tl.store(lse_out_ptr + g, lse_row[g])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Input checks
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1
        assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16

        device = q.device
        total_q, G, D = q.shape
        total_kv, GH, _ = k.shape
        assert G == 32 and D == 128 and GH == 8, "This Triton kernel expects G=32, D=128, GH=8"

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Outputs (float32 for accumulation)
        output = torch.empty((total_q, G, D), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, G), dtype=torch.float32, device=device)

        # Run kernel per batch block defined by indptr
        for b in range(0, qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            M = q_end - q_start
            N = kv_end - kv_start
            delta = N - M  # runtime int

            q_block = q_f32[q_start:q_end]      # [M, 32, 128]
            k_block = k_f32[kv_start:kv_end]    # [N, 8, 128]
            v_block = v_f32[kv_start:kv_end]    # [N, 8, 128]

            # Launch Triton kernel: one program per query index q_idx in [0, M)
            grid = (M,)
            _attention_block_kernel[grid](
                q_block, k_block, v_block,
                output, lse,
                float(sm_scale), q_start, q_start,
                M, N,
                D=D, G=G, GH=GH
            )

        # Return output cast to bfloat16 and LSE in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
