import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_single_segment_kernel(
    q_ptr, k_ptr, v_ptr,
    out_ptr, lse_ptr,
    Q: tl.int32, K: tl.int32,  # segment lengths
    sm_scale: tl.float32,       # scaling factor
    ln2: tl.float32,            # log(2)
    H: tl.constexpr,            # num_qo_heads = 32
    gqa_ratio: tl.constexpr,    # num_qo_heads // num_kv_heads = 4
    head_dim: tl.constexpr,     # 128
):
    # We will process one segment entirely inside the kernel.
    # Grid is 1D with 1 program per segment (handled by host loop).
    # i: query index, j: key index, h: head index
    # Compute per (i, h) the attention and accumulate output across heads.
    # We need to iterate i and j, and reduce across d=0..head_dim-1.

    # Prepare base pointers to segment starts (q, k, v) and output lse arrays.
    # Triton expects linear indexing; we pass pointers to the beginning of each segment.
    # We will compute linear offsets accordingly.

    # Note: Triton doesn't allow many control flows with dynamic bounds in static_range,
    # so we use Python for loops for i in range(Q), j in range(K). These work fine as long
    # as the loop body uses simple arithmetic and no unsupported constructs.

    # We compute logits[i, h, j] for all i, h, j and then apply mask, lse, softmax, and output.

    # Masks and constants
    delta = K - Q  # segment-specific delta

    # Loop over queries i
    for i in range(0, Q):
        # Loop over heads h
        for h in range(0, H):
            # Compute q_row vector across head_dim
            # q is laid out as [Q, H, head_dim], so offset = i * H * head_dim + h * head_dim
            q_row_base = i * H * head_dim + h * head_dim
            q_row = tl.zeros([head_dim], dtype=tl.float32)
            # Accumulate q[i, h, :]
            # Here we directly load each element to form a vector q_row
            # Triton vector ops: we can use tl.load(q_ptr + q_row_base + d, mask, other=0)
            for d in range(0, head_dim):
                q_row[d] = tl.load(q_ptr + q_row_base + d)

            # Initialize logits for this (i, h)
            logits = tl.full([K], -float("inf"), dtype=tl.float32)

            # Compute scores for all keys j
            for j in range(0, K):
                # Compute k_expanded[j, h, :] across d
                k_row_base = j * H * head_dim + h * head_dim
                k_row = tl.zeros([head_dim], dtype=tl.float32)
                for d in range(0, head_dim):
                    k_row[d] = tl.load(k_ptr + k_row_base + d)

                # Dot product across head_dim
                dot = 0.0
                for d in range(0, head_dim):
                    dot += q_row[d] * k_row[d]

                # Scale
                score = dot * sm_scale

                # Mask: allow j if j < (i + 1 + delta), else set to -inf
                valid = j < (i + 1 + delta)
                if valid:
                    logits[j] = score
                # else keep -inf

            # Compute lse for this (i, h): logsumexp in base-2
            # lse = max(logits) + log(sum(exp(logits - max))) / ln(2)
            max_logits = tl.max(logits)
            sum_exp = 0.0
            for j in range(0, K):
                sum_exp += tl.exp(logits[j] - max_logits)
            lse_val = max_logits + tl.log(sum_exp) / ln2

            # Store lse[i, h]
            # lse is laid out as [Q, H] contiguous; offset = i * H + h
            tl.store(lse_ptr + i * H + h, lse_val)

            # Now compute output[i, h, :] = sum_j attn[i, h, j] * v_expanded[j, h, :]
            # Compute denom[i, h] = sum_j exp(logits[j] - lse_val) / ln(2)
            denom = 0.0
            for j in range(0, K):
                numerator = tl.exp(logits[j] - lse_val) / ln2
                v_row_base = j * H * head_dim + h * head_dim
                v_row = tl.zeros([head_dim], dtype=tl.float32)
                for d in range(0, head_dim):
                    v_row[d] = tl.load(v_ptr + v_row_base + d)
                # Accumulate into output row for (i, h)
                out_row_base = i * H * head_dim + h * head_dim
                for d in range(0, head_dim):
                    out_ptr[out_row_base + d] += numerator * v_row[d]
                # denom += numerator
            # The kernel stores output[i, h, :] in out_ptr; we did it above. No need to track denom explicitly.
            # Note: We directly accumulate into output row as we compute numerator. We don't need denom for output.
            # If we needed denom, we could have summed numerator over j; but output already computed via numerator * v_row.


def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    # Constraints/assertions as in original
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert total_q == int(qo_indptr[-1].item())
    assert total_kv == int(kv_indptr[-1].item())

    device = q.device
    # Output in bfloat16; lse in float32
    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    ln2 = math.log(2.0)

    # Iterate over segments; per segment, launch the Triton kernel
    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            # No queries or KV for this batch element
            continue

        # Slicing to get segment tensors. Note: these are views, but we pass pointers to them.
        q_batch = q[q_start:q_end]
        k_batch = k[kv_start:kv_end]
        v_batch = v[kv_start:kv_end]

        # Allocate per-segment output and lse
        out_batch = torch.empty((q_end - q_start, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse_batch = torch.empty((q_end - q_start, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel for this segment
        # Grid is 1 program per segment. Triton expects pointers; we pass flat pointers.
        # We'll call the kernel with Q=q_end-q_start, K=kv_end-kv_start.
        Q = q_batch.shape[0]
        K = k_batch.shape[0]
        _attention_single_segment_kernel[(1,)](
            q_batch, k_batch, v_batch,
            out_batch, lse_batch,
            Q, K,
            sm_scale, ln2,
            H=32, gqa_ratio=4, head_dim=128,
            num_warps=4,
        )

        # Copy back to global output and lse
        output[q_start:q_end] = out_batch
        lse[q_start:q_end] = lse_batch

    return output, lse


def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).cuda()
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).cuda()
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return fused_operator(*args)


def run(*args):
    return ModelNew()(*args)
