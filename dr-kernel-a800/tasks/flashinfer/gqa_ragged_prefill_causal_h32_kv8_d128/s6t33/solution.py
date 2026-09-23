import math
import torch
import triton
import triton.language as tl


# Triton kernel: for each (i, h), compute softmax over j of logits[i, h, j] - lse[i, h]
# and accumulate output[i, h, d] += sum_j softmax * v_expanded[j, h, d].
# Assumptions: logits is [Q, H, N], v_expanded is [N, H, D], output is [Q, H, D].
@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr,       # *float32, shape [Q, H, N]
    lse_ptr,          # *float32, shape [Q, H]
    v_ptr,            # *float32, shape [N, H, D] (expanded v)
    out_ptr,          # *float32, shape [Q, H, D]
    Q: tl.constexpr,  # int
    N: tl.constexpr,  # int (key length)
    H: tl.constexpr,  # int (32)
    D: tl.constexpr,  # int (128)
    sm_scale,         # float32 scalar (unused here, kept for potential future use)
    BLOCK_J: tl.constexpr,  # tile size over j
):
    i = tl.program_id(0)  # query index
    h = tl.program_id(1)  # head index
    if (i >= Q) or (h >= H):
        return

    # Load lse[i, h]
    lse = tl.load(lse_ptr + i * H + h)

    # Initialize output accumulator for this (i, h)
    out_offset = i * H * D + h * D
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over j in chunks of BLOCK_J
    for j0 in tl.static_range(0, N, BLOCK_J):
        j_offsets = j0 + tl.arange(0, BLOCK_J)
        mask_j = j_offsets < N

        # Load logits chunk for (i, h, j_offsets): shape [BLOCK_J]
        logits_chunk = tl.load(logits_ptr + i * H * N + h * N + j_offsets, mask=mask_j, other=-float("inf"))
        # Compute softmax over this chunk: exp(logits - lse), masked invalid positions to 0 after exp
        w = tl.exp(logits_chunk - lse)
        # Mask invalid entries by setting them to 0
        w = tl.where(mask_j, w, 0.0)

        # Compute sum of weights for normalization
        sum_w = tl.sum(w, axis=0)
        # Normalize weights
        w = w / sum_w

        # Load v chunk: v[j_offsets, h, :] -> [BLOCK_J, D]
        v_chunk = tl.load(v_ptr + j_offsets[:, None] * H * D + h * D + tl.arange(0, D), mask=mask_j[:, None], other=0.0)

        # Accumulate: acc += sum_j w_j * v_chunk[j, :]
        for jj in tl.static_range(0, BLOCK_J):
            w_j = w[jj]
            v_row = v_chunk[jj, :]  # [D]
            acc += w_j * v_row

    # Store result
    tl.store(out_ptr + out_offset, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        device = q.device
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        # Original code uses bfloat16; we will compute in float32 and cast output back to bfloat16.
        q_f32 = q.contiguous().to(torch.float32)   # [Q, 32, 128]
        k_f32 = k.contiguous().to(torch.float32)   # [K, 8, 128]
        v_f32 = v.contiguous().to(torch.float32)   # [K, 8, 128]

        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        assert total_q == q.shape[0], "qo_indptr[-1] must equal q.size(0)"
        assert total_kv == k.shape[0], "kv_indptr[-1] must equal k.size(0)"

        # Output buffer (float32 for compute, cast to bfloat16 at end)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process segments defined by qo_indptr and kv_indptr
        for b in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            Q = q_end - q_start
            K = kv_end - kv_start

            q_slice = q_f32[q_start:q_end]                 # [Q, 32, 128]
            k_slice = k_f32[kv_start:kv_end]               # [K, 8, 128]
            v_slice = v_f32[kv_start:kv_end]               # [K, 8, 128]

            # Expand to 32 heads (GQA)
            k_expanded = k_slice.repeat_interleave(gqa_ratio, dim=1)  # [K, 32, 128]
            v_expanded = v_slice.repeat_interleave(gqa_ratio, dim=1)  # [K, 32, 128]

            # Compute logits[i, h, j] = q[i, h] * k[j, h] * sm_scale
            # Use torch.einsum for robustness and speed
            logits = torch.einsum('qhd,khd->qhk', q_slice, k_expanded) * sm_scale  # [Q, 32, K]
            # Ensure logits are contiguous [Q, 32, K]
            logits = logits.contiguous()

            # Compute lse[i, h] = logsumexp(logits[i,h,:]) / log(2)
            lse_slice = torch.logsumexp(logits, dim=-1)  # [Q, 32]
            lse_slice = lse_slice / math.log(2.0)         # divide by log(2) exactly as original
            # Assign to lse per segment
            lse[q_start:q_end] = lse_slice

            # Prepare output buffer for this segment
            out_slice = torch.empty((Q, num_qo_heads, head_dim), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute output per (i, h): accumulation with softmax over keys
            grid = (Q, num_qo_heads)
            _softmax_accum_output_kernel[grid](
                logits, lse[q_start:q_end], v_expanded, out_slice,
                Q=Q, N=K, H=num_qo_heads, D=head_dim,
                sm_scale=float(sm_scale),  # not used in kernel (lse already normalized)
                BLOCK_J=64,                # tile over key dimension; adjust if needed
                num_warps=4,
                num_stages=2,
            )

            # Accumulate into final output
            output[q_start:q_start + Q] = out_slice

        # Cast output to bfloat16 as in the original code
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
