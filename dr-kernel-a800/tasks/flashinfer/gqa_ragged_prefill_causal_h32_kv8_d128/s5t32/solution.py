import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_attention_kernel(
    q_ptr,            # *float32, shape [M, G, D], contiguous
    k_ptr,            # *float32, shape [N, GH, D], contiguous
    v_ptr,            # *float32, shape [N, GH, D]
    out_ptr,          # *bfloat16, shape [M, G, D]
    lse_ptr,          # *float32, shape [M, G]
    # indices: per-block device int32 tensors (shape [2]) derived from qo_indptr/kv_indptr
    qo_indptr_ptr,    # *int32, [q_start, q_end]
    kv_indptr_ptr,    # *int32, [kv_start, kv_end]
    # sizes (compile-time constants)
    G: tl.constexpr,           # num_qo_heads (e.g., 32)
    GH: tl.constexpr,          # num_kv_heads * gqa_ratio
    D: tl.constexpr,           # head_dim (e.g., 128)
    M: tl.constexpr,           # number of queries in this block (q_end - q_start)
    N: tl.constexpr,           # number of KV tokens in this block (kv_end - kv_start)
    SM_SCALE: tl.constexpr,    # scaling factor (e.g., 1.0 / sqrt(128))
    BLOCK_D: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    # Load q range [q_start, q_end)
    q_start = tl.load(qo_indptr_ptr + 0).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + 1).to(tl.int32)
    # For this kernel, we assume M == q_end - q_start and N == kv_end - kv_start

    # Process each query position q_idx in this block
    for q_idx in range(0, M):
        # Accumulator for LSE per head [G]
        lse_row = tl.full((G,), -float('inf'), tl.float32)
        # Output accumulator for this q_idx: [G, D]
        out_row = tl.zeros((G, D), dtype=tl.float32)

        # Base offset for this q_idx in q: ((q_idx * G) * D)
        base_q = (q_idx * G) * D

        # Process each output head g
        for g in range(0, G):
            # Build q_vec[g, :] by loading from q_ptr
            q_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                # offset: ((q_idx * G + g) * D + d)
                offset = base_q + g * D + d
                q_val = tl.load(q_ptr + offset)
                q_vec[d] = q_val

            # For each KV group (GQA group)
            for gh in range(0, GH):
                # Accumulator for logits for this (q_idx, g, gh): shape [N]
                logits = tl.zeros((N,), dtype=tl.float32)

                # Tile over N (KV length)
                for n0 in range(0, N, BLOCK_N):
                    n_offsets = n0 + tl.arange(0, BLOCK_N)
                    mask_n = n_offsets < N

                    # Load k_chunk and v_chunk as [BLOCK_N, D] by column-wise loads
                    k_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
                    v_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)

                    for d in range(0, D):
                        for nn in range(0, BLOCK_N):
                            n = n0 + nn
                            if n < N:
                                # offset for k_ptr: ((n * GH + gh) * D + d)
                                offset_k = (n * GH + gh) * D + d
                                k_chunk[nn, d] = tl.load(k_ptr + offset_k)
                                v_chunk[nn, d] = tl.load(v_ptr + offset_k)

                    # Compute logits[n] = sum_d q_vec[d] * k_chunk[n, d]
                    for dd in range(0, D):
                        q_val = q_vec[dd]
                        k_col = k_chunk[:, dd]
                        logits += k_col * q_val

                # Scale logits
                logits = logits * SM_SCALE

                # Apply causal mask: positions n >= q_idx + 1 + delta are -inf
                delta = N - M  # per-block delta
                for n0 in range(0, N, BLOCK_N):
                    n_offsets = n0 + tl.arange(0, BLOCK_N)
                    mask_n = n_offsets < N
                    cond = n_offsets >= (q_idx + 1 + delta)
                    logits = tl.where(mask_n & cond, -float('inf'), logits)

                # Accumulate LSE for this (q_idx, g) across all GH groups in base-2
                lse_row[g] += tl.logsumexp(logits, axis=0) / math.log(2.0)

                # Softmax over N for logits of this GH group
                max_logit = tl.max(logits, axis=0)
                logits = logits - max_logit
                exp_logits = tl.exp(logits)
                sum_exp = tl.sum(exp_logits, axis=0)
                softmax = exp_logits / sum_exp  # [BLOCK_N]

                # Accumulate output: out_row[g, :] += sum_n softmax[n] * v_chunk[n, :]
                for n0 in range(0, N, BLOCK_N):
                    n_offsets = n0 + tl.arange(0, BLOCK_N)
                    mask_n = n_offsets < N
                    for nn in range(0, BLOCK_N):
                        n = n0 + nn
                        if n < N:
                            s = softmax[nn]
                            v_row_base = (n * GH + gh) * D
                            for d in range(0, D):
                                out_row[g, d] += s * tl.load(v_ptr + v_row_base + d)

        # Store output and lse for this q_idx
        out_base = (q_start + q_idx) * G * D
        for g_out in range(0, G):
            for d in range(0, D):
                tl.store(out_ptr + out_base + g_out * D + d, out_row[g_out, d].to(tl.bfloat16))
            tl.store(lse_ptr + (q_start + q_idx) * G + g_out, lse_row[g_out].to(tl.float32))


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, num_qo_heads, head_dim] (bf16), CUDA tensor
        k: [total_kv, num_kv_heads, head_dim] (bf16), CUDA tensor
        v: [total_kv, num_kv_heads, head_dim] (bf16), CUDA tensor
        qo_indptr: [len_indptr] int32 (cumulative lengths per batch element)
        kv_indptr: [len_indptr] int32 (cumulative lengths per batch element)
        sm_scale: float32 scalar (e.g., 1.0 / sqrt(128))
        Returns: (output [total_q, num_qo_heads, head_dim] bf16, lse [total_q, num_qo_heads] float32)
        """
        if not TRITON_AVAILABLE or q.device.type != 'cuda':
            # Fallback to a PyTorch implementation (for safety if Triton/CUDA not available)
            total_q, num_qo_heads, head_dim = q.shape
            total_kv, num_kv_heads, _ = k.shape
            len_indptr = qo_indptr.shape[0]
            assert num_qo_heads == 32 and head_dim == 128 and num_kv_heads == 8
            assert total_q == qo_indptr[-1].item()
            assert total_kv == kv_indptr[-1].item()

            G = num_qo_heads
            GH = num_kv_heads * (G // num_kv_heads)  # gqa_ratio, should be 32
            q_f32 = q.to(torch.float32).contiguous()
            k_f32 = k.to(torch.float32).contiguous()
            v_f32 = v.to(torch.float32).contiguous()

            output = torch.zeros((total_q, G, head_dim), dtype=torch.bfloat16, device=q.device)
            lse = torch.full((total_q, G), -float("inf"), dtype=torch.float32, device=q.device)

            for b in range(len_indptr - 1):
                q_start = int(qo_indptr[b].item())
                q_end = int(qo_indptr[b + 1].item())
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())
                if q_start >= q_end or kv_start >= kv_end:
                    continue

                q_batch = q_f32[q_start:q_end]  # [M, G, D]
                k_batch = k_f32[kv_start:kv_end]  # [N, GH, D]
                v_batch = v_f32[kv_start:kv_end]  # [N, GH, D]
                gqa_ratio = G // num_kv_heads
                k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)
                v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)

                M = q_batch.shape[0]
                N = k_expanded.shape[0]

                logits = torch.einsum('qhd,khd->qhk', q_batch, k_expanded) * sm_scale  # [M, G, N]
                q_positions = torch.arange(M, device=q.device)
                kv_positions = torch.arange(N, device=q.device)
                causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + (N - M))
                logits = logits.masked_fill(~causal_mask, float('-inf'))
                lse[q_start:q_end] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
                attn = torch.softmax(logits, dim=-1)
                output_batch = torch.einsum('qhk,khd->qhd', attn, v_expanded)  # [M, G, D]
                output[q_start:q_end] = output_batch.to(torch.bfloat16)
            return output, lse

        # Ensure contiguity and float32 compute
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)

        total_q, G, D = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        assert G == 32 and D == 128 and num_kv_heads == 8
        GH = num_kv_heads * (G // num_kv_heads)  # should be 32

        output = torch.empty((total_q, G, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, G), dtype=torch.float32, device=q.device)

        # Launch one kernel per block
        for b in range(0, len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            M = q_end - q_start
            N = kv_end - kv_start

            # Prepare device int32 indices for this block
            qo_indptr_block = torch.tensor([q_start, q_end], dtype=torch.int32, device=q.device)
            kv_indptr_block = torch.tensor([kv_start, kv_end], dtype=torch.int32, device=q.device)

            # Launch Triton kernel for this block
            _block_attention_kernel[(1,)](
                q, k, v, output, lse,
                qo_indptr_block, kv_indptr_block,
                G=G, GH=GH, D=D, M=M, N=N,
                SM_SCALE=float(sm_scale),
                BLOCK_D=128, BLOCK_N=128,
                num_warps=4, num_stages=2
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
