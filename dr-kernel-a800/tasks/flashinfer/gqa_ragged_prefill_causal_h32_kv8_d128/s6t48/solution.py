import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: per-(i, h), scalar-loop over kv dimension
if TRITON_AVAILABLE:
    @triton.jit
    def _compute_logits_kernel(
        q_ptr,            # *f32, [N, 32, 128]
        k_ptr,            # *f32, [M, 32, 128]
        logits_ptr,       # *f32, [N, M]
        N: tl.constexpr,  # num_q_tokens
        M: tl.constexpr,  # num_kv_tokens
        sm_scale,         # f32 scalar
        q_stride_b,       # int
        q_stride_h,       # int
        q_stride_d,       # int
        k_stride_b,       # int
        k_stride_h,       # int
        k_stride_d,       # int
    ):
        pid = tl.program_id(axis=0)
        i = pid // 32
        h = pid % 32
        # If pid >= N*32, return (safety)
        if i >= N:
            return
        # Iterate over kv positions
        for j in range(0, M):
            # Load q[i, h]
            q_off = i * q_stride_b + h * q_stride_h
            q_val = tl.load(q_ptr + q_off)  # scalar f32
            # Load k[j, h]
            k_off = j * k_stride_b + h * k_stride_h
            k_val = tl.load(k_ptr + k_off)  # scalar f32
            # Compute score
            score = q_val * k_val * sm_scale
            # Causal mask: i can only attend to j < (i + 1 + delta)
            # delta = M - N
            # Triton allows int expressions; compute scalar condition
            valid = (j < (i + 1 + (M - N)))
            if not valid:
                score = -float('inf')
            # Store logits[i, j] (flattened over j)
            logits_off = i * M + j
            tl.store(logits_ptr + logits_off, score)


    @triton.jit
    def _lse_reduce_kernel(
        logits_ptr,       # *f32, [N, M]
        lse_ptr,          # *f32, [N]
        N: tl.constexpr,  # num_q_tokens
        M: tl.constexpr,  # num_kv_tokens
    ):
        pid = tl.program_id(axis=0)
        i = pid // 32
        h = pid % 32
        if i >= N:
            return
        # Compute logsumexp over j for this (i, h)
        m = -float('inf')
        # Find max
        for j in range(0, M):
            logits_off = i * M + j
            val = tl.load(logits_ptr + logits_off)
            m = tl.maximum(m, val)
        # Sum exp(val - m)
        s = 0.0
        for j in range(0, M):
            logits_off = i * M + j
            val = tl.load(logits_ptr + logits_off)
            s += tl.exp(val - m)
        lse = tl.log(s) + m  # standard logsumexp
        tl.store(lse_ptr + (i * 32 + h), lse)


    @triton.jit
    def _softmax_accum_output_kernel(
        logits_ptr,       # *f32, [N, M]
        lse_ptr,          # *f32, [N*32]
        v_ptr,            # *f32, [M, 32, 128]
        out_ptr,          # *f32, [N, 32, 128]
        N: tl.constexpr,  # num_q_tokens
        M: tl.constexpr,  # num_kv_tokens
        q_stride_b,       # int (unused but kept for symmetry)
        q_stride_h,       # int (unused but kept for symmetry)
        q_stride_d,       # int (unused but kept for symmetry)
        v_stride_b,       # int
        v_stride_h,       # int
        v_stride_d,       # int
        out_stride_b,     # int
        out_stride_h,     # int
        out_stride_d,     # int
    ):
        pid = tl.program_id(axis=0)
        i = pid // 32
        h = pid % 32
        if i >= N:
            return
        # Load lse[i, h]
        lse_val = tl.load(lse_ptr + (i * 32 + h))  # scalar f32
        # Accumulate output over j
        for j in range(0, M):
            # score = logits[i, j]
            logits_off = i * M + j
            score = tl.load(logits_ptr + logits_off)  # f32
            # Compute y = exp(score - lse_val)
            y = tl.exp(score - lse_val)
            # v_expanded[j, h, :] = v[j, h, :]
            # v_ptr points to [M, 32, 128] expanded for GQA ratio=1 here, but we need 32 heads expansion
            # For each j, v_expanded[j, h, :] equals v[j, h, :]. Since we expanded before (repeat_interleave 4), we must index the expanded v.
            # We pass v_ptr as [M, 32, 128] already expanded. Safe to index v[j, h, :].
            v_off = j * v_stride_b + h * v_stride_h
            d = tl.arange(0, 128)  # vector of dims
            v_vec = tl.load(v_ptr + v_off + d)  # [128] f32
            out_off = i * out_stride_b + h * out_stride_h + d * out_stride_d
            # Accumulate: out[i, h, d] += y * v_vec[d]
            # Triton supports elementwise operations; we broadcast y by multiplying per element
            out_vec = tl.load(out_ptr + out_off)
            out_vec += y * v_vec
            tl.store(out_ptr + out_off, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Fallback to PyTorch if Triton not available or tensors not on CUDA
        if (not TRITON_AVAILABLE) or (q.device.type != 'cuda'):
            # Replicate original PyTorch run logic for correctness
            total_q = int(qo_indptr[-1].item())
            total_kv = int(kv_indptr[-1].item())
            output = torch.zeros((total_q, 32, 128), dtype=torch.bfloat16, device=q.device)
            lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=q.device)

            for b in range(qo_indptr.shape[0] - 1):
                q_start = int(qo_indptr[b].item())
                q_end = int(qo_indptr[b + 1].item())
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())
                if q_start >= q_end or kv_start >= kv_end:
                    continue

                num_q_tokens = q_end - q_start
                num_kv_tokens = kv_end - kv_start

                q_batch = q[q_start:q_end].to(torch.float32)
                k_batch = k[kv_start:kv_end].to(torch.float32)
                v_batch = v[kv_start:kv_end].to(torch.float32)

                k_expanded = k_batch.repeat_interleave(4, dim=1)
                v_expanded = v_batch.repeat_interleave(4, dim=1)

                # Compute logits, LSE, and output via PyTorch (einsum) to match original behavior
                # logits: [num_q_tokens, 32, num_kv_tokens]
                # We need q[i,h] * k_expanded[j,h], but in PyTorch, use einsum:
                # Build q_expanded [num_q_tokens, 32, 128] is already q_batch
                # We can compute logits by repeating q_batch along kv dimension and k_expanded along q dimension via broadcasting, but better to replicate logic directly:
                # For correctness, use einsum:
                q_exp = q_batch[:, :, None, :]  # [N, 32, 1, 128]
                k_exp = k_expanded[None, :].expand(num_q_tokens, 32, -1, 128)  # [N, 32, M, 128]
                logits_ijk = torch.einsum('nhd,nkhd->nkh', q_exp, k_exp) * sm_scale  # [N, 32, M]

                # Mask causal: i can only attend to j < (i + 1 + delta), delta = M - N
                delta = num_kv_tokens - num_q_tokens
                mask = torch.arange(num_kv_tokens, device=q.device).unsqueeze(0) < (torch.arange(num_q_tokens, device=q.device).unsqueeze(1) + 1 + delta)
                logits_ijk = torch.where(mask, logits_ijk, torch.tensor(float('-inf'), dtype=logits_ijk.dtype, device=logits_ijk.device))

                # LSE per (i,h)
                lse_i = torch.logsumexp(logits_ijk, dim=-1).float()  # [N, 32]

                # Softmax and output: output[i,h,:] = sum_j exp(logits[i,h,j] - lse[i,h]) * v_expanded[j,h,:]
                attn = torch.exp(logits_ijk - lse_i[:, :, None])  # [N, 32, M]
                v_exp = v_expanded[None, :, :, :].expand(num_q_tokens, 32, -1, 128)  # [N, 32, M, 128]
                output_i = torch.einsum('njk,njkd->nd', attn, v_exp)  # [N, 32, 128]
                output[q_start:q_end] = output_i.to(torch.bfloat16)

                # Store LSE
                lse[q_start:q_end] = lse_i

            return output, lse

        # Triton path
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())

        # Output and LSE buffers (float32 for compute)
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        # Process each segment b
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice q, k, v
            q_batch = q[q_start:q_end].to(torch.float32)  # [N, 32, 128]
            k_batch = k[kv_start:kv_end].to(torch.float32)  # [M, 8, 128]
            v_batch = v[kv_start:kv_end].to(torch.float32)  # [M, 8, 128]

            # Expand k and v to 32 heads by repeating along head dim (GQA ratio 4)
            k_expanded = k_batch.repeat_interleave(4, dim=1)  # [M, 32, 128]
            v_expanded = v_batch.repeat_interleave(4, dim=1)  # [M, 32, 128]

            # Make contiguous
            q_f32 = q_batch.contiguous()
            k_f32 = k_expanded.contiguous()
            v_f32 = v_expanded.contiguous()

            # Allocate logits [N, M]
            logits = torch.empty((num_q_tokens, num_kv_tokens), dtype=torch.float32, device=q.device)

            # Launch _compute_logits_kernel: one program per (i,h)
            grid = (num_q_tokens * 32,)
            _compute_logits_kernel[grid](
                q_f32, k_f32, logits,
                num_q_tokens, num_kv_tokens, sm_scale,
                q_f32.stride(0), q_f32.stride(1), q_f32.stride(2),
                k_f32.stride(0), k_f32.stride(1), k_f32.stride(2),
                num_warps=1, num_stages=1
            )

            # Launch _lse_reduce_kernel to compute LSE[i,h]
            _lse_reduce_kernel[grid](
                logits, lse,
                num_q_tokens, num_kv_tokens,
                num_warps=1, num_stages=1
            )

            # Launch _softmax_accum_output_kernel to accumulate output[i,h,:]
            _softmax_accum_output_kernel[(num_q_tokens * 32,)](
                logits, lse, v_f32, output,
                num_q_tokens, num_kv_tokens,
                q_f32.stride(0), q_f32.stride(1), q_f32.stride(2),
                v_f32.stride(0), v_f32.stride(1), v_f32.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                num_warps=1, num_stages=1
            )

        # Convert output to bfloat16 as in original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
