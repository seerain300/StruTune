import math
import torch
import triton
import triton.language as tl


# Softmax with causal mask for a single row (vector). Out_ptr points to a 1D buffer with length=N.
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,            # currently unused; scale for multiplication (1.0)
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + idx, mask=idx < N, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = x * scale
    j = idx
    causal_mask = j > absolute_pos
    # For causal positions, set to -inf so exp(-inf)=0
    x = tl.where(causal_mask, -float("inf"), x)
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp
    tl.store(Out_ptr + row_id * N + idx, softmax, mask=idx < N)


# Base-2 logsumexp with causal mask for a single row (vector). Out_ptr points to a 1D buffer with length=1.
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,            # currently unused; scale for multiplication (1.0)
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)  # only one row is processed per launch
    idx = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + idx, mask=idx < N, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = x * scale
    j = idx
    causal_mask = j > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)
    sum_exp = tl.sum(tl.exp(x), axis=0)
    ln2 = 0.6931471805599453  # math.log(2.0)
    l = tl.log(sum_exp) / ln2
    tl.store(Out_ptr, l)  # store a single scalar l at Out_ptr[0]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA device (Triton requires CUDA tensors)
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA for Triton."

        total_q = q_nope.shape[0]
        device = q_nope.device
        num_qo_heads = 16
        head_dim_ckv = 512
        head_dim_kpe = 64

        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # We only use the first len_indptr - 1 batches; with provided inputs, len_indptr=2.
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # Key/value tokens for this batch
            page_beg = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            if page_beg >= kv_end:
                continue

            tok_idx = kv_indices[page_beg:kv_end]  # [kv_len]
            Kc = ckv_cache[tok_idx]  # [kv_len, 512]
            Kp = kpe_cache[tok_idx]  # [kv_len, 64]

            for i in range(q_len):
                abs_q = q_start + i

                # Compute scores_n = qn @ Kc.T, scores_p = qp @ Kp.T
                qn = q_nope[abs_q]  # [16, 512]
                qp = q_pe[abs_q]    # [16, 64]
                scores_n = triton.ops.matmul(qn, Kc.transpose(0, 1))  # [16, kv_len]
                scores_p = triton.ops.matmul(qp, Kp.transpose(0, 1))  # [16, kv_len]
                scores = scores_n + scores_p  # [16, kv_len]

                # Stable softmax with causal mask per head
                kv_len = scores.shape[1]
                prefix_len = (kv_end - page_beg) - q_len
                absolute_pos = prefix_len + i  # absolute position in the stream for this query

                # Softmax output for each head
                attn = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                # Launch per head (grid over num_qo_heads)
                for h in range(num_qo_heads):
                    # scores[h] is [kv_len], pass absolute_pos and N=kv_len
                    out_row = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    softmax_row_causal_kernel[(1,)](
                        scores[h], out_row,
                        kv_len, 1.0, absolute_pos,
                        BLOCK=kv_len
                    )
                    attn[h] = out_row

                # Compute output: out = attn @ Kc
                out = triton.ops.matmul(attn, Kc)  # [16, 512]
                output[abs_q] = out  # fp32

                # lse per head
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                for h in range(num_qo_heads):
                    lse_row[h] = lse_row_causal_kernel[(1,)](
                        scores[h], lse_row[h].new_empty(1),  # Triton expects Out_ptr to a 1-element tensor
                        kv_len, 1.0, absolute_pos,
                        BLOCK=kv_len
                    )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as per original requirements
        output = output.to(torch.bfloat16)
        # Return output [total_q, 16, 512] and lse [total_q, 16] (float32)
        # Note: original lse is shape (total_q, 16), so we reshape lse to match: keep as [total_q, 16]
        return output, lse


def run(*args):
    return ModelNew()(*args)
