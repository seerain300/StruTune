import math
import torch
import triton
import triton.language as tl


# Row-wise softmax with causal mask (stable).
# For each row, apply causal mask: positions j > (prefix_len + i) are set to -inf.
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,             # usually 1.0
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,          # power of 2, e.g., 16
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + row_id * N + offsets, mask=mask, other=-float("inf"))
    # Apply causal mask: set j > absolute_pos to -inf
    causal_mask = offsets > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)
    # Stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    out = exp_x / denom
    tl.store(Out_ptr + row_id * N + offsets, out, mask=mask)


# Row-wise logsumexp with causal mask (base-2). Computes lse per row as scalar.
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,             # 1.0 (no scaling)
    absolute_pos: tl.int32,       # prefix_len + i
    BLOCK: tl.constexpr,          # power of 2, e.g., 16
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + row_id * N + offsets, mask=mask, other=-float("inf"))
    # Apply causal mask
    causal_mask = offsets > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    lse = tl.log(sum_exp) / tl.log(2.0)  # base-2
    tl.store(Out_ptr + row_id, lse)  # one scalar per row


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move inputs to CUDA device (Triton requires CUDA tensors).
        assert q_nope.is_cuda and q_pe.is_cuda, "Inputs must be CUDA tensors for Triton."
        device = q_nope.device

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        assert num_qo_heads == 16, "num_qo_heads must be 16."
        head_dim_ckv = q_nope.shape[2]
        assert head_dim_ckv == 512, "head_dim_ckv must be 512."
        head_dim_kpe = q_pe.shape[2]
        assert head_dim_kpe == 64, "head_dim_kpe must be 64."

        # Preload Kc_all and Kp_all from caches
        # ckv_cache: [num_pages, 1, 512] -> Kc_all [num_pages, 512]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element b
        for b in range(1, qo_indptr.numel()):
            q_start = int(qo_indptr[b - 1].item())
            q_end = int(qo_indptr[b].item())
            q_len = q_end - q_start

            page_beg = int(kv_indptr[b - 1].item())
            page_end = int(kv_indptr[b].item())
            kv_len = page_end - page_beg

            # Skip if empty
            if q_len <= 0 or kv_len <= 0:
                continue

            # Collect Kc, Kp for this batch
            tok_idx = torch.arange(page_beg, page_end, device=device)  # indices into Kc_all
            Kc = Kc_all[tok_idx]  # [kv_len, 512], float32
            Kp = Kp_all[tok_idx]  # [kv_len, 64], float32

            # Process each query in this batch
            for i in range(q_len):
                abs_q = q_start + i
                # Load qn and qp
                qn = q_nope[abs_q].to(torch.float32)  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32)   # [16, 64]

                # Matmul: scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.matmul(qn, Kc.transpose(0, 1))  # [16, kv_len], float32
                # Matmul: scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.matmul(qp, Kp.transpose(0, 1))  # [16, kv_len], float32

                # Combine
                scores = scores_n + scores_p  # [16, kv_len]

                # Compute causal mask absolute position: prefix_len = kv_len - q_len
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i  # scalar int

                # Row-wise softmax with causal mask
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                # Ensure BLOCK is power of 2 (e.g., 16)
                softmax_row_causal_kernel[(16,)](
                    scores, attn,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=16, num_warps=1, num_stages=1
                )

                # Output = attn @ Kc -> [16, 512]
                out_row = torch.matmul(attn, Kc)  # [16, 512], float32
                # Save output
                output[abs_q] = out_row

                # lse per head (base-2)
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(16,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=16, num_warps=1, num_stages=1
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as original code does
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
