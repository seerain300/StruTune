import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_allheads_token_kernel(
    q_nope_ptr, q_pe_ptr, Kc_all_ptr, Kp_all_ptr, sparse_idx_ptr,
    out_ptr, lse_ptr,
    N, H, Dk, Dp, topk,
    sm_scale, inv_log2,
):
    # One Triton program per token
    t = tl.program_id(0)

    # Load indices for this token; sparse_idx_ptr is [N*topk]
    base = t * topk
    j = tl.arange(0, topk)
    idx_vals = tl.load(sparse_idx_ptr + base + j, mask=j < topk, other=-1)  # int32
    valid = idx_vals != -1  # [topk] boolean

    # Compute base offsets for q_nope, q_pe
    qno_base = t * (H * Dk)  # row start for this token in q_nope
    qpe_base = t * (H * Dp)  # row start for this token in q_pe

    for h in range(0, H):
        qno_h_base = qno_base + h * Dk
        qpe_h_base = qpe_base + h * Dp

        # Prepare vectors q_nope row and q_pe row
        qno_vec = tl.load(q_nope_ptr + qno_h_base + tl.arange(0, Dk))  # [Dk]
        qpe_vec = tl.load(q_pe_ptr + qpe_h_base + tl.arange(0, Dp))   # [Dp]

        # Load Kc and Kp vectors for each candidate with mask
        # Note: idx_vals may contain -1 for padding; we mask them out.
        # We'll still attempt loads but will ignore them via mask in reductions.
        # idx_vals is int32; convert to float for pointer arithmetic
        j_idx_f = idx_vals.to(tl.float32)  # [topk]
        # Load Kc_all and Kp_all for valid candidates
        Kc_vecs = tl.load(Kc_all_ptr + j_idx_f * Dk + tl.arange(0, Dk), mask=valid, other=0.0)  # [topk, Dk]
        Kp_vecs = tl.load(Kp_all_ptr + j_idx_f * Dp + tl.arange(0, Dp), mask=valid, other=0.0)  # [topk, Dp]

        # Compute logits per candidate: qno_vec @ Kc + qpe_vec @ Kp
        dot1 = (Kc_vecs * qno_vec[None, :]).sum(axis=1)  # [topk]
        dot2 = (Kp_vecs * qpe_vec[None, :]).sum(axis=1)  # [topk]
        logits = dot1 + dot2  # [topk], float32

        # Scale
        logit_scaled = logits * sm_scale  # [topk]

        # Compute logsumexp (scaled) for this head
        m = tl.max(logit_scaled, axis=0)                 # scalar
        sum_exp = tl.sum(tl.exp(logit_scaled - m), axis=0)  # scalar
        lse_h = m + tl.log(sum_exp) * inv_log2          # scalar
        tl.store(lse_ptr + t * H + h, lse_h)

        # Compute output row: out[h, :] = sum_j exp(logit_scaled - lse_h) * Kc[j, :]
        attn_scaled = logit_scaled - lse_h  # [topk]
        attn = tl.exp(attn_scaled)          # [topk]
        prod = attn[None, :] * Kc_vecs      # [topk, Dk]
        sum_prod = tl.sum(prod, axis=0)     # [Dk]
        out_row = sum_prod / sum_exp        # normalize by sum of attn

        # Store output row h for token t
        out_base = t * (H * Dk) + h * Dk
        tl.store(out_ptr + out_base + tl.arange(0, Dk), out_row)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Prepare data: no torch ops for computation
        device = q_nope.device
        # Ensure contiguity and dtypes: compute in fp32
        q_nope = q_nope.contiguous().to(torch.float32)  # [N, H, Dk]
        q_pe = q_pe.contiguous().to(torch.float32)     # [N, H, Dp]
        # Flatten paged KV cache: [num_pages, 64, Dk] -> [num_pages*64, Dk]
        Kc_all = ckv_cache.contiguous().view(-1, 512).to(torch.float32)  # [num_pages*64, 512]
        Kp_all = kpe_cache.contiguous().view(-1, 64).to(torch.float32)   # [num_pages*64, 64]
        sparse_indices = sparse_indices.contiguous().to(torch.int32)     # [N, topk]

        N = q_nope.shape[0]
        H = q_nope.shape[1]  # 16
        Dk = q_nope.shape[2] # 512
        Dp = q_pe.shape[2]   # 64
        topk = sparse_indices.shape[1]  # 2048

        # Allocate outputs
        out = torch.empty((N, H, Dk), dtype=torch.float32, device=device)  # compute in fp32
        lse = torch.empty((N, H), dtype=torch.float32, device=device)      # fp32

        # Launch Triton kernel: one program per token
        grid = (N,)
        inv_log2 = 1.0 / math.log(2.0)

        attention_allheads_token_kernel[grid](
            q_nope, q_pe, Kc_all, Kp_all, sparse_indices,
            out, lse,
            N, H, Dk, Dp, topk,
            sm_scale, inv_log2,
            num_warps=4,
            num_stages=2,
        )

        # Return: output in bfloat16, lse in float32
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
