import math
import torch
import triton
import triton.language as tl


# Triton kernel: gather rows from a flattened cache into out
# cache_ptr: flattened [num_pages * D], tok_idx_ptr: [num_tokens], out_ptr: [num_tokens * D]
@triton.jit
def gather_rows_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                        num_tokens: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * D
    for k in range(0, D):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * D + k, val)


# Triton kernel: compute per-head lse = logsumexp_base2(logit_row) = (m + log(sum_exp)) / ln(2),
# one program per head, passes over the row twice: compute m (max), then sum_exp (sum of exp(x - m))
# out_ptr: scalar per head lse as float32
@triton.jit
def lse_base2_row_kernel(logit_row_ptr, out_ptr,
                         L: tl.constexpr):
    i = tl.program_id(0)  # head index (unused since grid is 1, but kept for generality)
    m = -float("inf")
    # Pass 1: compute max
    for t in range(0, L):
        val = tl.load(logit_row_ptr + t)
        m = tl.maximum(m, val)
    # Pass 2: compute sum exp
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logit_row_ptr + t)
        sum_exp += tl.exp(val - m)
    lse_val = (m + math.log(sum_exp)) / math.log(2.0)
    tl.store(out_ptr, lse_val)


# Triton kernel: softmax over a row, one program handles the whole row
# logits_ptr: [L] per head
# attn_ptr: [L] per head
@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr,
                       L: tl.constexpr):
    m = -float("inf")
    # Pass 1: compute max
    for t in range(0, L):
        val = tl.load(logits_ptr + t)
        m = tl.maximum(m, val)
    # Pass 2: compute sum of exp
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_ptr + t)
        sum_exp += tl.exp(val - m)
    inv_sum = 1.0 / sum_exp
    # Write normalized softmax values
    for t in range(0, L):
        val = tl.load(logits_ptr + t)
        tl.store(attn_ptr + t, tl.exp(val - m) * inv_sum)


# Triton kernel: matvec per head: given attn_row[L] and Kc[L, D], produce out_vec[D]
# attn_ptr: [L] (we launch one program per head, so pass per-head vector)
# Kc_ptr: [L * D] flattened (contiguous) for the rows we gathered
# out_ptr: [D] per head
@triton.jit
def matvec_kernel(attn_ptr, Kc_ptr, out_ptr,
                  L: tl.constexpr, D: tl.constexpr):
    # One program per head; we pass attn_ptr as [L] and out_ptr as [D] for this head
    base_attn = 0  # in this kernel we assume vector passed directly
    for d in range(0, D):
        acc = 0.0
        for t in range(0, L):
            acc += tl.load(attn_ptr + t) * tl.load(Kc_ptr + t * D + d)
        tl.store(out_ptr + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, Dc], bfloat16
        q_pe:   [B, H, Dp], bfloat16
        ckv_cache: [P, 1, Dc], bfloat16
        kpe_cache: [P, 1, Dp], bfloat16
        kv_indptr: [B+1], int32
        kv_indices: [L_total], int32
        sm_scale: float32 scalar
        Returns:
        output: [B, H, Dc], bfloat16
        lse: [B, H], float32 (log-sum-exp in base-2)
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Triton requires CUDA tensors"
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Prepare flattened caches in float32 for computation
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Dp]

        # Output buffers
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens for this batch element
                for i in range(H):
                    output[b, i] = torch.zeros((Dc,), dtype=torch.bfloat16, device=device)
                lse[b] = torch.zeros((H,), dtype=torch.float32, device=device)
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from caches into Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=device)

            grid_gather = (L_tokens,)
            gather_rows_kernel[grid_gather](Kc_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)  # [L_tokens, Dc]

            gather_rows_kernel[grid_gather](Kp_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)  # [L_tokens, Dp]

            # 2) For each head i: compute logits_scaled, lse, attn, and out_vec
            for i in range(H):
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]

                # Compute logits per token: qn @ Kc.T + qp @ Kp.T -> [L_tokens]
                logits_qn = qn @ Kc.T  # [1, L_tokens]
                logits_qp = qp @ Kp.T  # [1, L_tokens]
                logits = (logits_qn + logits_qp).squeeze(0)  # [L_tokens]
                logits_scaled = logits * sm_scale  # [L_tokens]

                # 3) Compute lse per head using Triton (grid=(1,) since we compute per-head vector)
                lse[b, i] = lse_base2_row_kernel[(1,)](logits_scaled, lse[b, i], L_tokens)

                # 4) Compute attention weights via Triton softmax
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_row_kernel[(L_tokens,)](logits_scaled, attn, L_tokens)

                # 5) Final projection: attn @ Kc -> [Dc]
                out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
                # For matvec_kernel we need attn as [L], Kc as [L*D], out as [D]
                matvec_kernel[(1,)](attn, Kc.contiguous().view(-1), out_vec, L_tokens, Dc)

                # Store output[b, i, :] as bfloat16
                output[b, i] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
