import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dc: tl.constexpr):
    # Each program handles one token row
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    # Load token index
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    # Compute base offsets and copy the row into out
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val.to(tl.float32))


@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dp: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val.to(tl.float32))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Original constraints
        B, H, Dc = q_nope.shape
        assert H == 16, "num_qo_heads must be 16"
        assert Dc == 512, "head_dim_ckv must be 512"
        B2, H2, Dp = q_pe.shape
        assert B2 == B, "q_pe batch size must match q_nope"
        assert H2 == H, "q_pe num heads must match q_nope"
        assert Dp == 64, "head_dim_kpe must be 64"

        device = q_nope.device

        # Prepare caches: squeeze the size-1 dimension and cast to float32 for computation
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Dp]

        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                lse[b] = torch.zeros((H,), dtype=torch.float32, device=device)
                output[b] = torch.zeros((H, Dc), dtype=torch.bfloat16, device=device)
                continue

            # Token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from ckv_cache into Kc_flat (float32) -> [L_tokens, Dc]
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=device)
            grid_g = (L_tokens,)
            gather_rows_c_kernel[grid_g](Kc_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)  # [L_tokens, 512]

            # 2) Gather rows from kpe_cache into Kp_flat (float32) -> [L_tokens, Dp]
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=device)
            grid_g = (L_tokens,)
            gather_rows_p_kernel[grid_g](Kp_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)  # [L_tokens, 64]

            # 3) For each head i: compute logits_scaled, lse, attention, and output
            for i in range(H):
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [512]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [64]
                # GEMV: logits_qn = qn @ Kc.T -> [1, L_tokens]
                logits_qn = qn @ Kc.T
                # GEMV: logits_qp = qp @ Kp.T -> [1, L_tokens]
                logits_qp = qp @ Kp.T
                logits = (logits_qn + logits_qp).squeeze(0)      # [L_tokens]
                logits_scaled = logits * sm_scale                # [L_tokens], float32

                # Compute lse per head in base-2: lse = logsumexp(logits_scaled) / ln(2)
                ln2 = 1.4426950408889634
                lse[b, i] = torch.logsumexp(logits_scaled, dim=0) / ln2

                # Softmax over tokens (dim=0)
                attn = torch.softmax(logits_scaled, dim=0)       # [L_tokens]

                # Final projection: out_vec[i, :] = attn @ Kc -> [512]
                out_vec = attn @ Kc                             # [512], float32
                output[b, i] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
