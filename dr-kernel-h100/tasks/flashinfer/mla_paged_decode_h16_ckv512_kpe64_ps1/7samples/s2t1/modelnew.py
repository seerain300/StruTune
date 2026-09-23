import torch
import math
import triton
import triton.language as tl


@triton.jit
def _batch_elem_kernel(
    q_nope_b_ptr,      # pointer to q_nope[b] as [H, Dc] contiguous
    q_pe_b_ptr,        # pointer to q_pe[b] as [H, Dp] contiguous
    Kc_all_ptr,        # pointer to Kc_all [num_pages, Dc] contiguous
    Kp_all_ptr,        # pointer to Kp_all [num_pages, Dp] contiguous
    out_ptr,           # pointer to out[b, :, :] as [H, Dc] contiguous (bf16), will be written as f32 then cast
    lse_ptr,           # pointer to lse[b, :] as [H] float32 contiguous
    L_tokens,          # int32: number of tokens in this batch element
    sm_scale,          # float32
    H: tl.constexpr,   # number of heads, compile-time for loops
    Dc: tl.constexpr,  # head_dim_ckv, compile-time for loops
    Dp: tl.constexpr,  # head_dim_kpe, compile-time for loops
):
    # Each program handles one batch element b. H is constexpr, so we can loop over heads.
    # We assume q_nope_b_ptr points to [H, Dc], q_pe_b_ptr to [H, Dp],
    # Kc_all_ptr to [num_pages, Dc], Kp_all_ptr to [num_pages, Dp],
    # out_ptr to [H, Dc], lse_ptr to [H].

    # We will perform operations in float32 for stability, then store to out.

    # Loop over each head h
    for h in range(0, H):
        # We need to accumulate logits across tokens, then compute softmax and output.

        # Initialize max_log and accum for LSE
        max_log = -float('inf')
        accum = 0.0

        # We'll compute logits per token t and update max_log and accum
        for t in range(0, L_tokens):
            # Compute dot products:
            # qn = q_nope_b[h, :] and Kc_row = Kc_all[t, :]
            # qn_ptr = q_nope_b_ptr + h * Dc
            qn_ptr = q_nope_b_ptr + h * Dc
            # Load qn vector: [Dc] as float32
            qn_vec = tl.load(qn_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)
            qn_vec = qn_vec.to(tl.float32)

            # Kc_row[t, :]
            Kc_row_ptr = Kc_all_ptr + t * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)
            Kc_row = Kc_row.to(tl.float32)

            # dot1 = sum_i qn_vec[i] * Kc_row[i]
            dot1 = tl.sum(qn_vec * Kc_row, axis=0)

            # Kp_row[t, :]
            Kp_row_ptr = Kp_all_ptr + t * Dp
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)
            Kp_row = Kp_row.to(tl.float32)

            # qp = q_pe_b[h, :]
            qp_ptr = q_pe_b_ptr + h * Dp
            qp_vec = tl.load(qp_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)
            qp_vec = qp_vec.to(tl.float32)

            dot2 = tl.sum(qp_vec * Kp_row, axis=0)

            logits_t = dot1 + dot2
            # update max_log and accum for LSE
            # We will track in float32 for stability
            max_log = tl.maximum(max_log, logits_t)
            accum += tl.exp((logits_t - max_log) * sm_scale)

        # Compute logsumexp scaled by ln(2): lse = max_log + log(accum) / ln(2)
        ln2 = 0.6931471805599453  # math.log(2)
        lse_h = max_log + tl.log(accum) / ln2
        # store lse for this head
        tl.store(lse_ptr + h, lse_h)

        # Now compute attn[t] for each t and accumulate out[h, :] = sum_t attn[t] * Kc[t, :]
        # We'll do it in chunks of Dc for numerical stability and to write vectors.
        for t in range(0, L_tokens):
            qn_ptr = q_nope_b_ptr + h * Dc
            qn_vec = tl.load(qn_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)
            qn_vec = qn_vec.to(tl.float32)

            Kc_row_ptr = Kc_all_ptr + t * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)
            Kc_row = Kc_row.to(tl.float32)

            dot1 = tl.sum(qn_vec * Kc_row, axis=0)

            Kp_row_ptr = Kp_all_ptr + t * Dp
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)
            Kp_row = Kp_row.to(tl.float32)

            qp_ptr = q_pe_b_ptr + h * Dp
            qp_vec = tl.load(qp_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)
            qp_vec = qp_vec.to(tl.float32)

            dot2 = tl.sum(qp_vec * Kp_row, axis=0)

            logits_t = dot1 + dot2
            attn_t = tl.exp((logits_t - max_log) * sm_scale) / accum  # scalar

            # out[h, :] += attn_t * Kc_row
            out_row_ptr = out_ptr + h * Dc
            out_vec = tl.load(out_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0)
            out_vec = out_vec.to(tl.float32)
            contrib = attn_t * Kc_row
            out_vec += contrib
            tl.store(out_row_ptr + tl.arange(0, Dc), out_vec.to(tl.bfloat16), mask=tl.arange(0, Dc) < Dc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, Dc], bfloat16
        q_pe: [B, H, Dp], bfloat16
        ckv_cache: [N, 1, Dc], bfloat16, squeezed -> [N, Dc]
        kpe_cache: [N, 1, Dp], bfloat16, squeezed -> [N, Dp]
        kv_indptr: [B+1], int32
        kv_indices: [T], int32 (T = num_kv_indices)
        sm_scale: float32
        Returns: output [B, H, Dc], bfloat16; lse [B, H], float32
        """
        device = q_nope.device
        B, H, Dc = q_nope.shape
        _, H_pe, Dp = q_pe.shape
        assert H == H_pe, "num_qo_heads must match between q_nope and q_pe"
        N = ckv_cache.shape[0]
        assert Dc == 512, "head_dim_ckv must be 512"
        assert Dp == 64, "head_dim_kpe must be 64"
        assert H == 16, "num_qo_heads must be 16"

        # Prepare Kc_all and Kp_all as contiguous
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)

        # Prepare outputs
        out = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # For each batch element b, compute its token range and load kv_indices slice
        # But since the kernel will be launched per b, we can pass per-batch views directly.
        for b in range(B):
            # Compute L_tokens and tok_idx for this batch b
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No tokens for this batch element, zero output and skip
                out[b].zero_()
                lse[b].zero_()
                continue

            # We need to gather Kc and Kp rows corresponding to these indices.
            # Note: For the provided test, with len_indptr == B+1, tok_idx = [start..end-1].
            # But to be robust, we’ll gather using kv_indices[start:end].
            tok_idx = kv_indices[start:end].contiguous()

            # Create per-batch contiguous views for q_nope[b] and q_pe[b]
            q_nope_b = q_nope[b].contiguous().to(torch.float32)  # [H, Dc]
            q_pe_b = q_pe[b].contiguous().to(torch.float32)      # [H, Dp]

            # Select Kc rows and Kp rows according to tok_idx
            # Kc_all[tok_idx, :] -> [L_tokens, Dc], Kp_all[tok_idx, :] -> [L_tokens, Dp]
            Kc_sel = Kc_all[tok_idx].contiguous()  # [L_tokens, Dc]
            Kp_sel = Kp_all[tok_idx].contiguous()  # [L_tokens, Dp]

            # Launch Triton kernel for this batch element
            # Grid: one program per batch element
            _batch_elem_kernel[(1,)](
                q_nope_b, q_pe_b, Kc_sel, Kp_sel,
                out[b], lse[b],
                L_tokens, sm_scale,
                H=H, Dc=Dc, Dp=Dp,
                num_warps=4,  # small problem sizes; 4 warps is fine
                num_stages=2
            )

        return out, lse