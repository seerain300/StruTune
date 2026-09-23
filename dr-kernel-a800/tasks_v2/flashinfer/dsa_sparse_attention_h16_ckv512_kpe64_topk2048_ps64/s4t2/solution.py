import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_all_heads_kernel(
    q_nope_ptr, q_pe_ptr,   # pointers for this token: [num_qo_heads, 512] and [num_qo_heads, 64]
    Kc_ptr, Kp_ptr,         # [num_valid, 512] and [num_valid, 64], flattened
    out_ptr,                # pointer to out[t] matrix [num_qo_heads, num_valid]
    tok_idx_ptr,            # pointer to valid token indices for this token (int32), length num_valid
    num_valid: tl.constexpr,
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,  # 512
    head_dim_kpe: tl.constexpr,  # 64
    BLOCK_V: tl.constexpr,
):
    # One program instance per token. Loop over heads and chunks of V.
    for h in range(num_qo_heads):
        out_base = out_ptr + h * num_valid  # points to out[t, h, :]
        for chunk in range(0, tl.cdiv(num_valid, BLOCK_V)):
            v_start = chunk * BLOCK_V
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < num_valid
            tok_idx_chunk = tl.load(tok_idx_ptr + v_offsets, mask=mask, other=0).to(tl.int32)
            for i in range(BLOCK_V):
                v = v_start + i
                if v < num_valid:
                    tok_idx = tok_idx_chunk[i]
                    acc1 = 0.0
                    acc2 = 0.0
                    # Dot with Kc: sum over d in [0, head_dim_ckv)
                    for d in range(head_dim_ckv):
                        q_val = tl.load(q_nope_ptr + h * head_dim_ckv + d)
                        kc_val = tl.load(Kc_ptr + tok_idx * head_dim_ckv + d)
                        acc1 += q_val * kc_val
                    # Dot with Kp: sum over d in [0, head_dim_kpe)
                    for d in range(head_dim_kpe):
                        q_val = tl.load(q_pe_ptr + h * head_dim_kpe + d)
                        kp_val = tl.load(Kp_ptr + tok_idx * head_dim_kpe + d)
                        acc2 += q_val * kp_val
                    tl.store(out_base + v, acc1 + acc2)


@triton.jit
def compute_lse_base2_row_kernel(
    logits_ptr,            # pointer to logits vector [num_valid] for one head
    out_lse_ptr,           # pointer to scalar lse for that head
    num_valid: tl.constexpr,
    sm_scale: tl.constexpr,      # scalar scale
    inv_ln2: tl.constexpr,       # 1 / ln(2) scalar
):
    # Compute logsumexp_base2 = log(sum exp(sm_scale * logits)) / ln(2)
    sum_exp = 0.0
    for v in range(num_valid):
        x = tl.load(logits_ptr + v)
        x = x * sm_scale
        sum_exp += tl.exp(x * inv_ln2)  # exp(x / ln(2))
    lse = tl.log(sum_exp) / inv_ln2    # log(sum_exp) * ln(2)
    tl.store(out_lse_ptr, lse)


@triton.jit
def compute_softmax_row_kernel(
    logits_ptr,            # pointer to logits vector [num_valid] for one head
    sm_scale: tl.constexpr,      # scalar scale
    out_softmax_ptr,       # pointer to softmax vector [num_valid]
    num_valid: tl.constexpr,
    inv_ln2: tl.constexpr,
):
    # Compute softmax of (logits * sm_scale) with exponent base 2:
    # softmax_i = exp(sm_scale * logits_i / ln(2)) / sum_j exp(sm_scale * logits_j / ln(2))
    sum_exp = 0.0
    for v in range(num_valid):
        x = tl.load(logits_ptr + v)
        x = x * sm_scale
        sum_exp += tl.exp(x * inv_ln2)

    for v in range(num_valid):
        x = tl.load(logits_ptr + v)
        x = x * sm_scale
        softmax_v = tl.exp(x * inv_ln2) / sum_exp
        tl.store(out_softmax_ptr + v, softmax_v)


@triton.jit
def compute_attention_output_kernel(
    softmax_ptr,           # pointer to softmax vector [num_valid]
    Kc_ptr,                # pointer to selected Kc rows [num_valid, 512], flattened
    out_out_ptr,           # pointer to output vector [512] for this head
    num_valid: tl.constexpr,
    head_dim_ckv: tl.constexpr,   # 512
    BLOCK_V: tl.constexpr,
):
    # Output for head: out = softmax @ Kc_selected
    for d in range(head_dim_ckv):
        acc = 0.0
        for i in range(0, tl.cdiv(num_valid, BLOCK_V)):
            v_start = i * BLOCK_V
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < num_valid
            softmax_chunk = tl.load(softmax_ptr + v_offsets, mask=mask, other=0.0)
            Kc_chunk = tl.load(Kc_ptr + v_offsets * head_dim_ckv + d, mask=mask, other=0.0)
            # Reduce chunk to scalar
            acc += tl.sum(softmax_chunk * Kc_chunk, axis=0)
        tl.store(out_out_ptr + d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure tensors are on CUDA for Triton
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Triton requires CUDA tensors."

        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, page_size, _ = ckv_cache.shape
        topk = sparse_indices.shape[-1]

        # Flatten and cast selected cache to float32
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [num_tokens * num_pages * page_size, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [num_tokens * num_pages * page_size, 64]

        # Allocate output buffers
        logits_out = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        inv_ln2 = 1.0 / math.log(2.0)  # constant for Triton

        # For each token, compute valid indices and run Triton kernels to compute logits for all heads
        for t in range(num_tokens):
            idx = sparse_indices[t]  # [topk]
            valid = (idx != -1)      # [topk]
            tok_idx_list = idx[valid].to(torch.int32)  # [num_valid_used]
            num_valid_used = int(tok_idx_list.numel())

            # Initialize per-token output for all heads
            out_all = torch.empty((num_qo_heads, num_valid_used), dtype=torch.float32, device=device)

            # Launch Triton kernel: one program per head, computing all chunks of V
            BLOCK_V = 256
            grid = (num_qo_heads,)
            compute_logits_all_heads_kernel[grid](
                q_nope[t], q_pe[t],
                Kc_all, Kp_all,
                out_all, tok_idx_list,
                num_valid=num_valid_used,
                num_qo_heads=num_qo_heads,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                BLOCK_V=BLOCK_V,
                num_warps=4,
            )

            # Store logits_out[t, h, :] = out_all[h, :] for valid entries; pad invalid with zeros (they won't be used in softmax)
            logits_out[t, :, :num_valid_used] = out_all  # rest of logits_out[t, :, :] remains zeros

            # Compute lse: logsumexp base-2 of logits_out[t, h, :]
            # We only need the reduced dimension per head. For invalid positions, we left them as zeros (pad), which are fine for lse.
            logits_scaled = logits_out[t] * sm_scale  # [num_qo_heads, topk]

            # Launch Triton lse kernel per head
            out_lse = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            for h in range(num_qo_heads):
                compute_lse_base2_row_kernel[(1,)](
                    logits_scaled[h], out_lse[h],
                    num_valid=topk,
                    sm_scale=sm_scale,
                    inv_ln2=inv_ln2,
                    num_warps=1,
                )
            lse[t] = out_lse  # [num_qo_heads]

        # Final output: compute attention and matmul in Triton per token
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)

        for t in range(num_tokens):
            idx = sparse_indices[t]
            valid = (idx != -1)
            tok_idx_list_t = idx[valid].to(torch.int32)  # [num_valid_used]
            num_valid_used = int(tok_idx_list_t.numel())

            # Per-head attention output
            for h in range(num_qo_heads):
                # softmax vector for this head and token over valid entries only
                logits_scaled_h = logits_out[t, h] * sm_scale  # [topk]
                softmax_vec = torch.empty((num_valid_used,), dtype=torch.float32, device=device)

                # Launch Triton softmax kernel
                compute_softmax_row_kernel[(1,)](
                    logits_scaled_h, sm_scale,
                    softmax_vec,
                    num_valid=num_valid_used,
                    inv_ln2=inv_ln2,
                    num_warps=1,
                )

                # Select corresponding Kc rows
                Kc_selected = Kc_all[tok_idx_list_t]  # [num_valid_used, 512]

                # Compute output for this head: softmax_vec @ Kc_selected (no torch.matmul)
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                compute_attention_output_kernel[(1,)](
                    softmax_vec, Kc_selected,
                    out_vec,
                    num_valid=num_valid_used,
                    head_dim_ckv=head_dim_ckv,
                    BLOCK_V=256,
                    num_warps=4,
                )

                output[t, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
