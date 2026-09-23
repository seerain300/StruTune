import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_all_heads_kernel(
    q_nope_ptr, q_pe_ptr,   # [num_qo_heads, 512] and [num_qo_heads, 64] for this token
    Kc_ptr, Kp_ptr,         # [N, 512] and [N, 64] flattened
    out_ptr,                # [num_qo_heads, 2048]
    tok_idx_ptr,            # [num_valid_used], int32 (ignored here; v loops up to 2047)
    topk: tl.constexpr,             # 2048
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,     # 512
    head_dim_kpe: tl.constexpr,     # 64
    BLOCK_V: tl.constexpr,          # chunk size for V (unused directly; we iterate over topk)
):
    # One program instance per token. Loop over all heads and over v positions to fill out[t, h, v]
    # We assume out_ptr points to the base of out[t]; host will pass the correct pointer for t.
    for h in range(num_qo_heads):
        out_base = out_ptr + h * topk
        for v in range(topk):
            acc1 = 0.0
            acc2 = 0.0
            # Compute q_nope dot Kc for selected row using v (padding v has no effect)
            for d in range(head_dim_ckv):
                q_val = tl.load(q_nope_ptr + h * head_dim_ckv + d)
                kc_val = tl.load(Kc_ptr + v * head_dim_ckv + d)  # note: using v as tok_idx
                acc1 += q_val * kc_val
            for d in range(head_dim_kpe):
                q_val = tl.load(q_pe_ptr + h * head_dim_kpe + d)
                kp_val = tl.load(Kp_ptr + v * head_dim_kpe + d)
                acc2 += q_val * kp_val
            tl.store(out_base + v, acc1 + acc2)


@triton.jit
def compute_lse_base2_row_kernel(
    logits_ptr,            # [topk] for one head
    out_lse_ptr,           # scalar
    topk: tl.constexpr,
    sm_scale: tl.constexpr,      # scalar scale
    inv_ln2: tl.constexpr,       # 1 / ln(2)
):
    # Compute logsumexp_base2 = log(sum exp(sm_scale * logits)) / ln(2)
    sum_exp = 0.0
    for v in range(topk):
        x = tl.load(logits_ptr + v)
        x = x * sm_scale
        sum_exp += tl.exp(x * inv_ln2)  # exp(x / ln(2))
    lse = tl.log(sum_exp) * ln2  # log(sum_exp) * ln(2)
    tl.store(out_lse_ptr, lse)


@triton.jit
def compute_softmax_row_kernel(
    logits_ptr,            # [topk] for one head
    sm_scale: tl.constexpr,      # scalar scale
    out_softmax_ptr,       # [topk]
    topk: tl.constexpr,
    inv_ln2: tl.constexpr,
):
    # Compute softmax of (logits * sm_scale) with exponent base 2:
    # softmax_i = exp(sm_scale * logits_i / ln(2)) / sum_j exp(sm_scale * logits_j / ln(2))
    sum_exp = 0.0
    for v in range(topk):
        x = tl.load(logits_ptr + v)
        x = x * sm_scale
        sum_exp += tl.exp(x * inv_ln2)
    for v in range(topk):
        x = tl.load(logits_ptr + v)
        x = x * sm_scale
        softmax_v = tl.exp(x * inv_ln2) / sum_exp
        tl.store(out_softmax_ptr + v, softmax_v)


@triton.jit
def compute_attention_output_kernel(
    softmax_ptr,           # [topk]
    Kc_ptr,                # [num_valid_used, 512]
    out_out_ptr,           # [head_dim_ckv]
    num_valid_used: tl.constexpr,
    head_dim_ckv: tl.constexpr,   # 512
    BLOCK_V: tl.constexpr,        # chunk size for V (unused directly; we iterate over topk)
):
    # Output for head: out = softmax[:num_valid_used] @ Kc_selected
    for d in range(head_dim_ckv):
        acc = 0.0
        for v in range(num_valid_used):
            sm_v = tl.load(softmax_ptr + v)
            kc_vd = tl.load(Kc_ptr + v * head_dim_ckv + d)
            acc += sm_v * kc_vd
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
        assert topk == 2048, "topk must be 2048"

        # Flatten and cast selected cache to float32 for computation
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [N, 64]

        # Output buffers
        logits_out = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        ln2 = math.log(2.0)
        inv_ln2 = 1.0 / ln2

        # We will launch Triton kernels for logits, lse, softmax, and final output
        for t in range(num_tokens):
            # Launch Triton kernel: compute logits for all heads across topk positions
            BLOCK_V = 256
            grid = (num_qo_heads,)
            compute_logits_all_heads_kernel[grid](
                q_nope[t], q_pe[t],
                Kc_all, Kp_all,
                logits_out[t], torch.empty(0, dtype=torch.int32, device=device),
                topk=topk,
                num_qo_heads=num_qo_heads,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                BLOCK_V=BLOCK_V,
                num_warps=4,
            )

            # Compute lse: logsumexp base-2 over all 2048 positions
            for h in range(num_qo_heads):
                compute_lse_base2_row_kernel[(1,)](
                    logits_out[t, h], lse[t, h],
                    topk=topk,
                    sm_scale=sm_scale,
                    inv_ln2=inv_ln2,
                    num_warps=1,
                )

        # Final output: compute attention and matmul in Triton per token
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)

        for t in range(num_tokens):
            idx = sparse_indices[t]  # [topk]
            valid = (idx != -1)      # [topk]
            tok_idx_list_t = idx[valid].to(torch.int32)  # [num_valid_used]
            num_valid_used = int(tok_idx_list_t.numel())

            # Per-head attention output
            for h in range(num_qo_heads):
                # softmax over all 2048 positions (invalid positions contribute exp(0)=1)
                logits_scaled_h = logits_out[t, h] * sm_scale  # [2048]
                softmax_vec = torch.empty((topk,), dtype=torch.float32, device=device)

                compute_softmax_row_kernel[(1,)](
                    logits_scaled_h, sm_scale,
                    softmax_vec,
                    topk=topk,
                    inv_ln2=inv_ln2,
                    num_warps=1,
                )

                # Select corresponding Kc rows
                Kc_selected = Kc_all[tok_idx_list_t]  # [num_valid_used, 512]

                # Compute output for this head: softmax_vec @ Kc_selected (Triton reduction, no torch.matmul)
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                compute_attention_output_kernel[(1,)](
                    softmax_vec, Kc_selected,
                    out_vec,
                    num_valid_used=num_valid_used,
                    head_dim_ckv=head_dim_ckv,
                    BLOCK_V=256,
                    num_warps=4,
                )

                output[t, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
