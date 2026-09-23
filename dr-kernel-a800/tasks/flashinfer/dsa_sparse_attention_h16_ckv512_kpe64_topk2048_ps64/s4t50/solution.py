import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_all_heads_kernel(
    q_nope_ptr, q_pe_ptr,   # [num_qo_heads, 512] and [num_qo_heads, 64] for this token
    Kc_ptr, Kp_ptr,         # [N, 512] and [N, 64] flattened
    out_ptr,                # [num_qo_heads, 2048]
    tok_idx_ptr,            # [num_valid_used], int32
    num_valid_used: tl.constexpr,   # number of valid indices (<= 2048)
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,     # 512
    head_dim_kpe: tl.constexpr,     # 64
    topk: tl.constexpr,             # 2048
):
    # One program instance per token. Loop over heads and v in [0..topk-1].
    for h in range(num_qo_heads):
        out_base = out_ptr + h * topk
        for v in range(topk):
            # For v < num_valid_used: compute dot products; else write 0
            if v < num_valid_used:
                tok_idx = tl.load(tok_idx_ptr + v)
                acc1 = 0.0
                # Dot with Kc: sum over d in [0, head_dim_ckv)
                for d in range(head_dim_ckv):
                    q_val = tl.load(q_nope_ptr + h * head_dim_ckv + d)
                    kc_val = tl.load(Kc_ptr + tok_idx * head_dim_ckv + d)
                    acc1 += q_val * kc_val
                acc2 = 0.0
                # Dot with Kp: sum over d in [0, head_dim_kpe)
                for d in range(head_dim_kpe):
                    q_val = tl.load(q_pe_ptr + h * head_dim_kpe + d)
                    kp_val = tl.load(Kp_ptr + tok_idx * head_dim_kpe + d)
                    acc2 += q_val * kp_val
                val = acc1 + acc2
            else:
                val = 0.0
            tl.store(out_base + v, val)


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
    # softmax_v = exp(sm_scale * logits_v / ln(2)) / sum_w exp(sm_scale * logits_w / ln(2))
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
    BLOCK_V: tl.constexpr,        # chunk size for V
):
    # Output for head: out = softmax[:num_valid_used] @ Kc_selected
    for d in range(head_dim_ckv):
        acc = 0.0
        for i in range(0, tl.cdiv(num_valid_used, BLOCK_V)):
            v_start = i * BLOCK_V
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < num_valid_used
            softmax_chunk = tl.load(softmax_ptr + v_offsets, mask=mask, other=0.0)
            Kc_chunk = tl.load(Kc_ptr + v_offsets * head_dim_ckv + d, mask=mask, other=0.0)
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
        assert topk == 2048, "topk must be 2048"

        # Flatten and cast caches to float32 for computation
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [N, 64]

        # Output buffers
        logits_out = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        ln2 = math.log(2.0)
        inv_ln2 = 1.0 / ln2

        for t in range(num_tokens):
            idx = sparse_indices[t]  # [2048]
            valid = (idx != -1)      # [2048]
            tok_idx_list = idx[valid].to(torch.int32)  # [num_valid_used]
            num_valid_used = int(tok_idx_list.numel())

            # Launch Triton kernel: compute logits for all heads across topk positions
            compute_logits_all_heads_kernel[(1,)](
                q_nope[t], q_pe[t],
                Kc_all, Kp_all,
                logits_out[t], tok_idx_list,
                num_valid_used=num_valid_used,
                num_qo_heads=num_qo_heads,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                topk=topk,
                num_warps=4,
            )

            # Compute lse: logsumexp base-2 over all 2048 entries
            for h in range(num_qo_heads):
                compute_lse_base2_row_kernel[(1,)](
                    logits_out[t, h], lse[t, h],
                    topk=topk,
                    sm_scale=sm_scale,
                    inv_ln2=inv_ln2,
                    num_warps=1,
                )

        # Final output: compute attention and matvec in Triton per token
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)

        for t in range(num_tokens):
            idx = sparse_indices[t]
            valid = (idx != -1)
            tok_idx_list_t = idx[valid].to(torch.int32)  # [num_valid_used]
            num_valid_used = int(tok_idx_list_t.numel())

            # Per-head attention output
            for h in range(num_qo_heads):
                # softmax over all 2048 entries
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

                # Compute output for this head: softmax_vec @ Kc_selected via Triton reduction
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
