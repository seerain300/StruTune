import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_all_heads_kernel(
    q_nope_ptr, q_pe_ptr,   # [num_qo_heads, 512] and [num_qo_heads, 64] for this token
    Kc_ptr, Kp_ptr,         # [N, 512] and [N, 64] flattened
    out_ptr,                # [num_qo_heads, 2048] logits output
    tok_idx_ptr,            # [num_valid_used], int32
    num_valid_used: tl.constexpr,   # number of valid indices (<= 2048)
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,     # 512
    head_dim_kpe: tl.constexpr,     # 64
    topk: tl.constexpr,             # 2048
):
    # One program instance per head. Loop over v in [0..topk-1]; compute dot for valid v, else write 0.
    for h in range(num_qo_heads):
        out_base = out_ptr + h * topk
        for v in range(topk):
            if v < num_valid_used:
                tok_idx = tl.load(tok_idx_ptr + v)
                acc1 = 0.0
                # Dot with Kc selected row: sum over d in [0, head_dim_ckv)
                for d in range(head_dim_ckv):
                    q_val = tl.load(q_nope_ptr + h * head_dim_ckv + d)
                    kc_val = tl.load(Kc_ptr + tok_idx * head_dim_ckv + d)
                    acc1 += q_val * kc_val
                acc2 = 0.0
                # Dot with Kp selected row: sum over d in [0, head_dim_kpe)
                for d in range(head_dim_kpe):
                    q_val = tl.load(q_pe_ptr + h * head_dim_kpe + d)
                    kp_val = tl.load(Kp_ptr + tok_idx * head_dim_kpe + d)
                    acc2 += q_val * kp_val
                val = acc1 + acc2
            else:
                val = 0.0
            tl.store(out_base + v, val)


@triton.jit
def compute_lse_base2_valid_kernel(
    logits_ptr,            # [topk] for one head
    out_lse_ptr,           # scalar
    num_valid_used: tl.constexpr,
    sm_scale: tl.constexpr,      # scalar scale
    inv_ln2: tl.constexpr,       # 1 / ln(2)
):
    # Reduce over valid entries only
    sum_exp = 0.0
    for v in range(num_valid_used):
        x = tl.load(logits_ptr + v)
        x = x * sm_scale
        sum_exp += tl.exp(x * inv_ln2)  # exp(x / ln(2))
    lse = tl.log(sum_exp) * ln2  # log(sum_exp) * ln(2)
    tl.store(out_lse_ptr, lse)


@triton.jit
def compute_softmax_valid_kernel(
    logits_ptr,            # [topk] for one head
    sm_scale: tl.constexpr,      # scalar scale
    out_softmax_ptr,       # [num_valid_used]
    num_valid_used: tl.constexpr,
):
    # Compute softmax over valid entries only: exp(sm_scale * logits) / sum
    sum_exp = 0.0
    for v in range(num_valid_used):
        x = tl.load(logits_ptr + v)
        x = x * sm_scale
        sum_exp += tl.exp(x)
    for v in range(num_valid_used):
        x = tl.load(logits_ptr + v)
        x = x * sm_scale
        softmax_v = tl.exp(x) / sum_exp
        tl.store(out_softmax_ptr + v, softmax_v)


@triton.jit
def attention_output_reduce_kernel(
    softmax_ptr,           # [num_valid_used], float
    idx_list_ptr,          # [num_valid_used], int32 (tok indices)
    Kc_ptr,                # [N, 512] flattened
    out_out_ptr,           # [head_dim_ckv]
    num_valid_used: tl.constexpr,
    head_dim_ckv: tl.constexpr,   # 512
):
    # out = softmax @ Kc_selected, where Kc_selected rows are Kc_all[idx_list_ptr[v], :]
    for d in range(head_dim_ckv):
        acc = 0.0
        for v in range(num_valid_used):
            softmax_v = tl.load(softmax_ptr + v)
            idx = tl.load(idx_list_ptr + v)  # int32
            kc_val = tl.load(Kc_ptr + idx * head_dim_ckv + d)
            acc += softmax_v * kc_val
        tl.store(out_out_ptr + d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Triton requires CUDA tensors
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Triton requires CUDA tensors."

        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, page_size, _ = ckv_cache.shape
        topk = sparse_indices.shape[-1]
        assert topk == 2048, "topk must be 2048"

        # Flatten and cast selected cache to float32
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
            compute_logits_all_heads_kernel[(num_qo_heads,)](
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

            # Compute lse: logsumexp base-2 over valid entries only
            for h in range(num_qo_heads):
                compute_lse_base2_valid_kernel[(1,)](
                    logits_out[t, h], lse[t, h],
                    num_valid_used=num_valid_used,
                    sm_scale=sm_scale,
                    inv_ln2=inv_ln2,
                    num_warps=1,
                )

            # Final output: compute attention and matvec in Triton per token, per head
            output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
            for h in range(num_qo_heads):
                # softmax over valid entries only
                logits_scaled_h = logits_out[t, h] * sm_scale  # [2048]; only first num_valid_used are valid
                softmax_vec = torch.empty((num_valid_used,), dtype=torch.float32, device=device)
                compute_softmax_valid_kernel[(1,)](
                    logits_scaled_h, sm_scale,
                    softmax_vec,
                    num_valid_used=num_valid_used,
                    num_warps=1,
                )

                # Prepare idx_list_int tensor for Triton
                idx_list_int = tok_idx_list.contiguous()

                # Compute output vector for this head using Triton reduction
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                attention_output_reduce_kernel[(1,)](
                    softmax_vec, idx_list_int, Kc_all, out_vec,
                    num_valid_used=num_valid_used,
                    head_dim_ckv=head_dim_ckv,
                    num_warps=1,
                )

                # Assign to output and cast to bfloat16
                output[t, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
