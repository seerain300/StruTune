import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: gather selected rows from global cache into contiguous per-batch buffers
@triton.jit
def gather_tokens_kernel(
    K_src_ptr,        # *f32, flattened [num_pages, head_dim]
    idx_ptr,          # *i32, [L_tokens] token indices
    out_ptr,          # *f32, contiguous [L_tokens, head_dim]
    num_pages: tl.constexpr,  # int
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    src_linear = tok_id * head_dim + offs
    # copy row tok_id into out_ptr[pid, :]
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + pid * head_dim + offs, vals)


# Triton kernel: compute per-head logits vector for a batch: q @ K.T
@triton.jit
def forward_attention_kernel(
    q_ptr,            # *f32, [q_dim]
    K_ptr,            # *f32, [L_tokens, head_dim] contiguous per-batch
    out_ptr,          # *f32, [L_tokens]
    q_dim: tl.constexpr,      # int
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # which element in output vector
    offs_k = tl.arange(0, head_dim)
    acc = 0.0
    # loop over K dimension tiles
    for k in range(0, q_dim, head_dim):
        offs_q = k + offs_k
        mask_q = offs_q < q_dim
        q_vals = tl.load(q_ptr + offs_q, mask=mask_q, other=0.0)
        K_vals = tl.load(K_ptr + pid * head_dim + offs_k)
        acc += tl.sum(q_vals * K_vals, axis=0)
    tl.store(out_ptr + pid, acc)


# Triton kernel: softmax over a vector (x is logits_scaled)
@triton.jit
def softmax_kernel(
    x_ptr,            # *f32, [L_tokens]
    out_ptr,          # *f32, [L_tokens]
    L_tokens: tl.constexpr,   # int
):
    # compute max for numerical stability
    max_val = -float("inf")
    for i in range(0, L_tokens):
        max_val = tl.maximum(max_val, tl.load(x_ptr + i))
    # sum of exp(x - max)
    sum_exp = 0.0
    for i in range(0, L_tokens):
        val = tl.load(x_ptr + i)
        sum_exp += tl.exp(val - max_val)
    # write normalized
    for i in range(0, L_tokens):
        val = tl.load(x_ptr + i)
        out_val = tl.exp(val - max_val) / sum_exp
        tl.store(out_ptr + i, out_val)


# Triton kernel: matvec out = attn @ K -> [head_dim]
@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32, [L_tokens]
    K_ptr,            # *f32, [L_tokens, head_dim] contiguous
    out_ptr,          # *f32, [head_dim]
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
):
    offs = tl.arange(0, head_dim)
    acc = 0.0
    for i in range(0, L_tokens):
        attn_i = tl.load(attn_ptr + i)
        K_i = tl.load(K_ptr + i * head_dim + offs)
        acc += attn_i * K_i
    tl.store(out_ptr + offs, acc)


# Triton kernel: per-head logsumexp of a vector (x is logits_scaled)
@triton.jit
def lse_per_head_kernel(
    x_ptr,            # *f32, [L_tokens]
    out_ptr,          # *f32 scalar per head
    L_tokens: tl.constexpr,   # int
):
    max_val = -float("inf")
    for i in range(0, L_tokens):
        max_val = tl.maximum(max_val, tl.load(x_ptr + i))
    sum_exp = 0.0
    for i in range(0, L_tokens):
        val = tl.load(x_ptr + i)
        sum_exp += tl.exp(val - max_val)
    lse_val = max_val + tl.log(sum_exp)
    tl.store(out_ptr, lse_val)


# Triton kernel: reduce per-batch lse across heads
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,         # *f32, [batch_size, num_qo_heads]
    out_ptr,          # *f32, [batch_size]
    num_heads: tl.constexpr,  # int
):
    b = tl.program_id(0)
    sum_lse = 0.0
    for h in range(0, num_heads):
        sum_lse += tl.load(lse_ptrs + b * num_heads + h)
    avg_lse = sum_lse / num_heads
    tl.store(out_ptr + b, avg_lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]
        device = q_nope.device

        # Prepare output and lse buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)
        lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)

        # Compute L_tokens and token indices per batch (len_indptr is [batch_size+1])
        L_tokens_list = []
        for b in range(batch_size):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            L_tokens_list.append(L_tokens)
        # Ensure device compatibility for indices
        tok_idx_list = [kv_indices[i:i + L_tokens].to(torch.int32).to(device) for i, L_tokens in enumerate(L_tokens_list) if L_tokens > 0]
        # We need to gather per batch. To keep Triton-only, we loop b and launch kernels per batch.

        for b in range(batch_size):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens for this batch element, output zeros
                output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                # lse defaults are fine (we'll reduce later)
                continue

            # Gather Kc_tmp and Kp_tmp: [L_tokens, head_dim]
            Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
            Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]
            Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
            Kp_tmp = torch.empty((L_tokens, head_dim_kpe), device=device)

            # Launch gather for Kc and Kp
            grid_gather = (L_tokens,)
            gather_tokens_kernel[grid_gather](
                K_src_ptr=Kc_all,
                idx_ptr=tok_idx_list[b],
                out_ptr=Kc_tmp,
                num_pages=num_pages,
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
            )
            grid_gather[0] = (L_tokens,)
            gather_tokens_kernel[grid_gather](
                K_src_ptr=Kp_all,
                idx_ptr=tok_idx_list[b],
                out_ptr=Kp_tmp,
                num_pages=num_pages,
                head_dim=head_dim_kpe,
                L_tokens=L_tokens,
            )

            # Compute output and lse per head
            for h in range(num_qo_heads):
                qn = q_nope[b, h].to(torch.float32)  # [head_dim_ckv]
                qp = q_pe[b, h].to(torch.float32)   # [head_dim_kpe]

                # logits_scaled = qn @ Kc_tmp.T + qp @ Kp_tmp.T
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                # Launch forward_attention_kernel for qn and Kc_tmp
                grid_f = (L_tokens,)
                forward_attention_kernel[grid_f](
                    q_ptr=qn,
                    K_ptr=Kc_tmp,
                    out_ptr=logits,
                    q_dim=head_dim_ckv,
                    head_dim=head_dim_ckv,
                    L_tokens=L_tokens,
                )
                # Launch for qp and Kp_tmp
                forward_attention_kernel[grid_f](
                    q_ptr=qp,
                    K_ptr=Kp_tmp,
                    out_ptr=logits,
                    q_dim=head_dim_kpe,
                    head_dim=head_dim_kpe,
                    L_tokens=L_tokens,
                )

                # Scale
                logits_scaled = logits * sm_scale

                # Softmax
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_kernel[(L_tokens,)](
                    x_ptr=logits_scaled,
                    out_ptr=attn,
                    L_tokens=L_tokens,
                )

                # Matvec: output[h] = attn @ Kc_tmp -> [head_dim_ckv]
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_kernel[(head_dim_ckv,)](
                    attn_ptr=attn,
                    K_ptr=Kc_tmp,
                    out_ptr=out_vec,
                    head_dim=head_dim_ckv,
                    L_tokens=L_tokens,
                )
                output[b, h] = out_vec

                # Per-head lse: logsumexp of logits_scaled
                lse_val = torch.empty((), dtype=torch.float32, device=device)
                lse_per_head_kernel[(L_tokens,)](
                    x_ptr=logits_scaled,
                    out_ptr=lse_val,
                    L_tokens=L_tokens,
                )
                lse_per_head[b, h] = lse_val

        # Reduce lse across heads per batch and convert to base-2
        lse_reduce_kernel[(batch_size,)](
            lse_ptrs=lse_per_head,
            out_ptr=lse_base2,
            num_heads=num_qo_heads,
        )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)

        return output, lse_base2


def run(*args):
    return ModelNew()(*args)
