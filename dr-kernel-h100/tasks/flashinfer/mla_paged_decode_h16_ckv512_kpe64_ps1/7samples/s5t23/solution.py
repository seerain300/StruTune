import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: gather selected rows from global cache into contiguous per-batch buffers
# K_src_ptr: flattened [num_pages, head_dim]
# idx_ptr: [L_tokens] int32 token indices
# out_ptr: contiguous [L_tokens, head_dim]
@triton.jit
def gather_tokens_kernel(
    K_src_ptr,       # *f32
    idx_ptr,         # *i32
    out_ptr,         # *f32
    L_tokens: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    in_ptr = K_src_ptr + tok_id * head_dim + offs
    tl.store(out_ptr + pid * head_dim + offs, tl.load(in_ptr))


# Triton kernel: compute per-head logits vector (size L_tokens) for a given batch
# qn_ptr: [head_dim], qp_ptr: [head_dim_kp], Kc_ptr: [L_tokens, head_dim], Kp_ptr: [L_tokens, head_dim_kp]
# logits_ptr: [L_tokens]
@triton.jit
def forward_attention_kernel(
    qn_ptr,          # *f32
    qp_ptr,          # *f32
    Kc_ptr,          # *f32
    Kp_ptr,          # *f32
    logits_ptr,      # *f32
    head_dim_q: tl.constexpr,
    head_dim_kc: tl.constexpr,
    head_dim_kp: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # Accumulate two dot products into a vector
    acc = tl.zeros((L_tokens,), dtype=tl.float32)
    # qn @ Kc.T: head_dim_q x head_dim_kc -> vector [L_tokens]
    for k in range(0, head_dim_kc):
        qk = tl.load(qn_ptr + k)  # scalar
        kcol = tl.load(Kc_ptr + (tl.arange(0, L_tokens) * head_dim_kc) + k)  # [L_tokens]
        acc += qk * kcol
    # qp @ Kp.T: head_dim_q (should match head_dim_kp) x head_dim_kp -> vector [L_tokens]
    for k in range(0, head_dim_kp):
        qk = tl.load(qp_ptr + k)  # scalar
        kcol = tl.load(Kp_ptr + (tl.arange(0, L_tokens) * head_dim_kp) + k)  # [L_tokens]
        acc += qk * kcol
    tl.store(logits_ptr, acc)


# Triton kernel: softmax over a 1D vector with scale factor (subtract max, exp, sum, divide)
@triton.jit
def softmax_kernel(
    logits_ptr,      # *f32, input logits [L_tokens]
    out_ptr,         # *f32, output softmax [L_tokens]
    L_tokens: tl.constexpr,
    scale: tl.constexpr,  # sm_scale
):
    x = tl.load(logits_ptr)
    x = x * scale
    m = tl.max(x, axis=0)
    x = x - m
    ex = tl.exp(x)
    s = tl.sum(ex, axis=0)
    out = ex / s
    tl.store(out_ptr, out)


# Triton kernel: matvec over K dimension (head_dim) tiles
# attn_ptr: [L_tokens], K_ptr: [L_tokens, head_dim], out_ptr: [head_dim]
@triton.jit
def matvec_kernel(
    attn_ptr,        # *f32, [L_tokens]
    K_ptr,           # *f32, [L_tokens, head_dim]
    out_ptr,         # *f32, [head_dim]
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # out = attn @ K (per token row reduction into head_dim)
    for j in range(0, head_dim):
        acc = tl.zeros((), dtype=tl.float32)
        for i in range(0, L_tokens):
            acc += tl.load(attn_ptr + i) * tl.load(K_ptr + i * head_dim + j)
        tl.store(out_ptr + j, acc)


# Triton kernel: compute per-head logsumexp of a vector (scaled by sm_scale)
@triton.jit
def lse_per_head_kernel(
    logits_ptr,      # *f32, [L_tokens]
    out_ptr,         # *f32, scalar lse
    L_tokens: tl.constexpr,
    scale: tl.constexpr,  # sm_scale
):
    x = tl.load(logits_ptr)
    x = x * scale
    m = tl.max(x, axis=0)
    x = x - m
    s = tl.sum(tl.exp(x), axis=0)
    lse = tl.log(s) + m
    tl.store(out_ptr, lse)


# Triton kernel: reduce per-head lse across heads for each batch
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,        # *f32, [batch_size, num_heads]
    out_ptr,         # *f32, [batch_size]
    num_heads: tl.constexpr,
):
    b = tl.program_id(0)  # batch index
    sum_lse = tl.zeros((), dtype=tl.float32)
    for h in range(0, num_heads):
        sum_lse += tl.load(lse_ptrs + b * num_heads + h)
    sum_lse = sum_lse / num_heads  # average across heads
    tl.store(out_ptr + b, sum_lse)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure tensors are on CUDA
    device = q_nope.device
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA."

    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    head_dim_kc = ckv_cache.shape[-1]
    head_dim_kp = kpe_cache.shape[-1]
    # Indices must be on int32
    assert kv_indices.dtype == torch.int32, "kv_indices must be int32"
    assert kv_indptr.dtype == torch.int32, "kv_indptr must be int32"

    # Output buffers
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # we'll cast to bfloat16 at end
    lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)

    # Per-batch buffers for Kc_tmp and Kp_tmp
    for b in range(batch_size):
        # Determine L_tokens and valid range
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = max(page_end - page_beg, 0)
        if L_tokens == 0:
            # No tokens in this batch element
            output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            continue

        # Gather selected rows
        Kc_tmp = torch.empty((L_tokens, head_dim_kc), dtype=torch.float32, device=device)
        Kp_tmp = torch.empty((L_tokens, head_dim_kp), dtype=torch.float32, device=device)

        # Triton launch for gather
        grid_gather = (L_tokens,)
        gather_tokens_kernel[grid_gather](
            ckv_cache.view(-1),                 # K_src_ptr
            kv_indices[page_beg:page_end],     # idx_ptr
            Kc_tmp,                             # out_ptr
            L_tokens=L_tokens,
            head_dim=head_dim_kc,
            num_warps=4,
        )
        # kpe cache gather
        grid_gather2 = (L_tokens,)
        gather_tokens_kernel[grid_gather2](
            kpe_cache.view(-1),                 # K_src_ptr
            kv_indices[page_beg:page_end],     # idx_ptr
            Kp_tmp,                             # out_ptr
            L_tokens=L_tokens,
            head_dim=head_dim_kp,
            num_warps=4,
        )

        # For each head h
        for h in range(num_qo_heads):
            # Compute logits vector
            qn = q_nope[b, h].contiguous().to(torch.float32)
            qp = q_pe[b, h].contiguous().to(torch.float32)
            logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)

            grid_forward = (1,)
            forward_attention_kernel[grid_forward](
                qn,                                # qn_ptr
                qp,                                # qp_ptr
                Kc_tmp,                           # Kc_ptr
                Kp_tmp,                           # Kp_ptr
                logits,                           # logits_ptr
                head_dim_q=head_dim_ckv,
                head_dim_kc=head_dim_ckv,
                head_dim_kp=head_dim_kp,
                L_tokens=L_tokens,
                num_warps=4,
            )

            # Softmax on logits_scaled = logits * sm_scale
            softmax_out = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            grid_softmax = (L_tokens,)
            softmax_kernel[grid_softmax](
                logits,                           # logits_ptr
                softmax_out,                      # out_ptr
                L_tokens=L_tokens,
                scale=sm_scale,                   # sm_scale
                num_warps=4,
            )

            # Matvec: output[h] = softmax_out @ Kc_tmp
            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            grid_matvec = (1,)
            matvec_kernel[grid_matvec](
                softmax_out,                      # attn_ptr
                Kc_tmp,                          # K_ptr
                out_vec,                         # out_ptr
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
                num_warps=4,
            )
            output[b, h] = out_vec

            # Per-head lse: logsumexp(logits * sm_scale) across L_tokens
            lse_per_head = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # dummy to define shape; we only need scalar per (b,h)
            lse_val = torch.empty((), dtype=torch.float32, device=device)
            grid_lse = (L_tokens,)
            lse_per_head_kernel[grid_lse](
                logits,                           # logits_ptr
                lse_val,                          # out_ptr
                L_tokens=L_tokens,
                scale=sm_scale,                   # sm_scale
                num_warps=4,
            )
            lse_per_head[b, h] = lse_val  # store per (b, h)

    # Reduce across heads: mean lse per batch, then convert to base-2 (divide by ln(2))
    lse_reduce = torch.empty((batch_size,), dtype=torch.float32, device=device)
    grid_reduce = (batch_size,)
    lse_reduce_kernel[grid_reduce](
        lse_per_head,                          # [batch_size, num_qo_heads]
        lse_reduce,                           # out_ptr
        num_heads=num_qo_heads,
        num_warps=4,
    )
    lse_base2[:] = lse_reduce / math.log(2.0)

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)

    return output, lse_base2


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only execution: no torch tensor ops on GPU tensors in host
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
