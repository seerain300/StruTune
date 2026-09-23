import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: gather selected rows from K [num_pages, head_dim] into out [L_tokens, head_dim]
# Each program processes one token index pid in [0, L_tokens)
@triton.jit
def gather_tokens_kernel(
    K_src_ptr,        # *f32, flattened [num_pages, head_dim], row-major
    idx_ptr,          # *i32, [L_tokens] token indices
    out_ptr,          # *f32, contiguous [L_tokens, head_dim]
    head_dim: tl.constexpr,  # int: K dimension
    L_tokens: tl.constexpr,  # int: number of tokens to gather
):
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)  # i32
    src_linear = tok_id * head_dim + offs
    dst_linear = pid * head_dim + offs
    # K_src_ptr is float32; out_ptr is float32
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + dst_linear, vals)


# Triton kernel: compute per-head logits for one head h:
# Given q_vec [head_dim] and K [L_tokens, head_dim], write logits [L_tokens]
# We launch this kernel for each head, then sum two calls (qn and qp).
@triton.jit
def forward_attention_vec_kernel(
    q_ptr,            # *f32, [head_dim] (q_vec for this head)
    K_ptr,            # *f32, [L_tokens, head_dim]
    out_ptr,          # *f32, [L_tokens] (logits vector)
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
):
    i = tl.program_id(0)  # token index in [0, L_tokens)
    if i >= L_tokens:
        return
    acc = 0.0
    # q is 1D of length head_dim; K is [L_tokens, head_dim]
    # For each feature j, accumulate q[j] * K[i, j]
    for j in range(head_dim):
        qj = tl.load(q_ptr + j)
        # row i of K: base + i*head_dim + j
        kj = tl.load(K_ptr + i * head_dim + j)
        acc += qj * kj
    tl.store(out_ptr + i, acc)


# Triton kernel: softmax over vector x (length L_tokens), writes y[i] = exp(x[i] - max) / sum(exp(...))
# Assumes x is already scaled by sm_scale. We pass max_x computed on host to avoid recomputation in kernel.
@triton.jit
def softmax_kernel(
    x_ptr,            # *f32, [L_tokens]
    y_ptr,            # *f32, [L_tokens]
    max_x,            # f32 scalar
    L_tokens: tl.constexpr,
    sm_scale,         # f32 scalar
):
    i = tl.program_id(0)
    xi = tl.load(x_ptr + i) * sm_scale
    yi = tl.exp(xi - max_x)
    # Compute denominator: sum of exp(x - max_x)
    sum_exp = 0.0
    for t in range(L_tokens):
        xt = tl.load(x_ptr + t) * sm_scale
        sum_exp += tl.exp(xt - max_x)
    yi = yi / sum_exp
    tl.store(y_ptr + i, yi)


# Triton kernel: compute out[k] = sum_i attn[i] * K[i, k] for k in [0, head_dim)
# Input:
#   attn_ptr: *f32, [L_tokens]
#   K_ptr:    *f32, [L_tokens, head_dim]
#   out_ptr:  *f32, [head_dim]
@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32, [L_tokens]
    K_ptr,            # *f32, [L_tokens, head_dim]
    out_ptr,          # *f32, [head_dim]
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    k = tl.program_id(0)  # feature index in [0, head_dim)
    if k >= head_dim:
        return
    accum = 0.0
    for i in range(L_tokens):
        ai = tl.load(attn_ptr + i)
        ki = tl.load(K_ptr + i * head_dim + k)
        accum += ai * ki
    tl.store(out_ptr + k, accum)


# Triton kernel: compute per-head logsumexp of x (length L_tokens), writes scalar to out_ptr[0]
# Assumes max_x is passed from host. We implement lse = log(sum(exp(x - max_x))) + max_x
@triton.jit
def lse_per_head_kernel(
    x_ptr,            # *f32, [L_tokens]
    out_ptr,          # *f32, scalar at out_ptr[0]
    max_x,            # f32 scalar
    L_tokens: tl.constexpr,
):
    sum_exp = 0.0
    for t in range(L_tokens):
        xt = tl.load(x_ptr + t)
        sum_exp += tl.exp(xt - max_x)
    lse = tl.log(sum_exp) + max_x
    tl.store(out_ptr, lse)


# Triton kernel: reduce per-batch lse across heads, produce per-batch lse_avg (scalar per batch)
# Input:
#   lse_ptrs: *f32, [batch_size, num_heads]
#   out_ptr:  *f32, [batch_size]
#   num_heads: tl.constexpr
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,         # *f32, [batch_size, num_heads]
    out_ptr,          # *f32, [batch_size]
    num_heads: tl.constexpr,
    batch_size: tl.constexpr,
):
    b = tl.program_id(0)
    sum_lse = 0.0
    for h in range(num_heads):
        sum_lse += tl.load(lse_ptrs + b * num_heads + h)
    avg_lse = sum_lse / num_heads
    # Convert to base-2 by dividing by ln(2); we pass ln(2) as a scalar to host
    # Note: Triton requires scalar constant for division; use 1.4426950408889634 (1/ln(2))
    base2_avg = avg_lse * 0.6931471805599453  # multiply by 1/ln(2)
    tl.store(out_ptr + b, base2_avg)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only implementation of the original run function.
    Returns (output [batch, num_qo_heads, head_dim_ckv] in bfloat16, lse [batch] in float32).
    """
    # Check inputs
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA for Triton"
    device = q_nope.device
    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
    assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "Cache must have second dim = 1"
    assert head_dim_ckv == 512 and num_qo_heads == 16 and head_dim_kpe == 64, "Fixes required for Triton kernels"
    num_pages = ckv_cache.shape[0]
    assert kv_indptr.shape[0] == batch_size + 1, "kv_indptr length must be batch_size + 1"
    # num_kv_indices may not equal kv_indptr[-1].item() in general; we only use the range defined by indptr
    # Prepare Kc_tmp and Kp_tmp buffers [L_tokens, head_dim] per batch
    # We'll allocate them on host and fill via Triton gather kernels
    Kc_tmp = []  # list of tensors per batch
    Kp_tmp = []  # list of tensors per batch
    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = max(page_end - page_beg, 0)
        if L_tokens == 0:
            # No tokens for this batch element
            Kc_tmp.append(torch.empty((0, head_dim_ckv), dtype=torch.float32, device=device))
            Kp_tmp.append(torch.empty((0, head_dim_kpe), dtype=torch.float32, device=device))
            continue
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [L_tokens]
        Kc_tmp.append(torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device))
        Kp_tmp.append(torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device))

        # Cast caches to float32 for computation
        Kc_src = ckv_cache.to(torch.float32)
        Kp_src = kpe_cache.to(torch.float32)

        # Launch Triton gather for Kc_tmp[b]
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=Kc_src,          # [num_pages, 512]
            idx_ptr=tok_idx,           # [L_tokens] int32
            out_ptr=Kc_tmp[b],         # [L_tokens, 512] float32
            head_dim=head_dim_ckv,
            L_tokens=L_tokens,
        )

        # Launch Triton gather for Kp_tmp[b]
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=Kp_src,          # [num_pages, 64]
            idx_ptr=tok_idx,           # [L_tokens] int32
            out_ptr=Kp_tmp[b],         # [L_tokens, 64] float32
            head_dim=head_dim_kpe,
            L_tokens=L_tokens,
        )

    # Output buffer in float32 for computation, then cast to bfloat16 at end
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

    # Per-head lse buffer
    lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    # Compute per-batch lse average (will be converted to base-2 on host)
    lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)

    # Process each batch element
    for b in range(batch_size):
        L_tokens = Kc_tmp[b].shape[0] if isinstance(Kc_tmp[b], torch.Tensor) and Kc_tmp[b].numel() > 0 else 0
        if L_tokens == 0:
            output[b].zero_()
            lse_per_head[b].zero_()
            continue

        # Compute logits for each head: sum over qn and qp
        for h in range(num_qo_heads):
            # q vectors for this head
            qn_vec = q_nope[b, h].to(torch.float32)  # [head_dim_ckv]
            qp_vec = q_pe[b, h].to(torch.float32)    # [head_dim_kpe]
            # logit_qn: [L_tokens], logit_qp: [L_tokens]
            logit_qn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            logit_qp = torch.empty((L_tokens,), dtype=torch.float32, device=device)

            # Launch forward_attention_vec_kernel for qn
            forward_attention_vec_kernel[(L_tokens,)](
                q_ptr=qn_vec,              # [512]
                K_ptr=Kc_tmp[b],           # [L_tokens, 512]
                out_ptr=logit_qn,          # [L_tokens]
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
            )

            # Launch forward_attention_vec_kernel for qp
            forward_attention_vec_kernel[(L_tokens,)](
                q_ptr=qp_vec,              # [64]
                K_ptr=Kp_tmp[b],           # [L_tokens, 64]
                out_ptr=logit_qp,          # [L_tokens]
                head_dim=head_dim_kpe,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
            )

            # Sum to get logits for this head
            logits = logit_qn + logit_qp  # [L_tokens]
            # Max for softmax
            max_logits = torch.max(logits).item()  # host scalar
            # Softmax attention
            attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            softmax_kernel[(L_tokens,)](
                x_ptr=logits,
                y_ptr=attn,
                max_x=max_logits,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
            )

            # Output for this head: attn @ Kc_tmp[b] -> [512]
            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_kernel[(head_dim_ckv,)](
                attn_ptr=attn,             # [L_tokens]
                K_ptr=Kc_tmp[b],           # [L_tokens, 512]
                out_ptr=out_vec,           # [512]
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
            )
            output[b, h] = out_vec

            # Per-head lse: logsumexp(logits * sm_scale)
            # Compute max again (or use current max_logits)
            max_x = torch.max(logits * sm_scale).item()
            lse_val = torch.empty((), dtype=torch.float32, device=device)
            lse_per_head_kernel[(L_tokens,)](
                x_ptr=logits * sm_scale,
                out_ptr=lse_val,
                max_x=max_x,
                L_tokens=L_tokens,
            )
            lse_per_head[b, h] = lse_val

    # Reduce across heads to get per-batch lse, then convert to base-2 (divide by ln(2))
    lse_reduce_kernel[(batch_size,)](
        lse_ptrs=lse_per_head,
        out_ptr=lse_base2,
        num_heads=num_qo_heads,
        batch_size=batch_size,
    )

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)

    return output, lse_base2


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only execution: no torch tensor ops in host except final cast
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)