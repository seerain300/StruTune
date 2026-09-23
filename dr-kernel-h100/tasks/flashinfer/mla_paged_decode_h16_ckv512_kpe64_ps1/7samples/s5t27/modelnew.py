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
    K_src_ptr,        # *f32, flattened [num_pages * head_dim]
    idx_ptr,          # *i32, [L_tokens]
    out_ptr,          # *f32, contiguous [L_tokens, head_dim]
    num_pages: tl.constexpr,  # int
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)  # int32 token index
    src_linear = tok_id * head_dim + offs
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + pid * head_dim + offs, vals)


# Triton kernel: compute per-head logits = qn[h] @ Kc_tmp.T + qp[h] @ Kp_tmp.T
@triton.jit
def forward_attention_vec_kernel(
    qn_ptr,           # *f32, [head_dim]
    qp_ptr,           # *f32, [head_dim_kpe]
    Kc_ptr,           # *f32, [L_tokens, head_dim]
    Kp_ptr,           # *f32, [L_tokens, head_dim_kpe]
    logits_ptr,       # *f32, [L_tokens]
    head_dim: tl.constexpr,       # int (e.g., 512)
    head_dim_kpe: tl.constexpr,   # int (e.g., 64)
    L_tokens: tl.constexpr,       # int
):
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    # Load qn and qp vectors
    qn_vec = tl.load(qn_ptr + tl.arange(0, head_dim))             # [head_dim]
    qp_vec = tl.load(qp_ptr + tl.arange(0, head_dim_kpe))         # [head_dim_kpe]

    # Accumulate two dot products: qn @ Kc.T and qp @ Kp.T
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)

    # Loop over tokens to compute dot products
    # For each row i in [0, L_tokens), we load Kc[i] and Kp[i] and dot with qn, qp.
    # Note: pid is the row index.
    # Compute Kc[i] dot qn: sum_j Kc[i, j] * qn[j]
    # Compute Kp[i] dot qp: sum_j Kp[i, j] * qp[j]
    # We access row pid directly.
    # Kc_ptr points to [L_tokens, head_dim] contiguous; Kp_ptr points to [L_tokens, head_dim_kpe].
    kc_row = tl.load(Kc_ptr + pid * head_dim + tl.arange(0, head_dim))    # [head_dim]
    kp_row = tl.load(Kp_ptr + pid * head_dim_kpe + tl.arange(0, head_dim_kpe))  # [head_dim_kpe]

    acc1 += tl.sum(kc_row * qn_vec, axis=0)
    acc2 += tl.sum(kp_row * qp_vec, axis=0)

    tl.store(logits_ptr + pid, acc1 + acc2)


# Triton kernel: apply softmax to logits_scaled (subtract max, exp, sum, divide)
@triton.jit
def softmax_kernel(
    logits_ptr,       # *f32, [L_tokens]
    attn_ptr,         # *f32, [L_tokens]
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,  # float
):
    offs = tl.arange(0, L_tokens)
    logits = tl.load(logits_ptr + offs)
    m = tl.max(logits, axis=0)
    logits_shift = logits - m
    exp_logits = tl.exp(logits_shift)
    sum_exp = tl.sum(exp_logits, axis=0)
    attn = exp_logits / sum_exp
    # Apply scaling factor if needed (already applied before kernel in host; here assume sm_scale=1 or handled)
    tl.store(attn_ptr + offs, attn)


# Triton kernel: matvec output = attn @ Kc_tmp, i.e., sum_i attn[i] * Kc[i, :]
@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32, [L_tokens]
    K_ptr,            # *f32, [L_tokens, head_dim]
    out_vec_ptr,      # *f32, [head_dim]
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    offs = tl.arange(0, head_dim)
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    for i in range(L_tokens):
        ai = tl.load(attn_ptr + i)                  # scalar
        Ki = tl.load(K_ptr + i * head_dim + offs)  # [head_dim]
        acc += ai * Ki
    tl.store(out_vec_ptr + offs, acc)


# Triton kernel: per-head logsumexp of logits_scaled
@triton.jit
def lse_per_head_kernel(
    logits_ptr,       # *f32, [L_tokens]
    out_ptr,          # *f32, scalar per (b,h)
    L_tokens: tl.constexpr,
):
    offs = tl.arange(0, L_tokens)
    logits = tl.load(logits_ptr + offs)
    m = tl.max(logits, axis=0)
    sum_exp = tl.sum(tl.exp(logits - m), axis=0)
    lse_val = m + tl.log(sum_exp)
    tl.store(out_ptr, lse_val)


# Triton kernel: reduce per-head lse across heads for each batch
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,         # *f32, [batch_size, num_heads]
    out_ptr,          # *f32, [batch_size]
    num_heads: tl.constexpr,
    batch_size: tl.constexpr,
):
    b = tl.program_id(0)  # 0..batch_size-1
    sum_lse = tl.zeros((), dtype=tl.float32)
    # Sum across heads: lse_ptrs[b, 0..num_heads-1]
    for h in range(num_heads):
        sum_lse += tl.load(lse_ptrs + b * num_heads + h)
    avg = sum_lse / num_heads
    tl.store(out_ptr + b, avg)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Assume inputs are on CUDA device and Triton available; original code uses .to(torch.float32) in kernel
    assert TRITON_AVAILABLE, "Triton is not available"
    device = q_nope.device

    # Extract shapes (constants in original): num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64, num_pages=989669, kpe_cache has 1
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    num_pages = ckv_cache.shape[0]
    batch_size = q_nope.shape[0]

    # Prepare output and lse tensors
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    # Process each batch element
    for b in range(batch_size):
        # Determine token range
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        L_tokens = end - start
        if L_tokens <= 0:
            # No tokens for this batch element
            output[b].zero_()
            lse_per_head[b].zero_()
            continue

        # Gather Kc_tmp and Kp_tmp
        Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
        Kp_tmp = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

        # Flatten pointers for gather
        Kc_src = ckv_cache.contiguous().view(-1)                # [num_pages * head_dim_ckv]
        Kp_src = kpe_cache.contiguous().view(-1)                # [num_pages * head_dim_kpe]

        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=Kc_src,
            idx_ptr=kv_indices[start:end].contiguous(),         # [L_tokens], int32
            out_ptr=Kc_tmp,
            num_pages=num_pages,
            head_dim=head_dim_ckv,
            L_tokens=L_tokens,
            num_warps=4,
        )

        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=Kp_src,
            idx_ptr=kv_indices[start:end].contiguous(),         # [L_tokens], int32
            out_ptr=Kp_tmp,
            num_pages=num_pages,
            head_dim=head_dim_kpe,
            L_tokens=L_tokens,
            num_warps=4,
        )

        # For each head, compute forward attention, softmax, matvec, and per-head lse
        for h in range(num_qo_heads):
            # Prepare qn[h], qp[h] vectors as 1D (contiguous)
            qn_vec = q_nope[b, h].contiguous().to(torch.float32)          # [head_dim_ckv]
            qp_vec = q_pe[b, h].contiguous().to(torch.float32)            # [head_dim_kpe]

            logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)

            forward_attention_vec_kernel[(L_tokens,)](
                qn_ptr=qn_vec,
                qp_ptr=qp_vec,
                Kc_ptr=Kc_tmp,
                Kp_ptr=Kp_tmp,
                logits_ptr=logits,
                head_dim=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                L_tokens=L_tokens,
                num_warps=4,
            )

            # Scale logits by sm_scale
            logits_scaled = logits * sm_scale

            attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)

            softmax_kernel[(L_tokens,)](
                logits_ptr=logits_scaled,
                attn_ptr=attn,
                L_tokens=L_tokens,
                sm_scale=sm_scale,  # declared in kernel signature
                num_warps=4,
            )

            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)

            matvec_kernel[(head_dim_ckv,)](
                attn_ptr=attn,
                K_ptr=Kc_tmp,  # multiply attn[i] with Kc_tmp[i, :]
                out_vec_ptr=out_vec,
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
                num_warps=4,
            )

            # Store output for this head
            output[b, h] = out_vec

            # Per-head lse: logsumexp of logits_scaled
            lse_val = torch.empty((), dtype=torch.float32, device=device)
            lse_per_head_kernel[(L_tokens,)](
                logits_ptr=logits_scaled,
                out_ptr=lse_val,
                L_tokens=L_tokens,
                num_warps=4,
            )
            lse_per_head[b, h] = lse_val

    # Reduce per-head lse across heads to get per-batch average, then convert to base-2
    lse_avg = torch.empty((batch_size,), dtype=torch.float32, device=device)
    lse_reduce_kernel[(batch_size,)](
        lse_ptrs=lse_per_head,
        out_ptr=lse_avg,
        num_heads=num_qo_heads,
        batch_size=batch_size,
        num_warps=4,
    )

    lse_base2 = lse_avg / math.log(2.0)  # convert to base-2

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)
    return output, lse_base2


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only execution
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)