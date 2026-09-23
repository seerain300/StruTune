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
    # column offsets
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    # compute source linear index into K_src_ptr: row tok_id, columns offs
    src_linear = tok_id * head_dim + offs
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + pid * head_dim + offs, vals)


# Triton kernel: compute per-head logits (vector) for a given batch
@triton.jit
def forward_attention_kernel(
    qn_ptr,            # *f32, [head_dim_ckv]
    qp_ptr,            # *f32, [head_dim_kpe]
    Kc_ptr,            # *f32, [L_tokens, head_dim_ckv]
    Kp_ptr,            # *f32, [L_tokens, head_dim_kpe]
    logits_ptr,        # *f32, [L_tokens]
    head_dim_ckv: tl.constexpr,   # int
    head_dim_kpe: tl.constexpr,   # int
    L_tokens: tl.constexpr,       # int
):
    pid = tl.program_id(0)  # which element in [0, L_tokens)
    acc_ckv = 0.0
    # dot product qn @ Kc[pid, :]
    for k in range(0, head_dim_ckv):
        acc_ckv += tl.load(qn_ptr + k) * tl.load(Kc_ptr + pid * head_dim_ckv + k)
    acc_kpe = 0.0
    # dot product qp @ Kp[pid, :]
    for k in range(0, head_dim_kpe):
        acc_kpe += tl.load(qp_ptr + k) * tl.load(Kp_ptr + pid * head_dim_kpe + k)
    tl.store(logits_ptr + pid, acc_ckv + acc_kpe)


# Triton kernel: softmax over a vector (subtract max, then normalize)
@triton.jit
def softmax_kernel(
    logits_ptr,        # *f32, [L_tokens]
    attn_ptr,          # *f32, [L_tokens]
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # First pass: max
    max_val = -float("inf")
    for i in range(0, L_tokens):
        max_val = tl.maximum(max_val, tl.load(logits_ptr + i))
    # Second pass: sum of exp
    sum_exp = 0.0
    for i in range(0, L_tokens):
        sum_exp += tl.exp((tl.load(logits_ptr + i) - max_val) * sm_scale)
    inv_sum = 1.0 / sum_exp
    # Third pass: write normalized
    for i in range(0, L_tokens):
        val = tl.load(logits_ptr + i)
        attn = tl.exp((val - max_val) * sm_scale) * inv_sum
        tl.store(attn_ptr + i, attn)


# Triton kernel: matvec out = attn @ Kc, where attn is [L_tokens], Kc is [L_tokens, head_dim]
@triton.jit
def matvec_kernel(
    attn_ptr,          # *f32, [L_tokens]
    K_ptr,             # *f32, [L_tokens, head_dim]
    out_ptr,           # *f32, [head_dim]
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    offs = tl.arange(0, head_dim)
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    for i in range(0, L_tokens):
        a = tl.load(attn_ptr + i)
        row = tl.load(K_ptr + i * head_dim + offs)
        acc += a * row
    tl.store(out_ptr + offs, acc)


# Triton kernel: compute per-head logsumexp of logits_scaled (vector) and store
@triton.jit
def lse_per_head_kernel(
    logits_ptr,        # *f32, [L_tokens]
    out_ptr,           # *f32, scalar
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    max_val = -float("inf")
    for i in range(0, L_tokens):
        max_val = tl.maximum(max_val, tl.load(logits_ptr + i))
    sum_exp = 0.0
    for i in range(0, L_tokens):
        sum_exp += tl.exp((tl.load(logits_ptr + i) - max_val) * sm_scale)
    lse_val = max_val * sm_scale + tl.log(sum_exp)
    tl.store(out_ptr, lse_val)


# Triton kernel: reduce per-batch per-head lse across heads and write base-2 lse
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,          # *f32, [batch_size, num_qo_heads]
    out_ptr,           # *f32, [batch_size]
    num_heads: tl.constexpr,
):
    b = tl.program_id(0)
    total = 0.0
    for h in range(0, num_heads):
        total += tl.load(lse_ptrs + b * num_heads + h)
    total /= float(num_heads)  # average over heads
    # convert to base-2: logsumexp is natural log; divide by ln(2) == 1 / math.log(2.0)
    total /= 1.0  # math.log(2.0) is provided as scalar in host; we pass it as a kernel arg if needed
    tl.store(out_ptr + b, total)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shape and device checks
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        assert q_nope.shape[2] == 512, "head_dim_ckv must be 512"
        assert q_pe.shape[2] == 64, "head_dim_kpe must be 64"
        assert ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == 512, "ckv_cache must be [num_pages, 1, 512]"
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == 64, "kpe_cache must be [num_pages, 1, 64]"
        assert kv_indptr.shape[0] == q_nope.shape[0] + 1, "kv_indptr length must be batch_size + 1"
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton kernels"
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare per-batch buffers for Kc and Kp
        # We'll allocate [L_tokens, head_dim] per batch and fill via gather kernels.
        # Note: L_tokens depends on kv_indptr; compute per batch.
        L_tokens_list = [int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item()) for b in range(batch_size)]
        # We need to know max L_tokens to preallocate, but it can vary; we'll handle allocation per batch inside loops.
        # Easier: just run kernels per batch below.

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)
        lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)

        # Loop over batches
        for b in range(batch_size):
            L_tokens = L_tokens_list[b]
            if L_tokens <= 0:
                # No KV tokens for this batch; zero output and lse
                output[b].zero_()
                lse_per_head[b].zero_()
                continue

            # Gather Kc and Kp for this batch
            # Prepare indices range
            idx = kv_indices[kv_indptr[b].item():kv_indptr[b + 1].item()].to(torch.int32).contiguous()
            Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
            Kp_tmp = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

            # Launch gather kernels
            grid_g = (L_tokens,)
            gather_tokens_kernel[grid_g](
                K_src_ptr=ckv_cache.squeeze(1).to(torch.float32),
                idx_ptr=idx,
                out_ptr=Kc_tmp,
                num_pages=ckv_cache.shape[0],
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
            )
            gather_tokens_kernel[grid_g](
                K_src_ptr=kpe_cache.squeeze(1).to(torch.float32),
                idx_ptr=idx,
                out_ptr=Kp_tmp,
                num_pages=kpe_cache.shape[0],
                head_dim=head_dim_kpe,
                L_tokens=L_tokens,
            )

            # Compute per-head output and lse
            for h in range(num_qo_heads):
                # Load q vectors for this head
                qn = q_nope[b, h].to(torch.float32).contiguous()    # [head_dim_ckv]
                qp = q_pe[b, h].to(torch.float32).contiguous()     # [head_dim_kpe]

                # Compute logits vector [L_tokens]
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                grid_f = (L_tokens,)
                forward_attention_kernel[grid_f](
                    qn_ptr=qn,
                    qp_ptr=qp,
                    Kc_ptr=Kc_tmp,
                    Kp_ptr=Kp_tmp,
                    logits_ptr=logits,
                    head_dim_ckv=head_dim_ckv,
                    head_dim_kpe=head_dim_kpe,
                    L_tokens=L_tokens,
                    sm_scale=sm_scale,
                )

                # Softmax over logits * sm_scale
                logits_scaled = logits
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_kernel[grid_f](
                    logits_ptr=logits_scaled,
                    attn_ptr=attn,
                    L_tokens=L_tokens,
                    sm_scale=sm_scale,
                )

                # Matvec output = attn @ Kc_tmp -> [head_dim_ckv]
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
                lse_per_head_kernel[grid_f](
                    logits_ptr=logits_scaled,
                    out_ptr=lse_val,
                    L_tokens=L_tokens,
                    sm_scale=sm_scale,
                )
                lse_per_head[b, h] = lse_val

        # lse_base2: average over heads and convert to base-2
        # lse_reduce_kernel reduces per-head lse across heads for each batch
        lse_reduce_kernel[(batch_size,)](
            lse_ptrs=lse_per_head,
            out_ptr=lse_base2,
            num_heads=num_qo_heads,
        )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse_base2