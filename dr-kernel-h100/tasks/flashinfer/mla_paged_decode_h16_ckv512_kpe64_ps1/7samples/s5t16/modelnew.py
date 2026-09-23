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
    L_tokens: tl.constexpr,   # int
    head_dim: tl.constexpr,   # int
):
    # Each program handles one row (one token index)
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    # columns offsets
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    # compute source linear index into K_src_ptr: row tok_id, columns offs
    src_linear = tok_id * head_dim + offs
    vals = tl.load(K_src_ptr + src_linear)
    dest_linear = pid * head_dim + offs
    tl.store(out_ptr + dest_linear, vals)


# Triton kernel: forward attention per head -> compute logits vector [L_tokens]
# logits = qn @ Kc.T + qp @ Kp.T
@triton.jit
def forward_attention_kernel(
    qn_ptr,           # *f32 [head_dim_ckv] or [head_dim_kpe]
    qn_dim: tl.constexpr,  # int (length of qn)
    K_ptr,            # *f32 [L_tokens, head_dim]
    L_tokens: tl.constexpr,   # int
    head_dim: tl.constexpr,   # int
    out_ptr,          # *f32 [L_tokens]
):
    # Single program computes entire logits vector
    pid = tl.program_id(0)  # should be 0
    offs = tl.arange(0, head_dim)  # K dimension
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    # qn is 1D; we multiply qn[offs] with K[:, offs] and accumulate
    # Note: this assumes K is [L_tokens, head_dim] and we dot over K columns
    for k in range(L_tokens):
        # load qn and K row k
        qn_k = tl.load(qn_ptr + offs)  # qn over K dim
        K_row = tl.load(K_ptr + k * head_dim + offs)
        acc += qn_k * K_row
    tl.store(out_ptr, acc)


# Triton kernel: softmax over a vector x (size L_tokens), out_ptr stores softmax(x)
@triton.jit
def softmax_kernel(
    in_ptr,           # *f32 [L_tokens]
    out_ptr,          # *f32 [L_tokens]
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,  # float scalar
):
    # Compute in-place softmax over vector in_ptr -> out_ptr
    x = tl.load(in_ptr)
    max_val = tl.max(x)
    x_shifted = x - max_val
    exp_x = tl.exp(x_shifted * sm_scale)
    sum_exp = tl.sum(exp_x)
    attn = exp_x / sum_exp
    tl.store(out_ptr, attn)


# Triton kernel: matvec out = attn @ K -> [head_dim]
@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32 [L_tokens]
    K_ptr,            # *f32 [L_tokens, head_dim]
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
    out_ptr,          # *f32 [head_dim]
):
    offs = tl.arange(0, head_dim)
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    for k in range(L_tokens):
        attn_k = tl.load(attn_ptr + k)
        K_row = tl.load(K_ptr + k * head_dim + offs)
        acc += attn_k * K_row
    tl.store(out_ptr + offs, acc)


# Triton kernel: per-head logsumexp of a vector x (size L_tokens) -> scalar
@triton.jit
def lse_per_head_kernel(
    in_ptr,           # *f32 [L_tokens]
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,  # float scalar
    out_ptr,          # *f32 scalar
):
    x = tl.load(in_ptr)
    max_val = tl.max(x)
    sum_exp = tl.sum(tl.exp((x - max_val) * sm_scale))
    lse = max_val * sm_scale + tl.log(sum_exp)
    tl.store(out_ptr, lse)


# Triton kernel: reduce per-batch lse across heads: out[b] = sum(heads)/num_heads
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,         # *f32 [batch_size, num_heads]
    num_heads: tl.constexpr,  # int
    out_ptr,          # *f32 [batch_size]
    batch_size: tl.constexpr, # int
):
    b = tl.program_id(0)
    total = tl.zeros((), dtype=tl.float32)
    for h in range(num_heads):
        total += tl.load(lse_ptrs + b * num_heads + h)
    avg = total / num_heads
    tl.store(out_ptr + b, avg)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes from original code: batch_size, num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        device = q_nope.device

        # Ensure inputs are contiguous and float32 for Triton
        q_nope_f32 = q_nope.contiguous().to(torch.float32)
        q_pe_f32 = q_pe.contiguous().to(torch.float32)
        ckv_cache_f32 = ckv_cache.contiguous().to(torch.float32).squeeze(1)  # [num_pages, 512]
        kpe_cache_f32 = kpe_cache.contiguous().to(torch.float32).squeeze(1)  # [num_pages, 64]
        kv_indptr_i32 = kv_indptr.contiguous().to(torch.int32)
        kv_indices_i32 = kv_indices.contiguous().to(torch.int32)

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(batch_size):
            # Determine L_tokens and indices for this batch
            L_tokens = int(kv_indptr_i32[b + 1].item()) - int(kv_indptr_i32[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse_per_head[b].zero_()
                continue

            tok_idx = kv_indices_i32[int(kv_indptr_i32[b].item()):int(kv_indptr_i32[b + 1].item())]  # [L_tokens]
            # Gather Kc_tmp and Kp_tmp into contiguous buffers
            Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
            Kp_tmp = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)
            # Launch gather kernels
            grid_gather = (L_tokens,)
            gather_tokens_kernel[grid_gather](
                K_src_ptr=ckv_cache_f32,
                idx_ptr=tok_idx,
                out_ptr=Kc_tmp,
                L_tokens=L_tokens,
                head_dim=head_dim_ckv,
            )
            # Note: kpe_cache_f32 has shape [num_pages, 64]; we need only 64 dim. Here we assume kpe per token matches head_dim_kpe.
            gather_tokens_kernel[grid_gather](
                K_src_ptr=kpe_cache_f32,  # only using the 64-dim per token
                idx_ptr=tok_idx,
                out_ptr=Kp_tmp,
                L_tokens=L_tokens,
                head_dim=head_dim_kpe,
            )

            # Compute output per head
            for h in range(num_qo_heads):
                # qn and qp for this head
                qn = q_nope_f32[b, h]                     # [512]
                qp = q_pe_f32[b, h]                      # [64]

                # logits = qn @ Kc_tmp.T + qp @ Kp_tmp.T -> [L_tokens]
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                forward_attention_kernel[(1,)](
                    qn_ptr=qn,
                    qn_dim=head_dim_ckv,                 # actually using both qn and qp dims; reusing head_dim_ckv for qn
                    K_ptr=Kc_tmp,
                    L_tokens=L_tokens,
                    head_dim=head_dim_ckv,
                    out_ptr=logits,
                    sm_scale=float(sm_scale),           # pass sm_scale to kernel (fixes previous error)
                )
                # Second part: qp @ Kp_tmp.T
                forward_attention_kernel[(1,)](
                    qn_ptr=qp,
                    qn_dim=head_dim_kpe,
                    K_ptr=Kp_tmp,
                    L_tokens=L_tokens,
                    head_dim=head_dim_kpe,
                    out_ptr=logits,
                    sm_scale=float(sm_scale),
                )
                # Softmax over logits
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_kernel[(L_tokens,)](
                    in_ptr=logits,
                    out_ptr=attn,
                    L_tokens=L_tokens,
                    sm_scale=float(sm_scale),
                )
                # Matvec: output[b, h] = attn @ Kc_tmp -> [512]
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_kernel[(head_dim_ckv,)](
                    attn_ptr=attn,
                    K_ptr=Kc_tmp,
                    head_dim=head_dim_ckv,
                    L_tokens=L_tokens,
                    out_ptr=out_vec,
                )
                output[b, h] = out_vec

                # Per-head logsumexp of logits (scaled)
                lse_val = torch.empty((), dtype=torch.float32, device=device)
                lse_per_head_kernel[(L_tokens,)](
                    in_ptr=logits,
                    L_tokens=L_tokens,
                    sm_scale=float(sm_scale),
                    out_ptr=lse_val,
                )
                lse_per_head[b, h] = lse_val

        # Reduce lse across heads per batch and convert to base-2 (divide by ln(2))
        lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)
        lse_reduce_kernel[(batch_size,)](
            lse_ptrs=lse_per_head,
            num_heads=num_qo_heads,
            out_ptr=lse_base2,
            batch_size=batch_size,
        )
        lse_base2 = lse_base2 / math.log(2.0)

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse_base2