import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: gather selected rows from global cache into contiguous per-batch buffers
# K_src_ptr: *f32, flattened [num_pages, head_dim]
# idx_ptr:   *i32, [L_tokens] token indices (for this batch)
# out_ptr:   *f32, contiguous [L_tokens, head_dim]
@triton.jit
def gather_tokens_kernel(
    K_src_ptr,        # *f32
    idx_ptr,          # *i32
    out_ptr,          # *f32
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    src_linear = tok_id * head_dim + offs
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + pid * head_dim + offs, vals)


# Triton kernel: compute per-head logits: qn @ Kc_tmp.T + qp @ Kp_tmp.T
# Kc_tmp_ptr: *f32, [L_tokens, head_dim_ckv]
# Kp_tmp_ptr: *f32, [L_tokens, head_dim_kpe]
# qn_ptr:     *f32, [head_dim_ckv]
# qp_ptr:     *f32, [head_dim_kpe]
# logits_ptr: *f32, [L_tokens]
@triton.jit
def forward_attention_kernel(
    Kc_tmp_ptr,       # *f32
    Kp_tmp_ptr,       # *f32
    qn_ptr,           # *f32
    qp_ptr,           # *f32
    logits_ptr,       # *f32
    head_dim_ckv: tl.constexpr,  # int
    head_dim_kpe: tl.constexpr,  # int
    L_tokens: tl.constexpr,      # int
):
    offs = tl.arange(0, L_tokens)
    # Dot product with Kc_tmp
    acc_ckv = tl.zeros((), dtype=tl.float32)
    for j in range(head_dim_ckv):
        qj = tl.load(qn_ptr + j)  # scalar
        Kj = tl.load(Kc_tmp_ptr + offs * head_dim_ckv + j)  # [L_tokens]
        acc_ckv += qj * Kj
    # Dot product with Kp_tmp
    acc_kpe = tl.zeros((), dtype=tl.float32)
    for j in range(head_dim_kpe):
        qj = tl.load(qp_ptr + j)  # scalar
        Kj = tl.load(Kp_tmp_ptr + offs * head_dim_kpe + j)  # [L_tokens]
        acc_kpe += qj * Kj
    tl.store(logits_ptr + offs, acc_ckv + acc_kpe)


# Triton kernel: softmax on a 1D vector (logits_scaled)
# logits_ptr: *f32, [L_tokens]
# out_ptr:    *f32, [L_tokens]
@triton.jit
def softmax_kernel(
    logits_ptr,       # *f32
    out_ptr,          # *f32
    L_tokens: tl.constexpr,  # int
    sm_scale: tl.float32,    # scalar
):
    offs = tl.arange(0, L_tokens)
    logits = tl.load(logits_ptr + offs) * sm_scale
    m = tl.max(logits, axis=0)
    logits = logits - m
    exp_logits = tl.exp(logits)
    denom = tl.sum(exp_logits, axis=0)
    out = exp_logits / denom
    tl.store(out_ptr + offs, out)


# Triton kernel: matvec (per head): out = attn @ Kc_tmp
# attn_ptr:   *f32, [L_tokens]
# Kc_ptr:     *f32, [L_tokens, head_dim]
# out_ptr:    *f32, [head_dim]
@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32
    Kc_ptr,           # *f32
    out_ptr,          # *f32
    head_dim: tl.constexpr,    # int (e.g., 512)
    L_tokens: tl.constexpr,    # int
):
    offs_k = tl.arange(0, head_dim)
    acc = tl.zeros((), dtype=tl.float32)
    for i in range(L_tokens):
        attn_i = tl.load(attn_ptr + i)
        Kj = tl.load(Kc_ptr + i * head_dim + offs_k)
        acc += attn_i * Kj
    tl.store(out_ptr + offs_k, acc)


# Triton kernel: per-head lse = logsumexp(logits * sm_scale)
# logits_ptr: *f32, [L_tokens]
# out_ptr:    *f32, scalar (we store into 1-element tensor)
@triton.jit
def lse_per_head_kernel(
    logits_ptr,       # *f32
    out_ptr,          # *f32, single element
    L_tokens: tl.constexpr,  # int
    sm_scale: tl.float32,    # scalar
):
    offs = tl.arange(0, L_tokens)
    logits_scaled = tl.load(logits_ptr + offs) * sm_scale
    m = tl.max(logits_scaled, axis=0)
    z = tl.sum(tl.exp(logits_scaled - m), axis=0)
    lse = tl.log(z) + m
    tl.store(out_ptr, lse)


# Triton kernel: reduce lse across heads per batch
# lse_ptrs:   *f32, [batch_size, num_heads]
# out_ptr:    *f32, [batch_size]
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,         # *f32, [batch_size, num_heads]
    out_ptr,          # *f32, [batch_size]
    num_heads: tl.constexpr,  # int
):
    b = tl.program_id(0)
    lse_vec = tl.load(lse_ptrs + b * num_heads + tl.arange(0, num_heads))
    sum_lse = tl.sum(lse_vec, axis=0)
    mean_lse = sum_lse / num_heads
    tl.store(out_ptr + b, mean_lse)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure inputs are on CUDA and contiguous
    device = q_nope.device
    if not TRITON_AVAILABLE:
        # Fallback: original PyTorch behavior
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        # Reference tensors
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)
        output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        for b in range(batch_size):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                output[b].zero_()
                continue

            L_tokens = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)

            Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
            Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]
            qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
            qp = q_pe[b].to(torch.float32)    # [num_qo_heads, head_dim_kpe]

            logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, L_tokens]
            logits_scaled = logits * sm_scale

            # lse per head
            lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, L_tokens]
            out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
            output[b] = out.to(torch.bfloat16)

        return output, lse

    # Triton path
    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]  # ensure last dim matches q_pe

    # Prepare per-batch per-head Kc_tmp and Kp_tmp
    Kc_tmp = torch.empty((batch_size, 0), dtype=torch.float32, device=device)  # placeholder; real size determined in loop
    Kp_tmp = torch.empty((batch_size, 0), dtype=torch.float32, device=device)

    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    # We need to recompute per-batch L_tokens and indices
    L_tokens_list = []
    for b in range(batch_size):
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        L_tokens_list.append(L_tokens)

    # We will build per-batch K_tmp buffers dynamically using kernel (Kc_tmp, Kp_tmp) per batch
    # But for Triton kernels, we launch gather per batch. We need to allocate per-batch buffers first.
    # Allocate empty placeholders of correct shape (will be overwritten inside per-batch loop)
    Kc_tmp_buffers = [torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device) for L_tokens in L_tokens_list]
    Kp_tmp_buffers = [torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device) for L_tokens in L_tokens_list]

    for b in range(batch_size):
        L_tokens = L_tokens_list[b]
        tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32)

        # Launch gather tokens for Kc and Kp
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=ckv_cache.squeeze(1),  # shape [num_pages, head_dim_ckv]
            idx_ptr=tok_idx,
            out_ptr=Kc_tmp_buffers[b],
            head_dim=head_dim_ckv,
            L_tokens=L_tokens,
        )
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=kpe_cache.squeeze(1),  # shape [num_pages, head_dim_kpe]
            idx_ptr=tok_idx,
            out_ptr=Kp_tmp_buffers[b],
            head_dim=head_dim_kpe,
            L_tokens=L_tokens,
        )

        # Compute logits per head using Triton
        for h in range(num_qo_heads):
            qn = q_nope[b, h].to(torch.float32)       # [head_dim_ckv]
            qp = q_pe[b, h].to(torch.float32)         # [head_dim_kpe]
            logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)

            forward_attention_kernel[(L_tokens,)](
                Kc_tmp_ptr=Kc_tmp_buffers[b],
                Kp_tmp_ptr=Kp_tmp_buffers[b],
                qn_ptr=qn,
                qp_ptr=qp,
                logits_ptr=logits,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                L_tokens=L_tokens,
            )

            # Softmax over logits_scaled
            attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            softmax_kernel[(L_tokens,)](
                logits_ptr=logits,
                out_ptr=attn,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
            )

            # Matvec: output[h] = attn @ Kc_tmp[b]
            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_kernel[(head_dim_ckv,)](
                attn_ptr=attn,
                Kc_ptr=Kc_tmp_buffers[b],
                out_ptr=out_vec,
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
            )

            # Store output
            output[b, h] = out_vec

            # Per-head lse = logsumexp(logits * sm_scale)
            lse_val = torch.empty((), dtype=torch.float32, device=device)
            lse_per_head_kernel[(L_tokens,)](
                logits_ptr=logits,
                out_ptr=lse_val,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
            )
            lse_per_head[b, h] = lse_val

    # Reduce lse across heads and convert to base-2
    lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)
    lse_reduce_kernel[(batch_size,)](
        lse_ptrs=lse_per_head,
        out_ptr=lse_base2,
        num_heads=num_qo_heads,
    )

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)

    return output, lse_base2


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only execution
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
