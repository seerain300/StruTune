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
    head_dim: tl.constexpr,   # int (columns in each cache row)
    L_tokens: tl.constexpr,   # int (number of tokens selected for this batch)
):
    pid = tl.program_id(0)  # which token row to gather
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    src_linear = tok_id * head_dim + offs
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + pid * head_dim + offs, vals)


# Triton kernel: compute per-head logits = qn @ Kc.T + qp @ Kp.T -> [L_tokens]
@triton.jit
def forward_attention_kernel(
    qn_ptr,           # *f32, [head_dim_ckv]
    qp_ptr,           # *f32, [head_dim_kpe]
    Kc_ptr,           # *f32, [L_tokens, head_dim_ckv]
    Kp_ptr,           # *f32, [L_tokens, head_dim_kpe]
    logits_ptr,       # *f32, [L_tokens]
    L_tokens: tl.constexpr,       # int
    head_dim_ckv: tl.constexpr,   # int
    head_dim_kpe: tl.constexpr,   # int
    sm_scale: tl.constexpr,       # float scale
):
    offs = tl.arange(0, L_tokens)
    # qn @ Kc.T
    acc = tl.zeros((head_dim_ckv,), dtype=tl.float32)
    for i in range(L_tokens):
        q_row = tl.load(qn_ptr + offs[i] * head_dim_ckv)  # [head_dim_ckv]
        K_row = tl.load(Kc_ptr + i * head_dim_ckv + offs)  # [head_dim_ckv]
        acc += q_row * K_row
    # qp @ Kp.T
    acc2 = tl.zeros((head_dim_kpe,), dtype=tl.float32)
    for i in range(L_tokens):
        q_row = tl.load(qp_ptr + offs[i] * head_dim_kpe)  # [head_dim_kpe]
        K_row = tl.load(Kp_ptr + i * head_dim_kpe + offs)  # [head_dim_kpe]
        acc2 += q_row * K_row
    # combine and scale
    acc = acc + acc2  # [head_dim_ckv]
    acc = acc * sm_scale
    # store as 1D logits
    tl.store(logits_ptr + offs, acc)


# Triton kernel: softmax over a vector (size L_tokens), logits scaled by sm_scale
@triton.jit
def softmax_kernel(
    logits_scaled_ptr,  # *f32, [L_tokens]
    attn_ptr,           # *f32, [L_tokens]
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    offs = tl.arange(0, L_tokens)
    logits = tl.load(logits_scaled_ptr + offs)
    # subtract max for numerical stability
    m = tl.max(logits, axis=0)
    logits = logits - m
    e = tl.exp(logits)
    denom = tl.sum(e, axis=0)
    attn = e / denom
    tl.store(attn_ptr + offs, attn)


# Triton kernel: matvec attn @ Kc -> out_vec, Kc shape [L_tokens, head_dim_ckv], attn [L_tokens]
@triton.jit
def matvec_kernel(
    attn_ptr,          # *f32, [L_tokens]
    K_ptr,             # *f32, [L_tokens, head_dim_ckv]
    out_ptr,           # *f32, [head_dim_ckv]
    head_dim: tl.constexpr,        # int
    L_tokens: tl.constexpr,        # int
):
    offs = tl.arange(0, head_dim)
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    for i in range(L_tokens):
        attn_i = tl.load(attn_ptr + i)          # scalar
        K_row = tl.load(K_ptr + i * head_dim + offs)  # [head_dim]
        acc += attn_i * K_row
    tl.store(out_ptr + offs, acc)


# Triton kernel: per-head logsumexp of logits_scaled -> scalar lse
@triton.jit
def lse_per_head_kernel(
    logits_scaled_ptr,  # *f32, [L_tokens]
    out_ptr,            # *f32, scalar output
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    offs = tl.arange(0, L_tokens)
    logits = tl.load(logits_scaled_ptr + offs)
    m = tl.max(logits, axis=0)
    sum_exp = tl.sum(tl.exp((logits - m) * sm_scale), axis=0)
    lse = m * sm_scale + tl.log(sum_exp)
    tl.store(out_ptr, lse)


# Triton kernel: reduce per-batch lse across heads (num_heads is passed as int), divide by ln(2)
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,          # *f32, [batch_size, num_qo_heads]
    out_ptr,           # *f32, [batch_size]
    num_heads: tl.constexpr,       # int
    inv_ln2: tl.constexpr,         # float: 1 / ln(2)
):
    pid = tl.program_id(0)  # batch id
    sum_val = 0.0
    for h in range(num_heads):
        sum_val += tl.load(lse_ptrs + pid * num_heads + h)
    avg = sum_val / num_heads
    avg = avg * inv_ln2  # convert to base-2
    tl.store(out_ptr + pid, avg)


# Host function implementing ModelNew.forward, Triton-only
def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Shapes
    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    num_pages = ckv_cache.shape[0]
    device = q_nope.device

    # Sanity checks mirroring original
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    # For given inputs, num_pages == 989669
    # We will use the provided kv_indptr and kv_indices

    # Output tensors
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)
    lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    # Precompute K_all for ckv and kpe as float32, no torch ops in host
    # Note: squeeze(1) is not a torch op in host; we just work with ckv_cache and kpe_cache directly.
    # For Triton gather, we pass the original tensors.

    # For each batch b, compute L_tokens and idx_ptr
    for b in range(batch_size):
        # token range for this batch
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            # no tokens, set output to zeros and skip
            output[b].zero_()
            lse_per_head[b, :] = -float('inf')
            continue
        tok_idx = kv_indices[b:b + L_tokens].to(torch.int32)

        # Allocate per-batch contiguous buffers for Kc_tmp and Kp_tmp
        Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
        Kp_tmp = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

        # Gather tokens into per-batch buffers
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=ckv_cache.view(-1),           # [num_pages * head_dim_ckv]
            idx_ptr=tok_idx,                        # [L_tokens]
            out_ptr=Kc_tmp,                        # [L_tokens, head_dim_ckv]
            head_dim=head_dim_ckv,
            L_tokens=L_tokens,
        )
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=kpe_cache.view(-1),           # [num_pages * head_dim_kpe]
            idx_ptr=tok_idx,                        # [L_tokens]
            out_ptr=Kp_tmp,                        # [L_tokens, head_dim_kpe]
            head_dim=head_dim_kpe,
            L_tokens=L_tokens,
        )

        # Compute per-head outputs
        for h in range(num_qo_heads):
            qn = q_nope[b, h].to(torch.float32)   # [head_dim_ckv]
            qp = q_pe[b, h].to(torch.float32)     # [head_dim_kpe]

            # Allocate logits vector [L_tokens]
            logits_scaled = torch.empty((L_tokens,), dtype=torch.float32, device=device)

            # Forward attention logits
            forward_attention_kernel[(1,)](
                qn_ptr=qn,                         # [head_dim_ckv]
                qp_ptr=qp,                         # [head_dim_kpe]
                Kc_ptr=Kc_tmp,                    # [L_tokens, head_dim_ckv]
                Kp_ptr=Kp_tmp,                    # [L_tokens, head_dim_kpe]
                logits_ptr=logits_scaled,         # [L_tokens]
                L_tokens=L_tokens,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                sm_scale=float(sm_scale),
            )

            # Softmax over logits_scaled
            attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            softmax_kernel[(L_tokens,)](
                logits_scaled_ptr=logits_scaled,
                attn_ptr=attn,
                L_tokens=L_tokens,
                sm_scale=float(sm_scale),
            )

            # Matvec: attn @ Kc_tmp -> [head_dim_ckv]
            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_kernel[(head_dim_ckv,)](
                attn_ptr=attn,                    # [L_tokens]
                K_ptr=Kc_tmp,                    # [L_tokens, head_dim_ckv]
                out_ptr=out_vec,                 # [head_dim_ckv]
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
            )

            # Store output
            output[b, h] = out_vec

            # Per-head lse
            lse_val = torch.empty((), dtype=torch.float32, device=device)
            lse_per_head_kernel[(L_tokens,)](
                logits_scaled_ptr=logits_scaled,
                out_ptr=lse_val,
                L_tokens=L_tokens,
                sm_scale=float(sm_scale),
            )
            lse_per_head[b, h] = lse_val

        # Reduce lse across heads to per-batch and convert to base-2
        lse_reduce_kernel[(batch_size,)](
            lse_ptrs=lse_per_head,
            out_ptr=lse_base2,
            num_heads=num_qo_heads,
            inv_ln2=1.0 / math.log(2.0),
        )

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)
    return output, lse_base2


# Entry point class ModelNew
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure all inputs are on the same CUDA device (Triton requires CUDA)
        if not q_nope.is_cuda or not q_pe.is_cuda or not ckv_cache.is_cuda or not kpe_cache.is_cuda or not kv_indptr.is_cuda or not kv_indices.is_cuda:
            # Move to default CUDA device if available
            if torch.cuda.is_available():
                device = torch.device("cuda")
                q_nope = q_nope.to(device)
                q_pe = q_pe.to(device)
                ckv_cache = ckv_cache.to(device)
                kpe_cache = kpe_cache.to(device)
                kv_indptr = kv_indptr.to(device)
                kv_indices = kv_indices.to(device)
            else:
                raise RuntimeError("CUDA device required for Triton kernels")
        # Call Triton-only implementation
        output, lse_base2 = _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)
        return output, lse_base2