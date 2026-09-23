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
    tl.store(out_ptr + pid * head_dim + offs, tl.load(K_src_ptr + src_linear))


# Triton kernel: compute logits per head (forward attention part)
# output: [L_tokens] vector in out_ptr
@triton.jit
def forward_attention_kernel(
    qn_ptr,           # *f32, [head_dim_ckv]
    qp_ptr,           # *f32, [head_dim_kpe]
    Kc_ptr,           # *f32, [L_tokens, head_dim_ckv]
    Kp_ptr,           # *f32, [L_tokens, head_dim_kpe]
    out_ptr,          # *f32, [L_tokens]
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    L_tokens: tl.constexpr,
):
    offs = tl.arange(0, L_tokens)
    acc_qn = tl.zeros((L_tokens,), dtype=tl.float32)
    acc_qp = tl.zeros((L_tokens,), dtype=tl.float32)

    # qn @ Kc.T: accumulate Kc rows dot qn
    for i in range(head_dim_ckv):
        q = tl.load(qn_ptr + i)  # scalar
        col = tl.load(Kc_ptr + offs * head_dim_ckv + i)  # vector [L_tokens]
        acc_qn += col * q

    # qp @ Kp.T: accumulate Kp rows dot qp
    for i in range(head_dim_kpe):
        p = tl.load(qp_ptr + i)  # scalar
        col = tl.load(Kp_ptr + offs * head_dim_kpe + i)  # vector [L_tokens]
        acc_qp += col * p

    # sum both contributions
    tl.store(out_ptr + offs, acc_qn + acc_qp)


# Triton kernel: softmax over a vector x of length L (in-place into out_ptr)
@triton.jit
def softmax_kernel(
    x_ptr,            # *f32, [L]
    out_ptr,          # *f32, [L]
    L: tl.constexpr,  # int
    sm_scale: tl.float32,
):
    offs = tl.arange(0, L)
    x = tl.load(x_ptr + offs)
    # subtract max for numerical stability
    m = tl.max(x, axis=0)
    x = x - m
    x = x * sm_scale
    expx = tl.exp(x)
    denom = tl.sum(expx, axis=0)
    out = expx / denom
    tl.store(out_ptr + offs, out)


# Triton kernel: matvec y = attn @ Kc, where attn is [L_tokens], Kc is [L_tokens, head_dim]
@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32, [L_tokens]
    K_ptr,            # *f32, [L_tokens, head_dim]
    out_ptr,          # *f32, [head_dim]
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    offs_dim = tl.arange(0, head_dim)
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    for i in range(L_tokens):
        a = tl.load(attn_ptr + i)  # scalar
        k = tl.load(K_ptr + i * head_dim + offs_dim)  # vector [head_dim]
        acc += k * a
    tl.store(out_ptr + offs_dim, acc)


# Triton kernel: per-head logsumexp over logits_scaled (vector of length L)
@triton.jit
def lse_per_head_kernel(
    logits_ptr,       # *f32, [L_tokens]
    out_ptr,          # *f32, scalar
    L: tl.constexpr,  # int
    sm_scale: tl.float32,
):
    offs = tl.arange(0, L)
    x = tl.load(logits_ptr + offs)
    m = tl.max(x, axis=0)
    x = x - m
    x = x * sm_scale
    sumexp = tl.sum(tl.exp(x), axis=0)
    lse = m * sm_scale + tl.log(sumexp)
    tl.store(out_ptr, lse)


# Triton kernel: reduce per-batch per-head lse across num_heads and divide by ln(2)
# input: lse_per_head[b, 0:num_heads], output: lse_base2[b]
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,         # *f32, [batch_size, num_heads]
    out_ptr,          # *f32, [batch_size]
    num_heads: tl.constexpr,  # int
):
    b = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for h in range(num_heads):
        lse = tl.load(lse_ptrs + b * num_heads + h)
        acc += lse
    acc = acc / tl.log(2.0)
    tl.store(out_ptr + b, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA/Triton execution
        assert TRITON_AVAILABLE, "Triton is not available"
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be CUDA"

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare per-batch token ranges
        B = batch_size
        num_pages = ckv_cache.shape[0]
        # Ensure kv_indptr, kv_indices are int32 on device
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.int32)

        # Output allocation
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

        # Per-batch per-head lse buffer (float32), then reduce
        lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)
        lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)

        # Loop over batch
        for b in range(B):
            # Compute L_tokens and gather indices
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous().to(torch.int32)

            # Prepare contiguous per-batch K buffers
            Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
            Kp_tmp = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

            # Launch gather kernel for this batch
            grid_gather = (L_tokens,)
            gather_tokens_kernel[grid_gather](
                ckv_cache[b].contiguous().view(-1),             # K_src_ptr: [num_pages*head_dim_ckv]
                kpe_cache[b].contiguous().view(-1),             # K_src_ptr: [num_pages*head_dim_kpe] for Kp
                Kc_tmp,                                          # out_ptr: [L_tokens, head_dim_ckv]
                num_pages, head_dim_ckv, L_tokens,
            )

            # For Kp, gather from kpe_cache[b]
            gather_tokens_kernel[grid_gather](
                kpe_cache[b].contiguous().view(-1),             # K_src_ptr: [num_pages*head_dim_kpe]
                tok_idx,                                         # idx_ptr: [L_tokens]
                Kp_tmp,                                          # out_ptr: [L_tokens, head_dim_kpe]
                num_pages, head_dim_kpe, L_tokens,
            )

            # Process each head
            for h in range(num_qo_heads):
                # q vectors
                qn = q_nope[b, h].contiguous().to(torch.float32)  # [head_dim_ckv]
                qp = q_pe[b, h].contiguous().to(torch.float32)    # [head_dim_kpe]

                # Compute logits vector [L_tokens]
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                grid_f = (L_tokens,)
                forward_attention_kernel[grid_f](
                    qn, qp, Kc_tmp, Kp_tmp, logits, head_dim_ckv, head_dim_kpe, L_tokens,
                )

                # Softmax over logits_scaled = logits * sm_scale
                logits_scaled = logits
                attn = torch.empty_like(logits_scaled, dtype=torch.float32, device=device)
                grid_s = (L_tokens,)
                softmax_kernel[grid_s](
                    logits_scaled, attn, L_tokens, sm_scale,
                )

                # Output per head: attn @ Kc_tmp -> [head_dim_ckv]
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_kernel[(head_dim_ckv,)](
                    attn, Kc_tmp, out_vec, head_dim_ckv, L_tokens,
                )
                output[b, h] = out_vec

                # Per-head lse
                lse_val = torch.empty((), dtype=torch.float32, device=device)
                lse_per_head_kernel[(L_tokens,)](
                    logits_scaled, lse_val, L_tokens, sm_scale,
                )
                lse_per_head[b, h] = lse_val

        # Reduce lse across heads and convert to base-2 (divide by ln(2))
        lse_reduce_kernel[(B,)](
            lse_per_head, lse_base2, num_qo_heads,
        )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse_base2