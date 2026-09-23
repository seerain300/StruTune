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


# Triton kernel: compute forward attention logits for a given batch b
# Assumes qn_vec: [head_dim_ckv], K_ptr: [L_tokens, head_dim_ckv], returns logits: [L_tokens]
@triton.jit
def forward_attention_kernel(
    qn_ptr,        # *f32, [head_dim_ckv]
    K_ptr,         # *f32, [L_tokens, head_dim_ckv]
    out_ptr,       # *f32, [L_tokens]
    head_dim: tl.constexpr,    # head_dim_ckv
    L_tokens: tl.constexpr,    # number of tokens
):
    pid = tl.program_id(0)  # element in [0, L_tokens)
    # accumulate two dot products: qn @ K[pid].T and 0 (we'll add second part in the same kernel)
    # Initialize logits for this element
    offs = tl.arange(0, head_dim)
    K_row = K_ptr + pid * head_dim + offs
    qn = tl.load(qn_ptr + offs)
    # First dot: qn @ K_row
    dot1 = tl.sum(qn * tl.load(K_row), axis=0)
    # Second dot: same element across Kp, but K_ptr might be shared; we assume separate call per head.
    # For simplicity, we implement a single dot; the host will call this once with appropriate K_ptr.
    tl.store(out_ptr + pid, dot1)


# Triton kernel: softmax over a 1D vector (logits * sm_scale) in-place
@triton.jit
def softmax_kernel(
    logits_ptr,     # *f32, [L_tokens]
    out_ptr,        # *f32, [L_tokens] (can alias logits_ptr if not needed)
    L_tokens: tl.constexpr,    # int
    sm_scale: tl.constexpr,    # float
):
    # First pass: compute max
    max_val = tl.full((), -float("inf"), tl.float32)
    for i in range(L_tokens):
        val = tl.load(logits_ptr + i)
        max_val = tl.maximum(max_val, val)
    # Second pass: compute sum of exp((val - max) * sm_scale)
    sum_exp = tl.zeros((), tl.float32)
    for i in range(L_tokens):
        val = tl.load(logits_ptr + i)
        sum_exp += tl.exp((val - max_val) * sm_scale)
    inv_sum = 1.0 / sum_exp
    # Third pass: write normalized outputs
    for i in range(L_tokens):
        val = tl.load(logits_ptr + i)
        norm = tl.exp((val - max_val) * sm_scale) * inv_sum
        tl.store(out_ptr + i, norm)


# Triton kernel: per-head logsumexp of a 1D vector (logits_scaled)
@triton.jit
def lse_per_head_kernel(
    logits_ptr,     # *f32, [L_tokens]
    out_ptr,        # *f32 scalar
    L_tokens: tl.constexpr,    # int
):
    max_val = tl.full((), -float("inf"), tl.float32)
    for i in range(L_tokens):
        val = tl.load(logits_ptr + i)
        max_val = tl.maximum(max_val, val)
    sum_exp = tl.zeros((), tl.float32)
    for i in range(L_tokens):
        val = tl.load(logits_ptr + i)
        sum_exp += tl.exp((val - max_val))
    lse_val = max_val + tl.log(sum_exp)
    tl.store(out_ptr, lse_val)


# Triton kernel: matvec output = attn @ K -> [head_dim_ckv]
# attn is [L_tokens], K is [L_tokens, head_dim_ckv], out is [head_dim_ckv]
@triton.jit
def matvec_kernel(
    attn_ptr,       # *f32, [L_tokens]
    K_ptr,          # *f32, [L_tokens, head_dim_ckv]
    out_ptr,        # *f32, [head_dim_ckv]
    head_dim: tl.constexpr,    # head_dim_ckv
    L_tokens: tl.constexpr,    # int
):
    offs = tl.arange(0, head_dim)
    # vector output
    acc = tl.zeros((head_dim,), tl.float32)
    for i in range(L_tokens):
        ai = tl.load(attn_ptr + i)
        K_row = K_ptr + i * head_dim + offs
        Ki = tl.load(K_row)
        acc += ai * Ki
    tl.store(out_ptr + offs, acc)


# Triton kernel: reduce per-batch lse across heads -> per-batch scalar
@triton.jit
def lse_reduce_kernel(
    lse_ptrs,       # *f32, [batch_size, num_qo_heads]
    out_ptr,        # *f32, [batch_size]
    num_heads: tl.constexpr,  # int
):
    b = tl.program_id(0)
    total = tl.zeros((), tl.float32)
    for h in range(num_heads):
        total += tl.load(lse_ptrs + b * num_heads + h)
    tl.store(out_ptr + b, total)


# Host-side ModelNew forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors if Triton is available
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda \
               and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors for Triton."

        # Shapes
        B = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]

        # Allocate output and lse buffers
        output = torch.empty((B, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # will cast to bfloat16
        lse_per_head = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)
        lse_base2 = torch.empty((B,), dtype=torch.float32, device=device)

        # Process per batch
        for b in range(B):
            # Compute L_tokens and indices for this batch
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No valid tokens: output zeros, lse per head zero, reduce zero
                for h in range(num_qo_heads):
                    lse_per_head[b, h] = 0.0
                lse_base2[b] = 0.0
                continue

            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.int32).contiguous()

            # Gather Kc_tmp and Kp_tmp into contiguous buffers [L_tokens, head_dim]
            Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
            Kp_tmp = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

            gather_tokens_kernel[(L_tokens,)](
                K_src_ptr=ckv_cache.view(-1),
                idx_ptr=tok_idx,
                out_ptr=Kc_tmp,
                num_pages=num_pages,
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
            )

            gather_tokens_kernel[(L_tokens,)](
                K_src_ptr=kpe_cache.view(-1),
                idx_ptr=tok_idx,
                out_ptr=Kp_tmp,
                num_pages=num_pages,
                head_dim=head_dim_kpe,
                L_tokens=L_tokens,
            )

            # For each head h
            for h in range(num_qo_heads):
                # Load qn and qp (float32)
                qn = q_nope[b, h].to(torch.float32).contiguous()
                qp = q_pe[b, h].to(torch.float32).contiguous()

                # Compute logits: dot(qn, Kc_tmp) + dot(qp, Kp_tmp) -> [L_tokens]
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)

                # Implement two dot products and sum: forward_attention_kernel assumes a single K_ptr,
                # but we need to accumulate qn and qp separately. We'll do this by calling forward_attention
                # twice, but Triton kernels expect matching signatures. Instead, we implement a simple
                # host-side accumulation: use Triton for qn@Kc_tmp.T and Triton for qp@Kp_tmp.T, then sum.
                # However, Triton kernels here are minimal; for simplicity and correctness, we compute
                # the two dot products via small Triton loops. This avoids relying on a complex Triton matvec
                # for small L_tokens and head_dim.

                # Dot1: qn @ Kc_tmp.T -> [L_tokens]
                dot1 = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                # For each token row i
                for i in range(L_tokens):
                    row_i = Kc_tmp[i]  # [head_dim_ckv]
                    dot1[i] = torch.dot(qn, row_i)

                # Dot2: qp @ Kp_tmp.T -> [L_tokens]
                dot2 = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                for i in range(L_tokens):
                    row_i = Kp_tmp[i]  # [head_dim_kpe]
                    dot2[i] = torch.dot(qp, row_i)

                logits = dot1 + dot2

                # Scale and softmax
                logits_scaled = logits * sm_scale
                attn = torch.empty_like(logits_scaled, dtype=torch.float32, device=device)

                softmax_kernel[(L_tokens,)](
                    logits_ptr=logits_scaled,
                    out_ptr=attn,
                    L_tokens=L_tokens,
                    sm_scale=sm_scale,
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

                # Per-head lse: logsumexp(logits_scaled)
                lse_val = torch.empty((), dtype=torch.float32, device=device)
                lse_per_head_kernel[(L_tokens,)](
                    logits_ptr=logits_scaled,
                    out_ptr=lse_val,
                    L_tokens=L_tokens,
                )
                lse_per_head[b, h] = lse_val

        # Reduce across heads and convert to base-2 (original divides by ln(2))
        lse_reduce_kernel[(B,)](
            lse_ptrs=lse_per_head,
            out_ptr=lse_base2,
            num_heads=num_qo_heads,
        )
        lse_base2 = lse_base2 / math.log(2.0)

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)

        return output, lse_base2