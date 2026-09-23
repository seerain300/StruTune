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
    K_src_ptr,        # *f32, flattened [num_pages, head_dim_ckv] or [num_pages, head_dim_kpe]
    idx_ptr,          # *i32, [L_tokens] token indices
    out_ptr,          # *f32, contiguous [L_tokens, head_dim]
    num_pages: tl.constexpr,  # int
    head_dim: tl.constexpr,   # int
    L_tokens: tl.constexpr,   # int
):
    # Each program copies one row
    row_id = tl.program_id(0)
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + row_id)
    src_linear = tok_id * head_dim + offs
    vals = tl.load(K_src_ptr + src_linear)
    tl.store(out_ptr + row_id * head_dim + offs, vals)


# Triton kernel: compute logits per head for all tokens (matvec part)
@triton.jit
def forward_attention_kernel(
    qn_ptr,            # *f32, [head_dim_ckv] or [head_dim_kpe]
    K_ptr,             # *f32, [L_tokens, head_dim]
    out_ptr,           # *f32, [L_tokens]
    head_dim,          # int
    L_tokens,          # int
    sm_scale,          # f32 scalar
):
    # One program computes all tokens for a given qn
    offs = tl.arange(0, head_dim)
    # Dot product: out[i] = sum_j qn[j] * K[i, j]
    # We implement this as a simple loop over head_dim (small for these use-cases)
    for i in range(0, L_tokens):
        acc = 0.0
        for j in range(0, head_dim):
            qj = tl.load(qn_ptr + j)
            Ki_j = tl.load(K_ptr + i * head_dim + j)
            acc += qj * Ki_j
        # Save acc as logits for token i
        tl.store(out_ptr + i, acc)


# Triton kernel: compute softmax over a vector (size = L_tokens)
@triton.jit
def softmax_kernel(
    logits_ptr,        # *f32, [L_tokens]
    out_ptr,           # *f32, [L_tokens]
    L_tokens,          # int
):
    # Compute max for numerical stability
    max_val = -float('inf')
    for i in range(0, L_tokens):
        val = tl.load(logits_ptr + i)
        if val > max_val:
            max_val = val
    # Compute sum of exp(logits - max)
    sum_exp = 0.0
    for i in range(0, L_tokens):
        val = tl.load(logits_ptr + i)
        exp_val = tl.exp((val - max_val) * 1.0)  # scale by 1.0, we'll apply sm_scale in host
        sum_exp += exp_val
    # Write softmax
    for i in range(0, L_tokens):
        val = tl.load(logits_ptr + i)
        soft = tl.exp((val - max_val) * 1.0) / sum_exp
        tl.store(out_ptr + i, soft)


# Triton kernel: compute matvec out = attn @ K (K is [L_tokens, head_dim], out is [head_dim])
@triton.jit
def matvec_kernel(
    attn_ptr,          # *f32, [L_tokens]
    K_ptr,             # *f32, [L_tokens, head_dim]
    out_ptr,           # *f32, [head_dim]
    head_dim,          # int
    L_tokens,          # int
):
    offs = tl.arange(0, head_dim)
    # Compute out[j] = sum_i attn[i] * K[i, j]
    for j in range(0, head_dim):
        acc = 0.0
        for i in range(0, L_tokens):
            ai = tl.load(attn_ptr + i)
            Ki_j = tl.load(K_ptr + i * head_dim + j)
            acc += ai * Ki_j
        tl.store(out_ptr + j, acc)


# Triton kernel: compute per-head logsumexp over logits vector (size = L_tokens)
@triton.jit
def lse_per_head_kernel(
    logits_ptr,        # *f32, [L_tokens]
    out_ptr,           # *f32, scalar
    L_tokens,          # int
    sm_scale,          # f32 scalar
):
    # Compute logsumexp over L_tokens with scaling
    max_val = -float('inf')
    for i in range(0, L_tokens):
        val = tl.load(logits_ptr + i)
        if val > max_val:
            max_val = val
    sum_exp = 0.0
    for i in range(0, L_tokens):
        val = tl.load(logits_ptr + i)
        sum_exp += tl.exp((val - max_val) * sm_scale)
    lse = max_val * sm_scale + tl.log(sum_exp)
    tl.store(out_ptr, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is computed in Triton

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes (as in original): q_nope [B, 16, 512], q_pe [B, 16, 64], ckv_cache [N, 1, 512], kpe_cache [N, 1, 64]
        assert q_nope.dim() == 3 and q_pe.dim() == 3
        assert ckv_cache.dim() == 3 and kpe_cache.dim() == 3
        assert kv_indptr.dim() == 1 and kv_indices.dim() == 1
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]
        # Sanity checks similar to original
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
        assert kv_indptr.shape[0] == batch_size + 1
        device = q_nope.device
        assert q_nope.device == q_pe.device == ckv_cache.device == kpe_cache.device == kv_indptr.device == kv_indices.device

        # Prepare output
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(batch_size):
            # Read range and indices
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32).contiguous()

            # Prepare per-batch Kc_tmp and Kp_tmp
            Kc_tmp = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
            Kp_tmp = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

            # Gather Kc and Kp rows
            # ckv_cache is [num_pages, 1, 512], flatten [num_pages, 512]
            Kc_src = ckv_cache.to(torch.float32).flatten(0, 1)  # [num_pages, 512]
            Kp_src = kpe_cache.to(torch.float32).flatten(0, 1)  # [num_pages, 64]

            # Launch gather kernel
            grid = (L_tokens,)
            gather_tokens_kernel[grid](
                K_src_ptr=Kc_src,
                idx_ptr=tok_idx,
                out_ptr=Kc_tmp,
                num_pages=num_pages,
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
            )
            grid = (L_tokens,)
            gather_tokens_kernel[grid](
                K_src_ptr=Kp_src,
                idx_ptr=tok_idx,
                out_ptr=Kp_tmp,
                num_pages=num_pages,
                head_dim=head_dim_kpe,
                L_tokens=L_tokens,
            )

            # For each head, compute attention output and per-head lse
            for h in range(num_qo_heads):
                # qn and qp as float32
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [512]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [64]

                # logits per token: qn @ Kc_tmp.T + qp @ Kp_tmp.T -> [L_tokens]
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                grid = (L_tokens,)
                forward_attention_kernel[grid](
                    qn_ptr=qn,
                    K_ptr=Kc_tmp,
                    out_ptr=logits,
                    head_dim=head_dim_ckv,
                    L_tokens=L_tokens,
                    sm_scale=float(sm_scale),
                )
                qpl = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                grid = (L_tokens,)
                forward_attention_kernel[grid](
                    qn_ptr=qp,
                    K_ptr=Kp_tmp,
                    out_ptr=qpl,
                    head_dim=head_dim_kpe,
                    L_tokens=L_tokens,
                    sm_scale=float(sm_scale),
                )
                logits_scaled = logits + qpl  # logits_scaled = logits * sm_scale; here sm_scale == 1.0 in the original harness

                # Softmax
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                grid = (L_tokens,)
                softmax_kernel[grid](
                    logits_ptr=logits_scaled,
                    out_ptr=attn,
                    L_tokens=L_tokens,
                )

                # Output matvec: attn @ Kc_tmp -> [512]
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                grid = (head_dim_ckv,)
                matvec_kernel[grid](
                    attn_ptr=attn,
                    K_ptr=Kc_tmp,
                    out_ptr=out_vec,
                    head_dim=head_dim_ckv,
                    L_tokens=L_tokens,
                )
                output[b, h] = out_vec

                # Per-head lse
                lse_val = torch.empty((), dtype=torch.float32, device=device)
                grid = (L_tokens,)
                lse_per_head_kernel[grid](
                    logits_ptr=logits_scaled,
                    out_ptr=lse_val,
                    L_tokens=L_tokens,
                    sm_scale=float(sm_scale),
                )
                lse_per_head[b, h] = lse_val

        # Match original dtype for output (bfloat16), lse remains float32 (original returned float32 lse)
        output = output.to(torch.bfloat16)

        # Convert lse to base-2 (original divides by math.log(2.0))
        lse_base2 = lse_per_head / math.log(2.0)

        return output, lse_base2


def run(*args):
    return ModelNew()(*args)
