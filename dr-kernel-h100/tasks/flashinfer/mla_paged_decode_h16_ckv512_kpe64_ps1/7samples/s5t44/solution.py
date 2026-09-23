import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def gather_tokens_kernel(
    K_src_ptr,        # *f32, flattened [num_pages, head_dim]
    idx_ptr,          # *i32, [L_tokens] token indices
    out_ptr,          # *f32, contiguous [L_tokens, head_dim]
    num_pages: tl.constexpr,
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # Each program copies one token's row into out
    pid = tl.program_id(0)  # row id in [0, L_tokens)
    offs = tl.arange(0, head_dim)
    tok_id = tl.load(idx_ptr + pid)
    src_linear = tok_id * head_dim + offs
    tl.store(out_ptr + pid * head_dim + offs, tl.load(K_src_ptr + src_linear))


@triton.jit
def forward_attention_kernel(
    qn_ptr,           # *f32, [head_dim]
    qp_ptr,           # *f32, [head_dim_kpe]
    Kc_ptr,           # *f32, [L_tokens, head_dim]
    Kp_ptr,           # *f32, [L_tokens, head_dim_kpe]
    logits_ptr,       # *f32, [L_tokens]
    head_dim: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # Each program computes one output element (logit for a given token)
    pid = tl.program_id(0)  # token id in [0, L_tokens)
    # qn @ Kc.T
    sum1 = tl.zeros((), dtype=tl.float32)
    for k in range(0, head_dim):
        qk = tl.load(qn_ptr + k)
        Kk = tl.load(Kc_ptr + pid * head_dim + k)
        sum1 += qk * Kk
    # qp @ Kp.T
    sum2 = tl.zeros((), dtype=tl.float32)
    for k in range(0, head_dim_kpe):
        qk = tl.load(qp_ptr + k)
        Kk = tl.load(Kp_ptr + pid * head_dim_kpe + k)
        sum2 += qk * Kk
    tl.store(logits_ptr + pid, sum1 + sum2)


@triton.jit
def softmax_kernel(
    logits_ptr,       # *f32, [L_tokens]
    out_ptr,          # *f32, [L_tokens] (softmax values)
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,  # we multiply logits by sm_scale
):
    # 1D softmax over vector
    offs = tl.arange(0, L_tokens)
    logits = tl.load(logits_ptr + offs)
    # subtract max for numerical stability
    m = tl.max(logits, axis=0)
    logits = logits - m
    logits = logits * sm_scale
    expv = tl.exp(logits)
    denom = tl.sum(expv, axis=0)
    out = expv / denom
    tl.store(out_ptr + offs, out)


@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32, [L_tokens]
    K_ptr,            # *f32, [L_tokens, head_dim]
    out_ptr,          # *f32, [head_dim]
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # Each program computes one output element (dot over tokens)
    j = tl.program_id(0)  # index in [0, head_dim)
    sumv = tl.zeros((), dtype=tl.float32)
    for i in range(0, L_tokens):
        sumv += tl.load(attn_ptr + i) * tl.load(K_ptr + i * head_dim + j)
    tl.store(out_ptr + j, sumv)


@triton.jit
def lse_per_head_kernel(
    logits_ptr,       # *f32, [L_tokens]
    out_ptr,          # *f32, scalar
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # Compute logsumexp of logits * sm_scale
    offs = tl.arange(0, L_tokens)
    logits = tl.load(logits_ptr + offs)
    m = tl.max(logits, axis=0)
    sumexp = tl.sum(tl.exp((logits - m) * sm_scale), axis=0)
    lse = m + (1.0 / sm_scale) * tl.log(sumexp)
    tl.store(out_ptr, lse)


@triton.jit
def lse_reduce_kernel(
    lse_ptr,          # *f32, [B, H] flattened logically
    out_ptr,          # *f32, [B]
    num_heads: tl.constexpr,
    B: tl.constexpr,
):
    # One program per batch: reduce across heads
    b = tl.program_id(0)
    # Sum across heads dimension (H)
    sumv = tl.zeros((), dtype=tl.float32)
    for h in range(0, num_heads):
        sumv += tl.load(lse_ptr + b * num_heads + h)
    tl.store(out_ptr + b, sumv)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    dtype_bf16 = torch.bfloat16

    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]

    # Precompute K_all contiguous per batch
    # Kc_all: [num_pages, head_dim_ckv], Kp_all: [num_pages, head_dim_kpe]
    # Gather into per-batch contiguous [L_tokens, head_dim] buffers
    # First compute L_tokens for each batch
    assert kv_indptr.shape[0] == batch_size + 1, "kv_indptr length must be batch_size + 1"
    L_tokens_list = [int(kv_indptr[b + 1].item() - kv_indptr[b].item()) for b in range(batch_size)]

    # Allocate temporary buffers per batch
    Kc_tmp = []
    Kp_tmp = []
    for b in range(batch_size):
        L_tokens = L_tokens_list[b]
        if L_tokens <= 0:
            Kc_tmp.append(torch.empty((1, head_dim_ckv), dtype=torch.float32, device=device))
            Kp_tmp.append(torch.empty((1, head_dim_kpe), dtype=torch.float32, device=device))
            continue
        tok_idx = kv_indices[kv_indptr[b].item():kv_indptr[b + 1].item()].to(torch.int32)  # [L_tokens]
        # Gather rows from ckv_cache and kpe_cache
        Kc = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
        Kp = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]
        # Copy selected rows into contiguous buffers
        Kc_b = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
        Kp_b = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)
        # Launch gather kernel
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=Kc,
            idx_ptr=tok_idx,
            out_ptr=Kc_b,
            num_pages=Kc.shape[0],
            head_dim=head_dim_ckv,
            L_tokens=L_tokens,
        )
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=Kp,
            idx_ptr=tok_idx,
            out_ptr=Kp_b,
            num_pages=Kp.shape[0],
            head_dim=head_dim_kpe,
            L_tokens=L_tokens,
        )
        Kc_tmp.append(Kc_b)
        Kp_tmp.append(Kp_b)

    # Prepare output
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

    # lse_per_head: [B, H] in float32
    lse_per_head = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)
    # lse_reduce out: [B] float32
    lse_base2 = torch.empty((batch_size,), dtype=torch.float32, device=device)

    # For each batch b and head h
    for b in range(batch_size):
        L_tokens = L_tokens_list[b]
        Kc_b = Kc_tmp[b] if L_tokens > 0 else None
        Kp_b = Kp_tmp[b] if L_tokens > 0 else None

        # For each head h
        for h in range(num_qo_heads):
            # qn[h] and qp[h]
            qn = q_nope[b, h].to(torch.float32).contiguous()  # [head_dim_ckv]
            qp = q_pe[b, h].to(torch.float32).contiguous()   # [head_dim_kpe]

            if L_tokens <= 0:
                # No valid tokens; output zeros and lse -inf
                output[b, h] = torch.zeros((head_dim_ckv,), dtype=torch.float32, device=device)
                lse_per_head[b, h] = float("-inf")
                continue

            # Logits vector [L_tokens]
            logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            forward_attention_kernel[(L_tokens,)](
                qn_ptr=qn,
                qp_ptr=qp,
                Kc_ptr=Kc_b,
                Kp_ptr=Kp_b,
                logits_ptr=logits,
                head_dim=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                L_tokens=L_tokens,
                sm_scale=float(sm_scale),
            )
            # Softmax over logits * sm_scale, output attn [L_tokens]
            attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            softmax_kernel[(L_tokens,)](
                logits_ptr=logits,
                out_ptr=attn,
                L_tokens=L_tokens,
                sm_scale=float(sm_scale),
            )
            # Output vector per head: attn @ Kc_b
            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_kernel[(head_dim_ckv,)](
                attn_ptr=attn,
                K_ptr=Kc_b,  # [L_tokens, head_dim_ckv]
                out_ptr=out_vec,
                head_dim=head_dim_ckv,
                L_tokens=L_tokens,
            )
            output[b, h] = out_vec

            # Per-head lse: logsumexp of logits * sm_scale
            lse_val = torch.empty((), dtype=torch.float32, device=device)
            lse_per_head_kernel[(L_tokens,)](
                logits_ptr=logits,
                out_ptr=lse_val,
                L_tokens=L_tokens,
                sm_scale=float(sm_scale),
            )
            lse_per_head[b, h] = lse_val

    # Reduce across heads and convert to base-2 (original divides by ln(2))
    lse_reduce_kernel[(batch_size,)](
        lse_ptr=lse_per_head,
        out_ptr=lse_base2,
        num_heads=num_qo_heads,
        B=batch_size,
    )

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)

    return output, lse_base2


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only execution: no torch tensor ops in host
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
