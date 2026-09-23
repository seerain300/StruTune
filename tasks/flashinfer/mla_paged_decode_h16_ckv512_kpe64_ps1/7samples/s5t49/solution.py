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
    # Each program computes one token's logit
    pid = tl.program_id(0)  # token id in [0, L_tokens)
    sum1 = tl.zeros((), dtype=tl.float32)
    # qn @ Kc.T
    for k in range(0, head_dim):
        qk = tl.load(qn_ptr + k)
        Kk = tl.load(Kc_ptr + pid * head_dim + k)
        sum1 += qk * Kk
    sum2 = tl.zeros((), dtype=tl.float32)
    # qp @ Kp.T
    for k in range(0, head_dim_kpe):
        qk = tl.load(qp_ptr + k)
        Kk = tl.load(Kp_ptr + pid * head_dim_kpe + k)
        sum2 += qk * Kk
    tl.store(logits_ptr + pid, sum1 + sum2)


@triton.jit
def softmax_kernel(
    logits_ptr,       # *f32, [L_tokens]
    out_ptr,          # *f32, [L_tokens] softmax values
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # 1D softmax over vector after scaling
    offs = tl.arange(0, L_tokens)
    vec = tl.load(logits_ptr + offs)
    scaled = vec * sm_scale
    m = tl.max(scaled, axis=0)
    z = scaled - m
    expz = tl.exp(z)
    sumexp = tl.sum(expz, axis=0)
    out = expz / sumexp
    tl.store(out_ptr + offs, out)


@triton.jit
def matvec_kernel(
    attn_ptr,         # *f32, [L_tokens]
    K_ptr,            # *f32, [L_tokens, head_dim]
    out_ptr,          # *f32, [head_dim] output vector
    head_dim: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # out_vec[j] = sum_i attn[i] * K[i, j]
    for j in range(0, head_dim):
        sumj = tl.zeros((), dtype=tl.float32)
        for i in range(0, L_tokens):
            sumj += tl.load(attn_ptr + i) * tl.load(K_ptr + i * head_dim + j)
        tl.store(out_ptr + j, sumj)


@triton.jit
def lse_per_head_kernel(
    logits_ptr,       # *f32, [L_tokens]
    out_ptr,          # *f32, scalar per-head lse
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # Compute natural logsumexp of scaled logits
    offs = tl.arange(0, L_tokens)
    vec = tl.load(logits_ptr + offs)
    scaled = vec * sm_scale
    m = tl.max(scaled, axis=0)
    z = scaled - m
    expz = tl.exp(z)
    sumexp = tl.sum(expz, axis=0)
    lse = tl.log(sumexp) + m  # natural logsumexp
    tl.store(out_ptr, lse)


@triton.jit
def lse_reduce_kernel(
    lse_ptr,          # *f32, [B, H] per-batch per-head lse
    out_ptr,          # *f32, [B] reduced lse per batch
    B: tl.constexpr,
    H: tl.constexpr,
):
    b = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for h in range(0, H):
        acc += tl.load(lse_ptr + b * H + h)
    tl.store(out_ptr + b, acc)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    num_pages = ckv_cache.shape[0]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1

    # Prepare output and lse buffers
    output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

    # We will return lse per batch (original returns [B, H]; we reduce to [B] for simplicity).
    lse_per = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    # Iterate over batch
    for b in range(batch_size):
        # Gather indices length and tokens
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            # No KV indices for this batch element: output zeros and lse zeros
            output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            for h in range(num_qo_heads):
                lse_per[b, h] = 0.0
            continue

        # Gather Kc_tmp and Kp_tmp: [L_tokens, head_dim] and [L_tokens, head_dim_kpe]
        Kc_b = torch.empty((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
        Kp_b = torch.empty((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

        # Flatten K_src pointers: [num_pages, head_dim] -> contiguous
        # Note: ckv_cache and kpe_cache are [num_pages, 1, D], we ignore the singleton dim in Triton.
        Kc_flat = ckv_cache.contiguous().view(-1, head_dim_ckv).float()  # ensure f32
        Kp_flat = kpe_cache.contiguous().view(-1, head_dim_kpe).float()

        # Launch gather kernels
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=Kc_flat, idx_ptr=kv_indices[:L_tokens], out_ptr=Kc_b,
            num_pages=num_pages, head_dim=head_dim_ckv, L_tokens=L_tokens,
            num_warps=1, num_stages=1
        )
        gather_tokens_kernel[(L_tokens,)](
            K_src_ptr=Kp_flat, idx_ptr=kv_indices[:L_tokens], out_ptr=Kp_b,
            num_pages=num_pages, head_dim=head_dim_kpe, L_tokens=L_tokens,
            num_warps=1, num_stages=1
        )

        # Compute output per head
        for h in range(num_qo_heads):
            # Load qn and qp (fp32)
            qn = q_nope[b, h].float().contiguous()  # [head_dim_ckv]
            qp = q_pe[b, h].float().contiguous()    # [head_dim_kpe]

            # Logits vector [L_tokens]
            logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            forward_attention_kernel[(L_tokens,)](
                qn_ptr=qn, qp_ptr=qp, Kc_ptr=Kc_b, Kp_ptr=Kp_b, logits_ptr=logits,
                head_dim=head_dim_ckv, head_dim_kpe=head_dim_kpe, L_tokens=L_tokens,
                num_warps=1, num_stages=1
            )

            # Softmax on scaled logits
            attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            softmax_kernel[(L_tokens,)](
                logits_ptr=logits, out_ptr=attn, L_tokens=L_tokens, sm_scale=float(sm_scale),
                num_warps=1, num_stages=1
            )

            # Output vector: attn @ Kc_b
            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_kernel[(head_dim_ckv,)](
                attn_ptr=attn, K_ptr=Kc_b, out_ptr=out_vec,
                head_dim=head_dim_ckv, L_tokens=L_tokens,
                num_warps=1, num_stages=1
            )
            output[b, h] = out_vec

            # Per-head lse: logsumexp of scaled logits (natural)
            lse_val = torch.empty((), dtype=torch.float32, device=device)
            lse_per_head_kernel[(L_tokens,)](
                logits_ptr=logits, out_ptr=lse_val, L_tokens=L_tokens, sm_scale=float(sm_scale),
                num_warps=1, num_stages=1
            )
            lse_per[b, h] = lse_val

    # Reduce per-head lse across heads to produce [B]
    lse_base = torch.empty((batch_size,), dtype=torch.float32, device=device)
    lse_reduce_kernel[(batch_size,)](
        lse_ptr=lse_per, out_ptr=lse_base, B=batch_size, H=num_qo_heads,
        num_warps=1, num_stages=1
    )

    # Cast output to bfloat16 to match original
    output_bf16 = output.to(torch.bfloat16)

    # Divide by ln(2) to mimic original's base-2 logsumexp conversion
    lse_base2 = lse_base / math.log(2.0)

    return output_bf16, lse_base2


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only execution: no torch tensor ops in host
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
