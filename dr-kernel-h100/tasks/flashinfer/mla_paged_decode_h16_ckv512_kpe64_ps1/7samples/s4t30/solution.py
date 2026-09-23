import torch
import math
import triton
import triton.language as tl


@triton.jit
def lse_and_attention_kernel(
    qn_ptr,           # *float32, [B*N*Dc] but we pass per-head pointer by slicing
    qp_ptr,           # *float32, [B*N*Dp] same
    Kc_ptr,           # *float32, [P*Dc] contiguous view after squeezing
    Kp_ptr,           # *float32, [P*Dp] contiguous view after squeezing
    tok_idx_ptr,      # *int32, [M_b]
    attn_ptr,         # *float32, [B*N*M_b] where we store attention for each (b,h)
    lse_ptr,          # *float32, [B*N]
    B: tl.constexpr,      # batch size
    N: tl.constexpr,      # number of heads (16)
    Dc: tl.constexpr,     # 512
    Dp: tl.constexpr,     # 64
    M_b: tl.constexpr,    # tokens in this batch
    sm_scale: tl.constexpr,  # scaling
    BLOCK_N: tl.constexpr     # token tile (e.g., 128)
):
    # One program per (b,h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Base offsets for qn, qp
    base_qn = (pid_b * N + pid_h) * Dc
    base_qp = (pid_b * N + pid_h) * Dp

    # Load qn_vec and qp_vec for this head
    qn_vec = tl.load(qn_ptr + base_qn + tl.arange(0, Dc))
    qp_vec = tl.load(qp_ptr + base_qp + tl.arange(0, Dp))

    # Initialize max and sum for logsumexp
    max_log = tl.full([1], -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros([1], dtype=tl.float32)

    # Iterate over tokens in chunks
    for start in range(0, M_b, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < M_b
        tok = tl.load(tok_idx_ptr + offs, mask=mask, other=0)
        # Kc_ptrs and Kp_ptrs for this chunk: linear indices tok * Dc and tok * Dp
        Kc_chunk = tl.load(Kc_ptr + tok * Dc, mask=mask, other=0.0)  # [BLOCK_N, Dc]
        Kp_chunk = tl.load(Kp_ptr + tok * Dp, mask=mask, other=0.0)  # [BLOCK_N, Dp]

        # Compute logits per token
        # logits_vec[BLOCK_N] = dot(qn_vec, Kc_chunk.T) + dot(qp_vec, Kp_chunk.T)
        # tl.dot expects [Dc, 1] with [1, Dc] -> scalar
        logits_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
        for i in range(BLOCK_N):
            valid = mask[i]
            if valid:
                kci = Kc_chunk[i]  # [Dc]
                kpi = Kp_chunk[i]  # [Dp]
                # Ensure proper dtypes for dot
                qn = qn_vec.to(tl.float32)
                kp = kpi.to(tl.float32)
                kc = kci.to(tl.float32)
                logits_vec[i] = tl.sum(qn * kc) + tl.sum(qp_vec * kp)

        # Apply scaling
        logits_scaled = logits_vec * sm_scale

        # Update max and sum for logsumexp
        chunk_max = tl.max(logits_scaled, axis=0)
        new_max = tl.maximum(max_log, chunk_max)
        # sum_exp = sum_exp * exp(max_log - new_max) + sum(exp(logits_scaled - new_max))
        # Note: Triton’s tl.sum works over vectors; we use a scalar reduction
        sum_exp = sum_exp * tl.exp(max_log - new_max) + tl.sum(tl.exp(logits_scaled - new_max), axis=0)
        max_log = new_max

    # Final logsumexp (natural log), then convert to base-2
    lse_natural = tl.log(sum_exp) + max_log
    lse = lse_natural / math.log(2.0)

    # Store lse
    tl.store(lse_ptr + pid_b * N + pid_h, lse)

    # Store attention weights for all tokens
    # attn_ptr layout: [B, N, M_b] flattened as ((b*N + h)*M_b + tok)
    for tok in range(0, M_b):
        attn_val = tl.exp(logits_scaled[tok] - max_log) / tl.exp(max_log)  # softmax normalized
        tl.store(attn_ptr + (pid_b * N + pid_h) * M_b + tok, attn_val)


@triton.jit
def matvec_per_head_kernel(
    attn_ptr,         # *float32, [B*N*M_b]
    Kc_ptr,           # *float32, [P*Dc] contiguous view after squeezing
    out_ptr,          # *float32, [B*N*Dc]
    B: tl.constexpr,      # batch size
    N: tl.constexpr,      # number of heads (16)
    Dc: tl.constexpr,     # 512
    M_b: tl.constexpr,    # tokens in this batch
    BLOCK_D: tl.constexpr # tile over Dc
):
    # One program per (b,h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Base offsets
    base = pid_b * N + pid_h

    # We need attn vector for this (b,h): load from attn_ptr at index base * M_b + tok
    # out[h, :] = sum_{t} attn[base, t] * Kc[t, :]
    out_vec = tl.zeros([Dc], dtype=tl.float32)
    for start in range(0, Dc, BLOCK_D):
        offs_d = start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < Dc
        # Load Kc chunk
        Kc_chunk = tl.load(Kc_ptr + offs_d * M_b, mask=mask_d, other=0.0)  # incorrect; we need [BLOCK_D, M_b]
        # Instead, we loop over tokens:
        # Implement reduction: out_vec += sum_t attn[base*M_b + t] * Kc[t, offs_d]
        # We need to load attn scalars per t and Kc[t, offs_d] vectors. Use Python for loop to iterate t.
        for t in range(0, M_b):
            attn_val = tl.load(attn_ptr + base * M_b + t)  # scalar
            # Load Kc row t at columns offs_d
            Kc_row = tl.load(Kc_ptr + t * Dc + offs_d, mask=mask_d, other=0.0)
            out_vec[offs_d] += attn_val * Kc_row

    tl.store(out_ptr + base * Dc + tl.arange(0, Dc), out_vec, mask=tl.arange(0, Dc) < Dc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all computation is Triton kernels

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, unused=None):
        # Device and shapes
        device = q_nope.device
        B, N, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        # Ensure dtypes: we compute in float32, return in bfloat16
        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()

        # Prepare Kc and Kp as [P, D] contiguous views after squeezing the size-1 dim
        # Note: original uses ckv_cache.squeeze(1) -> [P, Dc], same for kpe_cache -> [P, Dp]
        # Here we treat the given [P, 1, D] tensors and flatten per token. To avoid torch.cat in host,
        # we operate directly with the 3D tensors by indexing each token. Triton kernels will load per tok.
        # We do not perform torch.squeeze or torch.cat; we pass pointers as-is and slice inside kernels.

        # Compute tok_idx and M_b for each batch b
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1
        M_bs = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M_bs.append(end - start)
        # We will launch kernels per b; pass M_b and tok_idx accordingly.

        # Allocate output and lse
        out = torch.empty((B, N, Dc), dtype=torch.float32, device=device)
        lse = torch.empty((B, N), dtype=torch.float32, device=device)

        # Launch Triton kernels per batch
        # Kernel 1: compute logits, lse (base-2), and attention for each head
        attn = torch.empty((B * N, M_bs[0]), dtype=torch.float32, device=device)  # dummy; not used because we write per (b,h) in kernel
        for b in range(B):
            M_b = M_bs[b]
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()

            # For Kc_ptr and Kp_ptr, pass the original [P, 1, D] tensors and let Triton index by tok* D.
            # We can pass any pointers; Triton will read from the provided memory. Since we don't need
            # to squeeze or cat, we pass the original tensors and let kernel access per tok via tok*stride.
            # However, Triton expects flat pointers; for correct indexing, we treat the [P,1,D] as [P,D]
            # by passing the underlying flat memory. In PyTorch, [P,1,D].reshape(-1) yields a contiguous
            # [P*D] vector. Since squeeze(1) is not available in Triton host code, we avoid torch ops.
            # Instead, we pass the original tensors and let the kernel compute indices correctly by assuming
            # the provided [P,1,D] tensors are contiguous. We do not use torch.squeeze in host.

            # Launch lse_and_attention_kernel for this batch
            grid = (B, N)
            lse_and_attention_kernel[grid](
                q_nope_f32, q_pe_f32, ckv_cache, kpe_cache,
                tok_idx,
                attn, lse,
                B, N, Dc, Dp, M_b, float(sm_scale),
                BLOCK_N=128
            )

        # Kernel 2: matvec per head: out[h, :] = attention[b,h,:] @ Kc_sub[b]
        # Since we computed attn inside the lse kernel per (b,h), we reconstruct attn here as zeros to
        # satisfy the kernel signature. However, to maintain correctness, we instead implement a per-(b,h)
        # matvec using Triton: we loop over tokens and perform out_vec += attn[t] * Kc[t, :]. This kernel
        # is per (b,h) and uses Kc_ptr corresponding to the [P,1,Dc] tensor.
        grid_proj = (B, N)
        matvec_per_head_kernel[grid_proj](
            attn,  # attention we stored in lse kernel under lse_ptr? We need a separate attn buffer.
            ckv_cache,  # [P,1,Dc]; kernel accesses per tok by tok*stride
            out,
            B, N, Dc, M_bs[0], BLOCK_D=128
        )

        # Convert output to bfloat16 as original returns
        out = out.to(torch.bfloat16)

        # lse is float32 as original; return both
        return out, lse


def run(*args):
    return ModelNew()(*args)
