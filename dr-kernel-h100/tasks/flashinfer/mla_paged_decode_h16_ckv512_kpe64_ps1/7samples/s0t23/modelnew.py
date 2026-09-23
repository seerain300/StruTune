import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_lse_out_kernel(
    qnh_ptr,        # *f32, [512]
    Kc_ptr,         # *f32, [L_TOKENS * 512] contiguous
    Kp_ptr,         # *f32, [L_TOKENS * 64] contiguous
    out_vec_ptr,    # *f32, [512]
    lse_ptr,        # *f32, [16] (we only write one element per head)
    L_TOKENS: tl.constexpr,  # compile-time constant
    sm_scale: tl.constexpr,  # compile-time constant (we can pass as f32)
    compute_lse: tl.constexpr,  # 0 or 1
    qph_ptr,        # *f32, [64]
):
    # This kernel processes one head h. We encode h via program_id(0), but since we launch one per (b,h),
    # program_id(0) will be used to select batch index and then we just run the kernel for that head.
    # In practice, forward will pass qnh_ptr, qph_ptr, out_vec_ptr, and lse_ptr per head, and we will
    # call the kernel once per (b, h).
    # However, Triton does not allow functions with program_id, so we structure the call as:
    # grid = (B * 16,), and pass qnh_ptr, qph_ptr per head using slicing on host.
    # Here, we assume the caller binds qnh_ptr, qph_ptr correctly.

    # Compute logits vector
    logits = tl.zeros([L_TOKENS], dtype=tl.float32)
    # Load qnh and qph once
    qnh = tl.load(qnh_ptr + tl.arange(0, 512), mask=tl.arange(0, 512) < 512, other=0.0)
    qph = tl.load(qph_ptr + tl.arange(0, 64), mask=tl.arange(0, 64) < 64, other=0.0)

    for t in tl.static_range(0, L_TOKENS):
        offs_kc = t * 512 + tl.arange(0, 512)
        offs_kp = t * 64 + tl.arange(0, 64)
        Kc_row = tl.load(Kc_ptr + offs_kc)
        Kp_row = tl.load(Kp_ptr + offs_kp)
        acc1 = tl.sum(qnh * Kc_row, axis=0)
        acc2 = tl.sum(qph * Kp_row, axis=0)
        logits[t] = acc1 + acc2

    # Compute lse if requested
    if compute_lse:
        m = tl.max(logits, axis=0)
        s = tl.sum(tl.exp(logits - m * sm_scale), axis=0)
        lse_val = m + (tl.log(s) * 1.4426950408889634)  # 1 / ln(2)
        # write lse to lse_ptr[h]
        # We don't have direct h index from here; assume caller sets pointer accordingly.
        # Since forward launches one program per (b,h), we can pass lse_ptr[h].
        # Triton will write scalar to lse_ptr[h] via its pointer. To ensure correctness,
        # forward will pass lse_ptr[h] explicitly.
        tl.store(lse_ptr + 0, lse_val)  # forward will pass correct lse_ptr[h]
    else:
        # Read lse from lse_ptr[h]; assume lse has been precomputed in host or compute via reading a dummy.
        # In our launch, we will pass compute_lse=1; so this branch won't be taken.
        pass

    # Compute attention output if not computing lse
    if not compute_lse:
        # We need lse value for this head. Assume forward precomputes lse or we recompute here.
        # Since we pass compute_lse=1, this branch won't be taken. For completeness, we keep it.
        pass


# Note: The above kernel assumes compute_lse=1 in our forward. To compute output, we can re-launch a variant
# where we first compute lse in a separate kernel, then compute output. To keep code concise, we implement
# a second kernel that computes output given qnh, qph, lse, Kc, Kp.
# However, Triton does not support re-defining kernels here; we implement the output-only kernel as a separate
# Triton function below.

@triton.jit
def _attention_output_only_kernel(
    qnh_ptr,        # *f32, [512]
    Kc_ptr,         # *f32, [L_TOKENS * 512] contiguous
    Kp_ptr,         # *f32, [L_TOKENS * 64] contiguous
    out_vec_ptr,    # *f32, [512]
    lse_val,        # f32 scalar (we pass lse[h])
    L_TOKENS: tl.constexpr,
    sm_scale: tl.constexpr,
    qph_ptr,        # *f32, [64]
):
    logits = tl.zeros([L_TOKENS], dtype=tl.float32)
    qnh = tl.load(qnh_ptr + tl.arange(0, 512), mask=tl.arange(0, 512) < 512, other=0.0)
    qph = tl.load(qph_ptr + tl.arange(0, 64), mask=tl.arange(0, 64) < 64, other=0.0)

    for t in tl.static_range(0, L_TOKENS):
        offs_kc = t * 512 + tl.arange(0, 512)
        offs_kp = t * 64 + tl.arange(0, 64)
        Kc_row = tl.load(Kc_ptr + offs_kc)
        Kp_row = tl.load(Kp_ptr + offs_kp)
        acc1 = tl.sum(qnh * Kc_row, axis=0)
        acc2 = tl.sum(qph * Kp_row, axis=0)
        logits[t] = acc1 + acc2

    # Compute softmax with given lse_val
    for t in tl.static_range(0, L_TOKENS):
        scaled = logits[t] * sm_scale
        numerator = tl.exp(scaled - lse_val)
        # denominator is sum over all tokens
        # We need to recompute denominator efficiently.
        # Triton does not support while loops; use static_range and accumulate into a scalar.
        # We can compute denominator in a separate small loop.
        # Compute denominator
        denom = 0.0
        for u in tl.static_range(0, L_TOKENS):
            denom += tl.exp(logits[u] * sm_scale - lse_val)
        attn = numerator / denom
        # accumulate output vector: out_vec += attn * Kc_row[t, :]
        offs_out = t * 512 + tl.arange(0, 512)
        Kc_row = tl.load(Kc_ptr + offs_out)  # incorrect: offs_out points to next row; fix by reloading correct row
        # Fix: compute row pointer for t
        offs_kc = t * 512 + tl.arange(0, 512)
        Kc_row = tl.load(Kc_ptr + offs_kc)
        out_vec = out_vec + attn * Kc_row

    # store out_vec
    tl.store(out_vec_ptr + tl.arange(0, 512), out_vec)


# In ModelNew.forward, we will:
# 1) Precompute lse per (b, h) using Triton kernel.
# 2) Then compute output per (b, h) using Triton kernel with lse.
# We will do this without any "run" function to avoid recursion.

class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Move inputs to CUDA if necessary and cast to float32 for compute
        device = q_nope.device
        B, NH, D = q_nope.shape  # NH = 16, D = 512
        DP = kpe_cache.shape[-1]  # 64

        # Ensure CUDA tensors
        if not q_nope.is_cuda:
            q_nope = q_nope.to('cuda')
        if not q_pe.is_cuda:
            q_pe = q_pe.to('cuda')
        if not ckv_cache.is_cuda:
            ckv_cache = ckv_cache.to('cuda')
        if not kpe_cache.is_cuda:
            kpe_cache = kpe_cache.to('cuda')
        if not kv_indptr.is_cuda:
            kv_indptr = kv_indptr.to('cuda')
        if not kv_indices.is_cuda:
            kv_indices = kv_indices.to('cuda')

        # Compute L_tokens per batch, gather indices, and prepare Kc_selected, Kp_selected
        # We'll allocate output and lse tensors
        output = torch.empty((B, NH, D), dtype=torch.float32, device=device)
        lse = torch.full((B, NH), float('-inf'), dtype=torch.float32, device=device)

        # We need L_tokens per batch. PyTorch .item() can be used since devices are CUDA.
        # For each batch b:
        for b in range(B):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens for this batch; output zeros and lse stays -inf (per original)
                lse[b] = torch.full((NH,), float('-inf'), dtype=torch.float32, device=device)
                output[b] = torch.zeros((NH, D), dtype=torch.float32, device=device)
                continue

            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + L_tokens]].to(torch.long).contiguous()
            # Gather selected keys
            Kc_selected = ckv_cache[tok_idx].to(torch.float32).contiguous()  # [L_tokens, D]
            Kp_selected = kpe_cache[tok_idx].to(torch.float32).contiguous()  # [L_tokens, DP]

            # For each head h
            for h in range(NH):
                qnh = q_nope[b, h].to(torch.float32).contiguous()  # [512]
                qph = q_pe[b, h].to(torch.float32).contiguous()    # [64]

                # First, compute lse[h] using Triton
                lse_val = torch.empty((), dtype=torch.float32, device=device)
                _attention_lse_out_kernel[(1,)](
                    qnh, Kc_selected, Kp_selected, output[b, h], lse[b, h],  # out_vec and lse buffers
                    L_TOKENS=L_tokens,
                    sm_scale=sm_scale,
                    compute_lse=1,
                    qph=torch.empty(0, dtype=torch.float32, device=device)  # placeholder, not used when compute_lse=1
                )

                # Then compute output[b, h, :] using Triton with precomputed lse[b, h]
                _attention_output_only_kernel[(1,)](
                    qnh, Kc_selected, Kp_selected, output[b, h], lse[b, h],
                    L_TOKENS=L_tokens,
                    sm_scale=sm_scale,
                    qph=qph
                )

        # Cast output to bfloat16 as in original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse