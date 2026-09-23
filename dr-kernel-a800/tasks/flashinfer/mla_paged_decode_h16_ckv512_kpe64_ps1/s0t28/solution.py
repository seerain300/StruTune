import torch
import triton
import triton.language as tl


@triton.jit
def attention_forward_kernel(
    q_nope_ptr,           # *bf16 [B, H, Dc]
    q_pe_ptr,             # *bf16 [B, H, Dp]
    ckv_cache_ptr,        # *bf16 [N, Dc]
    kpe_cache_ptr,        # *bf16 [N, Dp]
    kv_indptr_ptr,        # *int32 [B+1]
    kv_indices_ptr,       # *int32 [L]
    output_ptr,           # *bf16 [B, H, Dc]
    lse_ptr,              # *float32 [B*H]
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv (512)
    Dp: tl.constexpr,     # head_dim_kpe (64)
    SM_SCALE: tl.constexpr,  # scaling factor
):
    # One program per batch element
    b = tl.program_id(0)

    # Loop over heads
    for h in range(H):
        # Compute number of tokens for this batch element
        base = tl.load(kv_indptr_ptr + b)          # int32
        end = tl.load(kv_indptr_ptr + b + 1)       # int32
        L_tokens = end - base                       # number of tokens for this batch element

        # If no tokens, skip (lse = -inf; output zero)
        if L_tokens <= 0:
            lse_offset = b * H + h
            tl.store(lse_ptr + lse_offset, -float("inf"))
            out_offset = b * H * Dc + h * Dc
            tl.store(output_ptr + out_offset + tl.arange(0, Dc), tl.zeros((Dc,), dtype=tl.float32))
            continue

        # Initialize per-token logits (float32)
        logits = tl.full((L_tokens,), -float("inf"), tl.float32)

        # First pass: compute logits for each token
        for i in range(L_tokens):
            idx = tl.load(kv_indices_ptr + base + i)  # int32
            # Load q vectors for head h
            qn_vec_ptr = q_nope_ptr + b * H * Dc + h * Dc
            qn_vec = tl.load(qn_vec_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)

            qp_vec_ptr = q_pe_ptr + b * H * Dp + h * Dp
            # Note: Dp is 64, but q_pe shape is [B, H, Dp]; we use Dp here as the last dim
            # However, q_pe is [B, H, 64]; we should use the last dimension directly:
            # Recompute that we have q_pe [B, H, Dp] but our code expects Dp=64; adjust to actual usage:
            # In our setup, q_pe is [B, H, 64], so we load the vector directly:
            # We need to ensure q_pe_ptr indexing is correct:
            # The correct pointer arithmetic: q_pe_ptr has stride along H and Dp.
            # Since q_pe shape is [B, H, Dp], for a given b and h, the vector is contiguous along Dp.
            # The offset is b*H*Dp + h*Dp, but we already have H dimension separate; since we loop h, we need:
            # q_pe_ptr[b, h, :] contiguous -> offset = b*H*Dp + h*Dp. Not correct; fix below.

            # Correct q_pe_ptr indexing: q_pe_ptr layout is linearized as [B, H, Dp], so offset is b*H*Dp + h*Dp.
            # But we don't have H*Dp stride; instead, since q_pe is [B, H, Dp], we can compute offset as:
            # For a given b, h, we need to find the linear offset for q_pe[b, h, :].
            # PyTorch tensors are contiguous by default; we can compute offset as:
            # q_pe[b, h, :] is at offset = b*H*Dp + h*Dp, but we need to map to linear index.
            # Simpler: pass q_pe as [B*H, Dp] contiguous. We'll change host to make q_pe contiguous [B*H, Dp].
            # However, original q_pe is [B, H, Dp]; we'll adjust indexing accordingly.

            # Fix: q_pe is [B, H, Dp]; we can index q_pe_ptr + (b * H + h) * Dp to get vector.
            qp_vec_ptr = q_pe_ptr + (b * H + h) * Dp
            qp_vec = tl.load(qp_vec_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

            # Load Kc_row and Kp_row for idx
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)

            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

            # Compute dot products
            dot1 = tl.sum(qn_vec * Kc_row, axis=0)  # scalar
            dot2 = tl.sum(qp_vec * Kp_row, axis=0)  # scalar
            val = (dot1 + dot2) * SM_SCALE
            logits[i] = val

        # Compute stable logsumexp across tokens
        max_val = tl.full((), -float("inf"), tl.float32)
        for i in range(L_tokens):
            max_val = tl.maximum(max_val, logits[i])
        sum_exp = tl.full((), 0.0, tl.float32)
        for i in range(L_tokens):
            sum_exp += tl.exp(logits[i] - max_val)
        lse_val = tl.log(sum_exp) + max_val  # natural log
        # Convert to base-2 logsumexp: divide by ln(2)
        ln2 = 0.6931471805599453
        lse_base2 = lse_val / ln2

        # Second pass: compute attention and output
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(L_tokens):
            attn_i = tl.exp(logits[i] - lse_base2) / ln2  # softmax scaling factor for base-2
            Kc_row_ptr = ckv_cache_ptr + tl.load(kv_indices_ptr + base + i) * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
            out_vec += attn_i * Kc_row

        # Store output for head h
        out_offset = b * H * Dc + h * Dc
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec)

        # Store lse per (b, h)
        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse_base2)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
        - q_nope: [B, H, Dc], bfloat16
        - q_pe: [B, H, Dp], bfloat16 (Dp=64)
        - ckv_cache: [N, Dc], bfloat16 (N=num_pages, Dc=512)
        - kpe_cache: [N, Dp], bfloat16 (Dp=64)
        - kv_indptr: [B+1], int32
        - kv_indices: [L], int32
        - sm_scale: float32 scalar
        Returns:
        - output: [B, H, Dc], bfloat16
        - lse: [B, H], float32
        """
        B, H, Dc = q_nope.shape
        Dp = q_pe.shape[-1]
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be CUDA for Triton."
        assert kv_indptr.is_cuda and kv_indices.is_cuda, "kv_indptr and kv_indices must be CUDA."

        # Ensure inputs are contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()  # [B, H, Dp]
        ckv_cache = ckv_cache.contiguous()  # [N, Dc]
        kpe_cache = kpe_cache.contiguous()  # [N, Dp]
        kv_indptr = kv_indptr.contiguous()  # [B+1]
        kv_indices = kv_indices.contiguous()  # [L]

        # Allocate outputs
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_forward_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
            output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp, SM_SCALE=float(sm_scale),
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
