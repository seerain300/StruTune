import torch
import math
import triton
import triton.language as tl


@triton.jit
def _attention_forward_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    tok_idx_ptr,
    output_ptr, lse_ptr,
    batch_size, num_qo_heads,
    head_dim_ckv, head_dim_kpe,
    sm_scale,
    LN_INV: tl.constexpr,  # 1 / ln(2)
    BLOCK_N: tl.constexpr,  # tile size along token dimension
):
    # program ids: one per (batch, head)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # safety: guard if grid is larger than actual sizes
    if b >= batch_size or h >= num_qo_heads:
        return

    # Load q vectors for this batch and head
    # q_nope[b, h, :] and q_pe[b, h, :]
    qn = tl.load(q_nope_ptr + b * head_dim_ckv + h * head_dim_ckv + tl.arange(0, head_dim_ckv), mask=True, other=0.0)
    qp = tl.load(q_pe_ptr + b * head_dim_kpe + h * head_dim_kpe + tl.arange(0, head_dim_kpe), mask=True, other=0.0)
    # Cast to fp32 for compute
    qn = qn.to(tl.float32)
    qp = qp.to(tl.float32)

    # Determine L_tokens for this batch b
    # tok_idx is a 1D vector of indices for this batch
    # We need to read tok_idx[numel-1] to get the count, but Triton doesn't support dynamic vector size introspection here.
    # Instead, we pass L_tokens as a scalar argument. In the Python host code, we compute it before launch.
    # So we need to adjust signature to include L_tokens.
    # We'll add an argument L_tokens for the kernel.
    # (Below, we reintroduce L_tokens after fixing the signature.)

    # For now, we continue and fix the signature by adding L_tokens argument. Triton requires explicit args.
    # We'll also add L_tokens and mask logic per token.

    # We'll re-declare with L_tokens parameter. To keep code below compilable, we'll implement the main logic with L_tokens.
    # But since Triton compilation fails if we change signature mid-code, we write the kernel with explicit L_tokens param below.

    # Note: The following is a corrected and final kernel implementation with L_tokens as a parameter.


@triton.jit
def _attention_forward_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    tok_idx_ptr,
    output_ptr, lse_ptr,
    b_idx,  # we can use program_id(0) directly; b_idx = tl.program_id(0)
    L_tokens,  # number of tokens for this batch
    head_dim_ckv, head_dim_kpe,
    sm_scale,
    LN_INV: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per (batch, head)
    b = tl.program_id(0)
    h = tl.program_id(1)

    if b >= batch_size or h >= num_qo_heads:
        return

    # Load q vectors
    qn = tl.load(q_nope_ptr + b * head_dim_ckv + h * head_dim_ckv + tl.arange(0, head_dim_ckv), mask=True, other=0.0).to(tl.float32)
    qp = tl.load(q_pe_ptr + b * head_dim_kpe + h * head_dim_kpe + tl.arange(0, head_dim_kpe), mask=True, other=0.0).to(tl.float32)

    # Initialize scalars for LSE
    m = -float("inf")
    sum_exp = 0.0

    # We'll compute m and sum_exp across tokens in blocks of BLOCK_N
    # Iterate over token positions
    # Note: Triton supports loops over runtime values, but we tile with BLOCK_N.
    for start in range(0, L_tokens, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < L_tokens

        # Gather tok_idx vector for this block
        tok_ids = tl.load(tok_idx_ptr + offs, mask=mask, other=0)

        # Build Kc_block [BLOCK_N, head_dim_ckv] and Kp_block [BLOCK_N, head_dim_kpe]
        # Kc_all: [num_pages, head_dim_ckv], tok_ids index rows
        Kc_block = tl.load(Kc_all_ptr + tok_ids[:, None] * head_dim_ckv + tl.arange(0, head_dim_ckv), mask=mask[:, None], other=0.0).to(tl.float32)
        Kp_block = tl.load(Kp_all_ptr + tok_ids[:, None] * head_dim_kpe + tl.arange(0, head_dim_kpe), mask=mask[:, None], other=0.0).to(tl.float32)

        # Compute dot products: acc1 = qn @ Kc_block.T -> [BLOCK_N]
        # acc2 = qp @ Kp_block.T -> [BLOCK_N]
        acc1 = tl.sum(qn[None, :] * Kc_block, axis=1)  # [BLOCK_N]
        acc2 = tl.sum(qp[None, :] * Kp_block, axis=1)  # [BLOCK_N]

        combined = acc1 + acc2
        scaled = combined * sm_scale

        # Update running max m
        block_max = tl.max(tl.where(mask, scaled, -float("inf")))
        m = tl.maximum(m, block_max)

        # Update sum_exp in a numerically stable way
        # For masked entries, set to -inf so exp is 0
        exp_scaled = tl.exp(scaled - m)
        exp_scaled = tl.where(mask, exp_scaled, 0.0)
        sum_exp += tl.sum(exp_scaled)

    # Compute base-2 LSE and store
    # lse = log(sum_exp) / ln(2) + m
    lse_val = tl.log(sum_exp) * LN_INV + m
    tl.store(lse_ptr + b * num_qo_heads + h, lse_val)

    # Now compute output vector for this head: out[b, h, :]
    # out = softmax(scaled) @ Kc
    # We reconstruct it by iterating tokens in blocks and accumulating.
    out_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
    for start in range(0, L_tokens, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < L_tokens
        tok_ids = tl.load(tok_idx_ptr + offs, mask=mask, other=0)

        Kc_block = tl.load(Kc_all_ptr + tok_ids[:, None] * head_dim_ckv + tl.arange(0, head_dim_ckv), mask=mask[:, None], other=0.0).to(tl.float32)
        Kp_block = tl.load(Kp_all_ptr + tok_ids[:, None] * head_dim_kpe + tl.arange(0, head_dim_kpe), mask=mask[:, None], other=0.0).to(tl.float32)

        acc1 = tl.sum(qn[None, :] * Kc_block, axis=1)  # [BLOCK_N]
        acc2 = tl.sum(qp[None, :] * Kp_block, axis=1)  # [BLOCK_N]
        combined = acc1 + acc2
        scaled = combined * sm_scale

        attn = tl.exp(scaled - m)  # already masked via sum_exp handling
        attn = tl.where(mask, attn, 0.0)
        attn = attn / sum_exp  # normalize

        # out_vec += attn[:, None] @ Kc_block.T
        out_vec += tl.sum(attn[:, None] * Kc_block, axis=1)

    # Store output as bfloat16
    tl.store(output_ptr + b * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Device and dtype checks
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be CUDA tensors"
        device = q_nope.device

        # Extract shapes
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]
        # Provided code asserts head_dim_ckv == 512, head_dim_kpe == 64, num_qo_heads == 16
        assert head_dim_ckv == 512 and head_dim_kpe == 64 and num_qo_heads == 16, "Fixed head dimensions expected"

        # Prepare Kc_all and Kp_all (squeeze dummy dim)
        # Note: The original code asserts ckv_cache has shape [num_pages, 1, 512] etc. We can assume that.
        # Here, we keep the dummy dimension (size 1) and just read it. In many Triton examples, we remove 1; but original asserts expect 512 and 64.
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        # Prepare output and lse
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Loop over batches: for each batch, compute tok_idx and launch kernel per head
        for b in range(batch_size):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                # No tokens for this batch
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            L_tokens = (page_end - page_beg)
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [L_tokens]

            # Launch Triton kernel: grid = (batch_size, num_qo_heads)
            # Choose BLOCK_N. Since head_dim_ckv=512, using 128 or 256 works fine. We use 128 for token tiling.
            BLOCK_N = 128
            LN_INV = 1.0 / math.log(2.0)

            _attention_forward_kernel[(batch_size, num_qo_heads)](
                q_nope[b].contiguous(),  # pointer to this batch
                q_pe[b].contiguous(),    # pointer to this batch
                Kc_all,                  # [num_pages, 512]
                Kp_all,                  # [num_pages, 64]
                tok_idx,                 # [L_tokens]
                output, lse,             # outputs
                b, L_tokens,             # kernel arguments
                head_dim_ckv, head_dim_kpe,
                sm_scale,
                LN_INV=LN_INV,
                BLOCK_N=BLOCK_N,
            )

        return output, lse