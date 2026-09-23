import torch
import triton
import triton.language as tl


@triton.jit
def matvec_kernel(
    K_ptr,              # *bf16 [N, Dc] (but we pass a single-row view per call; we'll handle per-token below)
    attn_ptr,           # *float32 [L_tokens]
    out_ptr,            # *bf16 [Dc]
    Dc: tl.constexpr,   # head_dim_ckv
    L: tl.constexpr,    # number of tokens (L_tokens)
    SM_SCALE: tl.constexpr,  # scaling factor for logits (unused in this kernel but kept for signature symmetry)
):
    """
    This kernel computes out_vec = sum_i attn[i] * K[i, :] for a single head.
    - K_ptr: pointer to keys, treated as [L, Dc] by host code when launching per-token
    - attn_ptr: pointer to attention weights [L]
    - out_ptr: pointer to output vector [Dc]
    """
    # We implement a simple loop over tokens to accumulate the output vector.
    out_vec = tl.zeros((Dc,), dtype=tl.float32)

    for i in range(L):
        # Load attn_i
        attn_i = tl.load(attn_ptr + i).to(tl.float32)
        # Load K_row[i, :] of length Dc
        K_row_ptr = K_ptr + i * Dc
        K_row = tl.load(K_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        out_vec += attn_i * K_row

    # Store result as bfloat16
    tl.store(out_ptr + tl.arange(0, Dc), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-involved forward:
        - Inputs:
          * q_nope: [B, H, Dc], dtype bfloat16
          * q_pe: [B, H, Dp], dtype bfloat16
          * ckv_cache: [N, 1, Dc], dtype bfloat16
          * kpe_cache: [N, 1, Dp], dtype bfloat16
          * kv_indptr: [B+1], int32
          * kv_indices: [L], int32
          * sm_scale: float32 scalar
        - Returns:
          * output: [B, H, Dc], dtype bfloat16
          * lse: [B, H], dtype float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be on CUDA device for Triton kernel."

        B, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        N, _, _ = ckv_cache.shape
        assert kpe_cache.shape[2] == Dp, "kpe_cache last dim must match q_pe head_dim."
        assert kv_indptr.shape[0] == B + 1, "kv_indptr length must be B+1."
        L = kv_indices.numel()

        # Ensure contiguous for pointer arithmetic
        q_nope_c = q_nope.contiguous()
        q_pe_c = q_pe.contiguous()
        ckv_cache_c = ckv_cache.contiguous()
        kpe_cache_c = kpe_cache.contiguous()
        kv_indptr_c = kv_indptr.contiguous()
        kv_indices_c = kv_indices.contiguous()

        # Allocate outputs
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Process one batch element per program
        for b in range(B):
            base = int(kv_indptr_c[b].item())
            end = int(kv_indptr_c[b + 1].item())
            L_tokens = end - base

            if L_tokens <= 0:
                # No tokens for this batch element
                output[b].zero_()
                lse[b] = 0.0
                continue

            # Compute logits per head: logits[h, i] = (q_nope[b, h] @ Kc[i]) + (q_pe[b, h] @ Kp[i]) * sm_scale
            attn = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)

            for h in range(H):
                # Load q vectors for head h
                qn_vec = q_nope_c[b, h, :].to(torch.float32)  # [Dc]
                qp_vec = q_pe_c[b, h, :].to(torch.float32)    # [Dp]

                # Build K matrices for this segment
                # Kc_all: [L_tokens, Dc], Kp_all: [L_tokens, Dp]
                Kc_all = ckv_cache_c[kv_indices_c[base: end], 0, :].to(torch.float32)  # [L_tokens, Dc]
                Kp_all = kpe_cache_c[kv_indices_c[base: end], 0, :].to(torch.float32)  # [L_tokens, Dp]

                # Compute dot products for all tokens and sum
                # Note: This uses PyTorch optimized matmul for correctness and speed
                dot1 = qn_vec @ Kc_all.T                           # [1, L_tokens]
                dot2 = qp_vec @ Kp_all.T                          # [1, L_tokens]
                logits_h = (dot1 + dot2) * sm_scale              # [1, L_tokens]
                attn[h, :] = logits_h[0, :]                       # [L_tokens]

            # Compute lse per head in base-2
            lse_b = torch.logsumexp(attn, dim=1) / math.log(2.0)  # [H]
            lse[b, :] = lse_b

            # Compute attention weights (softmax along token dimension) in base-2
            attn_b = torch.softmax(attn / math.log(2.0), dim=1)   # [H, L_tokens]

            # Triton kernel: out[h, :] = attn_b[h, :] @ Kc_all
            for h in range(H):
                Kc_all_h = ckv_cache_c[kv_indices_c[base: end], 0, :].to(torch.float32)  # [L_tokens, Dc]
                # Prepare pointers: treat Kc_all_h as [L_tokens, Dc] rows
                # We need to call the kernel L_tokens times to accumulate, but that's fine for correctness.
                # Alternatively, build out vector as zeros and loop i.
                out_vec = torch.empty((Dc,), dtype=torch.float32, device=q_nope.device)
                # Launch Triton kernel once per head to accumulate out_vec
                grid = (1,)
                matvec_kernel[grid](
                    Kc_all_h.reshape(-1),          # pass as 1D [L_tokens*Dc]
                    attn_b[h, :].contiguous(),     # [L_tokens] float32
                    out_vec,                        # [Dc] float32
                    Dc=Dc, L=L_tokens, SM_SCALE=sm_scale,
                    num_warps=4, num_stages=2
                )
                # Store output as bfloat16
                output[b, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
