import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel_single(
    qn_ptr,              # *bf16 [H, Dc]
    qp_ptr,              # *bf16 [H, Dp]
    ckv_cache_ptr,       # *bf16 [N, 1, Dc]
    kpe_cache_ptr,       # *bf16 [N, 1, Dp]
    kv_indptr_ptr,       # *int32 [B+1]
    kv_indices_ptr,      # *int32 [L]
    out_ptr,             # *bf16 [H, Dc] (we will cast f32 -> bf16 on store)
    lse_ptr,             # *float32 [H]
    H: tl.constexpr,     # num_qo_heads
    Dc: tl.constexpr,    # head_dim_ckv
    Dp: tl.constexpr,    # head_dim_kpe
    MAX_TOKENS: tl.constexpr,  # upper bound (e.g., 1024)
    SM_SCALE: tl.constexpr,     # scaling factor for logits
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)          # int32
    end = tl.load(kv_indptr_ptr + b + 1)       # int32
    L_tokens = end - base                       # int32

    # If no tokens for this b, set lse to -inf and return
    if L_tokens <= 0:
        for h in range(H):
            tl.store(lse_ptr + h, -float('inf'))
        return

    # Loop over heads
    for h in range(H):
        # Load q vectors for this batch, head as bf16 then cast to f32
        qn_bf = tl.load(qn_ptr + h * Dc)  # [Dc], bf16
        qp_bf = tl.load(qp_ptr + h * Dp)  # [Dp], bf16
        qn = qn_bf.to(tl.float32)  # [Dc], f32
        qp = qp_bf.to(tl.float32)  # [Dp], f32

        # Initialize running max and sum_exp
        acc = -float('inf')  # scalar float32
        sum_exp = 0.0        # scalar float32

        # Compute scaled logits for each token index (up to MAX_TOKENS) with mask
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32 token index
                Kc_row_bf = tl.load(ckv_cache_ptr + idx * Dc)  # [Dc], bf16
                Kp_row_bf = tl.load(kpe_cache_ptr + idx * Dp)  # [Dp], bf16
                Kc_row = Kc_row_bf.to(tl.float32)               # [Dc], f32
                Kp_row = Kp_row_bf.to(tl.float32)               # [Dp], f32

                dot1 = 0.0
                for j in range(Dc):
                    dot1 += qn[j] * Kc_row[j]
                dot2 = 0.0
                for j in range(Dp):
                    dot2 += qp[j] * Kp_row[j]

                val = (dot1 + dot2) * SM_SCALE
                # Update running max
                acc = tl.maximum(acc, val)
                # Accumulate sum of exp(scaled_logits - max)
                sum_exp += tl.exp(val - acc)

        # Compute lse = log(sum_exp) + max_val, then scale by 1/ln(2)
        lse_val = tl.log(sum_exp) + acc
        lse_val = lse_val / tl.log(2.0)
        tl.store(lse_ptr + h, lse_val)

        # Compute final output vector: out[h, :] = sum_i attn_i * Kc_row_i
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32
                Kc_row_bf = tl.load(ckv_cache_ptr + idx * Dc)  # [Dc], bf16
                Kp_row_bf = tl.load(kpe_cache_ptr + idx * Dp)  # [Dp], bf16
                Kc_row = Kc_row_bf.to(tl.float32)               # [Dc], f32
                Kp_row = Kp_row_bf.to(tl.float32)               # [Dp], f32

                dot1 = 0.0
                for j in range(Dc):
                    dot1 += qn[j] * Kc_row[j]
                dot2 = 0.0
                for j in range(Dp):
                    dot2 += qp[j] * Kp_row[j]

                val = (dot1 + dot2) * SM_SCALE
                attn_i = tl.exp(val - lse_val)

                out_vec += attn_i * Kc_row

        # Store output for head h as bf16
        out_offset = h * Dc
        out_vec_bf = out_vec.to(tl.bfloat16)
        for j in range(Dc):
            tl.store(out_ptr + out_offset + j, out_vec_bf[j])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
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
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Inputs must be on CUDA for Triton."

        B, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        device = q_nope.device

        # Prepare outputs
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        MAX_TOKENS = 1024  # upper bound for tokens in segment

        # Launch one Triton program per batch element
        for b in range(B):
            # Slice q vectors for this batch element
            qn = q_nope[b]   # [H, Dc], bf16
            qp = q_pe[b]     # [H, Dp], bf16

            # Ensure contiguous
            qn = qn.contiguous()
            qp = qp.contiguous()
            ckv_cache = ckv_cache.contiguous()
            kpe_cache = kpe_cache.contiguous()
            kv_indptr = kv_indptr.contiguous()
            kv_indices = kv_indices.contiguous()

            # Output per head [Dc], bf16
            out_h = torch.empty((H, Dc), dtype=torch.bfloat16, device=device)

            # Launch kernel: one program per b
            attention_kernel_single[(1,)](
                qn, qp, ckv_cache, kpe_cache, kv_indptr, kv_indices,
                out_h, lse[b],  # lse[b] is a 1D tensor of length H for this b
                H=H, Dc=Dc, Dp=Dp, MAX_TOKENS=MAX_TOKENS, SM_SCALE=sm_scale,
                num_warps=4, num_stages=2
            )

            # Assign results to output[b]
            output[b] = out_h

        return output, lse


def run(*args):
    return ModelNew()(*args)
