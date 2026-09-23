import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel_single(
    qn_ptr,              # *f32 [H, Dc]
    qp_ptr,              # *f32 [H, Dp]
    ckv_cache_ptr,       # *bf16 [N, 1, Dc] (we will load rows with scalar idx)
    kpe_cache_ptr,       # *bf16 [N, 1, Dp]
    kv_indptr_ptr,       # *int32 [B+1]
    kv_indices_ptr,      # *int32 [L]
    out_ptr,             # *f32 [H, Dc] (will be cast to bfloat16 in host)
    lse_ptr,             # *f32 [H]
    H: tl.constexpr,     # num_qo_heads
    Dc: tl.constexpr,    # head_dim_ckv
    Dp: tl.constexpr,    # head_dim_kpe
    MAX_TOKENS: tl.constexpr,  # upper bound (e.g., 1024)
    SM_SCALE: tl.constexpr,     # float scaling factor
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)          # int32
    end = tl.load(kv_indptr_ptr + b + 1)       # int32
    L_tokens = end - base                       # int32

    # Handle empty segment: lse = -inf
    if L_tokens <= 0:
        for h in range(H):
            tl.store(lse_ptr + h, -float('inf'))
        return

    # Loop over heads
    for h in range(H):
        # Load q vectors for this batch, head; already f32 [Dc] and [Dp]
        qn = tl.load(qn_ptr + h * Dc + tl.arange(0, Dc)).to(tl.float32)  # [Dc]
        qp = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp)).to(tl.float32)  # [Dp]

        # Initialize running max and sum_exp for logsumexp
        acc = -float('inf')  # float32 scalar
        sum_exp = 0.0         # float32 scalar

        # Compute scaled logits for each token index (up to MAX_TOKENS) with mask
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32 token index

                # Load Kc and Kp rows as bf16 then cast to f32
                # We load scalar row elements using idx * D and vector arange for the feature dimension
                Kc_row_bf = tl.load(ckv_cache_ptr + idx * Dc + tl.arange(0, Dc))  # [Dc], bf16
                Kp_row_bf = tl.load(kpe_cache_ptr + idx * Dp + tl.arange(0, Dp))  # [Dp], bf16
                Kc_row = Kc_row_bf.to(tl.float32)
                Kp_row = Kp_row_bf.to(tl.float32)

                # Dot products
                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                val = (dot1 + dot2) * SM_SCALE

                # Update running max
                acc = tl.maximum(acc, val)
                # Accumulate sum of exp with stable scaling
                sum_exp += tl.exp(val - acc)

        # Compute lse = log(sum_exp) + max
        lse_val = tl.log(sum_exp) + acc  # float32

        # Compute final output vector: out[h, :] = sum_i attention_i * Kc_row_i
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)
                Kc_row_bf = tl.load(ckv_cache_ptr + idx * Dc + tl.arange(0, Dc))  # bf16
                Kc_row = Kc_row_bf.to(tl.float32)

                Kp_row_bf = tl.load(kpe_cache_ptr + idx * Dp + tl.arange(0, Dp))  # bf16
                Kp_row = Kp_row_bf.to(tl.float32)

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                val = (dot1 + dot2) * SM_SCALE
                attn_i = tl.exp(val - lse_val)  # attention for token i

                out_vec += attn_i * Kc_row

        # Store output for head h (float32, will be cast to bf16 in host)
        out_offset = h * Dc
        tl.store(out_ptr + out_offset + tl.arange(0, Dc), out_vec)

        # Store lse per head
        tl.store(lse_ptr + h, lse_val)


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
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA device."

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Pre-slice q_nope and q_pe per batch element to 2D [H, D] and cast to float32 for math
        qn_list = []  # list of [H, Dc] float32 tensors
        qp_list = []  # list of [H, Dp] float32 tensors
        for b in range(B):
            qn_list.append(q_nope[b].contiguous().to(torch.float32))  # [H, Dc]
            qp_list.append(q_pe[b].contiguous().to(torch.float32))    # [H, Dp]

        # Ensure ckv/kpe are contiguous
        ckv_cache_c = ckv_cache.contiguous()
        kpe_cache_c = kpe_cache.contiguous()
        kv_indptr_c = kv_indptr.contiguous()
        kv_indices_c = kv_indices.contiguous()

        # Allocate outputs (float32 in kernel, cast to bfloat16 after)
        out_f32 = torch.empty((B, H, Dc), dtype=torch.float32, device=q_nope.device)
        lse_f32 = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel_single[grid](
            qn_list[0], qn_list[1] if B > 1 else qn_list[0],  # pass as many qn tensors as B slots; if B=1, pass same
            qp_list[0], qp_list[1] if B > 1 else qp_list[0],  # same for qp
            ckv_cache_c, kpe_cache_c,
            kv_indptr_c, kv_indices_c,
            out_f32, lse_f32,
            H=H, Dc=Dc, Dp=Dp, MAX_TOKENS=1024, SM_SCALE=sm_scale
        )

        # Cast output to bfloat16 to match original behavior
        out_bf16 = out_f32.to(torch.bfloat16)
        return out_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
