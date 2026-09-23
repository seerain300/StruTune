import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel_single(
    qn_ptr,              # *bf16 [H, Dc] (per-batch pre-sliced)
    qp_ptr,              # *bf16 [H, Dp] (per-batch pre-sliced)
    ckv_cache_ptr,       # *bf16 [N, 1, Dc]
    kpe_cache_ptr,       # *bf16 [N, 1, Dp]
    kv_indptr_ptr,       # *int32 [B+1]
    kv_indices_ptr,      # *int32 [L]
    output_ptr,          # *f32 [H, Dc] (float32 for compute)
    lse_ptr,             # *f32 [H]
    H: tl.constexpr,     # num_qo_heads
    Dc: tl.constexpr,    # head_dim_ckv
    Dp: tl.constexpr,    # head_dim_kpe
    MAX_TOKENS: tl.constexpr,  # upper bound for tokens in segment (>= L_tokens)
    SM_SCALE: tl.constexpr,     # scaling factor (e.g., 1.0)
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)          # int32
    end = tl.load(kv_indptr_ptr + b + 1)       # int32
    L_tokens = end - base                       # int32

    # Handle empty segment: lse = -inf, output zero vector
    if L_tokens <= 0:
        for h in range(H):
            tl.store(lse_ptr + h, -float('inf'))
        return

    # Loop over heads
    for h in range(H):
        # Load q vectors for this batch, head; already bf16, cast to f32 for compute
        qn = tl.load(qn_ptr + h * Dc).to(tl.float32)  # [Dc], f32
        qp = tl.load(qp_ptr + h * Dp).to(tl.float32)  # [Dp], f32

        # Initialize running max and sum of exp for stable logsumexp
        acc = -float('inf')  # float32 scalar
        sum_exp = 0.0        # float32 scalar

        # Compute per-token logits, update running max and sum_exp
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32 token index
                # Load Kc and Kp rows as bf16 then cast to f32
                Kc_row_bf = tl.load(ckv_cache_ptr + idx * Dc + tl.arange(0, Dc))  # [Dc], bf16
                Kp_row_bf = tl.load(kpe_cache_ptr + idx * Dp + tl.arange(0, Dp))  # [Dp], bf16
                Kc_row = Kc_row_bf.to(tl.float32)  # [Dc], f32
                Kp_row = Kp_row_bf.to(tl.float32)  # [Dp], f32

                # Dot products: qn @ Kc_row and qp @ Kp_row
                dot1 = 0.0
                for d in range(Dc):
                    dot1 += qn[d] * Kc_row[d]
                dot2 = 0.0
                for d in range(Dp):
                    dot2 += qp[d] * Kp_row[d]

                val = (dot1 + dot2) * SM_SCALE
                acc = tl.maximum(acc, val)
                sum_exp += tl.exp(val - acc)

        # Compute logsumexp scaled by 1/ln(2)
        lse_val = tl.log(sum_exp) + acc
        lse_val = lse_val / tl.log(2.0)

        # Compute final output vector: out[h, :] = sum_i attn_i * Kc_row_i
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32
                Kc_row_bf = tl.load(ckv_cache_ptr + idx * Dc + tl.arange(0, Dc))
                Kc_row = Kc_row_bf.to(tl.float32)
                Kp_row_bf = tl.load(kpe_cache_ptr + idx * Dp + tl.arange(0, Dp))
                Kp_row = Kp_row_bf.to(tl.float32)

                dot1 = 0.0
                for d in range(Dc):
                    dot1 += qn[d] * Kc_row[d]
                dot2 = 0.0
                for d in range(Dp):
                    dot2 += qp[d] * Kp_row[d]

                val = (dot1 + dot2) * SM_SCALE
                attn_i = tl.exp(val - lse_val)

                for d in range(Dc):
                    out_vec[d] += attn_i * Kc_row[d]

        # Store output for head h
        out_offset = h * Dc
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec)

        # Store lse per head
        tl.store(lse_ptr + h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
        - Inputs:
          * q_nope: [B, H, Dc], dtype bfloat16 (B=1, H=16, Dc=512)
          * q_pe: [B, H, Dp], dtype bfloat16 (Dp=64)
          * ckv_cache: [N, 1, Dc], dtype bfloat16
          * kpe_cache: [N, 1, Dp], dtype bfloat16
          * kv_indptr: [B+1], int32
          * kv_indices: [L], int32
          * sm_scale: float32 scalar
        - Returns:
          * output: [B, H, Dc], dtype bfloat16
          * lse: [B, H], dtype float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Pre-slice q_nope and q_pe for each batch element into [H, D] and cast to bf16 for kernel input
        # We pass them as bf16 to kernel and cast to f32 inside the kernel.
        qn_list = []
        qp_list = []
        for b in range(B):
            qn_list.append(q_nope[b].to(torch.bfloat16).contiguous())
            qp_list.append(q_pe[b].to(torch.bfloat16).contiguous())
        qn_tensors = qn_list  # list of [H, Dc] bf16
        qp_tensors = qp_list  # list of [H, Dp] bf16

        # Allocate outputs in float32 for compute, then cast to bf16 at the end
        output_f32 = torch.empty((B, H, Dc), dtype=torch.float32, device=device)
        lse_out = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel_single[grid](
            qn_tensors[0],          # qn_ptr: per-batch pre-sliced [H, Dc] bf16
            qp_tensors[0],          # qp_ptr: per-batch pre-sliced [H, Dp] bf16
            ckv_cache.contiguous(), # *bf16 [N, 1, Dc]
            kpe_cache.contiguous(), # *bf16 [N, 1, Dp]
            kv_indptr.contiguous(), # *int32 [B+1]
            kv_indices.contiguous(), # *int32 [L]
            output_f32,             # *f32 [H, Dc] (we'll index by b,h in host to write b,h)
            lse_out,                # *f32 [H]
            H=H,                    # pass H as keyword (avoid positional duplicate)
            Dc=Dc,
            Dp=Dp,
            MAX_TOKENS=1024,        # upper bound; works for typical L_tokens
            SM_SCALE=sm_scale,      # scaling factor
        )

        # Cast output to bfloat16 as expected
        output_bf16 = output_f32.to(torch.bfloat16)
        # lse_out shape [B, H], dtype float32 as expected
        return output_bf16, lse_out


def run(*args):
    return ModelNew()(*args)
