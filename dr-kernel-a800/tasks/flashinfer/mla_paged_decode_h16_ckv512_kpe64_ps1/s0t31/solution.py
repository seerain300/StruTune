import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr,           # *bf16 [B, H, Dc]
    q_pe_ptr,             # *bf16 [B, H, Dp]
    ckv_cache_ptr,        # *bf16 [N, 1, Dc]
    kpe_cache_ptr,        # *bf16 [N, 1, Dp]
    kv_indptr_ptr,        # *int32 [B+1]
    kv_indices_ptr,       # *int32 [L]
    output_ptr,           # *bf16 [B, H, Dc]
    lse_ptr,              # *float32 [B*H]
    B: tl.constexpr,      # batch_size (for grid)
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv
    Dp: tl.constexpr,     # head_dim_kpe
    MAX_TOKENS: tl.constexpr,  # upper bound (e.g., 1024)
    SM_SCALE: tl.constexpr,     # float scaling factor
):
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)          # int32
    end = tl.load(kv_indptr_ptr + b + 1)       # int32
    L_tokens = end - base                       # int32

    # Handle empty segment: lse = -inf
    if L_tokens <= 0:
        lse_val = -float('inf')
        tl.store(lse_ptr + b * H, lse_val)
        return

    # Loop over heads
    for h in range(H):
        # Load q vectors for this batch, head; cast to f32
        qn_bf = tl.load(q_nope_ptr + b * H * Dc + h * Dc)   # [Dc], bf16
        qp_bf = tl.load(q_pe_ptr + b * H * Dp + h * Dp)     # [Dp], bf16
        qn = qn_bf.to(tl.float32)                           # [Dc], f32
        qp = qp_bf.to(tl.float32)                           # [Dp], f32

        # Initialize stable logsumexp
        max_val = -float('inf')    # scalar
        sum_exp = 0.0              # scalar

        # Compute logits for each token and update max + sum_exp
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32
                # Load Kc and Kp rows as bf16 then cast to f32
                Kc_row_bf = tl.load(ckv_cache_ptr + idx * Dc)  # [Dc], bf16
                Kp_row_bf = tl.load(kpe_cache_ptr + idx * Dp)  # [Dp], bf16
                Kc_row = Kc_row_bf.to(tl.float32)               # [Dc], f32
                Kp_row = Kp_row_bf.to(tl.float32)               # [Dp], f32

                # Dot products: qn @ Kc_row and qp @ Kp_row
                dot1 = 0.0
                for d in range(Dc):
                    dot1 += qn[d] * Kc_row[d]
                dot2 = 0.0
                for d in range(Dp):
                    dot2 += qp[d] * Kp_row[d]

                val = (dot1 + dot2) * SM_SCALE
                max_val = tl.maximum(max_val, val)
                sum_exp += tl.exp(val - max_val)

        # Compute logsumexp per head and scale by 1/ln(2)
        lse_val = tl.log(sum_exp) + max_val
        lse_val = lse_val / tl.log(2.0)

        # Store lse for this (b, h)
        tl.store(lse_ptr + b * H + h, lse_val)

        # Compute final output: out[b, h, :] = sum_i attn_i * Kc_rows[i, :]
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)       # int32
                Kc_row_bf = tl.load(ckv_cache_ptr + idx * Dc)  # [Dc], bf16
                Kp_row_bf = tl.load(kpe_cache_ptr + idx * Dp)  # [Dp], bf16
                Kc_row = Kc_row_bf.to(tl.float32)              # [Dc], f32
                Kp_row = Kp_row_bf.to(tl.float32)              # [Dp], f32

                dot1 = 0.0
                for d in range(Dc):
                    dot1 += qn[d] * Kc_row[d]
                dot2 = 0.0
                for d in range(Dp):
                    dot2 += qp[d] * Kp_row[d]

                val = (dot1 + dot2) * SM_SCALE
                attn_i = tl.exp(val - lse_val)

                out_vec += attn_i * Kc_row

        # Store output vector for this head as bfloat16
        out_offset = b * H * Dc + h * Dc
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
        - Inputs:
          * q_nope: [B, H, Dc], bfloat16
          * q_pe: [B, H, Dp], bfloat16
          * ckv_cache: [N, 1, Dc], bfloat16
          * kpe_cache: [N, 1, Dp], bfloat16
          * kv_indptr: [B+1], int32
          * kv_indices: [L], int32
          * sm_scale: float32 scalar
        - Returns:
          * output: [B, H, Dc], bfloat16
          * lse: [B, H], float32
        """
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Tensors must be on CUDA device"

        # Make inputs contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape

        # Allocate outputs
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp,
            MAX_TOKENS=1024,  # upper bound; typically much smaller than this
            SM_SCALE=sm_scale,
            num_warps=4,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
