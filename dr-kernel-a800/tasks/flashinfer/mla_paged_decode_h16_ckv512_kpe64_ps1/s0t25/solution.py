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
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv
    Dp: tl.constexpr,     # head_dim_kpe
    SM_SCALE: tl.constexpr,  # scaling factor (float)
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)        # int32
    end = tl.load(kv_indptr_ptr + b + 1)     # int32
    L_tokens = end - base                     # int32 scalar

    # We'll compute per-head outputs sequentially
    for h in range(H):
        # Scalar accumulators for logsumexp across tokens
        max_val = tl.full((), -float("inf"), tl.float32)
        sum_exp = tl.full((), 0.0, tl.float32)

        # Iterate tokens in this segment
        for i in range(L_tokens):
            idx = tl.load(kv_indices_ptr + base + i)  # int32

            # Load q vectors for head h: qn_vec [Dc], qp_vec [Dp]
            qn_vec_ptr = q_nope_ptr + b * H * Dc + h * Dc
            qn_vec = tl.zeros((Dc,), dtype=tl.float32)
            for j in range(Dc):
                qn_vec[j] = tl.load(qn_vec_ptr + j).to(tl.float32)

            qp_vec_ptr = q_pe_ptr + b * H * Dp + h * Dp
            qp_vec = tl.zeros((Dp,), dtype=tl.float32)
            for j in range(Dp):
                qp_vec[j] = tl.load(qp_vec_ptr + j).to(tl.float32)

            # Load key rows Kc_row [Dc], Kp_row [Dp]
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)

            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

            # Compute val = (qn_vec @ Kc_row + qp_vec @ Kp_row) * SM_SCALE
            dot1 = tl.sum(qn_vec * Kc_row, axis=0)  # scalar
            dot2 = tl.sum(qp_vec * Kp_row, axis=0)  # scalar
            val = (dot1 + dot2) * SM_SCALE         # scalar float32

            # Update logsumexp accumulators
            # Note: if val is -inf, tl.maximum(val, max_val) stays -inf
            new_max = tl.maximum(max_val, val)
            # sum_exp = sum_exp * exp(max_val - new_max) + exp(val - new_max)
            sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.exp(val - new_max)
            max_val = new_max

        # Compute lse per head: log(sum_exp) + max_val, scaled by 1/ln(2)
        lse_h = tl.log(sum_exp) + max_val
        lse_h = lse_h / tl.log(2.0)

        # Accumulate output vector: out[b, h, :] += exp(val - lse_h) * Kc_row for each token i
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(L_tokens):
            idx = tl.load(kv_indices_ptr + base + i)
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)

            # Recompute val for this token
            qn_vec_ptr = q_nope_ptr + b * H * Dc + h * Dc
            qn_vec = tl.zeros((Dc,), dtype=tl.float32)
            for j in range(Dc):
                qn_vec[j] = tl.load(qn_vec_ptr + j).to(tl.float32)

            qp_vec_ptr = q_pe_ptr + b * H * Dp + h * Dp
            qp_vec = tl.zeros((Dp,), dtype=tl.float32)
            for j in range(Dp):
                qp_vec[j] = tl.load(qp_vec_ptr + j).to(tl.float32)

            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

            dot1 = tl.sum(qn_vec * Kc_row, axis=0)  # scalar
            dot2 = tl.sum(qp_vec * Kp_row, axis=0)  # scalar
            val = (dot1 + dot2) * SM_SCALE         # scalar

            attn_i = tl.exp(val - lse_h)           # scalar
            out_vec += attn_i * Kc_row

        # Store output and lse for head h
        out_offset = b * H * Dc + h * Dc
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec)

        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse_h)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
        - q_nope: [B, H, Dc], bfloat16
        - q_pe: [B, H, Dp], bfloat16
        - ckv_cache: [N, 1, Dc], bfloat16
        - kpe_cache: [N, 1, Dp], bfloat16
        - kv_indptr: [B+1], int32
        - kv_indices: [L], int32
        - sm_scale: float32 scalar
        Returns:
          output: [B, H, Dc], bfloat16
          lse: [B, H], float32
        """
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        device = q_nope.device

        # Ensure tensors are contiguous and on the right device
        q_nope = q_nope.to(torch.bfloat16).contiguous()
        q_pe = q_pe.to(torch.bfloat16).contiguous()
        ckv_cache = ckv_cache.to(torch.bfloat16).contiguous()
        kpe_cache = kpe_cache.to(torch.bfloat16).contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp,
            SM_SCALE=float(sm_scale),
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
