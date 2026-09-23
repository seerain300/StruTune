import math
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
    output_ptr,           # *float32 [B, H, Dc]
    lse_ptr,              # *float32 [B*H]
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv
    Dp: tl.constexpr,     # head_dim_kpe
    SM_SCALE: tl.float32,  # scaling factor for logits
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)         # int32
    end = tl.load(kv_indptr_ptr + b + 1)      # int32
    L_tokens = end - base                      # number of tokens for this batch element

    # Prepare accumulators
    eps = 1e-20

    for h in range(H):
        # Compute q vectors for this head
        qn_offset = b * H * Dc + h * Dc
        qn = tl.load(q_nope_ptr + qn_offset + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)

        qp_offset = b * H * Dp + h * Dp
        qp = tl.load(q_pe_ptr + qp_offset + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        # Compute logsumexp of scaled logits over tokens
        max_val = -float("inf")
        sum_exp = 0.0

        for i in range(0, L_tokens):
            idx = tl.load(kv_indices_ptr + base + i)  # int32

            # Load key rows as float32
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)

            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

            # Dot products
            dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
            dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
            val = (dot1 + dot2) * SM_SCALE

            # Stable update of max and sum_exp
            if max_val == -float("inf") or val > max_val:
                # new max; sum_exp = exp(val - max_val)
                sum_exp = tl.exp(val - max_val)
                max_val = val
            else:
                # existing max; add exp(val - max_val)
                sum_exp = sum_exp + tl.exp(val - max_val)

        # lse = log(sum_exp) + max_val, then divide by ln(2)
        lse_val = tl.log(sum_exp) + max_val
        lse_val = lse_val / tl.log(2.0)

        # Compute final output vector: out[b, h, :] = sum_i attn_i * Kc_all[i, :]
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(0, L_tokens):
            idx = tl.load(kv_indices_ptr + base + i)

            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)

            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

            dot1 = tl.sum(qn * Kc_row, axis=0)
            dot2 = tl.sum(qp * Kp_row, axis=0)
            val = (dot1 + dot2) * SM_SCALE
            attn_i = tl.exp(val - lse_val)  # softmax

            out_vec += attn_i * Kc_row

        # Store output for head h
        out_offset = b * H * Dc + h * Dc
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec)

        # Store lse per (b, h)
        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse_val)


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
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Ensure contiguous memory
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B, H, Dc = q_nope.shape
        Dp = q_pe.shape[-1]
        N = ckv_cache.shape[0]
        device = q_nope.device

        # Output tensors in float32 for kernel; cast to bfloat16 after
        output_fp32 = torch.empty((B, H, Dc), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
            output_fp32, lse,
            B=B, H=H, Dc=Dc, Dp=Dp, SM_SCALE=float(sm_scale),
        )

        # Cast output to bfloat16 to match original function
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse


# Helper functions to match the original harness
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
