import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_logits_and_lse_kernel(
    qn_ptr,         # *f32, [num_qo_heads, head_dim_ckv] = [16, 512]
    qp_ptr,         # *f32, [num_qo_heads, head_dim_kpe] = [16, 64]
    Kc_ptr,         # *f32, [num_pages, head_dim_ckv] (we use a slice by tok_idx)
    Kp_ptr,         # *f32, [num_pages, head_dim_kpe] (we use a slice by tok_idx)
    tok_idx_ptr,    # *i32, [L_TOKENS]
    lse_ptr,        # *f32, [num_qo_heads] (per-head lse)
    sm_scale,       # f32 scalar
    D: tl.constexpr,     # head_dim_ckv, e.g., 512
    DP: tl.constexpr,    # head_dim_kpe, e.g., 64
    L_TOKENS: tl.constexpr,  # number of selected tokens for this batch
):
    # Each program handles one head
    h = tl.program_id(0)
    # q vectors for this head
    qn = tl.load(qn_ptr + h * D)          # [D]
    qp = tl.load(qp_ptr + h * DP)         # [DP]

    # Compute logits vector of length L_TOKENS
    # We build Kc_row and Kp_row vectors by gathering from Kc_ptr/Kp_ptr via tok_idx_ptr
    logits = tl.zeros((L_TOKENS,), dtype=tl.float32)
    for t in range(0, L_TOKENS):
        idx = tl.load(tok_idx_ptr + t)  # int32
        Kc_row = tl.load(Kc_ptr + idx * D)   # [D]
        Kp_row = tl.load(Kp_ptr + idx * DP)  # [DP]
        logits[t] = tl.dot(qn, Kc_row) + tl.dot(qp, Kp_row)

    # Scale and compute logsumexp base-2
    logits_scaled = logits * sm_scale
    m = tl.max(logits_scaled, axis=0)
    # subtract max for numerical stability
    exp_logits = tl.exp(logits_scaled - m)
    s = tl.sum(exp_logits, axis=0)
    lse = m + tl.log(s)  # natural log; divide by log(2) outside if needed
    # Store per-head lse
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_attention_output_kernel(
    qn_ptr,         # *f32, [num_qo_heads, head_dim_ckv] = [16, 512]
    qp_ptr,         # *f32, [num_qo_heads, head_dim_kpe] = [16, 64]
    Kc_ptr,         # *f32, [num_pages, head_dim_ckv] (we use a slice by tok_idx)
    Kp_ptr,         # *f32, [num_pages, head_dim_kpe] (we use a slice by tok_idx)
    tok_idx_ptr,    # *i32, [L_TOKENS]
    output_ptr,     # *f32, [num_qo_heads, head_dim_ckv]
    sm_scale,       # f32 scalar
    D: tl.constexpr,     # head_dim_ckv, e.g., 512
    DP: tl.constexpr,    # head_dim_kpe, e.g., 64
    L_TOKENS: tl.constexpr,  # number of selected tokens for this batch
):
    # Each program handles one head
    h = tl.program_id(0)

    qn = tl.load(qn_ptr + h * D)          # [D]
    qp = tl.load(qp_ptr + h * DP)         # [DP]

    # Compute logits vector (we reuse the same computation as above)
    logits = tl.zeros((L_TOKENS,), dtype=tl.float32)
    for t in range(0, L_TOKENS):
        idx = tl.load(tok_idx_ptr + t)  # int32
        Kc_row = tl.load(Kc_ptr + idx * D)   # [D]
        Kp_row = tl.load(Kp_ptr + idx * DP)  # [DP]
        logits[t] = tl.dot(qn, Kc_row) + tl.dot(qp, Kp_row)

    logits_scaled = logits * sm_scale
    m = tl.max(logits_scaled, axis=0)
    exp_logits = tl.exp(logits_scaled - m)
    s = tl.sum(exp_logits, axis=0)
    attn = exp_logits / s  # [L_TOKENS]

    # Compute output = attn @ Kc_selected
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(0, L_TOKENS):
        idx = tl.load(tok_idx_ptr + t)  # int32
        Kc_row = tl.load(Kc_ptr + idx * D)   # [D]
        out_vec += attn[t] * Kc_row

    # Store result for this head
    tl.store(output_ptr + h * D, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, 16, 512], bfloat16
        q_pe:   [B, 16, 64],  bfloat16
        ckv_cache: [P, 1, 512], bfloat16
        kpe_cache: [P, 1, 64],  bfloat16
        kv_indptr: [B+1], int32
        kv_indices: [num_tokens], int32
        sm_scale: float32
        returns:
        output: [B, 16, 512], bfloat16
        lse: [B, 16], float32
        """
        assert q_nope.dim() == 3 and q_nope.shape[1] == 16 and q_nope.shape[2] == 512
        assert q_pe.dim() == 3 and q_pe.shape[1] == 16 and q_pe.shape[2] == 64
        assert ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == 512
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == 64
        assert kv_indptr.shape[0] == q_nope.shape[0] + 1
        assert kv_indices.device.type == 'cuda' and q_nope.device.type == 'cuda' and ckv_cache.device.type == 'cuda' and kpe_cache.device.type == 'cuda'
        device = q_nope.device
        dtype_compute = torch.float32

        B = q_nope.shape[0]
        num_qo_heads = 16
        D = 512
        DP = 64

        # Ensure inputs are contiguous
        q_nope_f32 = q_nope.to(dtype_compute)
        q_pe_f32 = q_pe.to(dtype_compute)
        # Squeeze the singleton dim to [num_pages, dim]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(dtype_compute)
        Kp_all = kpe_cache.squeeze(1).contiguous().to(dtype_compute)

        # Prepare outputs
        output = torch.empty((B, num_qo_heads, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(B):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(page_end - page_beg, 0)
            if L_tokens == 0:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            tok_idx = kv_indices[page_beg:page_end].contiguous().to(torch.int32)
            # Gather selected rows
            Kc_selected = Kc_all[tok_idx]            # [L_tokens, 512]
            Kp_selected = Kp_all[tok_idx]            # [L_tokens, 64]

            # Launch Triton kernel to compute logits and lse (per head)
            grid = (num_qo_heads,)
            _compute_logits_and_lse_kernel[grid](
                q_nope_f32[b], q_pe_f32[b], Kc_selected, Kp_selected, tok_idx,
                lse[b], sm_scale,
                D=D, DP=DP, L_TOKENS=L_tokens
            )

            # Launch Triton kernel to compute attention output (per head)
            _compute_attention_output_kernel[grid](
                q_nope_f32[b], q_pe_f32[b], Kc_selected, Kp_selected, tok_idx,
                output[b], sm_scale,
                D=D, DP=DP, L_TOKENS=L_tokens
            )

        # Cast output to bfloat16 as in original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
