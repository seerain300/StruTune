import torch
import math
import triton
import triton.language as tl

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure contiguity and compute in float32
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        total_q = q_f32.shape[0]
        total_kv = k_f32.shape[0]
        assert q_f32.shape[1:] == (self.num_qo_heads, self.head_dim)
        assert k_f32.shape[1:] == (self.num_kv_heads, self.head_dim)
        assert v_f32.shape[1:] == (self.num_kv_heads, self.head_dim)
        assert qo_indptr.shape[0] == kv_indptr.shape[0] >= 2
        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch segment
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Expand K/V by GQA ratio (4)
            gqa_ratio = self.num_qo_heads // self.num_kv_heads
            k_expanded = k_f32[kv_start:kv_end].repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
            v_expanded = v_f32[kv_start:kv_end].repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

            # Compute logits = Q @ K^T using einsum (PyTorch). Shapes: q_batch [num_q_tokens, 32, 128], k_exp [num_kv_tokens, 32, 128]
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]
            logits_batch = torch.einsum('qhd,khd->qhk', q_batch, k_expanded)  # [num_q_tokens, 32, num_kv_tokens]

            # Compute lse = logsumexp(logits) / ln(2) per (q, h)
            lse_batch = torch.logsumexp(logits_batch, dim=-1)  # [num_q_tokens, 32]
            lse_batch = lse_batch / math.log(2.0)
            lse[q_start:q_end] = lse_batch

            # Triton softmax + output
            output_seg = _softmax_output_with_lse(logits_batch, v_expanded, lse[q_start:q_end], num_q_tokens, num_kv_tokens, self.head_dim)
            output[q_start:q_end] = output_seg.to(torch.bfloat16)

        return output, lse

@triton.jit
def _softmax_output_with_lse_kernel(
    LOGITS, V, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_stride_k, V_stride_h, V_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens / BLOCK_Q), heads). Set BLOCK_Q=1.
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offset = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # scalar offset; with BLOCK_Q=1, this is 0

    # Load lse for this (q,h): shape [1], since BLOCK_Q=1
    LSE_ptr = LSE + q_offset * LSE.stride(0) + h * LSE.stride(1)
    lse_val = tl.load(LSE_ptr, mask=(q_offset < num_q_tokens), other=0.0)  # scalar

    # Compute output[q,h,d] across D tiles
    for d0 in range(0, 128, BLOCK_D):  # constexpr loop
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offset * OUT_stride_q + h * OUT_stride_h + d_idx * OUT_stride_d  # [BLOCK_D]
        out_vals = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # Accumulate over K tiles
        for k0 in range(0, 128, BLOCK_K):  # constexpr loop
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            k_valid = k_idx < num_kv_tokens

            # Causal mask: allowed keys j < (i + 1 + delta), delta = num_kv_tokens - num_q_tokens
            delta = num_kv_tokens - num_q_tokens
            allowed = k_idx[None, :] < (q_offset[:, None] + 1 + delta)  # [1, BLOCK_K]

            LOGITS_ptrs = LOGITS + q_offset * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k  # [1, BLOCK_K]
            vals = tl.load(LOGITS_ptrs, mask=(q_offset < num_q_tokens)[:, None] & k_valid[None, :] & allowed, other=-float("inf"))  # [1, BLOCK_K]

            # Subtract lse for numerical stability
            vals = vals - lse_val  # [1, BLOCK_K]

            exp_vals = tl.exp(vals)              # [1, BLOCK_K]
            sum_exp = tl.sum(exp_vals, axis=1)   # [1]
            probs = exp_vals / sum_exp[:, None]  # [1, BLOCK_K]

            V_ptrs = V + k_idx[None, :] * V_stride_k + h * V_stride_h + d_idx[None, :] * V_stride_d  # [BLOCK_K, BLOCK_D]
            v_tile = tl.load(V_ptrs, mask=(k_valid[None, :] & d_valid[None, :]), other=0.0)  # [BLOCK_K, BLOCK_D]

            # out[d] += sum_k probs[k] * v_tile[k, d]
            contrib = tl.sum(probs * v_tile, axis=0)  # [BLOCK_D]
            out_vals += contrib

        tl.store(OUT_ptrs, out_vals, mask=d_valid)

# Helper to launch Triton kernel computing softmax + output given logits, v_expanded, lse
def _softmax_output_with_lse(logits: torch.Tensor, v_exp: torch.Tensor, lse: torch.Tensor, num_q_tokens: int, num_kv_tokens: int, head_dim: int) -> torch.Tensor:
    # logits: [num_q_tokens, 32, num_kv_tokens], v_exp: [num_kv_tokens, 32, head_dim], lse: [num_q_tokens, 32]
    output = torch.empty((num_q_tokens, 32, head_dim), dtype=torch.float32, device=logits.device)
    grid = (triton.cdiv(num_q_tokens, 1), 32)
    _softmax_output_with_lse_kernel[grid](
        logits, v_exp, lse, output,
        num_q_tokens, num_kv_tokens, head_dim,
        logits.stride(0), logits.stride(1), logits.stride(2),
        v_exp.stride(0), v_exp.stride(1), v_exp.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_Q=1, BLOCK_K=32, BLOCK_D=32
    )
    return output


def run(*args):
    return ModelNew()(*args)
