import math
import torch
import triton
import triton.language as tl


@triton.jit
def run_token_head_kernel(
    q_nope_ptr,      # *f32, [num_tokens, num_qo_heads, head_dim_ckv]
    q_pe_ptr,        # *f32, [num_tokens, num_qo_heads, head_dim_kpe]
    Kc_all_ptr,      # *f32, [num_total_kv, head_dim_ckv]
    Kp_all_ptr,      # *f32, [num_total_kv, head_dim_kpe]
    sparse_i32_ptr,  # *i32, [num_tokens, topk]
    out_ptr,         # *f32, [num_tokens, num_qo_heads, head_dim_ckv] output vector per head
    lse_ptr,         # *f32, [num_tokens, num_qo_heads]
    num_tokens: tl.int32,
    num_qo_heads: tl.constexpr,  # keep constexpr to allow loops
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    topk: tl.constexpr,
    sm_scale: tl.float32,
):
    # program ids
    t = tl.program_id(0)  # token id
    h = tl.program_id(1)  # head id

    # Base offsets
    base_q_n = t * num_qo_heads * head_dim_ckv + h * head_dim_ckv
    base_q_p = t * num_qo_heads * head_dim_kpe + h * head_dim_kpe

    # First pass: compute max and sum of exp(scaled - m) across all k
    m = -float('inf')
    sum_exp = 0.0

    for k in range(0, topk):
        idx = tl.load(sparse_i32_ptr + t * topk + k)
        valid = idx != -1

        if valid:
            # Compute contrib1 = q_nope[t, h, :] @ Kc_all[idx, :]
            contrib1 = 0.0
            # dot over head_dim_ckv
            for d in range(0, head_dim_ckv):
                q_elem = tl.load(q_nope_ptr + base_q_n + d)
                kc_elem = tl.load(Kc_all_ptr + idx * head_dim_ckv + d)
                contrib1 += q_elem * kc_elem

            # Compute contrib2 = q_pe[t, h, :] @ Kp_all[idx, :]
            contrib2 = 0.0
            for e in range(0, head_dim_kpe):
                qp_elem = tl.load(q_pe_ptr + base_q_p + e)
                kp_elem = tl.load(Kp_all_ptr + idx * head_dim_kpe + e)
                contrib2 += qp_elem * kp_elem

            scaled = (contrib1 + contrib2) * sm_scale
            # Update max and sum_exp
            m_new = tl.maximum(m, scaled)
            # sum_exp = sum_exp * exp(m - m_new) + exp(scaled - m_new)
            sum_exp = sum_exp * tl.exp(m - m_new) + tl.exp(scaled - m_new)
            m = m_new
        else:
            # invalid: contribute -inf to max, zero to sum after shifting
            m_new = tl.maximum(m, -float('inf'))
            # sum_exp remains unchanged because new term is 0 after shifting
            # but to keep consistent, compute:
            # we set m = m (no change), sum_exp += 0 via below:
            # We can skip, but keep logic consistent
            pass

    # Compute lse = m + log(sum_exp) / ln(2)
    ln2 = 1.4426950408889634  # math.log(2.0)
    lse = m + tl.log(sum_exp) / ln2
    tl.store(lse_ptr + t * num_qo_heads + h, lse)

    # Second pass: accumulate output[t, h, :] = sum_k p_k * Kc_all[idx, :]
    # where p_k = exp((scaled_k - m) * sm_scale - lse) if valid, else 0.
    for k in range(0, topk):
        idx = tl.load(sparse_i32_ptr + t * topk + k)
        valid = idx != -1

        if valid:
            contrib1 = 0.0
            for d in range(0, head_dim_ckv):
                q_elem = tl.load(q_nope_ptr + base_q_n + d)
                kc_elem = tl.load(Kc_all_ptr + idx * head_dim_ckv + d)
                contrib1 += q_elem * kc_elem

            contrib2 = 0.0
            for e in range(0, head_dim_kpe):
                qp_elem = tl.load(q_pe_ptr + base_q_p + e)
                kp_elem = tl.load(Kp_all_ptr + idx * head_dim_kpe + e)
                contrib2 += qp_elem * kp_elem

            scaled = (contrib1 + contrib2) * sm_scale
            p_k = tl.exp(scaled - lse)  # attention weight for this k

            # accumulate output vector over head_dim_ckv
            base_out = t * num_qo_heads * head_dim_ckv + h * head_dim_ckv
            acc = tl.zeros([head_dim_ckv], dtype=tl.float32)
            # chunk over d
            for d0 in range(0, head_dim_ckv, 128):
                offs = d0 + tl.arange(0, 128)
                mask = offs < head_dim_ckv
                # Load Kc_all[idx, offs]
                kc_vec = tl.load(Kc_all_ptr + idx * head_dim_ckv + offs, mask=mask, other=0.0)
                # For each scalar d in chunk, add p_k * kc_vec[d]
                for s in range(0, 128):
                    dd = d0 + s
                    if dd < head_dim_ckv:
                        acc[dd] += p_k * kc_vec[s]

            # store acc to out
            # out[t, h, :] already initialized to zeros by host
            # add acc to out[t, h, :]
            # Implement by loading and adding chunked
            for d0 in range(0, head_dim_ckv, 128):
                offs = d0 + tl.arange(0, 128)
                mask = offs < head_dim_ckv
                cur = tl.load(out_ptr + base_out + offs, mask=mask, other=0.0)
                cur += acc[offs]  # vectorized add
                tl.store(out_ptr + base_out + offs, cur, mask=mask)

        # else: invalid index, no contribution


class ModelNew(torch.nn.Module):
    def __init__(self, topk=2048, head_dim_ckv=512, head_dim_kpe=64, num_qo_heads=16, sm_scale=1.0):
        super().__init__()
        self.topk = topk
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.num_qo_heads = num_qo_heads
        self.sm_scale = sm_scale

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale=None):
        # Ensure dtype and contiguity; cast to float32 for computation
        q_nope = q_nope.contiguous().to(torch.float32)
        q_pe = q_pe.contiguous().to(torch.float32)
        # Flatten and cast caches to float32
        num_pages, _, _ = ckv_cache.shape
        num_tokens = q_nope.shape[0]
        head_dim_ckv = self.head_dim_ckv
        head_dim_kpe = self.head_dim_kpe
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).contiguous().to(torch.float32)  # [num_pages * 64, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).contiguous().to(torch.float32)  # [num_pages * 64, 64]

        # sparse_indices: int32, keep as is
        sparse_i32 = sparse_indices.to(torch.int32)

        # Allocate outputs
        output = torch.zeros((num_tokens, self.num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device)
        lse = torch.full((num_tokens, self.num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per (token, head)
        grid = (num_tokens, self.num_qo_heads)
        run_token_head_kernel[grid](
            q_nope, q_pe, Kc_all, Kp_all, sparse_i32, output, lse,
            num_tokens=num_tokens,
            num_qo_heads=self.num_qo_heads,
            head_dim_ckv=self.head_dim_ckv,
            head_dim_kpe=self.head_dim_kpe,
            topk=self.topk,
            sm_scale=(self.sm_scale if sm_scale is None else float(sm_scale)),
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 to match original Model's output dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
