import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_token_head_kernel(
    q_nope_ptr, q_pe_ptr, Kc_all_ptr, Kp_all_ptr, sparse_idx_ptr,
    out_ptr, lse_ptr,
    N, H, Dk, Dp, topk,
    sm_scale, inv_log2,
    BLOCK: tl.constexpr,
):
    # 2D grid: (token, head)
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Base offset for this token in sparse_indices
    base = t * topk

    # Load q vectors for this head
    q_nope_flat = tl.load(q_nope_ptr + t * H * Dk + h * Dk + tl.arange(0, Dk))  # [Dk]
    q_pe_flat = tl.load(q_pe_ptr + t * H * Dp + h * Dp + tl.arange(0, Dp))      # [Dp]

    # Online logsumexp state
    m = -float('inf')
    sum_exp = 0.0

    # First pass over candidates in blocks to compute lse
    for j_start in range(0, topk, BLOCK):
        j_offsets = j_start + tl.arange(0, BLOCK)
        mask_j = j_offsets < topk
        idx_vals = tl.load(sparse_idx_ptr + base + j_offsets, mask=mask_j, other=-1)  # int32
        valid_mask = idx_vals != -1  # [BLOCK], boolean

        for jj in range(0, BLOCK):
            j = j_start + jj
            if j >= topk:
                break
            if not valid_mask[jj]:
                continue

            row_kc = idx_vals[jj] * Dk
            row_kp = idx_vals[jj] * Dp

            Kc_row = tl.load(Kc_all_ptr + row_kc + tl.arange(0, Dk))  # [Dk]
            Kp_row = tl.load(Kp_all_ptr + row_kp + tl.arange(0, Dp))  # [Dp]

            # Dot products
            dot1 = tl.sum(q_nope_flat * Kc_row, axis=0)
            dot2 = tl.sum(q_pe_flat * Kp_row, axis=0)
            logit = dot1 + dot2
            y = logit * sm_scale

            # Online logsumexp update
            if y > m:
                sum_exp = sum_exp * tl.exp(m - y) + 1.0
                m = y
            else:
                sum_exp += tl.exp(y - m)

    # lse = m / log(2) = m * inv_log2
    lse_h = m * inv_log2
    tl.store(lse_ptr + t * H + h, lse_h)

    # Second pass: compute attention and accumulate output
    out_row_ptr = out_ptr + t * H * Dk + h * Dk
    for j_start in range(0, topk, BLOCK):
        j_offsets = j_start + tl.arange(0, BLOCK)
        mask_j = j_offsets < topk
        idx_vals = tl.load(sparse_idx_ptr + base + j_offsets, mask=mask_j, other=-1)
        valid_mask = idx_vals != -1

        for jj in range(0, BLOCK):
            j = j_start + jj
            if j >= topk:
                break
            if not valid_mask[jj]:
                continue

            row_kc = idx_vals[jj] * Dk
            row_kp = idx_vals[jj] * Dp

            Kc_row = tl.load(Kc_all_ptr + row_kc + tl.arange(0, Dk))  # [Dk]
            Kp_row = tl.load(Kp_all_ptr + row_kp + tl.arange(0, Dp))  # [Dp]

            # Recompute logits for attention
            q_nope_flat = tl.load(q_nope_ptr + t * H * Dk + h * Dk + tl.arange(0, Dk))
            q_pe_flat = tl.load(q_pe_ptr + t * H * Dp + h * Dp + tl.arange(0, Dp))

            dot1 = tl.sum(q_nope_flat * Kc_row, axis=0)
            dot2 = tl.sum(q_pe_flat * Kp_row, axis=0)
            logit = dot1 + dot2
            y = logit * sm_scale
            attn = tl.exp(y - lse_h)

            # Accumulate output vector
            for kd in range(0, Dk):
                val = tl.load(out_row_ptr + kd)
                val += attn * Kc_row[kd]
                tl.store(out_row_ptr + kd, val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Triton-only forward. We allocate outputs and lse in host and pass pointers to the kernel.
        N = q_nope.shape[0]
        H = q_nope.shape[1]  # 16
        Dk = q_nope.shape[2]  # 512
        Dp = q_pe.shape[2]    # 64
        topk = sparse_indices.shape[1]  # 2048

        inv_log2 = 1.0 / math.log(2.0)

        # Allocate outputs and lse
        # output: [N, H, Dk], bfloat16 (as in original)
        output = torch.zeros((N, H, Dk), dtype=torch.bfloat16, device=q_nope.device)
        # lse: [N, H], float32 (as in original)
        lse = torch.empty((N, H), dtype=torch.float32, device=q_nope.device)

        # Launch 2D grid over tokens and heads
        grid = (N, H)
        attention_token_head_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
            output, lse,
            N, H, Dk, Dp, topk,
            float(sm_scale), float(inv_log2),
            BLOCK=128,  # process indices in blocks
            num_warps=4,
        )

        # Return computed outputs and lse
        return [output, lse]


def run(*args):
    return ModelNew()(*args)
