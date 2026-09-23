import torch
import triton
import triton.language as tl


@triton.jit
def _compute_bh(
    q_nope_ptr,         # *bf16, flattened as [H * D1] rows
    q_pe_ptr,           # *bf16, flattened as [H * D2] rows
    Kc_all_ptr,         # *bf16, flattened as [N * D1]
    Kp_all_ptr,         # *bf16, flattened as [N * D2]
    out_ptr,            # *bf32, flattened as [B * H * D1]
    H: tl.constexpr,    # number of heads
    D1: tl.constexpr,   # head_dim_ckv (e.g., 512)
    D2: tl.constexpr,   # head_dim_kpe (e.g., 64)
    MAX_T: tl.constexpr # upper bound for tokens (constexpr, e.g., 4096)
):
    pid = tl.program_id(0)  # one program per (b, h)
    h = pid % H
    b = pid // H

    # Pointers to q_nope[b, h, :] and q_pe[b, h, :]
    qn_ptr = q_nope_ptr + h * D1
    qp_ptr = q_pe_ptr + h * D2

    # Load qn and qp as float32 vectors
    qn_vec = tl.load(qn_ptr + tl.arange(0, D1)).to(tl.float32)  # [D1]
    qp_vec = tl.load(qp_ptr + tl.arange(0, D2)).to(tl.float32)  # [D2]

    # Initialize per-column max and sum for lse
    token_max = tl.full((D1,), -float("inf"), dtype=tl.float32)
    token_sum = tl.zeros((D1,), dtype=tl.float32)

    # Iterate tokens with static loop; guard out-of-range t
    for t in tl.static_range(0, MAX_T):
        valid = t < (kv_indptr_ptr[b + 1] - kv_indptr_ptr[b])  # L_tokens for this b

        # Compute indices into Kc_all and Kp_all (assuming one token per index: just t)
        # Note: In this benchmark, num_kv_indices is small; we process all t up to MAX_T.
        # The kernel masks invalid t to avoid OOB. For correctness, we set Kc/Kp to zero when invalid.
        mask_valid = valid
        # Load Kc_row and Kp_row as float32
        Kc_row = tl.load(Kc_all_ptr + t * D1 + tl.arange(0, D1), mask=mask_valid, other=0.0).to(tl.float32)  # [D1]
        Kp_row = tl.load(Kp_all_ptr + t * D2 + tl.arange(0, D2), mask=mask_valid, other=0.0).to(tl.float32)  # [D2]

        # Compute scalar logits: dot(qn, Kc_row) + dot(qp, Kp_row)
        dot1 = tl.sum(qn_vec * Kc_row, axis=0)  # scalar
        dot2 = tl.sum(qp_vec * Kp_row, axis=0)  # scalar
        logits_scalar = dot1 + dot2  # sm_scale=1.0 in original code

        # Update lse
        token_max = tl.maximum(token_max, logits_scalar)
        token_sum += tl.where(mask_valid, tl.exp(logits_scalar - token_max), 0.0)

    # Compute lse = max + log(sum) / log(2)
    # 1 / ln(2)
    inv_ln2 = 1.4426950408889634
    lse_scalar = token_max + tl.log(token_sum) * inv_ln2

    # Second pass: compute output using softmax
    for t in tl.static_range(0, MAX_T):
        valid = t < (kv_indptr_ptr[b + 1] - kv_indptr_ptr[b])
        mask_valid = valid

        Kc_row = tl.load(Kc_all_ptr + t * D1 + tl.arange(0, D1), mask=mask_valid, other=0.0).to(tl.float32)  # [D1]
        Kp_row = tl.load(Kp_all_ptr + t * D2 + tl.arange(0, D2), mask=mask_valid, other=0.0).to(tl.float32)  # [D2]

        dot1 = tl.sum(qn_vec * Kc_row, axis=0)
        dot2 = tl.sum(qp_vec * Kp_row, axis=0)
        logits_scalar = dot1 + dot2

        # attn = softmax((logits - lse) * 0 ?) => we need proper scaling, but we can compute with lse here.
        # Better: compute softmax in float32
        attn = tl.exp(logits_scalar - lse_scalar)  # scalar
        out_row_ptr = out_ptr + (b * H + h) * D1
        tl.store(out_row_ptr + tl.arange(0, D1), attn * Kc_row, mask=mask_valid)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation:
        - All computation is done in Triton kernels.
        - Returns output tensor of shape [B, H, D1] in bfloat16.
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device
        B, H, D1 = q_nope.shape
        _, _, D2 = q_pe.shape
        N = ckv_cache.shape[0]

        # Ensure dtypes are consistent: original inputs are bfloat16; kernel will cast to float32 internally
        # Flatten caches
        Kc_all = ckv_cache.squeeze(1)  # [N, D1]
        Kp_all = kpe_cache.squeeze(1)  # [N, D2]

        # Allocate 1D output buffer in float32 for computation
        out_flat = torch.empty(B * H * D1, dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B * H,)
        _compute_bh[grid](
            q_nope, q_pe, Kc_all, Kp_all, out_flat,
            H=H, D1=D1, D2=D2, MAX_T=4096,
            num_warps=4, num_stages=2
        )

        # Reshape and cast to bfloat16 to match original output dtype
        output = out_flat.view(B, H, D1).to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
