import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_bh_kernel(
    # Input pointers
    q_nope_ptr,      # *float32, shape [H*D1] flattened
    q_pe_ptr,        # *float32, shape [H*D2] flattened
    ckv_cache_ptr,   # *float32, shape [N*D1]
    kpe_cache_ptr,   # *float32, shape [N*D2]
    kv_indptr_ptr,   # *int32, shape [B+1]
    kv_indices_ptr,  # *int32, shape [num_kv_indices]
    output_ptr,      # *float32, shape [B*H*D1] flattened
    lse_ptr,         # *float32, shape [B*H] flattened
    # Scalars / constexpr
    B: tl.constexpr,
    H: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    L_tokens: tl.constexpr,
    SM_SCALE: tl.constexpr,
    MAX_T: tl.constexpr,
    b_idx: tl.constexpr,
):
    # Compute base pointers for q_nope and q_pe for this batch element
    # We don't have b_idx directly as constexpr, but Triton supports scalar args; we pass it via grid
    # Compute base positions for q_nope and q_pe rows
    # We will iterate over heads h and tokens t inside the kernel
    for h in tl.static_range(0, H):
        # Initialize output vector for this (b, h)
        out_row_ptr = output_ptr + b_idx * H * D1 + h * D1  # start index for this (b, h) row
        # Initialize per-column max and sum for lse (float32)
        token_max = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum = tl.zeros((D1,), dtype=tl.float32)

        # Precompute base pointers
        qn_ptr = q_nope_ptr + h * D1
        qp_ptr = q_pe_ptr + h * D2

        # Loop over tokens in a static unrolled loop up to MAX_T
        for t in tl.static_range(0, MAX_T):
            valid = t < L_tokens
            # Compute idx for ckv/kpe using kv_indptr and kv_indices
            # Note: Triton supports integer math
            page_beg = tl.load(kv_indptr_ptr + b_idx)  # scalar int32
            idx = tl.load(kv_indices_ptr + (b_idx * L_tokens + t), mask=valid, other=0)  # scalar int32
            idx = idx  # if invalid, idx is 0, but mask guards usage

            # Load qn and qp vectors (float32)
            qn = tl.load(qn_ptr + tl.arange(0, D1)).to(tl.float32)  # [D1]
            qp = tl.load(qp_ptr + tl.arange(0, D2)).to(tl.float32)  # [D2]

            # Load Kc_row and Kp_row using idx (float32)
            Kc_row = tl.load(ckv_cache_ptr + idx * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
            Kp_row = tl.load(kpe_cache_ptr + idx * D2 + tl.arange(0, D2)).to(tl.float32)  # [D2]

            # Compute scalar logits: sum(qn * Kc_row) + sum(qp * Kp_row)
            dot1 = tl.zeros((), dtype=tl.float32)
            for j in tl.static_range(0, D1):
                dot1 += qn[j] * Kc_row[j]
            dot2 = tl.zeros((), dtype=tl.float32)
            for j2 in tl.static_range(0, D2):
                dot2 += qp[j2] * Kp_row[j2]
            logits = (dot1 + dot2) * SM_SCALE

            # Update output: output[b, h, :] += logits * Kc_row
            for j in tl.static_range(0, D1):
                tl.store(out_row_ptr + j, tl.load(out_row_ptr + j) + logits * Kc_row[j])

            # Accumulate for lse (per-column max and sum of exp(scaled))
            scaled = logits / math.log(2.0)  # Triton uses float; math.log(2.0) is a float
            token_max = tl.maximum(token_max, scaled)
            # Sum exp(scaled - max)
            token_sum += tl.exp(scaled - token_max)

        # Compute lse[b, h] = log(sum) + max
        # Write to lse_ptr[b*H + h]
        lse_offset = b_idx * H + h
        tl.store(lse_ptr + lse_offset, tl.log(token_sum) + token_max)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; Triton kernels perform computation

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure all tensors are on the same CUDA device and dtype handled
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, \
            "All tensors must be on CUDA device"

        # Shapes
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]
        N = ckv_cache.shape[0]

        # Allocate outputs without specifying device; Triton will write to them
        output = torch.empty((B, H, D1), dtype=torch.float32, device=device)  # compute in float32
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Prepare inputs: ensure contiguous and flattened for kernel
        q_nope_f = q_nope.contiguous().to(torch.float32).view(H * D1)
        q_pe_f = q_pe.contiguous().to(torch.float32).view(H * D2)
        ckv_cache_f = ckv_cache.contiguous().to(torch.float32)  # [N, D1]
        kpe_cache_f = kpe_cache.contiguous().to(torch.float32)  # [N, D2]
        kv_indptr_f = kv_indptr.to(torch.int32).contiguous()
        kv_indices_f = kv_indices.to(torch.int32).contiguous()

        # Ensure dtype for output pointer; kernel writes float32
        # Launch Triton kernel: one program per batch element
        grid = (B,)
        _compute_bh_kernel[grid](
            q_nope_f, q_pe_f, ckv_cache_f, kpe_cache_f, kv_indptr_f, kv_indices_f,
            output, lse,
            B=B, H=H, D1=D1, D2=D2,
            L_tokens=0,  # placeholder; kernel computes via kv_indptr and indices
            SM_SCALE=float(sm_scale),
            MAX_T=2048,  # static loop bound; masked by L_tokens
            b_idx=0,     # Triton handles grid; we rely on program_id(0) for b
        )

        # Note: The kernel above is simplified to operate with b_idx passed; however,
        # Triton requires grid mapping. To correctly map, we should iterate over b manually.
        # Implement correct mapping by launching per b: we redefine launch to use lambda.

        # Correct approach: define a grid function that maps program_id(0) to b
        # But Triton doesn't allow lambda in forward; re-define the kernel to take b_idx as constexpr
        # Alternatively, we can launch per b using a Python loop, which is fine for B=1 or modest B.
        # Here, for generality, we will use a grid function via re-declaring kernel call per b.

        # Since Triton doesn't support dynamic grid lambda, we loop over B and call the kernel for each b.
        # We'll redefine the kernel call as per-b to ensure correct usage:
        for b in range(B):
            # Recompute L_tokens for this b
            Lb = int(kv_indptr_f[b + 1].item()) - int(kv_indptr_f[b].item())
            # Launch kernel for this b
            _compute_bh_kernel[(1,)](
                q_nope_f, q_pe_f, ckv_cache_f, kpe_cache_f, kv_indptr_f, kv_indices_f,
                output, lse,
                B=B, H=H, D1=D1, D2=D2,
                L_tokens=Lb,
                SM_SCALE=float(sm_scale),
                MAX_T=2048,
                b_idx=b,
            )

        # Return output and lse; cast output to bfloat16 to match original expectations
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
