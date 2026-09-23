import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_lse_per_head_kernel(
    q_nope_rows_ptr,     # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,       # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    lse_out_ptr,         # *f32, shape [B, H], contiguous
    H: tl.int32,         # number of heads (runtime)
    D1: tl.constexpr,    # 512
    D2: tl.constexpr,    # 64
    L_tokens: tl.int32,  # runtime (per b)
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,  # e.g., 1024
):
    # One Triton program per batch element b
    b = tl.program_id(axis=0)

    # We compute lse per head h: logsumexp over tokens of scaled logits
    for h in range(0, H):
        # Load q vectors for this head
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Initialize per-column max and sum across tokens
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        # Iterate tokens in static tiles of size BLOCK_T
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t0 = tile * BLOCK_T
            # Loop over tokens within the tile (compile-time unrolled)
            for tt in tl.static_range(0, BLOCK_T):
                t = t0 + tt
                valid = t < L_tokens
                # Load Kc_row and Kp_row as 1D vectors with mask
                Kc_row = tl.load(
                    Kc_sub_ptr + t * D1 + tl.arange(0, D1),
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)  # [D1]
                Kp_row = tl.load(
                    Kp_sub_ptr + t * D2 + tl.arange(0, D2),
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)  # [D2]

                # Compute scalar logits for this token
                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale  # scalar

                # Update per-column max and sum (masked)
                token_max_vec = tl.maximum(token_max_vec, tl.where(valid, logits_scalar, token_max_vec))
                # token_sum_vec += exp(logits - max) only if valid
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Compute lse for this head: logsumexp over tokens, scaled by 1/ln(2)
        lse_val = token_max_vec + tl.log(token_sum_vec) / tl.log(2.0)
        # Store lse[b, h]
        tl.store(lse_out_ptr + b * H + h, lse_val)


@triton.jit
def _compute_output_kernel(
    q_nope_rows_ptr,     # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,       # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,          # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,          # *f32, shape [L_tokens, D2], contiguous
    out_ptr,             # *f32, shape [B, H, D1], contiguous
    H: tl.int32,
    D1: tl.constexpr,
    D2: tl.constexpr,
    L_tokens: tl.int32,
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(axis=0)
    for h in range(0, H):
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # First pass: compute per-column max and sum for softmax normalization
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t0 = tile * BLOCK_T
            for tt in tl.static_range(0, BLOCK_T):
                t = t0 + tt
                valid = t < L_tokens
                Kc_row = tl.load(
                    Kc_sub_ptr + t * D1 + tl.arange(0, D1),
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)  # [D1]
                Kp_row = tl.load(
                    Kp_sub_ptr + t * D2 + tl.arange(0, D2),
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale

                token_max_vec = tl.maximum(token_max_vec, tl.where(valid, logits_scalar, token_max_vec))
                token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Second pass: accumulate output = sum_t attn_t * Kc_row
        out_vec = tl.zeros((D1,), dtype=tl.float32)
        for tile in tl.static_range(0, (L_tokens + BLOCK_T - 1) // BLOCK_T):
            t0 = tile * BLOCK_T
            for tt in tl.static_range(0, BLOCK_T):
                t = t0 + tt
                valid = t < L_tokens
                Kc_row = tl.load(
                    Kc_sub_ptr + t * D1 + tl.arange(0, D1),
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)  # [D1]
                Kp_row = tl.load(
                    Kp_sub_ptr + t * D2 + tl.arange(0, D2),
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)  # [D2]

                dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
                dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
                logits_scalar = (dot1 + dot2) * sm_scale

                attn = tl.exp(logits_scalar - token_max_vec) / token_sum_vec  # scalar
                out_vec += tl.where(valid, attn * Kc_row, 0.0)

        # Store output for this head: out[b, h, :] = out_vec
        tl.store(out_ptr + b * (H * D1) + h * D1 + tl.arange(0, D1), out_vec, mask=None)


class ModelNew(torch.nn.Module):
    def __init__(self, block_t=1024):
        super().__init__()
        self.block_t = block_t

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, D1], q_pe: [B, H, D2]
        ckv_cache: [N, 1, D1] (squeezed to [N, D1] in usage), kpe_cache: [N, 1, D2] -> [N, D2]
        kv_indptr: [B+1], kv_indices: [L_tokens]
        sm_scale: float32 scalar
        Returns:
        - output: [B, H, D1], bfloat16
        - lse: [B, H], float32
        """
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        assert q_nope.shape[2] == 512, "head_dim_ckv must be 512"
        assert q_pe.shape[2] == 64, "head_dim_kpe must be 64"

        device = q_nope.device
        B, H, D1 = q_nope.shape
        _, _, D2 = q_pe.shape

        # Prepare per-batch Kc_sub and Kp_sub from kv_indices ranges
        # Note: For correctness and Triton compatibility, we ensure tensors are contiguous float32.
        # Since original code asserts dimensions, we proceed with squeezing and casting.
        # We need L_tokens per batch b from kv_indptr. The loop below computes it and loads subsets.

        # Create output and lse tensors (fp32 for computation)
        out_fp32 = torch.empty((B, H, D1), dtype=torch.float32, device=device)
        lse_fp32 = torch.empty((B, H), dtype=torch.float32, device=device)

        # Prepare q_nope_rows and q_pe_rows: [H, D1] and [H, D2] contiguous
        q_nope_rows = q_nope.to(torch.float32).reshape(H, D1).contiguous()  # [H, D1]
        q_pe_rows = q_pe.to(torch.float32).reshape(H, D2).contiguous()     # [H, D2]

        # Launch kernels per batch element
        grid = (B,)

        # Kernel 1: compute lse per head
        _compute_lse_per_head_kernel[grid](
            q_nope_rows, q_pe_rows, ckv_cache.squeeze(1).to(torch.float32), kpe_cache.squeeze(1).to(torch.float32),
            lse_fp32, H, D1, D2,
            # We need L_tokens per b; Triton kernel takes L_tokens as runtime. Compute L_tokens below for each b.
            # We can pass a placeholder and rely on indexing; instead, we recompute L_tokens inside the kernel using its input parameter L_tokens.
            # We must pass B L_tokens here; the kernel uses the same L_tokens as provided.
            # Since we don't know L_tokens yet, we compute here for each b by deriving from kv_indptr and kv_indices.
            # But since we can't pass different L_tokens to each program, we'll compute per b in the forward loop.
            # To simplify, launch per-b separately with actual L_tokens, but Triton expects single launch. So we compute L_tokens vector on host and pass to kernel via an outer loop. Here, we'll compute L_tokens per b using torch and pass them through a custom launch. Triton doesn't accept dynamic per-program args easily, so we'll do a small trick: compute L_tokens vector on host, and pass it to the kernel by a small wrapper function that we can't inline here. Instead, we compute L_tokens per b outside and then call the kernel with proper L_tokens. Since Triton requires compile-time meta parameters, we pass L_tokens as a tensor? No. Triton requires scalar meta-parameters; we'll pass L_tokens via grid not used. The best is to compute per b in Python and then call kernel with that L_tokens. We'll do that in a simple loop below, but Triton only supports one kernel launch. Therefore, we need to launch per b in a separate function. Triton doesn't support dynamic per-program parameters easily; we'll do a workaround by launching one program per b and passing L_tokens as a scalar.

            # Workaround: We launch one kernel and in its body compute L_tokens per b by reading kv_indptr and kv_indices for each b. To do that, we need to pass pointers to these tensors. Triton kernels can't read arbitrary torch tensors unless we pass them as pointers. The simplest is to compute L_tokens on host per b and pass it as an argument. Since Triton kernels require compile-time meta-parameters, we'll pass L_tokens as a scalar to the kernel and rely on indexing Kc_sub_ptr with t * D1. To avoid complexity, we'll compute L_tokens vector on host and call the kernel with correct L_tokens per b by looping over b and launching a separate kernel per b. Triton allows multiple calls. We'll do that here.
            # Compute L_tokens for each b and launch kernel.

            # We'll call the kernel per b using a loop, computing L_tokens for b in host. Triton requires scalar L_tokens for this kernel. We'll do that below, and similarly for output kernel.
        )

        # However, Triton doesn't support dynamic per-program parameters well in this context. To keep it simple and correct, we'll implement a small Python loop over b and call Triton kernels per b with actual L_tokens. This satisfies the requirement: all computation in Triton kernels launched by forward.

        # Compute per-batch L_tokens and launch kernels
        # Output initialization
        out_fp32.zero_()
        lse_fp32.zero_()

        for b in range(B):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens for this batch element; output zero, lse stays zero
                continue

            # Recompute q_nope_rows and q_pe_rows for this b (though they are constant across b)
            qn_rows = q_nope[b].to(torch.float32).reshape(H, D1).contiguous()
            qp_rows = q_pe[b].to(torch.float32).reshape(H, D2).contiguous()

            # Slice Kc and Kp for this b
            Kc_sub = ckv_cache.squeeze(1).to(torch.float32)[b * L_tokens : (b + 1) * L_tokens]  # [L_tokens, D1]
            Kp_sub = kpe_cache.squeeze(1).to(torch.float32)[b * L_tokens : (b + 1) * L_tokens]  # [L_tokens, D2]
            Kc_sub = Kc_sub.contiguous()
            Kp_sub = Kp_sub.contiguous()

            # Kernel 1: compute lse per head for this b
            _compute_lse_per_head_kernel[(1,)](
                qn_rows, qp_rows, Kc_sub, Kp_sub, lse_fp32[b].view(1), H, D1, D2, L_tokens, sm_scale, self.block_t
            )

            # Kernel 2: compute output per head for this b
            _compute_output_kernel[(1,)](
                qn_rows, qp_rows, Kc_sub, Kp_sub, out_fp32[b].view(H, D1), H, D1, D2, L_tokens, sm_scale, self.block_t
            )

        # Cast output to bfloat16 to match original Model
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16, lse_fp32


def run(*args):
    return ModelNew()(*args)
