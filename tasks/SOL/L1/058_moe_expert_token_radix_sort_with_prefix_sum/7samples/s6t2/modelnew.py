import torch
import triton
import triton.language as tl


@triton.jit
def _sort_token_stable_kernel(
    in_ptr,         # *int32, pointer to the token's vector (length = num_experts_per_tok)
    out_ptr,        # *int32, pointer to output permutation for this token (length = num_experts_per_tok)
    num_classes: tl.constexpr,   # typically 256, but not directly used here since we sort within token
    token_len: tl.constexpr      # num_experts_per_tok
):
    # We will sort the token's vector stably by (value, original position).
    # Each program handles one token (row). We implement a simple stable insertion-like method:
    # Maintain out_perm[0..token_len-1] and offsets[0..255] int32 vector on global memory.
    offsets = tl.zeros((256,), dtype=tl.int32)
    # Create out_perm in registers
    out_perm = tl.zeros((token_len,), dtype=tl.int32)  # indices 0..token_len-1

    # For each position i in the token, find its sorted place:
    # value = load in_ptr[i], pos = i. Place i at offsets[value], then offsets[value] += 1.
    # This is stable because we scan i in increasing order, and we append position as tie-breaker.
    for i in range(token_len):
        # Load value and original position; since position within token is i, we can use i as pos.
        # Note: in_ptr points to [0..token_len-1], and we don't have dynamic indexing easily here,
        # so we assume token_len is known at compile time and we pass the vector directly.
        # We simulate reading via indirect indexing using i and token_len structure by reloading
        # using the pointer + i. Triton supports vector loads; but here we rely on the host passing
        # the flattened vector per token.
        # To read value for position i: load scalar from in_ptr[i]
        v = tl.load(in_ptr + i)
        # Stable tie-breaker: append original position within token (i). Since we process i in increasing order,
        # placing i at offset[v] preserves stability for equal values.
        # Find current offset for class v. Since v is in [0, 255], we can just use offsets[v].
        idx_offset = offsets[v]
        # Store i at out_perm[idx_offset]
        tl.store(out_ptr + idx_offset, i)
        # Advance offset
        offsets[v] = idx_offset + 1

    # The out_perm vector is now the sorted permutation for this token.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original run function:
        - Sorts per token: for each (batch, seq), sort the vector of expert indices stably by value, tie-breaking by original position.
        - Returns:
          * sorted_token_indices: permutation of all tokens (flattened) of shape (N_total,), int32
          * expert_offsets: prefix sums of counts per expert, shape (num_experts + 1,), int32
        """
        # Ensure device is CUDA; Triton only runs on GPU. If CPU, fallback to device behavior.
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton execution."

        # Compute total number of tokens
        N_total = topk_idx.numel()
        # Prepare output permutation for all tokens
        out_idx = torch.empty(N_total, dtype=torch.int32, device=topk_idx.device)

        # We will process one token (i.e., one row across [seq_len, num_experts_per_tok]) at a time.
        # Shape: (batch, seq_len, num_experts_per_tok)
        B, S, N_per_tok = topk_idx.shape

        # For each batch and sequence position, sort the token vector and write into out_idx.
        # We need to map the linear index i in out_idx to its (b, s) row. However, since we write contiguous
        # for each row, we can compute base = row_start and fill consecutive indices. So we need to fill
        # out_idx for each row: base = b*S*N_per_tok + s*N_per_tok.
        # We'll run a Triton program per (b, s) to sort that vector, and then copy its permutation into out_idx.

        # Launch grid: one program per (b, s) row. Triton supports grid as a tuple; use (B*S,).
        # Inside kernel, we need token_len (num_experts_per_tok). Triton requires compile-time constants.
        # We pass token_len as a constexpr meta-parameter; Triton will JIT a specialized kernel per token_len.
        # But Triton kernel arguments don't accept Python loop over token_len at call; instead, we call once per (b,s).
        # So we implement a loop over rows in Python and call the kernel for each (b,s).
        for b in range(B):
            for s in range(S):
                # Load the token vector: topk_idx[b, s, :] as a contiguous 1D tensor of length N_per_tok
                token_vec = topk_idx[b, s, :].contiguous()
                # Output permutation for this token (length N_per_tok)
                out_perm = torch.empty(N_per_tok, dtype=torch.int32, device=topk_idx.device)
                # Compute base in out_idx where this token's sorted indices should go
                row_start = (b * S + s) * N_per_tok
                # Launch kernel: one program per (b,s) token. We pass in_ptr = token_vec,
                # out_ptr = out_perm, token_len = N_per_tok.
                # Note: Triton expects pointers; we pass the tensor directly. The kernel will read token_vec[i] for i in 0..N_per_tok-1.
                _sort_token_stable_kernel[(1,)](
                    token_vec, out_perm,
                    num_classes=256,  # not used in this specific kernel since we sort within token; kept for signature completeness
                    token_len=N_per_tok
                )
                # Write this token's permutation into out_idx at the corresponding row
                # out_idx[row_start + i] = out_perm[i] for i in 0..N_per_tok-1
                # To fill contiguous range, copy out_perm to out_idx[row_start:row_start+N_per_tok]
                out_idx[row_start:row_start + N_per_tok] = out_perm

        # 2) Compute expert offsets from original topk_idx (must match original behavior).
        # We count the number of tokens per expert across the entire dataset (all batches, seq, tokens).
        # Use torch.bincount on the flattened original tensor. Although we sorted the values, counts do not depend on order.
        all_vals = topk_idx.reshape(-1)
        counts256 = torch.bincount(all_vals.to(torch.int32), minlength=256)

        # expert_offsets: exclusive prefix sum of counts
        expert_offsets = torch.zeros(257, dtype=torch.int32, device=topk_idx.device)
        # Start from 1 to avoid the extra leading zero
        if counts256.numel() > 0:
            expert_offsets[1:] = torch.cumsum(counts256, dim=0)

        # Return sorted_token_indices (permutation of all tokens) and expert_offsets
        # sorted_token_indices has shape (N_total,), expert_offsets has shape (257,)
        return out_idx, expert_offsets[1:]  # exclude the leading zero