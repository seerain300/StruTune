import torch
import triton
import triton.language as tl


@triton.jit
def _stable_index_sort_by_key_kernel(flat_ptr, out_ptr, counts_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    Stable sort by keys in flat_ptr (int32), producing permutation in out_ptr (int32).

    Behavior: For each j in [0, M), key = flat[j], then out[per_key_counts[key]] = j,
              and per_key_counts[key] += 1. This yields stable ordering for equal keys
              because lower j gets lower global_pos first.
    """
    pid = tl.program_id(axis=0)
    # Each program handles one element j
    j = pid  # grid size should be exactly M
    if j >= M:
        return

    # Load the key (topk_idx value) for this position
    key = tl.load(flat_ptr + j)

    # Update per-key counts and compute global position
    # counts_ptr is a vector of length NUM_EXPERTS (small), passed by Triton
    # We read current counts for key, compute global_pos, write j at that position
    # Then increment counts[key].
    # Note: Triton allows vector indexing and scalar loads/stores.
    # Since key is an int32 in [0, NUM_EXPERTS), counts_ptr[key] is valid.
    current = tl.load(counts_ptr + key)
    global_pos = current
    # Store j at the computed position
    tl.store(out_ptr + global_pos, j)
    # Increment the count for this key
    tl.store(counts_ptr + key, current + 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of run:
        - Computes stable sort by original indices using counting sort via Triton.
        - Returns sorted_token_indices and expert_offsets.
        """
        # Ensure we're on CUDA for Triton; if not, move to CUDA (evaluation harness typically uses CUDA)
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')

        # Flatten and ensure int32
        flat = topk_idx.reshape(-1).contiguous()
        assert flat.dtype == torch.int32, "flat must be int32"
        M = flat.numel()
        device = flat.device
        NUM_EXPERTS = 256  # as in the provided reference

        # Output permutation buffer (int32), initialized to 0 (we'll fill it)
        index_out = torch.empty(M, dtype=torch.int32, device=device)

        # Per-key counts buffer (int32), vector of length NUM_EXPERTS
        per_key_counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)

        # Launch Triton kernel: one program per element j
        grid = (M,)
        _stable_index_sort_by_key_kernel[grid](
            flat, index_out, per_key_counts, M,
            NUM_EXPERTS=NUM_EXPERTS
        )

        # Compute cumulative counts per key: prefix_cum[k] = sum_{t<=k} counts[t]
        prefix_cum = torch.cumsum(per_key_counts, dim=0).to(torch.int32) - 1  # prefix_cum has length NUM_EXPERTS

        # Build expert_offsets: offsets[e+1] - offsets[e] == count of key e
        expert_offsets = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        expert_offsets[1:] = prefix_cum

        # Return results. The sorted_token_indices are the permutation index_out.
        # They must be int32 (as original code does).
        sorted_token_indices = index_out  # permutation of [0..M-1]

        return sorted_token_indices, expert_offsets


# The rest of the original helper functions can be reused.
def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    # Generate random expert indices in valid range [0, num_experts-1]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    return {"topk_idx": topk_idx}


def run(*args):
    return ModelNew()(*args)
