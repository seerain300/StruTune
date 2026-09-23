import torch
import triton
import triton.language as tl


@triton.jit
def counts_per_key_kernel(
    flat_ptr,                    # *const int32, length M
    counts_ptr,                 # *int32, length NUM_EXPERTS
    M: tl.constexpr,            # number of elements in flat
    NUM_EXPERTS: tl.constexpr   # number of expert categories
):
    """
    For each key k in [0..NUM_EXPERTS), count occurrences in flat.
    This kernel performs O(M * NUM_EXPERTS) work but ensures correctness and simple Triton logic.
    """
    for k in tl.static_range(0, NUM_EXPERTS):
        acc = tl.zeros((), dtype=tl.int32)
        for j in tl.static_range(0, M):
            val = tl.load(flat_ptr + j)  # int32
            if val == k:
                acc += 1
        tl.store(counts_ptr + k, acc)


@triton.jit
def inclusive_scan_kernel(
    counts_ptr,                 # *const int32, length NUM_EXPERTS
    prefix_ptr,                 # *int32, length NUM_EXPERTS
    NUM_EXPERTS: tl.constexpr
):
    """
    Compute inclusive prefix sums of counts:
    prefix[e] = sum_{t=0..e} counts[t]
    """
    running = tl.zeros((), dtype=tl.int32)
    for e in tl.static_range(0, NUM_EXPERTS):
        running += tl.load(counts_ptr + e)
        tl.store(prefix_ptr + e, running)


@triton.jit
def finalize_offsets_kernel(
    prefix_ptr,                 # *const int32, length NUM_EXPERTS
    total_ptr,                  # *const int32, length 1
    offsets_ptr,                # *int32, length (NUM_EXPERTS + 1)
    NUM_EXPERTS: tl.constexpr
):
    """
    Write expert_offsets:
    offsets[i] = prefix[i] for i in [0..NUM_EXPERTS-1]
    offsets[NUM_EXPERTS] = total_count + 1
    """
    for i in tl.static_range(0, NUM_EXPERTS):
        tl.store(offsets_ptr + i, tl.load(prefix_ptr + i))
    total = tl.load(total_ptr)
    tl.store(offsets_ptr + NUM_EXPERTS, total + 1)


class ModelDict(torch.nn.Module):
    @torch.no_grad()
    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          sorted_token_indices: permutation of [0..M-1] sorted by flat values (stable=True)
          expert_offsets: length (num_experts + 1), inclusive counts per expert plus +1
        """
        # Flatten values; these are the sorting keys (stable sort by these values).
        flat = topk_idx.reshape(-1).to(torch.int32)
        M = flat.numel()

        # Compute sorted_token_indices using PyTorch (exact semantics, stable=True).
        # Note: torch.sort returns (values, indices); here we need the permutation indices.
        # torch.sort(flat, stable=True) returns sorted values; to get indices, use argsort:
        # However, torch.sort(stable=True) also provides indices; for clarity:
        # Compute indices explicitly:
        sorted_token_indices = torch.argsort(flat, stable=True)  # shape [M], int64 indices

        # Determine num_experts dynamically: number of distinct categories in flat.
        # We can take NUM_EXPERTS as one plus the maximum value present in flat.
        max_val = int(torch.max(flat).item())
        num_experts = max_val + 1 if max_val >= 0 else 1

        # Allocate counts and run Triton kernel to count occurrences per key
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        counts_per_key_kernel[(1,)](flat, counts, M, num_experts)

        # Compute inclusive prefix sums via Triton
        prefix = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        inclusive_scan_kernel[(1,)](counts, prefix, num_experts)

        # Finalize expert_offsets using Triton: offsets[:num_experts] = prefix[:]
        # and offsets[num_experts] = total_count + 1
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        total_count = int(counts.sum().item())
        total_tensor = torch.tensor([total_count], dtype=torch.int32, device=flat.device)

        finalize_offsets_kernel[(1,)](prefix, total_tensor, offsets, num_experts)

        return sorted_token_indices, offsets


# Assign the required entry point ModelNew to the Triton-enabled module
ModelNew = ModelDict


def run(*args):
    return ModelNew()(*args)
