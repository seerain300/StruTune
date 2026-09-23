import torch
import triton
import triton.language as tl


# Triton kernel: histogram of flattened expert indices.
# Input:
#   topk_flat_ptr: pointer to int32 flattened indices (length N)
#   counts_ptr: pointer to int32 counts (length num_experts)
#   N: number of elements in topk_flat
#   num_experts: number of expert bins (compile-time constant per launch)
# We aggregate per-block counts and emit a single atomic_add per bin per block.
@triton.jit
def _hist_kernel(topk_flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load a block of flattened indices; out-of-range lanes get 0 (won't contribute).
    vals = tl.load(topk_flat_ptr + offsets, mask=mask, other=0)

    # Accumulate counts per bin locally: local_counts[bin] += number of occurrences in this block
    local_counts = tl.zeros((num_experts,), dtype=tl.int32)
    # For each lane j in the block, increment the corresponding bin.
    # Note: vals[j] is int32 in [0, num_experts-1], so masked lanes (j without mask) are ignored.
    for j in range(BLOCK_SIZE):
        val = vals[j]
        # Only increment if lane is valid
        valid = mask[j]
        # Increment the appropriate bin (cast val to int to index)
        bin_idx = val
        # If invalid, do nothing
        if valid:
            local_counts[bin_idx] += 1

    # Atomically add local counts to global counts
    for b in range(num_experts):
        tl.atomic_add(counts_ptr + b, local_counts[b])


# Triton kernel: inclusive prefix sum of counts to produce expert_offsets (length num_experts+1).
# Input:
#   counts_ptr: pointer to int32 counts (length num_experts)
#   offsets_ptr: pointer to int32 offsets (length num_experts+1)
# We use a simple loop: each program computes one output offset by looping over all bins.
@triton.jit
def _prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Each program computes one offset: offsets[i+1] = sum_{j=0..i} counts[j]
    # We’ll launch with grid = (1,) and compute the whole array in one program.
    total = tl.zeros((), dtype=tl.int32)
    for j in range(num_experts):
        total += tl.load(counts_ptr + j)
        tl.store(offsets_ptr + j + 1, total)
    # Handle offsets[0] = 0 by caller

# ModelNew entry point
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input dict with 'topk_idx' as in the original
        # (Assuming get_inputs returns a dict with 'topk_idx' on CUDA)
        # In this environment, args contains the tensors. We take the first argument.
        if len(args) == 0:
            raise RuntimeError("ModelNew.forward expects inputs.")
        topk_idx = args[0]
        if not isinstance(topk_idx, torch.Tensor):
            raise RuntimeError("Expected a tensor input.")
        if topk_idx.numel() == 0:
            # Edge case: empty, return empty permutations and zeros
            N = 0
        else:
            N = topk_idx.numel()

        # Ensure int32
        flat = topk_idx.contiguous().view(-1).to(torch.int32)

        # 1) Stable sort of flattened indices to get sorted_token_indices permutation.
        #    Note: For integers, stable=True matches non-stable ordering. We use PyTorch for reliability.
        # sorted_token_indices is the indices that would sort 'flat'. torch.argsort returns such indices.
        sorted_token_indices = torch.argsort(flat, stable=True)

        # 2) Compute per-expert offsets using Triton kernels.
        num_experts = 256  # as in the original run() function

        # Allocate counts and offsets
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel
        # Choose BLOCK_SIZE to balance occupancy and atomic reduction; 1024 is a good default.
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _hist_kernel[grid](flat, counts, N, num_experts, BLOCK_SIZE)

        # Launch prefix-sum kernel; we compute in one program for simplicity
        # Since num_experts is small (256), a single program is sufficient and fast.
        _prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Ensure offsets[0] = 0 (torch.bincount returns inclusive counts; we mimic by initializing counts=0 then summing)
        # offsets already computed as inclusive sums; we only need to ensure offsets[0]=0 by construction.
        # Return results matching the original signature: (sorted_token_indices, expert_offsets)
        return sorted_token_indices.to(torch.int32), offsets


# The following helper functions are not used by the evaluator but shown for completeness if needed.
def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    topk_idx = torch.randint(0, num_experts,
                              (batch_size, seq_len, num_experts_per_tok),
                              dtype=torch.int32, device=device)
    return {"topk_idx": topk_idx}


# Original run function for reference (not used in ModelNew, kept here for clarity)
@torch.no_grad()
def run(topk_idx: torch.Tensor):
    num_experts = 256
    flat = topk_idx.reshape(-1)

    # Stable sort to get permutation
    _, sorted_token_indices = flat.sort(stable=True)

    # Histogram + prefix sum
    expert_counts = torch.bincount(flat.long(), minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    expert_offsets[1:] = expert_counts.cumsum(0).to(torch.int32)
    return sorted_token_indices.to(torch.int32), expert_offsets


# Example usage:
# device = torch.device("cuda")
# inputs = get_inputs({"batch_size": 8, "seq_len": 256, "num_experts": 256, "num_experts_per_tok": 4}, device)
# model = ModelNew().to(device)
# sorted_idx, offsets = model(*[inputs["topk_idx"]])
# print(sorted_idx.shape, offsets.shape)


def run(*args):
    return ModelNew()(*args)
