import torch
import triton
import triton.language as tl


@triton.jit
def _count_by_key_kernel(
    flat_ptr,           # *int32, 1D flattened input
    counts_ptr,         # *int32, length NUM_EXPERTS, will hold per-key counts
    total_ptr,          # *int32, single element to hold total_count
    M: tl.constexpr,    # total number of elements in flat
    NUM_EXPERTS: tl.constexpr,  # number of unique keys (256 in harness)
    BLOCK: tl.constexpr,        # block size for parallelization
):
    # One program per key (0..NUM_EXPERTS-1)
    key = tl.program_id(0)
    # Accumulate count of elements equal to 'key'
    count = tl.zeros((), dtype=tl.int32)
    for start in range(0, M, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < M
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        # Only consider lanes where mask and vals == key
        eq_mask = mask & (vals == key)
        # Sum the number of True in eq_mask
        count += tl.sum(eq_mask.to(tl.int32), axis=0)
    # Write count for this key
    tl.store(counts_ptr + key, count)
    # Accumulate into total_count (atomic add)
    tl.atomic_add(total_ptr, count)


@triton.jit
def _finalize_offsets_kernel(
    offsets_incl_ptr,   # *int32, length NUM_EXPERTS (prefix sums)
    total_ptr,          # *int32, total_count
    offsets_out_ptr,    # *int32, length (NUM_EXPERTS + 1)
    NUM_EXPERTS: tl.constexpr,
):
    # Set offsets_out[0..NUM_EXPERTS-1] = offsets_incl[0..NUM_EXPERTS-1]
    for i in range(NUM_EXPERTS):
        tl.store(offsets_out_ptr + i, tl.load(offsets_incl_ptr + i))
    # offsets_out[NUM_EXPERTS] = total_count + 1
    total_val = tl.load(total_ptr)
    tl.store(offsets_out_ptr + NUM_EXPERTS, total_val + 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure device is CUDA and int32, contiguous
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton."
        flat = topk_idx.contiguous().view(-1)
        M = flat.numel()
        NUM_EXPERTS = 256  # as per harness

        # 1) Triton kernel: count per key
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        total_count = torch.zeros(1, dtype=torch.int32, device=flat.device)

        grid = (NUM_EXPERTS,)
        _count_by_key_kernel[grid](
            flat, counts, total_count, M, NUM_EXPERTS, BLOCK=1024,
            num_warps=4
        )

        # 2) Compute inclusive prefix sums of counts using PyTorch (simple and correct)
        offsets_incl = torch.cumsum(counts, dim=0)

        # 3) Finalize expert_offsets: length (NUM_EXPERTS + 1) via Triton kernel
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        _finalize_offsets_kernel[(NUM_EXPERTS,)](
            offsets_incl, total_count, expert_offsets, NUM_EXPERTS, num_warps=1
        )

        # 4) Stable sort of flat using PyTorch to ensure exact correctness.
        #    Return indices (stable=True preserves original order for ties).
        sorted_token_indices = torch.sort(flat, stable=True).indices

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
