# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r18 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(
    topk_ptr,
    expert_counts_ptr,
    num_tokens,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offsets = tile * BLOCK + lane
    mask = offsets < num_tokens

    expert_ids = tl.load(
        topk_ptr + offsets,
        mask=mask,
        other=0,
        cache_modifier=".ca",
    ).to(tl.int32)

    tl.atomic_add(
        expert_counts_ptr + expert_ids,
        1,
        mask=mask,
    )


@triton.jit
def _expert_offsets_kernel(
    expert_counts_ptr,
    expert_offsets_ptr,
    num_tokens,
):
    expert = tl.arange(0, 256)
    counts = tl.load(expert_counts_ptr + expert)
    offsets = tl.cumsum(counts, axis=0) - counts

    tl.store(expert_offsets_ptr + expert, offsets)
    tl.store(
        expert_offsets_ptr + 256,
        num_tokens,
        mask=expert == 0,
    )


@triton.jit
def _expert_stable_scatter_kernel(
    topk_ptr,
    expert_offsets_ptr,
    sorted_token_indices_ptr,
    num_tokens,
    num_tiles,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    expert_base = tl.load(expert_offsets_ptr + expert)
    running = tl.zeros((), dtype=tl.int32)

    for tile in range(0, num_tiles):
        offsets = tile * BLOCK + lane
        mask = offsets < num_tokens
        expert_ids = tl.load(
            topk_ptr + offsets,
            mask=mask,
            other=256,
            cache_modifier=".ca",
        ).to(tl.int32)

        matches = (expert_ids == expert) & mask
        match_int = matches.to(tl.int32)
        local_rank = tl.cumsum(match_int, axis=0) - match_int

        tl.store(
            sorted_token_indices_ptr + expert_base + running + local_rank,
            offsets.to(tl.int32),
            mask=matches,
        )
        running += tl.sum(match_int, axis=0)


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    num_tokens = flat.numel()

    block = 256
    num_tiles = triton.cdiv(num_tokens, block)

    sorted_token_indices = torch.empty(
        num_tokens,
        dtype=torch.int32,
        device=flat.device,
    )
    expert_offsets = torch.empty(
        257,
        dtype=torch.int32,
        device=flat.device,
    )
    expert_counts = torch.zeros(
        256,
        dtype=torch.int32,
        device=flat.device,
    )

    _histogram_kernel[(num_tiles,)](
        flat,
        expert_counts,
        num_tokens,
        BLOCK=block,
        num_warps=4,
    )

    _expert_offsets_kernel[(1,)](
        expert_counts,
        expert_offsets,
        num_tokens,
        num_warps=8,
    )

    _expert_stable_scatter_kernel[(256,)](
        flat,
        expert_offsets,
        sorted_token_indices,
        num_tokens,
        num_tiles,
        BLOCK=block,
        num_warps=4,
    )

    return sorted_token_indices, expert_offsets