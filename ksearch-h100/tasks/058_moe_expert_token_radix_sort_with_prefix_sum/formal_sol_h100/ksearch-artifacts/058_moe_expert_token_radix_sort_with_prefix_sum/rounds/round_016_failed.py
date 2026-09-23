# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r16 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _tile_histogram_kernel(
    topk_ptr,
    tile_hist_ptr,
    num_tokens,
    num_tiles,
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
        tile_hist_ptr + tile * 256 + expert_ids,
        1,
        mask=mask,
    )


@triton.jit
def _tile_prefix_kernel(
    tile_hist_ptr,
    tile_prefix_ptr,
    expert_offsets_ptr,
    num_tiles,
    num_tokens,
):
    expert = tl.program_id(0)
    running = tl.zeros((), dtype=tl.int32)

    for tile in range(0, num_tiles):
        hist = tl.load(tile_hist_ptr + tile * 256 + expert)
        tl.store(tile_prefix_ptr + tile * 256 + expert, running)
        running += hist

    tl.store(expert_offsets_ptr + expert, running - running)

    for prior_expert in range(0, 256):
        if prior_expert < expert:
            prior_count = tl.load(expert_offsets_ptr + prior_expert + 1)
            running += prior_count

    tl.store(expert_offsets_ptr + expert, running)

    if expert == 255:
        tl.store(expert_offsets_ptr + 256, num_tokens)


@triton.jit
def _tile_stable_scatter_kernel(
    topk_ptr,
    tile_prefix_ptr,
    expert_offsets_ptr,
    sorted_token_indices_ptr,
    num_tokens,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    expert = tl.program_id(1)
    lane = tl.arange(0, BLOCK)

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

    tile_base = tl.load(tile_prefix_ptr + tile * 256 + expert)
    expert_base = tl.load(expert_offsets_ptr + expert)

    tl.store(
        sorted_token_indices_ptr + expert_base + tile_base + local_rank,
        offsets.to(tl.int32),
        mask=matches,
    )


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    num_tokens = flat.numel()
    block = 256
    num_tiles = (num_tokens + block - 1) // block

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
    tile_hist = torch.zeros(
        (num_tiles, 256),
        dtype=torch.int32,
        device=flat.device,
    )
    tile_prefix = torch.empty(
        (num_tiles, 256),
        dtype=torch.int32,
        device=flat.device,
    )

    _tile_histogram_kernel[(num_tiles,)](
        flat,
        tile_hist,
        num_tokens,
        num_tiles,
        BLOCK=block,
        num_warps=4,
    )

    _tile_prefix_kernel[(256,)](
        tile_hist,
        tile_prefix,
        expert_offsets,
        num_tiles,
        num_tokens,
        num_warps=4,
    )

    _tile_stable_scatter_kernel[(num_tiles, 256)](
        flat,
        tile_prefix,
        expert_offsets,
        sorted_token_indices,
        num_tokens,
        BLOCK=block,
        num_warps=4,
    )

    return sorted_token_indices, expert_offsets