# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r20 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _tile_histogram_kernel(
    topk_ptr,
    tile_data_ptr,
    num_tokens,
    NUM_TILES: tl.constexpr,
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

    counts = tl.histogram(expert_ids, 256)
    valid = tl.minimum(BLOCK, num_tokens - tile * BLOCK)
    counts -= (lane == 0).to(tl.int32) * (BLOCK - valid)

    tl.store(tile_data_ptr + tile * 256 + lane, counts)


@triton.jit
def _tile_prefix_kernel(
    tile_data_ptr,
    expert_offsets_ptr,
    num_tokens,
    NUM_TILES: tl.constexpr,
    TILES_PAD: tl.constexpr,
):
    expert = tl.arange(0, 256)[:, None]
    tile = tl.arange(0, TILES_PAD)[None, :]
    mask = tile < NUM_TILES

    counts = tl.load(
        tile_data_ptr + tile * 256 + expert,
        mask=mask,
        other=0,
    ).to(tl.int32)

    prior_tiles = tl.cumsum(counts, axis=1) - counts
    local_expert_start = tl.cumsum(counts, axis=0) - counts

    tl.store(
        tile_data_ptr + tile * 256 + expert,
        prior_tiles - local_expert_start,
        mask=mask,
    )

    totals = tl.sum(counts, axis=1)
    expert_offsets = tl.cumsum(totals, axis=0) - totals
    tl.store(expert_offsets_ptr + expert, expert_offsets)
    tl.store(expert_offsets_ptr + 256, num_tokens)


@triton.jit
def _tile_stable_scatter_kernel(
    topk_ptr,
    tile_data_ptr,
    expert_offsets_ptr,
    sorted_token_indices_ptr,
    num_tokens,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    tile_base = tile * BLOCK
    offsets = tile_base + lane
    mask = offsets < num_tokens

    expert_ids = tl.load(
        topk_ptr + offsets,
        mask=mask,
        other=256,
        cache_modifier=".ca",
    ).to(tl.int32)

    keys = expert_ids * BLOCK + lane
    sorted_keys = tl.sort(keys, dim=0)

    sorted_experts = sorted_keys // BLOCK
    source_lanes = sorted_keys % BLOCK
    valid = lane < tl.minimum(BLOCK, num_tokens - tile_base)

    expert_base = tl.load(
        expert_offsets_ptr + sorted_experts,
        mask=valid,
        other=0,
    )
    adjustment = tl.load(
        tile_data_ptr + tile * 256 + sorted_experts,
        mask=valid,
        other=0,
    )

    destinations = expert_base + adjustment + lane
    token_indices = tile_base + source_lanes

    tl.store(
        sorted_token_indices_ptr + destinations,
        token_indices.to(tl.int32),
        mask=valid,
    )


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    num_tokens = flat.numel()

    block = 256
    num_tiles = triton.cdiv(num_tokens, block)
    tiles_pad = triton.next_power_of_2(num_tiles)

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
    tile_data = torch.empty(
        (num_tiles, 256),
        dtype=torch.int32,
        device=flat.device,
    )

    _tile_histogram_kernel[(num_tiles,)](
        flat,
        tile_data,
        num_tokens,
        NUM_TILES=num_tiles,
        BLOCK=block,
        num_warps=8,
    )

    _tile_prefix_kernel[(1,)](
        tile_data,
        expert_offsets,
        num_tokens,
        NUM_TILES=num_tiles,
        TILES_PAD=tiles_pad,
        num_warps=8,
    )

    _tile_stable_scatter_kernel[(num_tiles,)](
        flat,
        tile_data,
        expert_offsets,
        sorted_token_indices,
        num_tokens,
        BLOCK=block,
        num_warps=8,
    )

    return sorted_token_indices, expert_offsets