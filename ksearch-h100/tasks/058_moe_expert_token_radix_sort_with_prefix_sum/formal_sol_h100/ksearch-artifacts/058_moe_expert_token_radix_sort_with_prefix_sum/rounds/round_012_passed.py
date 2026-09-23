# solution=GPT-5.6-Sol_058_moe_expert_token_radix_sort_with_prefix_sum_triton_optimized_r5 score=7.0504621728954335 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _sort_experts_kernel(
    topk_idx_ptr,
    sorted_token_indices_ptr,
    expert_offsets_ptr,
    num_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pair = tl.program_id(0)
    expert0 = pair * 2
    expert1 = expert0 + 1

    output_start0 = tl.zeros((), dtype=tl.int32)
    output_start1 = tl.zeros((), dtype=tl.int32)

    for start in tl.range(0, num_tokens, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_tokens
        expert_ids = tl.load(topk_idx_ptr + offsets, mask=mask, other=256)

        output_start0 += tl.sum(
            (expert_ids < expert0).to(tl.int32),
            axis=0,
        )
        output_start1 += tl.sum(
            (expert_ids < expert1).to(tl.int32),
            axis=0,
        )

    tl.store(expert_offsets_ptr + expert0, output_start0)
    tl.store(expert_offsets_ptr + expert1, output_start1)

    if pair == 127:
        tl.store(expert_offsets_ptr + 256, num_tokens)

    written0 = tl.zeros((), dtype=tl.int32)
    written1 = tl.zeros((), dtype=tl.int32)

    for start in tl.range(0, num_tokens, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_tokens
        expert_ids = tl.load(topk_idx_ptr + offsets, mask=mask, other=-1)

        matches0 = mask & (expert_ids == expert0)
        matches1 = mask & (expert_ids == expert1)

        match_int0 = matches0.to(tl.int32)
        match_int1 = matches1.to(tl.int32)

        local_rank0 = tl.cumsum(match_int0, axis=0) - 1
        local_rank1 = tl.cumsum(match_int1, axis=0) - 1

        tl.store(
            sorted_token_indices_ptr + output_start0 + written0 + local_rank0,
            offsets.to(tl.int32),
            mask=matches0,
        )
        tl.store(
            sorted_token_indices_ptr + output_start1 + written1 + local_rank1,
            offsets.to(tl.int32),
            mask=matches1,
        )

        written0 += tl.sum(match_int0, axis=0)
        written1 += tl.sum(match_int1, axis=0)


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    num_experts = 256
    num_tokens = topk_idx.numel()

    sorted_token_indices = torch.empty(
        num_tokens,
        dtype=torch.int32,
        device=topk_idx.device,
    )
    expert_offsets = torch.empty(
        num_experts + 1,
        dtype=torch.int32,
        device=topk_idx.device,
    )

    _sort_experts_kernel[(num_experts // 2,)](
        topk_idx,
        sorted_token_indices,
        expert_offsets,
        num_tokens,
        BLOCK_SIZE=2048,
        num_warps=8,
    )

    return sorted_token_indices, expert_offsets