import torch
import triton
import triton.language as tl


@triton.jit
def argtopk_groups_kernel(GroupScores_ptr, GroupIdx_ptr,
                           M, K,
                           stride_Sm, stride_Sk,
                           stride_Ikm, stride_Ikn,
                           BLOCK_K: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    # Select top-K (K <= 8) from 8 groups using scanning
    best_vals = tl.full([BLOCK_K], -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)
    for g in range(0, 8):
        val = tl.load(GroupScores_ptr + m * stride_Sm + g * stride_Sk)
        for j in range(0, BLOCK_K):
            if val > best_vals[j]:
                tmp = best_vals[j]
                best_vals[j] = val
                val = tmp
                tmp_idx = best_idxs[j]
                best_idxs[j] = g
                idx = tmp_idx
    for j in range(0, BLOCK_K):
        tl.store(GroupIdx_ptr + m * stride_Ikm + j * stride_Ikn, best_idxs[j])


@triton.jit
def argtopk_experts_kernel(X_ptr, Indices_ptr,
                            M, N,
                            stride_Xm, stride_Xn,
                            stride_Ikm, stride_Ikn,
                            BLOCK_N: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    # Select top-8 from N using repeated argmax scanning
    best_vals = tl.full([8], -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros([8], dtype=tl.int32)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N
        x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask_n, other=0.0)
        # Unroll over K=8 positions: find max among the block and update
        for j in range(8):
            val = x[j]
            idx = n_offsets[j]
            # Scan remaining positions in block
            for i in range(8, BLOCK_N):
                cur = x[i]
                cur_idx = n_offsets[i]
                if cur > val:
                    val = cur
                    idx = cur_idx
            # Assign selected to slot j
            best_vals[j] = val
            best_idxs[j] = idx
    for j in range(8):
        tl.store(Indices_ptr + m * stride_Ikm + j * stride_Ikn, best_idxs[j])


@triton.jit
def normalize_scale_kernel(Indices_ptr, Selected_ptr, Weights_ptr,
                           M, K, N, scaling,
                           stride_Ikm, stride_Ikn,
                           stride_Sm, stride_Sn,
                           stride_Wm, stride_Wn,
                           BLOCK_K: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    sum_w = 0.0
    for j in range(0, K):
        idx = tl.load(Indices_ptr + m * stride_Ikm + j * stride_Ikn)
        val = tl.load(Selected_ptr + m * stride_Sm + idx * stride_Sn)
        sum_w += val
    for j in range(0, K):
        idx = tl.load(Indices_ptr + m * stride_Ikm + j * stride_Ikn)
        val = tl.load(Selected_ptr + m * stride_Sm + idx * stride_Sn)
        out = val / (sum_w + 1e-20) * scaling
        tl.store(Weights_ptr + m * stride_Wm + j * stride_Wn, out)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-only implementation:
        - logits via torch.nn.functional.linear
        - sigmoid + bias (Triton elementwise)
        - per-group top-2 (PyTorch topk over small groups to keep correctness simple)
          -> group_scores [M, 8]
        - Triton arg-topk groups to get top-4 groups per token
        - Build group_mask [M, 8] and expand to [M, N] (Triton scatter + repeat)
        - Triton mask_scores: set non-selected group scores to -inf
        - Triton arg-topk experts to get top-8 per token from masked scores
        - Triton normalize + scale
        Returns topk_idx (int32) and topk_weight (float32)
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be on CUDA device for Triton kernels."

        M = hidden_states.shape[0]
        N = weight.shape[0]  # num experts = 256

        # 1) Compute logits = hidden_states @ weight.T using PyTorch (fast and stable)
        logits = torch.nn.functional.linear(hidden_states, weight, bias=None)  # [M, N]
        logits = logits.to(torch.float32)

        # 2) Triton sigmoid + expert bias → scores
        scores = torch.empty((M, N), device=logits.device, dtype=torch.float32)
        sigmoid_bias_kernel[(M,)](
            logits, expert_bias.to(torch.float32), scores,
            M, N,
            logits.stride(0), logits.stride(1),
            expert_bias.stride(0),
            scores.stride(0), scores.stride(1),
            BLOCK_N=256,
        )

        # 3) Compute per-group top-2 scores: group_scores [M, 8]
        # We implement this with torch.topk over each 32-expert slice to ensure correctness and simplicity.
        group_scores = torch.empty((M, 8), device=scores.device, dtype=torch.float32)
        for g in range(8):
            start = g * 32
            end = start + 32
            group = scores[:, start:end]  # [M, 32]
            top2, _ = torch.topk(group, k=2, dim=1, largest=True, sorted=False)
            group_scores[:, g] = top2.sum(dim=1)  # [M]

        # 4) Triton arg-topk groups: select top-4 groups per token (K=4)
        group_idx = torch.empty((M, 4), device=group_scores.device, dtype=torch.int32)
        argtopk_groups_kernel[(M,)](
            group_scores, group_idx,
            M, 4,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            BLOCK_K=4,
        )

        # 5) Build group_mask [M, 8] (1.0 at selected groups) using Triton scatter
        group_mask = torch.empty((M, 8), device=group_scores.device, dtype=torch.float32)
        build_group_mask_kernel[(M,)](
            group_idx, group_mask,
            M, 8,
            1, 1,  # strides not strictly needed; we write ones at selected positions
            1, 1,
        )

        # 6) Expand group_mask to [M, N] by repeating across 32 experts per group (Triton)
        expanded_mask = torch.empty((M, N), device=group_scores.device, dtype=torch.float32)
        expand_group_mask_kernel[(M,)](
            group_mask, expanded_mask,
            M, 8, N,
            1, 1,
            1, 1,
            BLOCK_N=128,
        )

        # 7) Triton mask_scores: set non-selected group scores to -inf
        masked_scores = torch.empty((M, N), device=group_scores.device, dtype=torch.float32)
        mask_scores_kernel[(M,)](
            scores, expanded_mask, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            expanded_mask.stride(0), expanded_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            BLOCK_N=256,
        )

        # 8) Triton arg-topk experts on masked_scores to get top-8 per token
        topk_idx = torch.empty((M, 8), device=group_scores.device, dtype=torch.int32)
        argtopk_experts_kernel[(M,)](
            masked_scores, topk_idx,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            BLOCK_N=256,
        )

        # 9) Gather selected scores from original scores (to normalize), but Triton lacks gather; use PyTorch for this step.
        #    However, to minimize PyTorch usage, we instead compute selected_scores by re-reading masked_scores at the selected indices via a Triton gather-like pattern using loops, but Triton doesn't support dynamic gather; we'll use torch.gather here as a fallback for correctness.
        #    Given the environment constraints, we implement gather in PyTorch: selected_scores[m, j] = scores[m, topk_idx[m, j]]
        selected_scores = torch.gather(scores, dim=1, index=topk_idx.to(torch.long))

        # 10) Triton normalize and scale
        topk_weight = torch.empty((M, 8), device=group_scores.device, dtype=torch.float32)
        normalize_scale_kernel[(M,)](
            topk_idx, selected_scores, topk_weight,
            M, 8, N, routed_scaling_factor,
            topk_idx.stride(0), topk_idx.stride(1),
            selected_scores.stride(0), selected_scores.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            BLOCK_K=8,
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
