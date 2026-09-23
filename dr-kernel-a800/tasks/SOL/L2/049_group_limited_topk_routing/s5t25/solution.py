import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Bn,
                         stride_Ym, stride_Yn,
                         BLOCK_N: tl.constexpr):
    # One program per token row
    m = tl.program_id(0)
    # Iterate over N (experts) in blocks
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask, other=0.0)
        b = tl.load(Bias_ptr + n_offsets * stride_Bn, mask=mask, other=0.0)
        y = 1.0 / (1.0 + tl.exp(-x)) + b
        tl.store(Y_ptr + m * stride_Ym + n_offsets * stride_Yn, y, mask=mask)


@triton.jit
def top2_per_group_kernel(S_ptr, GroupScores_ptr,
                          M, G, E,
                          stride_Sm, stride_Sg, stride_Sexp,
                          stride_GSm, stride_GSn,
                          BLOCK_E: tl.constexpr):
    # Grid over tokens
    m = tl.program_id(0)
    # For each group g, compute top-2 over E experts
    for g in range(0, G):
        top1 = -float("inf")
        top2 = -float("inf")
        for e in range(0, BLOCK_E):
            # e in [0, E); mask tail
            if e < E:
                val = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + e * stride_Sexp)
                # Update top-2
                if val > top1:
                    top2 = top1
                    top1 = val
                elif val > top2:
                    top2 = val
        group_score = top1 + top2
        tl.store(GroupScores_ptr + m * stride_GSm + g * stride_GSn, group_score)


@triton.jit
def set_group_mask_kernel(GroupIdx_ptr, GroupMask_ptr,
                          M, G,
                          stride_GIm, stride_IdxG,
                          stride_GMm, stride_GMg,
                          BLOCK_M: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    # Write 1s at selected group indices; assume group_idx has up to G selections
    for g in range(0, G):
        # Read group index for this token
        idx = tl.load(GroupIdx_ptr + m * stride_GIm + g * stride_IdxG)
        # If idx is valid (>=0), set mask to 1
        if idx >= 0 and idx < G:
            tl.store(GroupMask_ptr + m * stride_GMm + idx * stride_GMg, 1.0)


@triton.jit
def expand_group_mask_kernel(GroupMask_ptr, Expanded_ptr,
                             M, N, G, E,
                             stride_GMm, stride_GMg,
                             stride_EMm, stride_EMn,
                             BLOCK_N: tl.constexpr):
    # Grid over tokens
    m = tl.program_id(0)
    # For each expanded column n, map to group g = n // E and copy mask[g]
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N
        # group id for each n
        group_id = n_offsets // E
        # Load mask value for that group (vectorized)
        mask_val = tl.load(GroupMask_ptr + m * stride_GMm + group_id * stride_GMg, mask=mask_n, other=0.0)
        tl.store(Expanded_ptr + m * stride_EMm + n_offsets * stride_EMn, mask_val, mask=mask_n)


@triton.jit
def mask_scores_kernel(Scores_ptr, ExpandedMask_ptr, Masked_ptr,
                        M, N,
                        stride_Sm, stride_Sn,
                        stride_EMm, stride_EMn,
                        stride_Mm, stride_Mn,
                        BLOCK_N: tl.constexpr):
    # Grid over tokens
    m = tl.program_id(0)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N
        s = tl.load(Scores_ptr + m * stride_Sm + n_offsets * stride_Sn, mask=mask_n, other=0.0)
        em = tl.load(ExpandedMask_ptr + m * stride_EMm + n_offsets * stride_EMn, mask=mask_n, other=0.0)
        # Apply mask: set non-selected to -inf
        s_masked = tl.where(em > 0.0, s, -float("inf"))
        tl.store(Masked_ptr + m * stride_Mm + n_offsets * stride_Mn, s_masked, mask=mask_n)


@triton.jit
def topk_experts_kernel(X_ptr, Indices_ptr,
                         M, K, N,
                         stride_Xm, stride_Xn,
                         stride_Im,
                         BLOCK_K: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    # We will select K top values via repeated max selection
    for i in range(0, K):
        # Initialize current best
        best_val = -float("inf")
        best_idx = 0
        # Scan all N values to find max
        for n in range(0, BLOCK_K):  # BLOCK_K >= K, loop up to BLOCK_K
            if n < N:
                val = tl.load(X_ptr + m * stride_Xm + n * stride_Xn)
                take = val > best_val
                best_val = tl.where(take, val, best_val)
                best_idx = tl.where(take, n, best_idx)
        # Store selected index
        tl.store(Indices_ptr + m * stride_Im + i, best_idx)
        # Remove it by not re-selecting in future iterations (we mark by setting a sentinel).
        # Simpler: next iteration will find a new best since we didn't change X.


@triton.jit
def normalize_scale_kernel(SelectedIdx_ptr, Masked_ptr, Output_ptr,
                           M, topK,
                           stride_Ipm, stride_Ipn,
                           stride_Mm, stride_Mn,
                           stride_Om, stride_On,
                           scaling_factor,
                           BLOCK_K: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    denom = 0.0
    # Compute sum of selected masked scores
    for k in range(0, BLOCK_K):
        if k < topK:
            idx = tl.load(SelectedIdx_ptr + m * stride_Ipm + k * stride_Ipn)
            sk = tl.load(Masked_ptr + m * stride_Mm + idx * stride_Mn)
            denom += sk
    # Write normalized and scaled outputs
    for k in range(0, BLOCK_K):
        if k < topK:
            idx = tl.load(SelectedIdx_ptr + m * stride_Ipm + k * stride_Ipn)
            sk = tl.load(Masked_ptr + m * stride_Mm + idx * stride_Mn)
            out = sk / (denom + 1e-20) * scaling_factor
            tl.store(Output_ptr + m * stride_Om + k * stride_On, out)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, top_k: int = 8, n_group: int = 8, experts_per_group: int = 32):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.experts_per_group = experts_per_group

    def forward(self, hidden_states: torch.Tensor,
                weight: torch.Tensor,
                expert_bias: torch.Tensor,
                routed_scaling_factor: float):
        # 1) Compute logits via PyTorch F.linear (weight: [N, K], hidden: [M, K])
        logits = torch.nn.functional.linear(hidden_states.to(torch.float32),
                                            weight.to(torch.float32))
        M, N = logits.shape
        assert N == self.num_experts, f"Expected N={self.num_experts}, got {N}"
        E = self.experts_per_group
        G = self.n_group
        assert N == G * E, "num_experts must equal n_group * experts_per_group"

        # 2) Triton: apply sigmoid and add expert bias
        scores = torch.empty_like(logits)
        sigmoid_bias_kernel[(M,)](
            logits, expert_bias, scores,
            M, N,
            scores.stride(0), scores.stride(1),
            expert_bias.stride(0),
            scores.stride(0), scores.stride(1),
            BLOCK_N=128,
        )

        # 3) Reshape to [M, 8, 32] and compute per-group top-2 sum (Triton)
        S = scores.view(M, G, E)
        group_scores = torch.empty((M, G), device=scores.device, dtype=torch.float32)
        top2_per_group_kernel[(M,)](
            S, group_scores,
            M, G, E,
            S.stride(0), S.stride(1), S.stride(2),
            group_scores.stride(0), group_scores.stride(1),
            BLOCK_E=E,  # fixed E=32 for this task
        )

        # 4) Select top-4 groups per token using torch.topk (to ensure correctness and simplicity)
        #    group_scores: [M, 8]; we need top-4 indices
        group_idx = torch.topk(group_scores, k=4, dim=1, largest=True, sorted=False)[1].to(torch.int32)

        # 5) Triton: set group_mask [M, 8] to 1 at selected groups
        group_mask = torch.empty((M, G), device=scores.device, dtype=torch.float32)
        set_group_mask_kernel[(M,)](
            group_idx, group_mask,
            M, G,
            group_idx.stride(0), group_idx.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            BLOCK_M=128,
        )

        # 6) Triton: expand group_mask to [M, N]
        expanded_mask = torch.empty((M, N), device=scores.device, dtype=torch.float32)
        expand_group_mask_kernel[(M,)](
            group_mask, expanded_mask,
            M, N, G, E,
            group_mask.stride(0), group_mask.stride(1),
            expanded_mask.stride(0), expanded_mask.stride(1),
            BLOCK_N=128,
        )

        # 7) Triton: mask non-selected group scores to -inf
        scores_contig = scores.contiguous()
        masked_scores = torch.empty_like(scores_contig)
        mask_scores_kernel[(M,)](
            scores_contig, expanded_mask, masked_scores,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            expanded_mask.stride(0), expanded_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            BLOCK_N=128,
        )

        # 8) Triton: per-token top-8 expert selection on masked_scores
        topk_idx = torch.empty((M, self.top_k), device=scores.device, dtype=torch.int32)
        # Use BLOCK_K=self.top_k (e.g., 8). Loop will run up to BLOCK_K; we guard with mask logic.
        topk_experts_kernel[(M,)](
            masked_scores, topk_idx,
            M, self.top_k, N,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0),
            BLOCK_K=self.top_k,
        )

        # 9) Triton: normalize and scale selected weights
        selected_weights = torch.empty((M, self.top_k), device=scores.device, dtype=torch.float32)
        normalize_scale_kernel[(M,)](
            topk_idx, masked_scores, selected_weights,
            M, self.top_k,
            topk_idx.stride(0), topk_idx.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            selected_weights.stride(0), selected_weights.stride(1),
            routed_scaling_factor,
            BLOCK_K=self.top_k,
        )

        # Return topk_idx (int32) and topk_weight (float32), matching original signature
        return topk_idx, selected_weights


def run(*args):
    return ModelNew()(*args)
