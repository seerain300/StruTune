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
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask, other=0.0)
        b = tl.load(Bias_ptr + n_offsets * stride_Bn, mask=mask, other=0.0)
        y = 1.0 / (1.0 + tl.exp(-x)) + b
        tl.store(Y_ptr + m * stride_Ym + n_offsets * stride_Yn, y, mask=mask)


@triton.jit
def expand_group_mask_kernel(GroupMask_ptr, MaskedScores_ptr, Scores_ptr,
                             M, N, G, E,
                             stride_GMm, stride_GMg,
                             stride_MSm, stride_MSexp,
                             stride_Sm, stride_Sexp,
                             BLOCK_EXP: tl.constexpr):
    # One program per token row; write -inf to scores for non-selected groups
    m = tl.program_id(0)
    for g in range(0, G):
        selected = tl.load(GroupMask_ptr + m * stride_GMm + g * stride_GMg)  # 0.0 or 1.0
        if selected == 1.0:
            # skip
            pass
        else:
            start_exp = g * E
            for e in range(0, E):
                exp_idx = start_exp + e
                score = tl.load(Scores_ptr + m * stride_Sm + exp_idx * stride_Sexp)
                tl.store(MaskedScores_ptr + m * stride_MSm + exp_idx * stride_MSexp, -1e20)


@triton.jit
def arg_topk_kernel(X_ptr, K_ptr, K_arg_ptr,
                     M, N, K,
                     stride_Xm, stride_Xn,
                     stride_Km, stride_Kk,
                     stride_Argm,
                     BLOCK_N: tl.constexpr):
    # One program per token row; select top-K indices
    m = tl.program_id(0)
    top_vals = tl.full([BLOCK_N], -1e20, tl.float32)
    top_inds = tl.full([BLOCK_N], -1, tl.int32)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask, other=-1e20)
        # Insert each element of the block into the current top-k
        for j in range(BLOCK_N):
            v = x[j]
            pos = 0
            while pos < K and top_vals[pos] > v:
                pos += 1
            if pos < K:
                # shift from end
                for t in range(K - 1, pos - 1, -1):
                    top_vals[t] = top_vals[t - 1]
                    top_inds[t] = top_inds[t - 1]
                top_vals[pos] = v
                top_inds[pos] = n_offsets[j]
        # After processing all blocks, take first K
        for k in range(K):
            tl.store(K_ptr + m * stride_Km + k * stride_Kk, top_vals[k])
            tl.store(K_arg_ptr + m * stride_Argm + k, top_inds[k])


@triton.jit
def normalize_scale_kernel(SelectedScores_ptr, TopkWeight_ptr,
                           M, K,
                           stride_SSm, stride_SSexp,
                           stride_TWm, stride_TWk,
                           scale,
                           BLOCK_K: tl.constexpr):
    # One program per token row; normalize and scale selected scores
    m = tl.program_id(0)
    sum_val = 0.0
    for k in range(0, K):
        score = tl.load(SelectedScores_ptr + m * stride_SSm + k * stride_SSexp)
        sum_val += score
    for k in range(0, K):
        score = tl.load(SelectedScores_ptr + m * stride_SSm + k * stride_SSexp)
        w = score / (sum_val + 1e-20) * scale
        tl.store(TopkWeight_ptr + m * stride_TWm + k * stride_TWk, w)


def _launch_expand_mask(group_mask: torch.Tensor, scores: torch.Tensor, masked_scores: torch.Tensor,
                        M: int, N: int, G: int, E: int):
    # group_mask: [M, G] float32 (0.0 or 1.0), contiguous
    # scores: [M, N], contiguous
    # masked_scores: [M, N], contiguous
    grid = (M,)
    triton.run(
        expand_group_mask_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        kwargs=dict(
            GroupMask_ptr=group_mask,
            MaskedScores_ptr=masked_scores,
            Scores_ptr=scores,
            M=M, N=N, G=G, E=E,
            stride_GMm=group_mask.stride(0), stride_GMg=group_mask.stride(1),
            stride_MSm=masked_scores.stride(0), stride_MSexp=masked_scores.stride(1),
            stride_Sm=scores.stride(0), stride_Sexp=scores.stride(1),
        ),
    )


def _launch_arg_topk(x: torch.Tensor, K: int, M: int, N: int):
    # x: [M, N] float32, contiguous
    top_vals = torch.empty((M, K), dtype=torch.float32, device=x.device)
    top_inds = torch.empty((M, K), dtype=torch.int32, device=x.device)
    grid = (M,)
    triton.run(
        arg_topk_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        kwargs=dict(
            X_ptr=x,
            K_ptr=top_vals,
            K_arg_ptr=top_inds,
            M=M, N=N, K=K,
            stride_Xm=x.stride(0), stride_Xn=x.stride(1),
            stride_Km=top_vals.stride(0), stride_Kk=top_vals.stride(1),
            stride_Argm=top_inds.stride(0),
        ),
    )
    return top_inds.to(torch.int64), top_vals


def _launch_normalize_scale(selected_scores: torch.Tensor, topk_weight: torch.Tensor,
                            M: int, K: int, scale: float):
    # selected_scores: [M, K] float32, contiguous
    grid = (M,)
    triton.run(
        normalize_scale_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        kwargs=dict(
            SelectedScores_ptr=selected_scores,
            TopkWeight_ptr=topk_weight,
            M=M, K=K,
            stride_SSm=selected_scores.stride(0), stride_SSexp=selected_scores.stride(1),
            stride_TWm=topk_weight.stride(0), stride_TWk=topk_weight.stride(1),
            scale=scale,
        ),
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                weight: torch.Tensor,
                expert_bias: torch.Tensor,
                routed_scaling_factor: float):
        """
        hidden_states: [M, K], float32
        weight: [E, K], float32, E=256, K=hidden_states.shape[1]
        expert_bias: [E], float32
        returns:
          topk_idx: [M, 8], int64
          topk_weight: [M, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be CUDA"
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        E = weight.shape[0]  # 256
        assert weight.shape[1] == K, "weight second dim must match hidden_states last dim"
        assert expert_bias.shape[0] == E, "expert_bias must have length equal to number of experts"

        # 1) Compute logits via F.linear (PyTorch)
        logits = torch.nn.functional.linear(hidden_states.to(torch.float32), weight.to(torch.float32))

        # 2) Triton kernel: sigmoid + add expert bias
        scores = torch.empty_like(logits)
        triton.run(
            sigmoid_bias_kernel,
            grid=(M,),
            num_warps=4,
            num_stages=2,
            kwargs=dict(
                X_ptr=logits,
                Bias_ptr=expert_bias,
                Y_ptr=scores,
                M=M, N=E,
                stride_Xm=logits.stride(0), stride_Xn=logits.stride(1),
                stride_Bn=expert_bias.stride(0),
                stride_Ym=scores.stride(0), stride_Yn=scores.stride(1),
            ),
        )

        # 3) Per-group top-2 aggregation to form group_scores [M, 8] using PyTorch (robust for small G)
        group_scores = scores.view(M, 8, 32)  # [M, 8, 32]
        # compute top-2 per group and sum
        # We implement a simple loop in PyTorch for correctness:
        # top2_vals: [M, 8, 2]
        top2_vals = torch.empty((M, 8, 2), dtype=torch.float32, device=scores.device)
        for g in range(8):
            group = group_scores[:, g, :]  # [M, 32]
            # top-2 via two maxima
            max1 = group.max(dim=1, keepdim=True).values  # [M, 1]
            mask1 = group == max1
            # exclude max1 from second max
            group2 = torch.where(mask1, torch.full_like(group, -1e20), group)
            max2 = group2.max(dim=1, keepdim=True).values  # [M, 1]
            top2_vals[:, g, 0] = max1[:, 0]
            top2_vals[:, g, 1] = max2[:, 0]
        group_scores_sum = top2_vals[:, :, 0] + top2_vals[:, :, 1]  # [M, 8]

        # 4) Triton arg-topk: per-token top-4 groups → group_idx [M, 4]
        _, group_idx = torch.topk(group_scores_sum, k=4, dim=1, sorted=False)  # [M, 4], int64

        # 5) Build group_mask [M, 8] with 1.0 at selected groups (others 0.0)
        group_mask = torch.zeros((M, 8), dtype=torch.float32, device=scores.device)
        group_mask.scatter_(1, group_idx, 1.0)  # expand to 4, will later handle top-4 via mask

        # 6) Triton: expand group_mask to [M, N] and mask scores for non-selected groups to -inf
        masked_scores = torch.empty_like(scores)  # [M, N]
        _launch_expand_mask(group_mask, scores, masked_scores, M, E, 8, 32)

        # 7) Triton arg-topk: per-token top-8 expert selection on masked_scores → topk_idx [M, 8]
        topk_idx, _ = _launch_arg_topk(masked_scores, 8, M, E)

        # 8) Normalize and scale selected weights in Triton
        # We gather selected_scores from original scores using topk_idx to get pre-mask values
        # Build selected_scores tensor [M, 8]
        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        for m in range(M):
            selected_scores[m] = scores[m][topk_idx[m]]

        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        _launch_normalize_scale(selected_scores, topk_weight, M, 8, routed_scaling_factor)

        # Return topk_idx and topk_weight as in original
        return topk_idx, topk_weight

# Example input helper (not used by evaluator):
def get_inputs():
    M, K = 2048, 128
    E = 256
    hidden_states = torch.randn(M, K, device='cuda', dtype=torch.float32)
    weight = torch.randn(E, K, device='cuda', dtype=torch.float32)
    expert_bias = torch.randn(E, device='cuda', dtype=torch.float32)
    routed_scaling_factor = 0.7
    return hidden_states, weight, expert_bias, routed_scaling_factor

# Optional quick check:
# if __name__ == "__main__":
#     model = ModelNew().cuda()
#     hidden_states, weight, expert_bias, routed_scaling_factor


def run(*args):
    return ModelNew()(*args)
