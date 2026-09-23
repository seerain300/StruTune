import torch
import triton
import triton.language as tl


@triton.jit
def _linear_proj_kernel(
    hidden_ptr,         # *ptr to hidden_states [N, K]
    weight_ptr,         # *ptr to weight [E, K]
    out_ptr,            # *ptr to output logits [N, E]
    N, E, K,            # sizes
    stride_hn, stride_hk,
    stride_we, stride_wk,
    stride_on, stride_oe,
    TILE_E: tl.constexpr,  # tile size over E
    TILE_K: tl.constexpr,  # tile size over K
):
    pid_n = tl.program_id(0)   # token row
    pid_te = tl.program_id(1)  # tile id over E

    # Compute E tile
    e_start = pid_te * TILE_E
    e_offsets = e_start + tl.arange(0, TILE_E)
    e_mask = e_offsets < E

    # Accumulator for this tile of E
    acc = tl.zeros((TILE_E,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, TILE_K):
        k_offsets = k0 + tl.arange(0, TILE_K)
        k_mask = k_offsets < K

        # Load hidden row slice: [TILE_K]
        h = tl.load(hidden_ptr + pid_n * stride_hn + k_offsets * stride_hk, mask=k_mask, other=0.0)

        # Load weight tile: [TILE_E, TILE_K]
        w = tl.load(weight_ptr + e_offsets[:, None] * stride_we + k_offsets[None, :] * stride_wk,
                    mask=e_mask[:, None] & k_mask[None, :], other=0.0)

        # acc += sum_k w[:, k] * h[k]
        # For each expert lane, compute dot with h
        for j in range(TILE_E):
            acc[j] += tl.sum(w[j, :] * h, axis=0)

    # Store results
    tl.store(out_ptr + pid_n * stride_on + e_offsets * stride_oe, acc, mask=e_mask)


@triton.jit
def _sigmoid_bias_kernel(
    logits_ptr,          # *ptr to [N, E] float32
    bias_ptr,            # *ptr to [E] float32
    out_ptr,             # *ptr to [N, E] float32
    N, E,
    stride_ln, stride_le,
    stride_bn,
    stride_on, stride_oe,
):
    pid = tl.program_id(0)
    # Compute n and e from linear index (flattened)
    cols = tl.arange(0, E)
    row = pid // E
    col = pid % E
    # If row >= N, return
    if row >= N:
        return
    # Load logits[row, col]
    l = tl.load(logits_ptr + row * stride_ln + col * stride_le)
    b = tl.load(bias_ptr + col * stride_bn)
    # Sigmoid and add bias
    s = 1.0 / (1.0 + tl.exp(-l))
    out = s + b
    tl.store(out_ptr + row * stride_on + col * stride_oe, out)


@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,          # *ptr to [N, E] float32
    group_scores_ptr,    # *ptr to [N, 8] float32
    N, E,
    stride_sn, stride_se,
    stride_gn, stride_ge,
    experts_per_group: tl.constexpr,   # 32
    n_group: tl.constexpr,             # 8
):
    pid_n = tl.program_id(0)  # one program per token
    if pid_n >= N:
        return
    # Compute group scores: for each group g, take 32 experts, find top-2, sum
    for g in range(n_group):
        base = g * experts_per_group
        # Initialize top-1 and top-2
        top1 = tl.full((), -float('inf'), dtype=tl.float32)
        top2 = tl.full((), -float('inf'), dtype=tl.float32)
        # Scan 32 experts in the group
        for j in range(experts_per_group):
            val = tl.load(scores_ptr + pid_n * stride_sn + (base + j) * stride_se)
            # update top-2
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        group_score = top1 + top2
        tl.store(group_scores_ptr + pid_n * stride_gn + g * stride_ge, group_score)


@triton.jit
def _select_top4_groups_kernel(
    scores_ptr,          # *ptr to [N, E] float32
    group_scores_ptr,    # *ptr to [N, 8] float32
    group_idx_ptr,       # *ptr to [N, 4] int32 (output)
    N, E,
    stride_sn, stride_se,
    stride_gn, stride_ge,
    stride_in, stride_ie,
    n_group: tl.constexpr,          # 8
    topk_group: tl.constexpr,       # 4
):
    pid_n = tl.program_id(0)  # one program per token
    if pid_n >= N:
        return
    # Initialize top-4 slots
    top_vals = [tl.full((), -float('inf'), dtype=tl.float32) for _ in range(topk_group)]
    top_idx = [tl.full((), -1, dtype=tl.int32) for _ in range(topk_group)]
    # Scan groups
    for g in range(n_group):
        score = tl.load(group_scores_ptr + pid_n * stride_gn + g * stride_ge)
        # Check and insert if better than current
        for k in range(topk_group):
            if score > top_vals[k]:
                # shift down
                for kk in range(topk_group - 1, k, -1):
                    top_vals[kk] = top_vals[kk - 1]
                    top_idx[kk] = top_idx[kk - 1]
                # insert
                top_vals[k] = score
                top_idx[k] = g
                break
    # Write to output
    for k in range(topk_group):
        tl.store(group_idx_ptr + pid_n * stride_in + k * stride_ie, top_idx[k])


@triton.jit
def _top8_mask_select_kernel(
    scores_ptr,            # *ptr to [N, E] float32
    group_idx_ptr,         # *ptr to [N, 4] int32
    out_idx_ptr,           # *ptr to [N, 8] int32 (output top-8 indices)
    out_weight_ptr,        # *ptr to [N, 8] float32 (output normalized weights * scaling)
    N, E,
    stride_sn, stride_se,
    stride_in, stride_ie,
    stride_on, stride_oe,
    routed_scaling_factor,  # float
    n_group: tl.constexpr,          # 8
    topk_group: tl.constexpr,       # 4
    topk_total: tl.constexpr,       # 8
    experts_per_group: tl.constexpr # 32
):
    pid_n = tl.program_id(0)  # one program per token
    if pid_n >= N:
        return
    # Build mask for selected groups: mask[j] = 1 if group of expert j is in top4, else 0
    mask = tl.zeros((E,), dtype=tl.int32)
    for k in range(topk_group):
        g = tl.load(group_idx_ptr + pid_n * stride_in + k * stride_ie)  # int32
        base = g * experts_per_group
        # Set mask for all experts in this group
        for j in range(experts_per_group):
            mask[base + j] = 1

    # Initialize top-8 slots
    top_vals = [tl.full((), -float('inf'), dtype=tl.float32) for _ in range(topk_total)]
    top_idx = [tl.full((), -1, dtype=tl.int32) for _ in range(topk_total)]

    # Scan all experts to fill top-8
    for j in range(E):
        # Only consider if mask[j] == 1 (selected group)
        m = mask[j]
        if m != 0:
            val = tl.load(scores_ptr + pid_n * stride_sn + j * stride_se)
            # insert into top-8
            for k in range(topk_total):
                if val > top_vals[k]:
                    for kk in range(topk_total - 1, k, -1):
                        top_vals[kk] = top_vals[kk - 1]
                        top_idx[kk] = top_idx[kk - 1]
                    top_vals[k] = val
                    top_idx[k] = j
                    break

    # Write output indices and normalized weights scaled
    scale = 1e-20  # small epsilon for normalization
    for k in range(topk_total):
        idx = top_idx[k]
        # write index
        tl.store(out_idx_ptr + pid_n * stride_on + k * stride_oe, idx)
        # write normalized weight * scaling
        val = top_vals[k]
        denom = tl.sum([top_vals[i] for i in range(topk_total)]) + scale
        norm = val / denom
        tl.store(out_weight_ptr + pid_n * stride_on + k * stride_oe, norm * routed_scaling_factor)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-only implementation of the original routing. No PyTorch ops in forward.
        Returns:
          - topk_idx: [num_tokens, 8] LongTensor of expert indices chosen
          - topk_weight: [num_tokens, 8] float32 normalized weights scaled
        """
        assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_dim]"
        assert weight.dim() == 2, "weight must be [num_experts, hidden_dim]"
        assert expert_bias.dim() == 1, "expert_bias must be [num_experts]"

        device = hidden_states.device
        num_tokens, hidden_dim = hidden_states.shape
        num_experts = weight.shape[0]
        assert weight.shape[1] == hidden_dim, "weight second dim must equal hidden_dim"
        assert hidden_dim == 256 and num_experts == 256, "This Triton implementation assumes hidden_dim=num_experts=256"

        # Ensure dtype float32 for Triton kernels
        hidden_f32 = hidden_states.contiguous().to(torch.float32)
        weight_f32 = weight.contiguous().to(torch.float32)
        expert_bias_f32 = expert_bias.contiguous().to(torch.float32)

        # 1) Compute logits = hidden @ weight.T using Triton GEMM
        logits = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=device)
        stride_hn, stride_hk = hidden_f32.stride()
        stride_we, stride_wk = weight_f32.stride()
        stride_on, stride_oe = logits.stride()

        # Tile sizes
        TILE_E = 32
        TILE_K = 64
        grid = (num_tokens, triton.cdiv(num_experts, TILE_E))
        _linear_proj_kernel[grid](
            hidden_f32, weight_f32, logits,
            num_tokens, num_experts, hidden_dim,
            stride_hn, stride_hk,
            stride_we, stride_wk,
            stride_on, stride_oe,
            TILE_E=TILE_E, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) Apply sigmoid and add expert bias using Triton elementwise kernel
        scores = torch.empty_like(logits)
        stride_ln, stride_le = logits.stride()
        stride_on_sb, stride_oe_sb = scores.stride()
        _sigmoid_bias_kernel[(num_tokens * num_experts,)](
            logits, expert_bias_f32, scores,
            num_tokens, num_experts,
            stride_ln, stride_le,
            expert_bias_f32.stride(0),
            stride_on_sb, stride_oe_sb,
            num_warps=1, num_stages=1,
        )

        # 3) Compute group scores (sum of top-2 per group) using Triton kernel: [num_tokens, 8]
        group_scores = torch.empty((num_tokens, 8), dtype=torch.float32, device=device)
        stride_sn, stride_se = scores.stride()
        stride_gn, stride_ge = group_scores.stride()
        _group_top2_sum_kernel[(num_tokens,)](
            scores, group_scores,
            num_tokens, num_experts,
            stride_sn, stride_se,
            stride_gn, stride_ge,
            experts_per_group=32, n_group=8,
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token using Triton kernel: [num_tokens, 4]
        group_idx = torch.empty((num_tokens, 4), dtype=torch.int32, device=device)
        stride_in, stride_ie = group_idx.stride()
        _select_top4_groups_kernel[(num_tokens,)](
            scores, group_scores, group_idx,
            num_tokens, num_experts,
            stride_sn, stride_se,
            stride_gn, stride_ge,
            stride_in, stride_ie,
            n_group=8, topk_group=4,
            num_warps=1, num_stages=1,
        )

        # 5) Perform masking and final top-8 selection, and normalize + scale using Triton kernel
        # Prepare output buffers
        topk_idx = torch.empty((num_tokens, 8), dtype=torch.int32, device=device)
        topk_weight = torch.empty((num_tokens, 8), dtype=torch.float32, device=device)

        stride_on_mask, stride_oe_mask = topk_idx.stride()
        stride_on_weight, stride_oe_weight = topk_weight.stride()

        _top8_mask_select_kernel[(num_tokens,)](
            scores, group_idx, topk_idx, topk_weight,
            num_tokens, num_experts,
            stride_sn, stride_se,
            stride_in, stride_ie,
            stride_on_mask, stride_oe_mask,
            routed_scaling_factor,
            n_group=8, topk_group=4, topk_total=8, experts_per_group=32,
            num_warps=1, num_stages=1,
        )

        # Return as requested: topk_idx (LongTensor) and topk_weight (FloatTensor)
        return topk_idx.to(torch.long), topk_weight


def run(*args):
    return ModelNew()(*args)
