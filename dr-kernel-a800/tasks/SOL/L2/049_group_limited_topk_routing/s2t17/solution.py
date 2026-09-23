import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim], float32, CUDA tensor
        # weight: [num_experts, hidden_dim], float32, CUDA tensor (num_experts must be 256)
        # expert_bias: [num_experts], float32, CUDA tensor
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"
        assert weight.shape[1] == hidden_dim, "weight's last dim must match hidden_states' last dim"

        # Prepare output buffers
        # We'll use Triton to fill these:
        logits = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=hidden_states.device)
        scores_for_routing = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=hidden_states.device)

        # Compute logits via Triton: scores[token, e] = sum_j hidden_states[token, j] * weight[e, j]
        @triton.jit
        def compute_logits_kernel(hidden_states_ptr, weight_ptr, logits_ptr,
                                   num_tokens, hidden_dim, num_experts):
            t = tl.program_id(0)
            e = tl.program_id(1)
            # Bounds check
            if (t >= num_tokens) or (e >= num_experts):
                return
            # Accumulate dot product
            acc = 0.0
            for j in range(0, hidden_dim):
                hs = tl.load(hidden_states_ptr + t * hidden_dim + j)
                w = tl.load(weight_ptr + e * hidden_dim + j)
                acc += hs * w
            tl.store(logits_ptr + t * num_experts + e, acc)

        grid = (num_tokens, num_experts)
        compute_logits_kernel[grid](
            hidden_states, weight, logits,
            num_tokens, hidden_dim, num_experts,
        )

        # Apply sigmoid to logits and add expert bias
        # Triton elementwise: scores_for_routing = sigmoid(logits) + expert_bias
        @triton.jit
        def apply_sigmoid_add_bias_kernel(logits_ptr, bias_ptr, scores_ptr,
                                          num_tokens, num_experts):
            t = tl.program_id(0)
            e = tl.program_id(1)
            if (t >= num_tokens) or (e >= num_experts):
                return
            l = tl.load(logits_ptr + t * num_experts + e)
            b = tl.load(bias_ptr + e)
            # sigmoid
            s = 1.0 / (1.0 + tl.exp(-l))
            tl.store(scores_ptr + t * num_experts + e, s + b)

        apply_sigmoid_add_bias_kernel[grid](
            logits, expert_bias, scores_for_routing,
            num_tokens, num_experts,
        )

        # Now, compute group_scores via top-2 per group, sum.
        # group_scores: [num_tokens, 8]
        # We'll implement the top-2/group selection in Triton by repeated scanning.
        # However, Triton does not allow writing to 2D outputs in a vectorized way; we'll use host to hold small outputs for indices and mask.

        # Output tensors for selected group indices (int32): [num_tokens, 4]
        group_idx_selected = torch.empty((num_tokens, 4), dtype=torch.int32, device=hidden_states.device)
        group_idx_selected.fill_(-1)

        @triton.jit
        def select_top4_groups_kernel(scores_ptr, out_ptr,
                                      num_tokens, num_experts, n_groups, experts_per_group):
            # Each program handles one token
            t = tl.program_id(0)
            if t >= num_tokens:
                return
            # Initialize candidates as (-inf, -1) for 8 groups
            candidates = tl.full((n_groups,), -float('inf'), dtype=tl.float32)
            indices = tl.full((n_groups,), -1, dtype=tl.int32)

            # First scan: fill candidates with max per group
            for g in range(0, n_groups):
                start = g * experts_per_group
                # Reduce max over this group: loop over 32
                max_val = -float('inf')
                max_idx = -1
                for i in range(0, experts_per_group):
                    e = start + i
                    val = tl.load(scores_ptr + t * num_experts + e)
                    if val > max_val:
                        max_val = val
                        max_idx = e
                # store candidate
                candidates[g] = max_val
                indices[g] = max_idx

            # Now select top-4 by repeated scanning: find max, store index, set it to -inf
            # We can store up to 4 selected indices into out_ptr[t, :]
            k = 0
            # Loop up to 4
            while k < 4:
                best = -float('inf')
                sel_idx = -1
                # Scan candidates to find max
                for g in range(0, n_groups):
                    if candidates[g] > best:
                        best = candidates[g]
                        sel_idx = indices[g]
                # If found, store and mark as removed
                if sel_idx != -1:
                    tl.store(out_ptr + t * 4 + k, sel_idx)
                    # Mark removed: set its candidate to -inf
                    candidates = candidates
                    # No need to update indices here (we won't read it later)
                    k += 1
                # Continue scanning for next selection

        # Launch kernel to get top-4 groups per token
        select_top4_groups_kernel[(num_tokens,)](
            scores_for_routing, group_idx_selected,
            num_tokens, num_experts, 8, 32
        )

        # Build group_mask [num_tokens, 8] in host from selected indices (1 for selected groups, 0 otherwise)
        # We'll initialize group_mask with zeros and scatter 1.0 at selected positions
        group_mask = torch.zeros((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)
        # Scatter selected group indices
        selected_cols = (group_idx_selected >= 0) & (group_idx_selected < 8)
        # For rows where any selected, write 1 at those columns
        for t in range(num_tokens):
            if selected_cols[t].any().item():
                group_mask[t, selected_cols[t]] = 1.0

        # Expand group_mask to per-expert mask and apply to scores: non-selected groups set to -inf
        # We emulate masking in Triton by comparing each expert e to group_start and using group_mask to decide.
        # Create a new tensor masked_scores_for_routing
        masked_scores_for_routing = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=hidden_states.device)
        # Fill with -inf first, then copy selected groups from scores_for_routing
        masked_scores_for_routing.fill_(float('-inf'))

        @triton.jit
        def apply_group_mask_kernel(scores_ptr, mask_ptr, out_ptr,
                                    num_tokens, num_experts, n_groups, experts_per_group):
            # Each program handles one token
            t = tl.program_id(0)
            if t >= num_tokens:
                return
            # Iterate groups
            for g in range(0, n_groups):
                # Read mask for this group
                m = tl.load(mask_ptr + t * n_groups + g)  # float32 0 or 1
                start = g * experts_per_group
                # If m == 1, copy this group; else leave as -inf
                # Loop over 32 experts in this group
                for i in range(0, experts_per_group):
                    e = start + i
                    val = tl.load(scores_ptr + t * num_experts + e)
                    # Use scalar m as flag
                    if m > 0.0:
                        tl.store(out_ptr + t * num_experts + e, val)
                    else:
                        tl.store(out_ptr + t * num_experts + e, float('-inf'))

        apply_group_mask_kernel[(num_tokens,)](
            scores_for_routing, group_mask, masked_scores_for_routing,
            num_tokens, num_experts, 8, 32
        )

        # Final top-8 selection from masked_scores_for_routing in Triton via repeated scanning
        topk_idx_out = torch.empty((num_tokens, 8), dtype=torch.int32, device=hidden_states.device)
        topk_idx_out.fill_(-1)

        @triton.jit
        def select_top8_kernel(scores_ptr, out_ptr,
                               num_tokens, num_experts):
            # Each program handles one token
            t = tl.program_id(0)
            if t >= num_tokens:
                return
            # Initialize candidates as (-inf, -1) for 256 experts
            candidates = tl.full((num_experts,), -float('inf'), dtype=tl.float32)
            indices = tl.full((num_experts,), -1, dtype=tl.int32)

            # First pass: fill candidates with max of scores_ptr
            for e in range(0, num_experts):
                val = tl.load(scores_ptr + t * num_experts + e)
                candidates[e] = val
                indices[e] = e

            # Repeated scanning to pick top-8
            k = 0
            while k < 8:
                best = -float('inf')
                sel_idx = -1
                # Scan candidates to find max
                for e in range(0, num_experts):
                    if candidates[e] > best:
                        best = candidates[e]
                        sel_idx = indices[e]
                if sel_idx != -1:
                    tl.store(out_ptr + t * 8 + k, sel_idx)
                    # Mark removed by setting candidate to -inf
                    candidates[sel_idx] = -float('inf')
                    k += 1

        select_top8_kernel[(num_tokens,)](
            masked_scores_for_routing, topk_idx_out,
            num_tokens, num_experts
        )

        # Now compute original logits for the selected indices (up to 8), normalize, and apply routed_scaling_factor
        # We'll recompute dot-products in Triton for those selected indices
        # Prepare selected_logits: [num_tokens, 8]
        selected_logits = torch.empty((num_tokens, 8), dtype=torch.float32, device=hidden_states.device)
        selected_logits.fill_(float('-inf'))

        @triton.jit
        def recompute_and_normalize_kernel(hs_ptr, weight_ptr, idx_ptr, out_ptr,
                                           num_tokens, hidden_dim, num_experts, scaling):
            # Each program handles one token
            t = tl.program_id(0)
            if t >= num_tokens:
                return
            total = 0.0
            # First compute sum of original logits for selected idx
            for k in range(0, 8):
                idx = tl.load(idx_ptr + t * 8 + k)  # int32
                if idx != -1:
                    acc = 0.0
                    for j in range(0, hidden_dim):
                        hs = tl.load(hs_ptr + t * hidden_dim + j)
                        w = tl.load(weight_ptr + idx * hidden_dim + j)
                        acc += hs * w
                    total += acc
            # Compute normalized outputs for k = 0..7 (overwrite -inf with valid logits)
            for k in range(0, 8):
                idx = tl.load(idx_ptr + t * 8 + k)
                acc = 0.0
                if idx != -1:
                    for j in range(0, hidden_dim):
                        hs = tl.load(hs_ptr + t * hidden_dim + j)
                        w = tl.load(weight_ptr + idx * hidden_dim + j)
                        acc += hs * w
                    val = acc / (total + 1e-20) * scaling
                    tl.store(out_ptr + t * 8 + k, val)

        # Note: Triton loops over 8 are fine; hidden_dim is runtime but loops are valid.
        recompute_and_normalize_kernel[(num_tokens,)](
            hidden_states, weight, topk_idx_out, selected_logits,
            num_tokens, hidden_dim, num_experts, routed_scaling_factor
        )

        # Return: topk_idx_out (int32) and selected_logits (float32)
        # The original PyTorch code returns (topk_idx, topk_weight). Here, topk_weight is the normalized and scaled selected logits we just computed.
        # Cast topk_idx_out to int64 to match typical PyTorch API expectations
        topk_idx_out = topk_idx_out.to(torch.int64)

        return topk_idx_out, selected_logits


def run(*args):
    return ModelNew()(*args)
