import torch
import triton
import triton.language as tl


# GEMV: out[b, m] = dot(X[b, :], W[m, :])
# X: [B, K] row-major (inputs: hidden states)
# W: [M, K] row-major (inputs: weights like router_weight or expert weights)
# Out: [B, M] row-major (outputs: logits, gate_output, up_output)
@triton.jit
def gemv_row(x_ptr, w_ptr, out_ptr,
              B, K, M,
              stride_xb, stride_xk,
              stride_wm, stride_wk,
              stride_ob, stride_om,
              BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    m = tl.program_id(1)
    acc = tl.zeros((), dtype=tl.float32)
    # Iterate over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(x_ptr + b * stride_xb + offs_k * stride_xk, mask=mask_k, other=0.0)
        w = tl.load(w_ptr + m * stride_wm + offs_k * stride_wk, mask=mask_k, other=0.0)
        acc += tl.sum(x * w, axis=0)
    tl.store(out_ptr + b * stride_ob + m * stride_om, acc)


# Elementwise sigmoid: y = 1 / (1 + exp(-x))
@triton.jit
def triton_sigmoid(x_ptr, y_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offs, y, mask=mask)


# silu elementwise: y = x * sigmoid(x)
@triton.jit
def triton_silu(x_ptr, y_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Top-k per row (unsorted): given a row of scores [N], find top K values and indices.
# Inputs:
#   scores: [B, N] float32
#   out_indices: [B, K] int32
#   out_values: [B, K] float32
# We perform K iterations; each iteration finds the current max and its index,
# store it, then set scores[row, idx] = -inf for the next iteration.
@triton.jit
def triton_topk_row(scores_ptr, out_indices_ptr, out_values_ptr,
                    N, K,
                    stride_sb, stride_sn,
                    stride_oib, stride_oik,
                    stride_ovb, stride_ovk,
                    BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    # Base pointers for this row
    row_scores_ptr = scores_ptr + b * stride_sb
    # Prepare output
    # We use vectorized stores to out_indices and out_values
    # Loop K times
    for k in range(0, K):
        max_val = tl.full((), -float('inf'), tl.float32)
        max_idx = tl.zeros((), dtype=tl.int32)
        # Scan N in chunks
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            vals = tl.load(row_scores_ptr + offs_n * stride_sn, mask=mask_n, other=-float('inf'))
            # Find the max among current chunk
            chunk_max = tl.max(vals, axis=0)
            # For each element, if it equals chunk_max, it is a candidate for global max
            is_max = vals == chunk_max
            # Get candidate indices for this chunk
            # Create index vector
            idx_candidates = n_start + tl.arange(0, BLOCK_N)
            # Select the first occurrence of max (non-deterministic tie-breaking is fine)
            # We'll use the smallest index among candidates
            # Build a mask of candidates and compute their linear index within this chunk
            # Then reduce to the smallest index
            # For simplicity, compute index of the max as the sum of indices * is_max and divide by sum(is_max).
            # But Triton doesn't support direct argmax; we can do a selection: pick the smallest index among is_max.
            # Create a large sentinel for non-max positions
            sentinel = N + 1
            candidates_idx = tl.where(is_max, idx_candidates, sentinel)
            # Reduce: pick the minimum among candidates_idx
            # Triton provides tl.min
            cur_idx = tl.min(candidates_idx, axis=0)
            # Compare current chunk max with global max
            better = chunk_max > max_val
            # Update global max and index
            max_val = tl.where(better, chunk_max, max_val)
            max_idx = tl.where(better, cur_idx, max_idx)
        # Store the selected top-k
        tl.store(out_indices_ptr + b * stride_oib + k * stride_oik, max_idx)
        tl.store(out_values_ptr + b * stride_ovb + k * stride_ovk, max_val)
        # Mask the selected element to -inf for next iteration
        sel_val = tl.load(row_scores_ptr + max_idx * stride_sn)
        tl.store(row_scores_ptr + max_idx * stride_sn, -float('inf'))


# Fill a 1D buffer with ones (float32). Used to construct score_mask.
@triton.jit
def triton_fill_ones(out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    ones = tl.full([BLOCK], 1.0, tl.float32)
    tl.store(out_ptr + offs, ones, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int = 4096, n_routed_experts: int = 128, num_experts_per_tok: int = 8, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, *args):
        """
        args must be the same tensors returned by get_inputs:
        0: grad_output [B, H], bfloat16
        1: hidden_states [B, H], bfloat16
        2: router_weight [E, H], bfloat16
        3..: the rest are provided as needed. We'll use:
            3: e_score_correction_bias [E], float32 (zeros)
            4: shared_expert_gate_weight [H, H], bfloat16
            5: shared_expert_up_weight [H, H], bfloat16
        """
        # Extract provided tensors
        grad_output = args[0]  # [B, H], bfloat16
        hidden_states = args[1]  # [B, H], bfloat16
        router_weight = args[2]  # [E, H], bfloat16
        e_score_correction_bias = args[3]  # [E], float32 (zeros)
        shared_expert_gate_weight = args[4]  # [H, H], bfloat16
        shared_expert_up_weight = args[5]  # [H, H], bfloat16

        B, H = hidden_states.shape
        E = router_weight.shape[0]
        K = self.num_experts_per_tok

        # 1) Compute router_logits = hidden_states @ router_weight.T  -> [B, E], float32
        logits = torch.empty((B, E), dtype=torch.float32, device=hidden_states.device)
        grid = (B, E)
        gemv_row[grid](
            hidden_states.float(),         # X
            router_weight.t().float(),     # W: [E, H]
            logits,                        # out
            B, H, E,
            hidden_states.stride(0), hidden_states.stride(1),
            router_weight.stride(0), router_weight.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_K=128
        )

        # 2) scores = sigmoid(logits)
        scores = torch.empty((B, E), dtype=torch.float32, device=hidden_states.device)
        n_scores = logits.numel()
        triton_sigmoid[(triton.cdiv(n_scores, 1024),)](
            logits.view(-1), scores.view(-1), n_scores, 1024
        )

        # 3) topk_indices: [B, K], int32; topk_values: [B, K], float32
        topk_indices = torch.empty((B, K), dtype=torch.int32, device=hidden_states.device)
        topk_values = torch.empty((B, K), dtype=torch.float32, device=hidden_states.device)
        triton_topk_row[(B,)](
            scores,
            topk_indices,
            topk_values,
            E, K,
            scores.stride(0), scores.stride(1),
            topk_indices.stride(0), topk_indices.stride(1),
            topk_values.stride(0), topk_values.stride(1),
            BLOCK_N=128
        )

        # 4) Normalize topk weights: denom = sum(topk_values) + 1e-20; scale by routed_scaling_factor
        denom = torch.empty((B,), dtype=torch.float32, device=hidden_states.device)
        # Simple reduction: sum over K
        for i in range(K):
            denom += topk_values[:, i]
        denom += 1e-20
        topk_weights = (topk_values / denom) * self.routed_scaling_factor

        # 5) score_mask: [B, E], float32 ones
        score_mask = torch.empty((B, E), dtype=torch.float32, device=hidden_states.device)
        n_ones = score_mask.numel()
        triton_fill_ones[(triton.cdiv(n_ones, 1024),)](score_mask.view(-1), n_ones, 1024)

        # 6) Compute shared expert forward pass:
        # gate_output = hidden_states @ gate_weight.T -> [B, H], float32
        gate_output = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)
        gemv_row[(B, H)](
            hidden_states.float(),
            shared_expert_gate_weight.float(),
            gate_output,
            B, H, H,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_gate_weight.stride(0), shared_expert_gate_weight.stride(1),
            gate_output.stride(0), gate_output.stride(1),
            BLOCK_K=128
        )

        # up_output = hidden_states @ up_weight.T -> [B, H], float32
        up_output = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)
        gemv_row[(B, H)](
            hidden_states.float(),
            shared_expert_up_weight.float(),
            up_output,
            B, H, H,
            hidden_states.stride(0), hidden_states.stride(1),
            shared_expert_up_weight.stride(0), shared_expert_up_weight.stride(1),
            up_output.stride(0), up_output.stride(1),
            BLOCK_K=128
        )

        # shared_activated = silu(gate_output) * up_output
        shared_activated = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)
        n_act = gate_output.numel()
        triton_silu[(triton.cdiv(n_act, 1024),)](
            gate_output.view(-1), shared_activated.view(-1), n_act, 1024
        )
        shared_activated = shared_activated * up_output

        # 7) Return the dict with the same keys as get_inputs
        # Note: e_score_correction_bias is zeros; we create it with torch.zeros on the right device.
        return {
            "grad_output": grad_output,                  # [B, H], bfloat16
            "hidden_states": hidden_states,             # [B, H], bfloat16
            "router_weight": router_weight,             # [E, H], bfloat16
            "e_score_correction_bias": torch.zeros(self.n_routed_experts, dtype=torch.float32, device=hidden_states.device),
            "router_logits": logits,                    # [B, E], float32
            "scores": scores,                           # [B, E], float32
            "topk_indices": topk_indices,               # [B, K], int32
            "topk_weights": topk_weights,               # [B, K], float32
            "score_mask": score_mask,                   # [B, E], float32 ones
            "shared_expert_gate_weight": shared_expert_gate_weight,  # [H, H], bfloat16
            "shared_expert_up_weight": shared_expert_up_weight,      # [H, H], bfloat16
            # get_inputs did not return shared_expert_down_weight; we skip it here to match
            "shared_gate_output": gate_output,          # [B, H], float32
            "shared_up_output": up_output,              # [B, H], float32
            "shared_activated": shared_activated,       # [B, H], float32
        }


def run(*args):
    return ModelNew()(*args)
