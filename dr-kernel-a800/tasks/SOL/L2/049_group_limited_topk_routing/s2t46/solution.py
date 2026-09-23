import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [num_tokens, hidden_dim]
        # weight: [num_experts, hidden_dim] (num_experts = 256)
        # expert_bias: [num_experts]
        # routed_scaling_factor: float

        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]
        assert num_experts == 256, "num_experts must be 256"

        # Outputs: top-8 selected expert indices and normalized weights
        # Use torch.topk for the final selection to match PyTorch's behavior
        # We still perform all intermediate computations in Triton.
        # Allocate outputs
        # We'll compute in Triton and then call torch.topk to get final top-8 indices and scores
        # (to ensure correct tie-handling and output format).
        # Note: Triton kernel below does not call torch.topk; we call it after masked_scores are computed.

        # Ensure contiguous tensors for Triton
        hidden_ptr = hidden_states.contiguous()
        weight_ptr = weight.contiguous()
        bias_ptr = expert_bias.contiguous()

        # We need to compute masked_scores per token in Triton, then do torch.topk on that vector.
        # However, Triton cannot directly write a 1D vector of length 256 per token without careful structuring.
        # To keep the Triton kernel simple and robust, we compute per-token scores and then perform top-8 selection
        # in PyTorch based on that vector. Below is a Triton kernel that produces per-token scores and group_mask.

        # Triton kernel: per-token compute scores and group mask
        @triton.jit
        def compute_scores_and_group_mask_kernel(
            hidden_ptr,        # *float32, [num_tokens, hidden_dim], row-major
            weight_ptr,        # *float32, [num_experts, hidden_dim], row-major
            bias_ptr,          # *float32, [num_experts]
            scores_ptr,        # *float32, [num_tokens, num_experts]
            group_mask_ptr,    # *float32, [num_tokens, 8]
            num_tokens,        # int32
            hidden_dim,        # int32
            num_experts,       # int32
            EXPERTS_PER_GROUP: tl.constexpr,  # 32
            N_GROUPS: tl.constexpr,           # 8
            BLOCK_SIZE: tl.constexpr,         # e.g., 128
        ):
            pid = tl.program_id(axis=0)  # one program per token
            if pid >= num_tokens:
                return

            # Load hidden vector for this token: h[hidden_dim]
            h = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for d in range(0, hidden_dim):
                h[d] = tl.load(hidden_ptr + pid * hidden_dim + d)

            # Compute logits for all num_experts via dot-product
            scores = tl.zeros((num_experts,), dtype=tl.float32)
            for e in range(0, num_experts):
                acc = 0.0
                for d in range(0, hidden_dim):
                    w = tl.load(weight_ptr + e * hidden_dim + d)
                    acc += w * h[d]
                scores[e] = 1.0 / (1.0 + tl.exp(-acc))  # sigmoid

            # Add expert bias
            for e in range(0, num_experts):
                b = tl.load(bias_ptr + e)
                scores[e] += b

            # Store scores for this token to scores_ptr[pid, :]
            out_row = pid * num_experts
            for e in range(0, num_experts):
                tl.store(scores_ptr + out_row + e, scores[e])

            # Per-group top-2 and group scores
            group_top2 = tl.zeros((N_GROUPS,), dtype=tl.float32)
            for g in range(0, N_GROUPS):
                start = g * EXPERTS_PER_GROUP
                top1 = -float("inf")
                top2 = -float("inf")
                for i in range(0, EXPERTS_PER_GROUP):
                    e_idx = start + i
                    v = scores[e_idx]
                    # update top2 if v > top2
                    if v > top2:
                        top2 = v
                    # update top1 if v > top1; move top1 down to top2
                    if v > top1:
                        top2 = top1
                        top1 = v
                group_top2[g] = top1 + top2


def run(*args):
    return ModelNew()(*args)
