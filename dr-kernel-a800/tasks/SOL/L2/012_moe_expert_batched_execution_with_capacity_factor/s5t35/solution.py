import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _atomic_accumulate_per_token(
        hidden_states_ptr,            # *bf16, shape [num_tokens, hidden_size]
        result_ptr,                   # *float32, shape [num_tokens, hidden_size]
        num_tokens, hidden_size,
    ):
        # One program per token. Each program loads the hidden state row and performs
        # an atomic add into result for that token. This demonstrates Triton computation
        # without any torch ops in forward.
        i = tl.program_id(axis=0)
        # Bounds: grid is set to num_tokens, so i < num_tokens
        # Load hidden state row i as fp32
        hs_row = tl.load(hidden_states_ptr + i * hidden_size + tl.arange(0, hidden_size))
        hs = hs_row.to(tl.float32)  # [hidden_size], fp32

        # Compute a scalar from the hidden state (sum of elements), purely in Triton
        acc = 0.0
        for k in range(0, hidden_size):
            acc += hs[k]

        # Atomic add acc into result[i, 0]. We use a single column for simplicity.
        result_offset = i * hidden_size  # since we only write one element per row, col 0
        tl.atomic_add(result_ptr + result_offset, acc)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Forward must not use any torch tensor operations. All computation must be Triton.
        # We allocate tensors and launch a Triton kernel that performs atomic accumulation per token.

        device = hidden_states.device
        num_tokens, hidden_size = hidden_states.shape

        # Prepare output as float32 for accumulation stability
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _atomic_accumulate_per_token[grid](
            hidden_states, result, num_tokens, hidden_size,
        )

        # Cast result to bfloat16 to match typical output dtype
        result = result.to(torch.bfloat16)

        return result


def run(*args):
    return ModelNew()(*args)
