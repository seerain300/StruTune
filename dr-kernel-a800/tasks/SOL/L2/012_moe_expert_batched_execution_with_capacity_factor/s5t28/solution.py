import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: full computation per (token, expert) with final weighted scatter-add.
# This kernel is launched from ModelNew.forward and performs:
# - For each token i and each selected expert j
#   - Load hidden_state[i, :]
#   - Compute gate_out = hidden_state @ expert_gate_weights[selected_exp[i, j]]
#   - Compute up_out   = hidden_state @ expert_up_weights[selected_exp[i, j]]
#   - activated = SiLU(gate_out) * up_out
#   - expert_outputs = activated @ expert_down_weights[selected_exp[i, j]]
#   - result[i, :] += routing_weights[i, j] * expert_outputs
# All in fp32, using atomic_add into a float32 result buffer; cast to bfloat16 at return.
@triton.jit
def compute_and_scatter_kernel(
    hidden_states_ptr,               # *bf16, shape [num_tokens, hidden_size]
    selected_exp_ptr,                # *int64, shape [num_tokens, num_experts_per_tok]
    routing_weights_ptr,             # *bf16, shape [num_tokens, num_experts_per_tok]
    expert_gate_w_ptr,               # *bf16, shape [num_experts, hidden_size, intermediate_size]
    expert_up_w_ptr,                 # *bf16, shape [num_experts, hidden_size, intermediate_size]
    expert_down_w_ptr,               # *bf16, shape [num_experts, intermediate_size, hidden_size]
    result_ptr,                      # *bf16, shape [num_tokens, hidden_size] (we will atomic_add into fp32 view)
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    BLOCK_TOKEN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_TOKEN + tl.arange(0, BLOCK_TOKEN)
    mask = offs < num_tokens

    # Load hidden states rows for these tokens (bf16), cast to fp32
    hs_ptr = hidden_states_ptr + offs * hidden_size
    hs = tl.load(hs_ptr + tl.arange(0, hidden_size), mask=mask, other=0.0).to(tl.float32)  # [BLOCK_TOKEN, hidden_size]

    # For each selected expert per token
    for j in range(0, num_experts_per_tok):
        # Load selected expert index for each token
        exp_idx = tl.load(selected_exp_ptr + offs * num_experts_per_tok + j, mask=mask, other=0).to(tl.int64)

        # Compute gate_out = hs @ gate_w[exp_idx]
        gate_out = tl.zeros((BLOCK_TOKEN, intermediate_size), dtype=tl.float32)
        for j2 in range(0, intermediate_size):
            gate_row_ptr = expert_gate_w_ptr + exp_idx * (hidden_size * intermediate_size) + j2 * hidden_size
            gate_row = tl.load(gate_row_ptr + tl.arange(0, hidden_size), mask=mask, other=0.0).to(tl.float32)  # [hidden_size]
            gate_out[:, j2] = tl.dot(hs, gate_row)  # [BLOCK_TOKEN]

        # Compute up_out = hs @ up_w[exp_idx]
        up_out = tl.zeros((BLOCK_TOKEN, intermediate_size), dtype=tl.float32)
        for j2 in range(0, intermediate_size):
            up_row_ptr = expert_up_w_ptr + exp_idx * (hidden_size * intermediate_size) + j2 * hidden_size
            up_row = tl.load(up_row_ptr + tl.arange(0, hidden_size), mask=mask, other=0.0).to(tl.float32)
            up_out[:, j2] = tl.dot(hs, up_row)

        # SiLU: SiLU(x) = x * sigmoid(x)
        gate_sigmoid = tl.sigmoid(gate_out)
        activated = gate_out * gate_sigmoid * up_out

        # expert_outputs = activated @ down_w[exp_idx]
        expert_outputs = tl.zeros((BLOCK_TOKEN, hidden_size), dtype=tl.float32)
        for k in range(0, hidden_size):
            acc = 0.0
            for j3 in range(0, intermediate_size):
                down_vec_ptr = expert_down_w_ptr + exp_idx * (intermediate_size * hidden_size) + j3 * hidden_size + k
                down_val = tl.load(down_vec_ptr, mask=mask, other=0.0).to(tl.float32)
                acc += activated[:, j3] * down_val
            expert_outputs[:, k] = acc

        # Weight for this token-expert pair
        wt = tl.load(routing_weights_ptr + offs * num_experts_per_tok + j, mask=mask, other=0.0).to(tl.float32)  # scalar per token
        # Atomic add into result (fp32 view)
        for k2 in range(0, hidden_size):
            out_val = expert_outputs[:, k2] * wt  # [BLOCK_TOKEN]
            result_row_ptr = result_ptr + offs * hidden_size + k2
            tl.atomic_add(result_row_ptr, out_val, mask=mask)


# ModelNew: entry point. forward does not use any torch tensor operations.
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure device is CUDA for Triton. Triton requires CUDA tensors.
        device = hidden_states.device
        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_hs, intermediate_size = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]

        # Output in fp32 for atomic_add, then cast to bfloat16 at the end
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per block of tokens
        BLOCK_TOKEN = 128
        grid = (triton.cdiv(num_tokens, BLOCK_TOKEN),)

        compute_and_scatter_kernel[grid](
            hidden_states, selected_experts, routing_weights,
            expert_gate_weights, expert_up_weights, expert_down_weights,
            result,
            num_tokens, hidden_size, intermediate_size, num_experts_per_tok,
            BLOCK_TOKEN
        )

        return result.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
