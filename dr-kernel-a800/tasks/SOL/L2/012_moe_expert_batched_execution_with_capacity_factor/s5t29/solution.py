import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: full computation and final weighted scatter-add.
# Launch with grid = (num_tokens,) — one program per token.
@triton.jit
def compute_and_scatter_kernel(
    hidden_states_ptr,          # *bf16, shape [num_tokens, hidden_size]
    selected_experts_ptr,       # *int32, shape [num_tokens, num_experts_per_tok]
    routing_weights_ptr,        # *bf16, shape [num_tokens, num_experts_per_tok]
    expert_gate_w_ptr,          # *bf16, shape [num_experts, hidden_size, intermediate_size]
    expert_up_w_ptr,            # *bf16, shape [num_experts, hidden_size, intermediate_size]
    expert_down_w_ptr,          # *bf16, shape [num_experts, intermediate_size, hidden_size]
    result_fp32_ptr,            # *fp32, shape [num_tokens, hidden_size]
    num_tokens: tl.constexpr,   # int
    hidden_size: tl.constexpr,  # int
    intermediate_size: tl.constexpr,  # int
    num_experts_per_tok: tl.constexpr, # int
    num_experts: tl.constexpr,  # int
    ELEMS: tl.constexpr,        # num_tokens * hidden_size
):
    token = tl.program_id(axis=0)
    # Load hidden state row for this token as fp32
    # hidden_state_row: [hidden_size] bf16 -> fp32
    hs_row_bf16 = tl.load(hidden_states_ptr + token * hidden_size + tl.arange(0, hidden_size))
    hs = hs_row_bf16.to(tl.float32)  # fp32 vector of length hidden_size

    # Loop over selected experts per token
    for j in range(0, num_experts_per_tok):
        # Load selected expert id (int32) and routing weight (bf16) for this token,j
        # Note: selected_experts_ptr is [num_tokens, num_experts_per_tok] int32
        exp_i = tl.load(selected_experts_ptr + token * num_experts_per_tok + j).to(tl.int32)
        wt_bf16 = tl.load(routing_weights_ptr + token * num_experts_per_tok + j)  # bf16 scalar
        wt = wt_bf16.to(tl.float32)  # fp32 scalar

        # Load expert weights as fp32
        # gate_out: [hidden_size, intermediate_size]
        gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        for j_g in range(0, intermediate_size):
            gate_row = tl.load(expert_gate_w_ptr + exp_i * (hidden_size * intermediate_size) + j_g * hidden_size + tl.arange(0, hidden_size))
            gate_row = gate_row.to(tl.float32)
            gate_out[:, j_g] = tl.dot(hs, gate_row)  # fp32

        # up_out: [hidden_size, intermediate_size]
        up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        for j_u in range(0, intermediate_size):
            up_row = tl.load(expert_up_w_ptr + exp_i * (hidden_size * intermediate_size) + j_u * hidden_size + tl.arange(0, hidden_size))
            up_row = up_row.to(tl.float32)
            up_out[:, j_u] = tl.dot(hs, up_row)  # fp32

        # SiLU and multiply
        activated = tl.silu(gate_out) * up_out  # fp32

        # expert_outputs: [hidden_size]
        expert_outputs = tl.zeros((hidden_size,), dtype=tl.float32)
        for k in range(0, hidden_size):
            acc = 0.0
            for j_d in range(0, intermediate_size):
                down_vec = tl.load(expert_down_w_ptr + exp_i * (intermediate_size * hidden_size) + j_d * hidden_size + tl.arange(0, hidden_size))
                down_vec = down_vec.to(tl.float32)
                acc += activated[k, j_d] * down_vec[k]  # dot over hidden dimension
            expert_outputs[k] = acc  # fp32

        # Atomic add to result[token, :]
        # result is fp32 buffer
        for k in range(0, hidden_size):
            tl.atomic_add(result_fp32_ptr + token * hidden_size + k, expert_outputs[k] * wt)

    # Done for this token
    return


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args should be: hidden_states, selected_experts, routing_weights,
        # expert_gate_weights, expert_up_weights, expert_down_weights
        hidden_states, selected_experts, routing_weights, expert_gate_weights, expert_up_weights, expert_down_weights = args

        # Ensure CUDA tensors and contiguous
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda \
               and expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
               "All tensors must be on CUDA for Triton."

        hidden_states = hidden_states.contiguous()
        selected_experts = selected_experts.contiguous().to(torch.int32)
        routing_weights = routing_weights.contiguous()
        expert_gate_weights = expert_gate_weights.contiguous()
        expert_up_weights = expert_up_weights.contiguous()
        expert_down_weights = expert_down_weights.contiguous()

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts_per_tok = selected_experts.shape[1]
        num_experts = expert_gate_weights.shape[0]
        intermediate_size = expert_gate_weights.shape[2]

        # Output as fp32 for atomic_add, cast to bf16 at the end
        result_fp32 = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        compute_and_scatter_kernel[grid](
            hidden_states, selected_experts, routing_weights,
            expert_gate_weights, expert_up_weights, expert_down_weights,
            result_fp32,
            num_tokens, hidden_size, intermediate_size, num_experts_per_tok, num_experts,
            num_tokens * hidden_size,
            num_warps=4,
        )

        # Cast to bfloat16 for return
        return result_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
