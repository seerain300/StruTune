import torch
import triton
import triton.language as tl


@triton.jit
def _compute_output_without_padding(
    hidden_states_ptr,            # *bf16, (num_tokens, hidden_size)
    selected_experts_ptr,         # *int64, (num_tokens, K)
    routing_weights_ptr,          # *bf16, (num_tokens, K)
    expert_gate_ptr,              # *bf16, (num_experts, hidden_size, intermediate_size)
    expert_up_ptr,                # *bf16, (num_experts, hidden_size, intermediate_size)
    expert_down_ptr,              # *bf16, (num_experts, intermediate_size, hidden_size)
    out_ptr,                      # *bf16, (num_tokens, hidden_size)
    num_tokens: tl.int32,
    hidden_size: tl.constexpr,    # e.g., 4096
    num_experts: tl.int32,
    intermediate_size: tl.constexpr,  # e.g., 8192
    K: tl.int32,                  # number of selected experts per token (runtime)
):
    # Each program computes one output element (token, hidden_dim)
    pid_token = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_token >= num_tokens or pid_h >= hidden_size:
        return

    # Prepare accumulators
    out_acc = tl.zeros((), dtype=tl.float32)  # scalar accumulator for this token
    hs_row_ptr = hidden_states_ptr + pid_token * hidden_size

    # Base offset for this token in flattened selected_experts and routing_weights arrays
    base = pid_token * K

    # Loop over each selected expert for this token
    exp = 0
    while exp < K:
        selected_exp_id = tl.load(selected_experts_ptr + base + exp).to(tl.int32)
        routing_w = tl.load(routing_weights_ptr + base + exp).to(tl.float32)

        # Compute gate_out_k = sum_h hidden_states[token, h] * expert_gate[expert, h, :]
        gate_out_k = tl.zeros((), dtype=tl.float32)
        for h_start in range(0, hidden_size, 64):
            h_idx = h_start + tl.arange(0, 64)
            mask_h = h_idx < hidden_size
            hs_chunk = tl.load(hs_row_ptr + h_idx, mask=mask_h, other=0.0).to(tl.float32)

            # Load gate weights chunk: shape [64] for i in [0, intermediate_size), but we reduce over i later.
            # We'll load per h and i with mask, then reduce over i.
            partial = tl.zeros((64,), dtype=tl.float32)
            for i in range(0, intermediate_size, 64):
                i_idx = i + tl.arange(0, 64)
                mask_i = i_idx < intermediate_size
                # Address for gate: selected_exp_id * (H*I) + h * I + i
                gate_off = selected_exp_id * (hidden_size * intermediate_size) + h_idx * intermediate_size + i_idx
                gate_chunk = tl.load(expert_gate_ptr + gate_off, mask=mask_h & mask_i, other=0.0).to(tl.float32)
                partial += hs_chunk * gate_chunk
            gate_out_k += tl.sum(partial, axis=0)

        # Compute up_out_k = sum_h hidden_states[token, h] * expert_up[expert, h, :]
        up_out_k = tl.zeros((), dtype=tl.float32)
        for h_start in range(0, hidden_size, 64):
            h_idx = h_start + tl.arange(0, 64)
            mask_h = h_idx < hidden_size
            hs_chunk = tl.load(hs_row_ptr + h_idx, mask=mask_h, other=0.0).to(tl.float32)

            partial = tl.zeros((64,), dtype=tl.float32)
            for i in range(0, intermediate_size, 64):
                i_idx = i + tl.arange(0, 64)
                mask_i = i_idx < intermediate_size
                up_off = selected_exp_id * (hidden_size * intermediate_size) + h_idx * intermediate_size + i_idx
                up_chunk = tl.load(expert_up_ptr + up_off, mask=mask_h & mask_i, other=0.0).to(tl.float32)
                partial += hs_chunk * up_chunk
            up_out_k += tl.sum(partial, axis=0)

        # SiLU and gated-up
        sig = 1.0 / (1.0 + tl.exp(-gate_out_k))
        activated_k = gate_out_k * sig
        activated_k = activated_k * up_out_k  # scalar

        # Compute contribution to output at hidden dimension pid_h
        contribution = tl.zeros((), dtype=tl.float32)
        # down has shape (num_experts, intermediate_size, hidden_size)
        # contribution = sum_i down[selected_exp_id, i, pid_h] * activated_k
        for i in range(0, intermediate_size, 64):
            i_idx = i + tl.arange(0, 64)
            mask_i = i_idx < intermediate_size
            down_off = selected_exp_id * (intermediate_size * hidden_size) + i_idx * hidden_size + pid_h
            down_chunk = tl.load(expert_down_ptr + down_off, mask=mask_i, other=0.0).to(tl.float32)
            contribution += tl.sum(down_chunk * activated_k, axis=0)

        out_acc += contribution * routing_w
        exp += 1

    # Store the result for this token and hidden_dim
    out_index = pid_token * hidden_size + pid_h
    tl.store(out_ptr + out_index, out_acc)  # Triton will cast to bf16 if needed


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        # Ensure CUDA tensors for Triton
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda \
               and expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."

        # Shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, intermediate_size = expert_gate_weights.shape
        K = selected_experts.shape[1]

        # Cast inputs to bfloat16 as in original
        hidden_states = hidden_states.to(torch.bfloat16)
        routing_weights = routing_weights.to(torch.bfloat16)
        selected_experts = selected_experts.to(torch.int64)  # keep int64 for exact matching

        # Output tensor
        out = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device=hidden_states.device)

        # Launch Triton kernel: grid over (num_tokens, hidden_size)
        grid = (num_tokens, hidden_size)
        _compute_output_without_padding[grid](
            hidden_states, selected_experts, routing_weights,
            expert_gate_weights, expert_up_weights, expert_down_weights,
            out,
            num_tokens,
            hidden_size, num_experts, intermediate_size, K,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
