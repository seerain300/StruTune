import torch
import triton
import triton.language as tl


@triton.jit
def _moe_forward_triton_kernel(
    hidden_states_ptr,               # *fp32 [num_tokens * hidden_size] flattened row-major
    selected_experts_ptr,            # *int64 [num_tokens * num_experts_per_tok] flattened
    routing_weights_ptr,             # *fp32 [num_tokens * num_experts_per_tok] flattened
    expert_gate_weights_ptr,         # *fp32 [num_experts * hidden_size * intermediate_size] flattened
    expert_up_weights_ptr,           # *fp32 [num_experts * hidden_size * intermediate_size] flattened
    expert_down_weights_ptr,         # *fp32 [num_experts * intermediate_size * hidden_size] flattened
    result_ptr,                      # *fp32 [num_tokens * hidden_size] flattened
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    intermediate_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # token index
    if pid >= num_tokens:
        return

    token_offset_hs = pid * hidden_size

    # Iterate over selected experts for this token
    for j in range(num_experts_per_tok):
        # Load selected expert id (int64), and routing weight (fp32)
        expert_id = tl.load(selected_experts_ptr + pid * num_experts_per_tok + j)
        weight = tl.load(routing_weights_ptr + pid * num_experts_per_tok + j)

        # Load hidden_state row for this token (fp32), vector of length hidden_size
        hs = tl.load(hidden_states_ptr + token_offset_hs + tl.arange(0, hidden_size), mask=None, other=0.0)

        # Compute gate_out = hs @ expert_gate_weights[expert_id]
        # gate_out shape: [hidden_size, intermediate_size]
        gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        for i in range(hidden_size):
            for k in range(intermediate_size):
                acc = 0.0
                # expert_gate_weights_ptr is flattened; for fixed expert_id, row i across k dimension:
                # base = expert_id * (hidden_size * intermediate_size) + i * intermediate_size + k
                for p in range(hidden_size):
                    coeff = tl.load(expert_gate_weights_ptr + expert_id * (hidden_size * intermediate_size) + p * intermediate_size + k)
                    acc += hs[p] * coeff
                gate_out[i, k] = acc

        # Compute up_out = hs @ expert_up_weights[expert_id]
        up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        for i in range(hidden_size):
            for k in range(intermediate_size):
                acc = 0.0
                for p in range(hidden_size):
                    coeff = tl.load(expert_up_weights_ptr + expert_id * (hidden_size * intermediate_size) + p * intermediate_size + k)
                    acc += hs[p] * coeff
                up_out[i, k] = acc

        # SiLU(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
        silu_gate = gate_out * (1.0 / (1.0 + tl.exp(-gate_out)))
        activated = silu_gate * up_out

        # Compute final output vector for this token: expert_outputs = activated @ expert_down_weights[expert_id]
        # activated shape: [hidden_size, intermediate_size], down_weights shape: [intermediate_size, hidden_size]
        expert_outputs = tl.zeros((hidden_size,), dtype=tl.float32)
        for i in range(hidden_size):
            acc = 0.0
            for k in range(intermediate_size):
                # activated[i, k] * down_weights[k, i]
                coeff = tl.load(expert_down_weights_ptr + expert_id * (intermediate_size * hidden_size) + k * hidden_size + i)
                acc += activated[i, k] * coeff
            expert_outputs[i] = acc

        # Atomic add weighted contribution into result
        for i in range(hidden_size):
            tl.atomic_add(result_ptr + pid * hidden_size + i, expert_outputs[i] * weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, selected_experts: torch.Tensor, routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor, expert_up_weights: torch.Tensor, expert_down_weights: torch.Tensor) -> torch.Tensor:
        # All tensors must be on CUDA; forward will not use any torch ops for computation.
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda \
               and expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, "All tensors must be on CUDA."

        # Ensure contiguous and compute in fp32
        hidden_states_fp32 = hidden_states.contiguous().to(torch.float32)
        selected_experts_i64 = selected_experts.contiguous()  # keep int64
        routing_weights_fp32 = routing_weights.contiguous().to(torch.float32)
        expert_gate_weights_fp32 = expert_gate_weights.contiguous().to(torch.float32)
        expert_up_weights_fp32 = expert_up_weights.contiguous().to(torch.float32)
        expert_down_weights_fp32 = expert_down_weights.contiguous().to(torch.float32)

        num_tokens = hidden_states_fp32.shape[0]
        hidden_size = hidden_states_fp32.shape[1]
        num_experts_per_tok = selected_experts_i64.shape[1]
        intermediate_size = expert_gate_weights_fp32.shape[2]

        # Flatten hidden states for row loads
        hidden_states_flat = hidden_states_fp32.reshape(-1)

        # Output buffer in fp32: [num_tokens, hidden_size] flattened
        result_fp32 = torch.zeros((num_tokens * hidden_size,), dtype=torch.float32, device=hidden_states_fp32.device)

        # Launch Triton kernel: 1D grid over tokens
        grid = (num_tokens,)
        _moe_forward_triton_kernel[grid](
            hidden_states_flat,
            selected_experts_i64.reshape(-1),
            routing_weights_fp32.reshape(-1),
            expert_gate_weights_fp32.reshape(-1),
            expert_up_weights_fp32.reshape(-1),
            expert_down_weights_fp32.reshape(-1),
            result_fp32,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            num_warps=1,
            num_stages=1,
        )

        # Return result cast to bfloat16 to match original dtype
        return result_fp32.reshape(num_tokens, hidden_size).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
