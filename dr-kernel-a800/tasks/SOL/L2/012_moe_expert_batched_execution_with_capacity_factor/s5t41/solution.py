import math
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_forward_kernel(
    hidden_states_ptr,                # *fp32, flattened [num_tokens * hidden_size]
    selected_experts_ptr,             # *int64, flattened [num_tokens * num_experts_per_tok]
    routing_weights_ptr,              # *fp32, flattened [num_tokens * num_experts_per_tok]
    expert_gate_weights_ptr,          # *fp32, flattened [num_experts * hidden_size * intermediate_size]
    expert_up_weights_ptr,            # *fp32, flattened [num_experts * hidden_size * intermediate_size]
    expert_down_weights_ptr,          # *fp32, flattened [num_experts * intermediate_size * hidden_size]
    result_ptr,                       # *fp32, flattened [num_tokens * hidden_size]
    num_tokens: tl.constexpr,         # int
    hidden_size: tl.constexpr,        # int
    num_experts_per_tok: tl.constexpr,# int
    intermediate_size: tl.constexpr,  # int
    H: tl.constexpr,                  # alias for hidden_size
    K: tl.constexpr,                  # alias for intermediate_size
    BLOCK_H: tl.constexpr = 128,      # tile size along hidden dimension
    BLOCK_K: tl.constexpr = 64,       # tile size along intermediate dimension
):
    pid = tl.program_id(axis=0)  # token id
    # Guard if pid >= num_tokens (not necessary if grid = (num_tokens,))
    if pid >= num_tokens:
        return

    # Process each selected expert j for this token
    for j in range(0, num_experts_per_tok):
        # Load selected expert id and routing weight
        # Note: selected_experts is flattened as [num_tokens * num_experts_per_tok]
        expert_id = tl.load(selected_experts_ptr + pid * num_experts_per_tok + j)
        # routing_weights_ptr is flattened as [num_tokens * num_experts_per_tok]
        weight = tl.load(routing_weights_ptr + pid * num_experts_per_tok + j)

        # Load hidden state row for this token (convert to fp32 for computation)
        # hidden_states_ptr is flattened as [num_tokens * hidden_size]
        row_start = pid * hidden_size
        hidden_row = tl.zeros((H,), dtype=tl.float32)
        # Copy row into a vector using a loop
        for hh in range(0, H):
            hidden_row[hh] = tl.load(hidden_states_ptr + row_start + hh)

        # Compute gate_out = hidden_row @ expert_gate_weights[expert_id]
        # gate_out shape: [H, K]
        gate_out = tl.zeros((H, K), dtype=tl.float32)
        # Loop over hidden_size rows for gate weights: gate weights are [H, K] per expert
        # Compute offsets into expert_gate_weights_ptr which is flattened [num_experts * H * K]
        # For a given expert, gate weight rows are contiguous in K for each hh
        # gate weight per row hh: offset = expert_id * (H * K) + hh * K + kk
        for hh in range(0, H):
            for kk in range(0, K):
                # offset for gate weight row hh, col kk
                gate_offset = expert_id * (H * K) + hh * K + kk
                # gate weight scalar
                gate_weight = tl.load(expert_gate_weights_ptr + gate_offset)
                # hidden_row[hh] scalar
                hidden_val = hidden_row[hh]
                gate_out[hh, kk] = hidden_val * gate_weight

        # Compute up_out = hidden_row @ expert_up_weights[expert_id]
        up_out = tl.zeros((H, K), dtype=tl.float32)
        for hh in range(0, H):
            for kk in range(0, K):
                up_offset = expert_id * (H * K) + hh * K + kk
                up_weight = tl.load(expert_up_weights_ptr + up_offset)
                up_out[hh, kk] = hidden_row[hh] * up_weight

        # SiLU(x) = x * sigmoid(x)
        silu_gate = gate_out * tl.sigmoid(gate_out)

        # activated = SiLU(gate_out) * up_out
        activated = tl.zeros((H, K), dtype=tl.float32)
        for hh in range(0, H):
            for kk in range(0, K):
                activated[hh, kk] = silu_gate[hh, kk] * up_out[hh, kk]

        # expert_outputs = activated @ expert_down_weights[expert_id]
        # down weights per expert are [K, H], flattened as [num_experts * K * H]
        expert_outputs = tl.zeros((H,), dtype=tl.float32)
        for hh in range(0, H):
            # inner dot over K
            inner = tl.zeros((), dtype=tl.float32)
            for kk in range(0, K):
                down_offset = expert_id * (K * H) + kk * H + hh
                down_weight = tl.load(expert_down_weights_ptr + down_offset)
                inner += activated[hh, kk] * down_weight
            expert_outputs[hh] = inner

        # Atomic add weighted contribution into result (fp32)
        # result_ptr is flattened [num_tokens * hidden_size]
        # For this token, add across hidden_size positions
        # Atomic add requires integer offsets, here positions are contiguous
        for hh in range(0, H):
            result_offset = pid * hidden_size + hh
            # weight is scalar, expert_outputs[hh] is scalar
            contrib = weight * expert_outputs[hh]
            # Atomic add into result
            tl.atomic_add(result_ptr + result_offset, contrib)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        Triton-only forward: compute the same result as the original run function
        using Triton kernels. No torch tensor operations in forward.
        """
        # Ensure tensors are on CUDA and contiguous
        device = hidden_states.device
        assert device.type == 'cuda', "This Triton implementation requires CUDA tensors."

        # Work in fp32 for numeric stability and to support Triton math
        hidden_states_fp32 = hidden_states.contiguous().to(torch.float32)
        selected_experts_i64 = selected_experts.contiguous()
        routing_weights_fp32 = routing_weights.contiguous().to(torch.float32)
        expert_gate_weights_fp32 = expert_gate_weights.contiguous().to(torch.float32)
        expert_up_weights_fp32 = expert_up_weights.contiguous().to(torch.float32)
        expert_down_weights_fp32 = expert_down_weights.contiguous().to(torch.float32)

        # Flatten for pointer-based access in Triton
        num_tokens, hidden_size = hidden_states_fp32.shape
        num_experts_per_tok = selected_experts_i64.shape[1]
        num_experts_gate = expert_gate_weights_fp32.shape[0]
        intermediate_size = expert_gate_weights_fp32.shape[2]

        # Output result buffer in fp32: [num_tokens, hidden_size]
        result_fp32 = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=device)

        # Launch Triton kernel: 1D grid over tokens
        grid = (num_tokens,)
        _moe_forward_kernel[grid](
            hidden_states_fp32.reshape(-1),                      # [num_tokens * hidden_size]
            selected_experts_i64.reshape(-1),                   # [num_tokens * num_experts_per_tok]
            routing_weights_fp32.reshape(-1),                   # [num_tokens * num_experts_per_tok]
            expert_gate_weights_fp32.reshape(-1),               # [num_experts * hidden_size * intermediate_size]
            expert_up_weights_fp32.reshape(-1),                 # [num_experts * hidden_size * intermediate_size]
            expert_down_weights_fp32.reshape(-1),               # [num_experts * intermediate_size * hidden_size]
            result_fp32.reshape(-1),                           # [num_tokens * hidden_size]
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            H=hidden_size,
            K=intermediate_size,
            num_warps=4,
            num_stages=2,
        )

        # Return result cast to bfloat16 to match original dtype
        return result_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
