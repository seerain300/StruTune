import math
import torch
import triton
import triton.language as tl


# Triton kernel: for each token t and expert j, compute gate_out, up_out, activated, final_out,
# multiply by routing weight, and atomic-add into result[t, :].
@triton.jit
def _moe_forward_kernel(
    hidden_states_ptr,                # *bf16 [num_tokens, hidden_size]
    selected_experts_ptr,             # *int64 [num_tokens, num_experts_per_tok]
    routing_weights_ptr,              # *bf16 [num_tokens, num_experts_per_tok]
    expert_gate_weights_ptr,          # *bf16 [num_experts, hidden_size, intermediate_size]
    expert_up_weights_ptr,            # *bf16 [num_experts, hidden_size, intermediate_size]
    expert_down_weights_ptr,          # *bf16 [num_experts, intermediate_size, hidden_size]
    result_ptr,                       # *fp32 [num_tokens, hidden_size]
    num_tokens: tl.constexpr,         # int
    hidden_size: tl.constexpr,        # int
    num_experts_per_tok: tl.constexpr, # int
    intermediate_size: tl.constexpr,  # int
    H: tl.constexpr,                  # hidden_size (alias)
    M: tl.constexpr,                  # intermediate_size
):
    t = tl.program_id(0)              # token index
    j = tl.program_id(1)              # selected expert index

    # Guard if grid exceeds num_tokens
    if t >= num_tokens:
        return

    # Load hidden state row for token t: shape [H]
    # We treat hidden_states as 2D [num_tokens, H], row-major contiguous.
    hidden_row_ptr = hidden_states_ptr + t * H
    hidden_row = tl.load(hidden_row_ptr + tl.arange(0, H))
    # Cast to fp32 for computation
    hidden_row_fp32 = hidden_row.to(tl.float32)

    # Load selected expert id for this (t, j)
    se_ptr = selected_experts_ptr + t * num_experts_per_tok + j
    expert_id = tl.load(se_ptr)  # int64

    # Compute base offsets for expert weights
    # Gate weight: [M, H], Up weight: [M, H], Down weight: [H, M]
    # Cast expert_id to int64 for pointer arithmetic
    expert_id64 = tl.cast(expert_id, tl.int64)

    # Gate: hidden_row @ gate_weight[expert_id, :]
    gate_weight_ptr = expert_gate_weights_ptr + expert_id64 * (H * M)  # base for this expert
    gate_weight = tl.load(gate_weight_ptr + tl.arange(0, M)[:, None] * H + tl.arange(0, H)[None, :], mask=True)
    gate_weight_fp32 = gate_weight.to(tl.float32)

    # Up: hidden_row @ up_weight[expert_id, :]
    up_weight_ptr = expert_up_weights_ptr + expert_id64 * (H * M)
    up_weight = tl.load(up_weight_ptr + tl.arange(0, M)[:, None] * H + tl.arange(0, H)[None, :], mask=True)
    up_weight_fp32 = up_weight.to(tl.float32)

    # Compute gate_out and up_out: elementwise inner product
    # gate_out[m] = sum_k hidden_row[k] * gate_weight[m, k]
    gate_out = tl.zeros((M,), dtype=tl.float32)
    for k in range(0, H):
        gate_out += hidden_row_fp32[k] * gate_weight_fp32[:, k]

    # up_out[m] = sum_k hidden_row[k] * up_weight[m, k]
    up_out = tl.zeros((M,), dtype=tl.float32)
    for k in range(0, H):
        up_out += hidden_row_fp32[k] * up_weight_fp32[:, k]

    # SiLU(x) = x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-gate_out))
    activated = gate_out * sig
    activated = activated * up_out

    # Down: activated @ down_weight[expert_id, :]
    down_weight_ptr = expert_down_weights_ptr + expert_id64 * (M * H)
    down_weight = tl.load(down_weight_ptr + tl.arange(0, H)[:, None] * M + tl.arange(0, M)[None, :], mask=True)
    down_weight_fp32 = down_weight.to(tl.float32)

    final_out = tl.zeros((H,), dtype=tl.float32)
    for m in range(0, M):
        final_out += activated[m] * down_weight_fp32[m, :]

    # Load routing weight for this (t, j)
    rw_ptr = routing_weights_ptr + t * num_experts_per_tok + j
    rw = tl.load(rw_ptr)  # bf16
    rw_fp32 = rw.to(tl.float32)

    # Atomic add contribution to result[t, :]
    res_ptr = result_ptr + t * H
    for h in range(0, H):
        tl.atomic_add(res_ptr + h, final_out[h] * rw_fp32)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, axes_and_scalars: dict):
        # Extract axes
        num_tokens = int(axes_and_scalars["num_tokens"])
        hidden_size = int(axes_and_scalars["hidden_size"])
        moe_intermediate_size = int(axes_and_scalars["moe_intermediate_size"])
        num_experts = int(axes_and_scalars["num_experts"])
        num_experts_per_tok = int(axes_and_scalars["num_experts_per_tok"])

        # Create input tensors (ensure CUDA and correct dtype)
        device = torch.device("cuda")
        dtype_bf16 = torch.bfloat16

        # hidden_states: [num_tokens, hidden_size], bfloat16
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype_bf16, device=device)

        # selected_experts: [num_tokens, num_experts_per_tok], int64
        selected_experts = torch.zeros(num_tokens, num_experts_per_tok, dtype=torch.int64, device=device)
        for i in range(num_tokens):
            perm = torch.randperm(num_experts, device=device)[:num_experts_per_tok]
            selected_experts[i] = perm

        # routing_weights: [num_tokens, num_experts_per_tok], bfloat16
        routing_logits = torch.randn(num_tokens, num_experts_per_tok, dtype=dtype_bf16, device=device)
        routing_weights = routing_logits  # already bfloat16, no softmax

        # Expert weights: [num_experts, hidden_size, intermediate_size], bfloat16
        expert_gate_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype_bf16, device=device)
        expert_up_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype_bf16, device=device)
        expert_down_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, dtype=dtype_bf16, device=device)

        # Ensure contiguity
        hidden_states = hidden_states.contiguous()
        selected_experts = selected_experts.contiguous()
        routing_weights = routing_weights.contiguous()
        expert_gate_weights = expert_gate_weights.contiguous()
        expert_up_weights = expert_up_weights.contiguous()
        expert_down_weights = expert_down_weights.contiguous()

        # Output buffer in fp32: [num_tokens, hidden_size]
        result = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=device)

        # Launch Triton kernel: 2D grid over tokens and selected experts
        grid = (num_tokens, num_experts_per_tok)
        _moe_forward_kernel[grid](
            hidden_states,
            selected_experts,
            routing_weights,
            expert_gate_weights,
            expert_up_weights,
            expert_down_weights,
            result,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=moe_intermediate_size,
            H=hidden_size,
            M=moe_intermediate_size,
            num_warps=1,
            num_stages=1,
        )

        # Return result cast to bfloat16
        return result.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
