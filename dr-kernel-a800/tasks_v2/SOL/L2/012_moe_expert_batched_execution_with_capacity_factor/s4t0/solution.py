import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triton_dot(X_ptr, W_ptr, C_ptr,
                N, M, H,
                stride_x0, stride_x1,
                stride_w0, stride_w1,
                stride_c0, stride_c1,
                BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """
    Compute C = X @ W
    - X: [N, H], row-major with given strides
    - W: [H, M], row-major with given strides
    - C: [N, M], row-major with given strides
    All tensors are bf16. We accumulate in fp32 and store bf16.
    H is a compile-time constant for efficient unrolling.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Masks for boundaries
    n_mask = n_offsets < N
    m_mask = m_offsets < M

    # Initialize accumulator
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    # Loop over H (constexpr -> unrolled by Triton)
    for h in range(0, H):
        # Load X tile: shape [BLOCK_N, 1] -> we expand later via broadcasting
        x_ptrs = X_ptr + n_offsets[:, None] * stride_x0 + h * stride_x1
        x = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)  # bf16

        # Load W tile: shape [1, BLOCK_M]
        w_ptrs = W_ptr + h * stride_w0 + m_offsets[None, :] * stride_w1
        w = tl.load(w_ptrs, mask=m_mask[None, :], other=0.0)  # bf16

        # Accumulate: acc += x @ w  (x: [BN,1], w:[1,BM] => [BN,BM])
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    # Store result in C as bf16
    c_ptrs = C_ptr + n_offsets[:, None] * stride_c0 + m_offsets[None, :] * stride_c1
    c_mask = n_mask[:, None] & m_mask[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


class ModelNew(nn.Module):
    def forward(self,
                hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        Triton-optimized version that replaces torch.bmm with Triton GEMMs.
        We keep all host-side preprocessing and reductions identical to the original.
        """
        # Ensure device is CUDA for Triton
        assert hidden_states.is_cuda, "ModelNew requires CUDA tensors for Triton kernels"
        assert selected_experts.is_cuda and routing_weights.is_cuda and \
               expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
               "All tensors must be on CUDA for Triton kernels"

        # Same preprocessing as original
        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]

        # Flatten and sort by selected expert for stable ordering
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        flat_token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(num_experts_per_tok)

        # Stable sort by selected_experts
        sorted_experts, sorted_indices = flat_experts.sort(stable=True)
        sorted_weights = flat_weights[sorted_indices]
        sorted_token_ids = flat_token_ids[sorted_indices]

        # Capacity per expert
        # capacity = ceil( (num_tokens * num_experts_per_tok) / num_experts ) * 1.25
        capacity = max(int((num_tokens * num_experts_per_tok) / num_experts * 1.25), 1)

        # Counts of tokens per expert
        counts = torch.bincount(sorted_experts, minlength=num_experts)
        starts = torch.zeros(num_experts, dtype=torch.long, device=hidden_states.device)
        starts[1:] = counts[:-1].cumsum(0)

        # Within-expert positions
        within_pos = torch.arange(len(sorted_experts), device=hidden_states.device) - starts[sorted_experts]

        # Apply capacity (same semantics as original)
        valid = within_pos < capacity
        v_exp = sorted_experts[valid]
        v_pos = within_pos[valid]
        v_tok = sorted_token_ids[valid]
        v_wt = sorted_weights[valid]

        # Prepare expert inputs: [num_valid, hidden_size] (bf16), pad with zeros
        num_valid = v_exp.numel()
        if num_valid == 0:
            # No valid entries -> output zeros
            result = torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
            return result

        expert_inputs = torch.zeros(num_experts, capacity, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
        # Scatter valid hidden_states into expert_inputs
        expert_inputs[v_exp, v_pos] = hidden_states[v_tok]

        # Make contiguous for Triton kernels
        expert_inputs = expert_inputs.contiguous()

        # Allocate outputs for GEMMs
        gate_out = torch.empty((num_valid, moe_intermediate_size), dtype=hidden_states.dtype, device=hidden_states.device)
        up_out = torch.empty_like(gate_out)
        expert_outputs = torch.empty((num_valid, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch Triton GEMMs
        # G = hidden_inputs @ expert_gate_weights[e]
        # Note: For each expert e, we only need G for valid positions. We recompute G for each valid row here.
        # This mirrors the original per-expert computation but is more efficient than PyTorch bmm in this harness.

        # Gate GEMM: X [N, H] = expert_inputs [num_valid, hidden_size], W [H, M] = expert_gate_weights [hidden_size, moe_intermediate_size]
        N = num_valid
        H = hidden_size
        M = moe_intermediate_size

        # Grid tiling
        BLOCK_N = 64
        BLOCK_M = 64
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))

        triton_dot[grid](
            expert_inputs, expert_gate_weights,
            gate_out,
            N, M, H,
            expert_inputs.stride(0), expert_inputs.stride(1),
            expert_gate_weights.stride(0), expert_gate_weights.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M
        )

        # Up GEMM: same X and W = expert_up_weights
        triton_dot[grid](
            expert_inputs, expert_up_weights,
            up_out,
            N, M, H,
            expert_inputs.stride(0), expert_inputs.stride(1),
            expert_up_weights.stride(0), expert_up_weights.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M
        )

        # Activations: SiLU(gate_out) * up_out
        # SiLU(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
        gate_out_fp32 = gate_out.to(torch.float32)
        up_out_fp32 = up_out.to(torch.float32)
        activated = torch.nn.functional.silu(gate_out_fp32) * up_out_fp32  # [num_valid, M], float32
        # Cast to bf16 for final GEMM
        activated_bf16 = activated.to(hidden_states.dtype)

        # Final GEMM: Out = activated @ expert_down_weights[e] where expert_down_weights shape [M, H] = [moe_intermediate_size, hidden_size]
        # So C [N, H] with W [M, H]
        triton_dot[grid](
            activated_bf16, expert_down_weights,
            expert_outputs,
            N, H, M,  # N=num_valid, H=hidden_size, M=moe_intermediate_size (loop over M)
            activated_bf16.stride(0), activated_bf16.stride(1),
            expert_down_weights.stride(0), expert_down_weights.stride(1),
            expert_outputs.stride(0), expert_outputs.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M
        )

        # Weighted gather and index-add into result
        valid_out = expert_outputs[v_exp, v_pos]  # [num_valid, hidden_size]
        weighted_out = v_wt.unsqueeze(1) * valid_out  # [num_valid, hidden_size]

        result = torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
        result.index_add_(0, v_tok, weighted_out)

        return result


def run(*args):
    return ModelNew()(*args)
