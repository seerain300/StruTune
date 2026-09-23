import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_row_matmul(C_ptr,        # *bfloat16, flattened [num_tokens * hidden_size]
                      A_ptr,         # *bfloat16, input row [H]
                      B_ptr,         # *bfloat16, matrix [H, H]
                      H: tl.constexpr,
                      BLOCK: tl.constexpr):
    # One program handles one (token, expert) row-block and writes its C[H]
    pid = tl.program_id(axis=0)
    out_base = pid * H  # C offset for this (token, expert)

    # Initialize output accumulator
    acc = tl.zeros([H], dtype=tl.float32)

    # Tile over K (columns of A/B) in chunks of BLOCK
    for k in range(0, H, BLOCK):
        k_offsets = k + tl.arange(0, BLOCK)  # [BLOCK]
        k_mask = k_offsets < H

        # Load A_row[k:k+BLOCK]
        a = tl.load(A_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK]

        # Load B[k:k+BLOCK, :] as a matrix of shape [BLOCK, H]
        b = tl.zeros([BLOCK, H], dtype=tl.float32)
        for jj in range(0, H, BLOCK):
            j_offsets = jj + tl.arange(0, BLOCK)  # [BLOCK]
            j_mask = j_offsets < H
            # B is stored row-major: B_ptr + k_offsets[:, None] * H + j_offsets[None, :]
            b_part = tl.load(B_ptr + k_offsets[:, None] * H + j_offsets[None, :],
                             mask=k_mask[:, None] & j_mask[None, :],
                             other=0.0).to(tl.float32)
            b += b_part

        # Accumulate dot products: sum over BLOCK
        acc += tl.sum(b * a[:, None], axis=0)  # [H]

    # Store result to C (bfloat16)
    for i in range(0, H):
        tl.store(C_ptr + out_base + i, acc[i].to(tl.bfloat16))


@triton.jit
def triton_silu(x_ptr,          # *bfloat16, [H]
                out_ptr,         # *bfloat16, [H]
                H: tl.constexpr):
    for i in range(0, H):
        x = tl.load(x_ptr + i).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(out_ptr + i, y.to(tl.bfloat16))


@triton.jit
def triton_mul(a_ptr, b_ptr, out_ptr, H: tl.constexpr):
    for i in range(0, H):
        a = tl.load(a_ptr + i).to(tl.float32)
        b = tl.load(b_ptr + i).to(tl.float32)
        tl.store(out_ptr + i, (a * b).to(tl.bfloat16))


@triton.jit
def triton_atomic_weighted_add(weight_ptr,   # *bfloat16, [N], N = num_tokens * hidden_size
                               vec_ptr,      # *bfloat16, [N]
                               out_ptr,      # *bfloat16, flattened [num_tokens, hidden_size]
                               N: tl.constexpr):
    # Atomic add weight * vec into out flattened memory
    for i in range(0, N):
        w = tl.load(weight_ptr + i).to(tl.float32)
        v = tl.load(vec_ptr + i).to(tl.float32)
        o = tl.load(out_ptr + i)
        o += w * v
        tl.store(out_ptr + i, o)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        Inputs:
          - hidden_states: [num_tokens, hidden_size], bfloat16, CUDA
          - selected_experts: [num_tokens, num_experts_per_tok], long
          - routing_weights: [num_tokens, num_experts_per_tok], bfloat16
          - expert_gate_weights: [num_experts, hidden_size, intermediate_size], bfloat16
          - expert_up_weights: [num_experts, hidden_size, intermediate_size], bfloat16
          - expert_down_weights: [num_experts, intermediate_size, hidden_size], bfloat16
        Output:
          - result: [num_tokens, hidden_size], bfloat16
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda, "hidden_states must be on CUDA device"
        device = hidden_states.device
        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_in, gate_out = expert_gate_weights.shape
        _, up_in, up_out = expert_up_weights.shape
        _, down_in, down_out = expert_down_weights.shape

        # Infer intermediate_size: per reference, intermediate_size == hidden_size.
        # Original code uses num_experts_per_tok at runtime but not for preprocessing.
        # We will iterate over selected_experts[t, :] directly and compute per-expert outputs.
        # Output tensor (zeros)
        result = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

        # Prepare flattened output for atomic add
        out_flat = result.view(-1)  # [num_tokens * hidden_size]
        N = num_tokens * hidden_size

        # Loop over tokens and their selected_experts
        for t in range(num_tokens):
            # Gather selected experts for this token
            selected = selected_experts[t, :].to(torch.int32)  # Triton expects int32 indices

            # For each selected expert, compute the three GEMMs and do atomic add
            for k in range(selected.shape[0]):
                exp = int(selected[k].item())  # Triton expects scalar constants in kernel call
                if exp >= num_experts:
                    continue

                # hidden_inputs: row t
                A_row_gate = hidden_states[t, :].to(torch.bfloat16).contiguous()  # [H]
                # expert_gate_weights[exp] is [H, intermediate_size], and intermediate_size == H
                B_gate = expert_gate_weights[exp].contiguous()  # [H, H]
                B_flat_gate = B_gate.view(-1)  # [H*H]

                # Allocate intermediate C vectors (flattened)
                C_gate_flat = torch.empty(hidden_size, device=device, dtype=torch.bfloat16)

                # Launch Triton row matmul for gate_out
                triton_row_matmul[(1,)](C_gate_flat, A_row_gate, B_flat_gate, H=hidden_size, BLOCK=64)

                # A_row_up: hidden_inputs
                A_row_up = hidden_states[t, :].to(torch.bfloat16).contiguous()  # [H]
                # expert_up_weights[exp] is [H, intermediate_size], and intermediate_size == H
                B_up = expert_up_weights[exp].contiguous()  # [H, H]
                B_flat_up = B_up.view(-1)  # [H*H]
                C_up_flat = torch.empty(hidden_size, device=device, dtype=torch.bfloat16)

                triton_row_matmul[(1,)](C_up_flat, A_row_up, B_flat_up, H=hidden_size, BLOCK=64)

                # Compute activated = SiLU(gate_out)
                activated = torch.empty(hidden_size, device=device, dtype=torch.bfloat16)
                triton_silu[(hidden_size,)](C_gate_flat, activated, H=hidden_size)

                # Multiply activated and up_out: activated * up_out
                mul_result = torch.empty(hidden_size, device=device, dtype=torch.bfloat16)
                triton_mul[(hidden_size,)](activated, C_up_flat, mul_result, H=hidden_size)

                # Compute expert_outputs = mul_result @ expert_down_weights[exp]
                # expert_down_weights[exp] is [intermediate_size, hidden_size], and intermediate_size == hidden_size
                B_down = expert_down_weights[exp].contiguous()  # [H, H]
                B_flat_down = B_down.view(-1)  # [H*H]
                C_expert_flat = torch.empty(hidden_size, device=device, dtype=torch.bfloat16)

                triton_row_matmul[(1,)](C_expert_flat, mul_result, B_flat_down, H=hidden_size, BLOCK=64)

                # Atomic add routing_weights[t, k] * C_expert_flat into result
                weight_scalar = routing_weights[t, k].to(torch.bfloat16)
                # Prepare weight vector of length H with same value
                weight_vec = weight_scalar.expand(hidden_size).contiguous()
                # Atomic add into flattened result
                triton_atomic_weighted_add[(N,)](weight_vec, C_expert_flat, out_flat, N=N)

        # Return result (shape [num_tokens, hidden_size], bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
