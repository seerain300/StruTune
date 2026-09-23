import torch
import triton
import triton.language as tl


@triton.jit
def triton_row_matmul(C_ptr,        # *bfloat16, output vector [H]
                      A_ptr,         # *bfloat16, input row [H]
                      B_ptr,         # *bfloat16, matrix [H, H] row-major
                      H: tl.constexpr,
                      BLOCK: tl.constexpr):
    # Compute C = A_row @ B. One program per (token, expert) pair.
    acc = tl.zeros([H], dtype=tl.float32)

    for kk in range(0, H, BLOCK):
        k_offsets = kk + tl.arange(0, BLOCK)  # [BLOCK], along columns of A/B
        k_mask = k_offsets < H

        # Load A_row[kk : kk+BLOCK] as vector
        a_vec = tl.load(A_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK]

        # For each jj tile, load B[k_offsets, jj:jj+BLOCK] as [BLOCK, BLOCK]
        for jj in range(0, H, BLOCK):
            j_offsets = jj + tl.arange(0, BLOCK)  # [BLOCK]
            j_mask = j_offsets < H

            b_tile = tl.load(
                B_ptr + k_offsets[:, None] * H + j_offsets[None, :],
                mask=k_mask[:, None] & j_mask[None, :],
                other=0.0
            ).to(tl.float32)  # [BLOCK, BLOCK]

            # Accumulate outer-product contributions
            for b_col in range(0, BLOCK):
                a_val = a_vec[b_col]            # scalar
                b_row = b_tile[b_col, :]       # vector [BLOCK]
                acc += a_val * b_row

    # Store result as bfloat16
    for i in range(0, H):
        tl.store(C_ptr + i, acc[i].to(tl.bfloat16))


@triton.jit
def triton_silu(x_ptr, out_ptr, H: tl.constexpr):
    # Elementwise SiLU: y = x * sigmoid(x)
    for i in range(0, H):
        x = tl.load(x_ptr + i).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(out_ptr + i, y.to(tl.bfloat16))


@triton.jit
def triton_mul(a_ptr, b_ptr, out_ptr, H: tl.constexpr):
    # Elementwise multiply two vectors of length H
    for i in range(0, H):
        a = tl.load(a_ptr + i).to(tl.float32)
        b = tl.load(b_ptr + i).to(tl.float32)
        tl.store(out_ptr + i, (a * b).to(tl.bfloat16))


@triton.jit
def triton_atomic_add_weighted(weight_ptr, vec_ptr, out_ptr, N: tl.constexpr):
    # Atomic add weight * vec into out (flattened length N)
    for i in range(0, N):
        weight = tl.load(weight_ptr + i).to(tl.float32)
        vec = tl.load(vec_ptr + i).to(tl.float32)
        old = tl.load(out_ptr + i)
        new = old + weight * vec
        tl.store(out_ptr + i, new)


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
        hidden_states: [num_tokens, hidden_size], bfloat16
        selected_experts: [num_tokens, num_experts_per_tok], long
        routing_weights: [num_tokens, num_experts_per_tok], bfloat16
        expert_gate_weights: [num_experts, hidden_size, hidden_size], bfloat16
        expert_up_weights: [num_experts, hidden_size, hidden_size], bfloat16
        expert_down_weights: [num_experts, hidden_size, hidden_size], bfloat16
        """
        num_tokens, hidden_size = hidden_states.shape

        # Prepare output
        result = torch.zeros((num_tokens, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)

        # Iterate over tokens and selected experts
        for t in range(num_tokens):
            K = int(selected_experts.shape[1])  # num_experts_per_tok
            for j in range(K):
                exp = int(selected_experts[t, j].item())
                weight = float(routing_weights[t, j].item())

                # Hidden input for this token as 1D
                hidden_row = hidden_states[t, :].to(torch.bfloat16).contiguous()  # [H]

                # Load expert weights for this expert
                gate_w = expert_gate_weights[exp, :, :].to(torch.bfloat16).contiguous()  # [H, H]
                up_w = expert_up_weights[exp, :, :].to(torch.bfloat16).contiguous()     # [H, H]
                down_w = expert_down_weights[exp, :, :].to(torch.bfloat16).contiguous() # [H, H]

                # Buffers
                gate_out = torch.empty((hidden_size,), dtype=torch.bfloat16, device=hidden_states.device)
                up_out = torch.empty((hidden_size,), dtype=torch.bfloat16, device=hidden_states.device)
                gate_silu = torch.empty((hidden_size,), dtype=torch.bfloat16, device=hidden_states.device)
                activated = torch.empty((hidden_size,), dtype=torch.bfloat16, device=hidden_states.device)
                expert_out = torch.empty((hidden_size,), dtype=torch.bfloat16, device=hidden_states.device)

                # Compute gate_out = hidden_row @ gate_w
                triton_row_matmul[(1,)](
                    gate_out, hidden_row, gate_w, H=hidden_size, BLOCK=64, num_warps=4
                )

                # Compute up_out = hidden_row @ up_w
                triton_row_matmul[(1,)](
                    up_out, hidden_row, up_w, H=hidden_size, BLOCK=64, num_warps=4
                )

                # SiLU on gate_out
                triton_silu[(hidden_size,)](
                    gate_out, gate_silu, H=hidden_size
                )

                # Elementwise multiply: activated = SiLU(gate_out) * up_out
                triton_mul[(hidden_size,)](
                    gate_silu, up_out, activated, H=hidden_size
                )

                # Compute expert_outputs = activated @ down_w
                triton_row_matmul[(1,)](
                    expert_out, activated, down_w, H=hidden_size, BLOCK=64, num_warps=4
                )

                # Atomic add weight * expert_out to result[t, :]
                # Flatten result to 1D for atomic adds
                triton_atomic_add_weighted[(num_tokens * hidden_size,)](
                    (weight * expert_out), expert_out, result.view(-1), N=num_tokens * hidden_size
                )

        return result


def run(*args):
    return ModelNew()(*args)
