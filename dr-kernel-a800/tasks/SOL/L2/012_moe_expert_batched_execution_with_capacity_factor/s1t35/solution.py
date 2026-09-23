import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_kernel(
    X_ptr,  # *bf16, [B, H] but we pass B=1 so 1D vector
    W_ptr,  # *bf16, [H, M]
    Y_ptr,  # *bf16, [B, M] but we pass B=1 so 1D vector
    B: tl.constexpr,           # number of rows in X (always 1 here)
    H,                         # int, length of X rows
    M,                         # int, output dimension
    BLOCK_M: tl.constexpr,     # tile size along M
):
    row = tl.program_id(0)
    if row >= B:
        return
    cols = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Loop over K dimension (H) in blocks
    for k0 in range(0, H, BLOCK_M):
        k = k0 + cols
        mask_k = k < H
        # Load X[row, k] vector (row=0 when B=1)
        x = tl.load(X_ptr + row * H + k, mask=mask_k, other=0.0)
        # Load W[k, cols] vector
        w = tl.load(W_ptr + k * M + cols, mask=mask_k, other=0.0)
        # Dot product accumulate in fp32
        acc += tl.dot(x, w, out_dtype=tl.float32)
    # Store result as bf16
    tl.store(Y_ptr + row * M + cols, acc.to(tl.bfloat16), mask=(row < B) & (cols < M))


@triton.jit
def elementwise_silu_mul_kernel(
    Z_ptr,  # *bf16 or *fp32, [N]
    U_ptr,  # *bf16 or *fp32, [N]
    Out_ptr,  # *bf16, [N]
    N,  # int
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    z = tl.load(Z_ptr + offs, mask=mask, other=0.0)
    u = tl.load(U_ptr + offs, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-z))
    y = z * s
    y = y * u
    tl.store(Out_ptr + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def atomic_accumulate_kernel(
    Result_ptr,  # *bf16, [num_tokens, hidden_size]
    Final_ptr,   # *bf16, [num_valid, hidden_size]
    Weight_ptr,  # *bf16, [num_valid]
    Indices_ptr, # *int32, [num_valid] token indices
    num_valid: tl.constexpr,
    H,  # hidden_size
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_valid
    tok = tl.load(Indices_ptr + offs, mask=mask, other=0)
    w = tl.load(Weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Load final_out vector
    final_row = Final_ptr + offs * H
    h = tl.arange(0, BLOCK)  # vector along hidden dimension
    final_vec = tl.load(final_row + h, mask=mask, other=0.0)
    # Atomic add into result[tok, :]
    result_row = Result_ptr + tok * H
    final_vec = final_vec.to(tl.float32)
    tl.atomic_add(result_row + h, final_vec * w[:, None], mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        # Triton-only forward: no torch ops on tensors
        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]

        # Output result vector
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device)

        # For each token, process all selected experts (no torch.sort/bincount)
        for t in range(num_tokens):
            # Process up to num_experts_per_tok experts per token
            for j in range(num_experts_per_tok):
                # Get selected expert index for this token
                expert = int(selected_experts[t, j].item())

                # Hidden vector for this token
                hidden_vec = hidden_states[t]  # [hidden_size], bfloat16

                # 1) gate_out = hidden_vec @ expert_gate_weights[expert]
                gate_out = torch.empty(moe_intermediate_size, dtype=torch.bfloat16, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    hidden_vec,          # X_ptr: [hidden_size]
                    expert_gate_weights[expert],  # W_ptr: [hidden_size, moe_intermediate_size]
                    gate_out,            # Y_ptr: [moe_intermediate_size]
                    1, hidden_vec.numel(), gate_out.numel(),
                    128, num_warps=4
                )

                # 2) up_out = hidden_vec @ expert_up_weights[expert]
                up_out = torch.empty(moe_intermediate_size, dtype=torch.bfloat16, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    hidden_vec,
                    expert_up_weights[expert],  # W_ptr: [hidden_size, moe_intermediate_size]
                    up_out,                  # Y_ptr: [moe_intermediate_size]
                    1, hidden_vec.numel(), up_out.numel(),
                    128, num_warps=4
                )

                # 3) activated = silu(gate_out) * up_out
                activated = torch.empty(moe_intermediate_size, dtype=torch.bfloat16, device=hidden_states.device)
                elementwise_silu_mul_kernel[(1,)](
                    gate_out,
                    up_out,
                    activated,
                    activated.numel(),
                    128, num_warps=4
                )

                # 4) final_out = activated @ expert_down_weights[expert]
                final_out = torch.empty(hidden_size, dtype=torch.bfloat16, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    activated,                   # X_ptr: [moe_intermediate_size]
                    expert_down_weights[expert], # W_ptr: [moe_intermediate_size, hidden_size]
                    final_out,                  # Y_ptr: [hidden_size]
                    1, activated.numel(), final_out.numel(),
                    128, num_warps=4
                )

                # 5) Aggregate: result[t] += routing_weights[t, expert] * final_out
                weight = routing_weights[t, expert]
                atomic_accumulate_kernel[(1,)](
                    result,
                    final_out,     # [hidden_size]
                    weight,        # scalar bfloat16
                    torch.tensor(t, dtype=torch.int32, device=hidden_states.device),  # token index
                    1,
                    hidden_size,
                    256, num_warps=4
                )

        return result


def run(*args):
    return ModelNew()(*args)
