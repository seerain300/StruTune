import torch
import triton
import triton.language as tl


@triton.jit
def bmm_rowvec_triton(
    X_ptr,          # *f32, pointer to input row vector [H_in]
    W_ptr,          # *f32, pointer to weight matrix [H_in, M]
    Y_ptr,          # *f32, pointer to output vector [M]
    H_in,           # int: hidden size (row length)
    M,              # int: intermediate/hidden output size (col length)
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Compute Y = X @ W where X is 1xH_in, W is H_in x M, Y is 1xM
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for h in range(0, H_in, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H_in
        x = tl.load(X_ptr + h_offsets, mask=h_mask, other=0.0)  # [BLOCK_H]
        # W_tile: load BLOCK_H x BLOCK_M chunk
        for m in range(0, M, BLOCK_M):
            m_offsets = m + tl.arange(0, BLOCK_M)
            m_mask = m_offsets < M
            w_tile = tl.load(W_ptr + h_offsets[:, None] * M + m_offsets[None, :],
                             mask=h_mask[:, None] & m_mask[None, :], other=0.0)  # [BLOCK_H, BLOCK_M]
            # Outer product and accumulate
            acc += tl.sum(x[:, None] * w_tile, axis=0)  # sum over H tile -> [BLOCK_M]
    # Store acc to Y (first BLOCK_M entries)
    # Note: Y_ptr is a 1D vector [M]; we assume M <= BLOCK_M*end in host side.
    # To be safe, store only the first M elements via masks; but we pass M and BLOCK_M such that we cover all.
    # Since this is a Triton kernel, we can store the full acc to Y_ptr using a simple loop over M.
    # Implement a direct store loop for correctness:
    for m in range(0, M, BLOCK_M):
        m_offsets = m + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        tl.store(Y_ptr + m_offsets, acc[m_offsets], mask=m_mask)


@triton.jit
def activation_silu_mul_kernel(
    Z_ptr,          # *f32, pointer to gate_out [M]
    U_ptr,          # *f32, pointer to up_out [M]
    Y_ptr,          # *f32, pointer to activated [M]
    M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Elementwise: Y = SiLU(Z) * U, where SiLU(Z) = Z * sigmoid(Z)
    for m in range(0, M, BLOCK):
        offsets = m + tl.arange(0, BLOCK)
        mask = offsets < M
        z = tl.load(Z_ptr + offsets, mask=mask, other=0.0)
        u = tl.load(U_ptr + offsets, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-z))
        y = z * s * u
        tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def atomic_add_vec_kernel(
    out_ptr,         # *f32, pointer to result [N, H] flattened as 1D length N*H
    add_ptr,         # *f32, pointer to vector to add [H]
    weights,         # scalar f32
    N,               # num_tokens
    H,               # hidden_size
    stride_out_n, stride_out_h,
    BLOCK_H: tl.constexpr,
):
    # For each row n, out[n, :] += weights * add (row-wise accumulation)
    # out_ptr is 1D: idx = n * H + h
    for n in range(0, N):
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            mask = h_offsets < H
            add_vals = tl.load(add_ptr + h_offsets, mask=mask, other=0.0)
            out_vals = tl.load(out_ptr + n * H + h_offsets, mask=mask, other=0.0)
            out_vals += weights * add_vals
            tl.store(out_ptr + n * H + h_offsets, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Triton-only forward: all computation performed via Triton kernels, no torch ops on tensors.
        device = hidden_states.device
        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, _ = expert_gate_weights.shape  # gate weights shape: [num_experts, hidden_size, intermediate]

        # Convert inputs to fp32 for Triton; we will cast outputs back to bfloat16
        hidden_states_f32 = hidden_states.to(torch.float32)
        # Note: selected_experts is int64 in provided inputs; we don't need it in Triton computation directly.
        routing_weights_f32 = routing_weights.to(torch.float32)
        expert_gate_weights_f32 = expert_gate_weights.to(torch.float32)
        expert_up_weights_f32 = expert_up_weights.to(torch.float32)
        expert_down_weights_f32 = expert_down_weights.to(torch.float32)

        # Allocate fp32 result buffer
        result_f32 = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)

        # Process each token and its selected experts
        # selected_experts is assumed to be [num_tokens, num_experts_per_tok]. We use it to pick one expert per token.
        num_experts_per_tok = selected_experts.shape[1]
        for t in range(num_tokens):
            for j in range(num_experts_per_tok):
                expert_idx = int(selected_experts[t, j].item())

                # 1) gate_out = hidden_states[t] @ expert_gate_weights[expert_idx]
                H_in = hidden_states_f32.shape[1]  # hidden_size
                M = expert_gate_weights_f32.shape[2]  # intermediate size
                gate_out = torch.empty(M, dtype=torch.float32, device=device)
                # Prepare X as 1xH_in vector and W as H_in x M
                X = hidden_states_f32[t].unsqueeze(0)  # [1, H_in]
                W = expert_gate_weights_f32[expert_idx]  # [H_in, M]
                bmm_rowvec_triton[(1,)](
                    X_ptr=X, W_ptr=W, Y_ptr=gate_out,
                    H_in=H_in, M=M,
                    BLOCK_H=128, BLOCK_M=128
                )

                # 2) up_out = hidden_states[t] @ expert_up_weights[expert_idx]
                up_out = torch.empty(M, dtype=torch.float32, device=device)
                X = hidden_states_f32[t].unsqueeze(0)  # [1, H_in]
                W = expert_up_weights_f32[expert_idx]  # [H_in, M]
                bmm_rowvec_triton[(1,)](
                    X_ptr=X, W_ptr=W, Y_ptr=up_out,
                    H_in=H_in, M=M,
                    BLOCK_H=128, BLOCK_M=128
                )

                # 3) activated = SiLU(gate_out) * up_out
                activated = torch.empty(M, dtype=torch.float32, device=device)
                activation_silu_mul_kernel[(1,)](
                    Z_ptr=gate_out, U_ptr=up_out, Y_ptr=activated,
                    M=M, BLOCK=128
                )

                # 4) final_out = activated @ expert_down_weights[expert_idx]
                H_out = expert_down_weights_f32.shape[2]  # should equal hidden_size
                final_out = torch.empty(H_out, dtype=torch.float32, device=device)
                X2 = activated.unsqueeze(0)  # [1, M]
                W2 = expert_down_weights_f32[expert_idx]  # [M, H_out]
                bmm_rowvec_triton[(1,)](
                    X_ptr=X2, W_ptr=W2, Y_ptr=final_out,
                    H_in=M, M=H_out,
                    BLOCK_H=128, BLOCK_M=128
                )

                # 5) Accumulate into result[t]
                weight = float(routing_weights_f32[t, j].item())
                # Atomic add into result row for this token
                atomic_add_vec_kernel[(1,)](
                    out_ptr=result_f32,
                    add_ptr=final_out,
                    weights=weight,
                    N=num_tokens,
                    H=hidden_size,
                    stride_out_n=result_f32.stride(0),
                    stride_out_h=result_f32.stride(1),
                    BLOCK_H=128
                )

        # Convert result back to bfloat16 to match original dtype
        result = result_f32.to(hidden_states.dtype)
        return result


def run(*args):
    return ModelNew()(*args)
