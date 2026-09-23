import triton
import triton.language as tl


@triton.jit
def bmm_triton_vec(X_ptr, W_ptr, Y_ptr,
                    H, M,
                    stride_x_b, stride_x_h,
                    stride_w_h, stride_w_m,
                    stride_y_b, stride_y_m,
                    BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr):
    # Compute Y[b, m] = sum_h X[b, h] * W[h, m], here b=0, single batch element
    for m in range(0, M, BLOCK_M):
        m_offsets = m + tl.arange(0, BLOCK_M)
        acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            x = tl.load(X_ptr + 0 * stride_x_b + h_offsets * stride_x_h, mask=h_offsets < H, other=0.0)  # [BLOCK_H]
            w = tl.load(W_ptr + h_offsets[:, None] * stride_w_h + m_offsets[None, :] * stride_w_m,
                        mask=(h_offsets[:, None] < H) & (m_offsets[None, :] < M), other=0.0)  # [BLOCK_H, BLOCK_M]
            # Reduce along H axis: x is [BLOCK_H], w is [BLOCK_H, BLOCK_M] -> sum_h x[h] * w[h, m] -> [BLOCK_M]
            acc += tl.sum(x[:, None] * w, axis=0)
        tl.store(Y_ptr + 0 * stride_y_b + m_offsets * stride_y_m, acc, mask=m_offsets < M)


@triton.jit
def elementwise_silu_mul_triton(Z_ptr, U_ptr, Y_ptr, M, BLOCK: tl.constexpr):
    # Y = SiLU(Z) * U, SiLU(Z) = Z * sigmoid(Z), sigmoid(Z) = 1 / (1 + exp(-Z))
    for m in range(0, M, BLOCK):
        offsets = m + tl.arange(0, BLOCK)
        mask = offsets < M
        z = tl.load(Z_ptr + offsets, mask=mask, other=0.0)
        u = tl.load(U_ptr + offsets, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-z))
        y = (z * s) * u
        tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def bmm_triton_vec_down(Act_ptr, Wd_ptr, Y_ptr,
                         M, H_out,
                         stride_act_m, stride_act_h,
                         stride_wd_m, stride_wd_h,
                         stride_y_b, stride_y_h,
                         BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    # Compute Y[b, h] = sum_m Act[m, b] * Wd[m, h], here b=0, single batch element
    for h in range(0, H_out, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_offsets = m + tl.arange(0, BLOCK_M)
            act = tl.load(Act_ptr + m_offsets * stride_act_m + 0 * stride_act_h, mask=m_offsets < M, other=0.0)  # [BLOCK_M]
            wd = tl.load(Wd_ptr + m_offsets[:, None] * stride_wd_m + h_offsets[None, :] * stride_wd_h,
                         mask=(m_offsets[:, None] < M) & (h_offsets[None, :] < H_out), other=0.0)  # [BLOCK_M, BLOCK_H]
            acc += tl.sum(act[:, None] * wd, axis=0)
        tl.store(Y_ptr + 0 * stride_y_b + h_offsets * stride_y_h, acc, mask=h_offsets < H_out)


@triton.jit
def atomic_add_weighted_vec(out_ptr, add_ptr, weight, H,
                             stride_out_n, stride_out_h,
                             BLOCK_H: tl.constexpr):
    # Accumulate into out[n, :] += weight * add, using atomic add per row element
    # We assume out_ptr points to a 1D contiguous buffer of length num_tokens * hidden_size.
    # But since we don't have num_tokens here, we operate row-wise by passing stride_out_n and H.
    # Weight is scalar float32, add_ptr is [H], out_ptr is [num_tokens, H] row-wise.
    # We need num_tokens to iterate; here we assume the caller will launch per token.
    # This kernel is meant to be launched per token; we pass grid size accordingly.
    n = tl.program_id(0)  # token id
    for h in range(0, H, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask = h_offsets < H
        add_vals = tl.load(add_ptr + h_offsets, mask=mask, other=0.0)
        out_vals = tl.load(out_ptr + n * H + h_offsets, mask=mask, other=0.0)
        out_vals += weight * add_vals
        tl.store(out_ptr + n * H + h_offsets, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Triton-only forward: no torch ops on tensors
        device = hidden_states.device
        num_tokens, hidden_size = hidden_states.shape
        num_experts, hidden_in, intermediate = expert_gate_weights.shape
        assert hidden_in == hidden_size, "hidden_size mismatch between hidden_states and expert_gate_weights"

        # Cast inputs to float32 for computation in Triton
        hidden_states_f32 = hidden_states.to(torch.float32)           # [num_tokens, hidden_size]
        routing_weights_f32 = routing_weights.to(torch.float32)       # [num_tokens, num_experts_per_tok]
        expert_gate_weights_f32 = expert_gate_weights.to(torch.float32)  # [num_experts, hidden_size, intermediate]
        expert_up_weights_f32 = expert_up_weights.to(torch.float32)      # [num_experts, hidden_size, intermediate]
        expert_down_weights_f32 = expert_down_weights.to(torch.float32)  # [num_experts, intermediate, hidden_size]

        # Prepare output buffer as float32
        result_f32 = torch.empty(num_tokens, hidden_size, dtype=torch.float32, device=device)

        # Loop over tokens and selected_experts
        num_experts_per_tok = selected_experts.shape[1]
        selected_experts_f32 = selected_experts.to(torch.int32)  # kernel expects int indices for exp

        for t in range(num_tokens):
            hidden_vec = hidden_states_f32[t]  # [hidden_size], contiguous
            for j in range(num_experts_per_tok):
                e = int(selected_experts_f32[t, j].item())
                # 1) gate_out = hidden_vec @ expert_gate_weights[e]
                gate_out = torch.empty(intermediate, dtype=torch.float32, device=device)
                # Launch Triton bmm for gate
                bmm_triton_vec[(
                    triton.cdiv(hidden_size, 64), triton.cdiv(intermediate, 64)
                )](
                    hidden_vec, expert_gate_weights_f32[e], gate_out,
                    hidden_size, intermediate,
                    1, 1,                  # stride_x_b, stride_x_h
                    1, intermediate,      # stride_w_h, stride_w_m
                    1, 1,                 # stride_y_b, stride_y_m
                    BLOCK_H=64, BLOCK_M=64,
                )

                # 2) up_out = hidden_vec @ expert_up_weights[e]
                up_out = torch.empty(intermediate, dtype=torch.float32, device=device)
                bmm_triton_vec[(
                    triton.cdiv(hidden_size, 64), triton.cdiv(intermediate, 64)
                )](
                    hidden_vec, expert_up_weights_f32[e], up_out,
                    hidden_size, intermediate,
                    1, 1,
                    1, intermediate,
                    1, 1,
                    BLOCK_H=64, BLOCK_M=64,
                )

                # 3) activated = SiLU(gate_out) * up_out (elementwise, Triton)
                activated = torch.empty(intermediate, dtype=torch.float32, device=device)
                elementwise_silu_mul_triton[(triton.cdiv(intermediate, 128),)](
                    gate_out, up_out, activated,
                    intermediate, BLOCK=128,
                )

                # 4) final_out = activated @ expert_down_weights[e]
                final_out = torch.empty(hidden_size, dtype=torch.float32, device=device)
                bmm_triton_vec_down[(
                    triton.cdiv(intermediate, 64), triton.cdiv(hidden_size, 64)
                )](
                    activated, expert_down_weights_f32[e], final_out,
                    intermediate, hidden_size,
                    1, 1,                  # stride_act_m, stride_act_h
                    intermediate, hidden_size,  # stride_wd_m, stride_wd_h
                    1, 1,                 # stride_y_b, stride_y_h
                    BLOCK_M=64, BLOCK_H=64,
                )

                # 5) Accumulate result[t] += routing_weights[t, e] * final_out
                weight = float(routing_weights_f32[t, j].item())
                atomic_add_weighted_vec[(num_tokens,)](
                    result_f32, final_out, weight, hidden_size,
                    hidden_size, 1,  # stride_out_n = hidden_size, stride_out_h = 1
                    BLOCK_H=64,
                )

        # Return result cast to bfloat16 to match original dtype expectations
        return result_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
