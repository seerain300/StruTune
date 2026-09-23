import triton
import triton.language as tl


@triton.jit
def bmm_row_kernel(X_ptr, W_ptr, Y_ptr,
                   H, M,
                   stride_x, stride_w_h, stride_w_m, stride_y,
                   BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    # Single-row matmul: compute Y = X @ W where X is length-H row, W is [H, M], Y is length-M
    # We launch with grid along M tiles.
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for h in range(0, H, BLOCK_H):
        offs_h = h + tl.arange(0, BLOCK_H)
        # Load X slice [BLOCK_H]
        x = tl.load(X_ptr + offs_h * stride_x, mask=offs_h < H, other=0.0)
        # Load W slice [BLOCK_H, BLOCK_M]
        w = tl.load(W_ptr + offs_h[:, None] * stride_w_h + offs_m[None, :] * stride_w_m,
                    mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
        # Multiply-accumulate
        acc += tl.sum(x[:, None] * w, axis=0)
    # Store result Y[offs_m] = acc
    tl.store(Y_ptr + offs_m * stride_y, acc, mask=offs_m < M)


@triton.jit
def silu_mul_kernel(U_ptr, V_ptr, Z_ptr, N, BLOCK: tl.constexpr):
    # Elementwise: Z = silu(U) * V
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    u = tl.load(U_ptr + offs, mask=offs < N, other=0.0)
    v = tl.load(V_ptr + offs, mask=offs < N, other=0.0)
    # silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    z = u * tl.sigmoid(u) * v
    tl.store(Z_ptr + offs, z, mask=offs < N)


@triton.jit
def atomic_accum_kernel(ROW_ptr, W_ptr, OUT_ptr, N, BLOCK: tl.constexpr):
    # OUT[row] += W * ROW
    row_id = tl.program_id(0)  # one program per row
    # Load scalar weight W_ptr is a 1-element fp32 tensor on device
    w_scalar = tl.load(W_ptr)  # scalar in fp32
    offs = tl.arange(0, BLOCK)
    row = tl.load(ROW_ptr + offs, mask=offs < N, other=0.0)  # [BLOCK]
    out_ptr_row = OUT_ptr + row_id * N
    # Atomic add each element: out += row[i] * w_scalar
    for i in range(0, N):
        out = tl.load(out_ptr_row + i)
        out = out + row[i] * w_scalar
        tl.store(out_ptr_row + i, out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        # Ensure all tensors are contiguous
        hidden_states = hidden_states.contiguous()  # [num_tokens, hidden_size], bfloat16
        selected_experts = selected_experts.contiguous()  # [num_tokens, num_experts_per_tok], int64
        routing_weights = routing_weights.contiguous()  # [num_tokens, num_experts_per_tok], bfloat16
        expert_gate_weights = expert_gate_weights.contiguous()  # [num_experts, hidden_size, moe_intermediate_size], bfloat16
        expert_up_weights = expert_up_weights.contiguous()  # [num_experts, hidden_size, moe_intermediate_size], bfloat16
        expert_down_weights = expert_down_weights.contiguous()  # [num_experts, moe_intermediate_size, hidden_size], bfloat16

        num_tokens, hidden_size = hidden_states.shape
        num_experts, H_g, M = expert_gate_weights.shape  # H_g == hidden_size
        _, H_u, _ = expert_up_weights.shape  # H_u == hidden_size
        _, M_d, H_out = expert_down_weights.shape  # M_d == M, H_out == hidden_size
        num_experts_per_tok = selected_experts.shape[1]

        # Output in fp32 for accumulation
        result = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=hidden_states.device)

        # Iterate tokens and selected_experts
        for t in range(num_tokens):
            # Process each selected expert j
            for j in range(num_experts_per_tok):
                e = int(selected_experts[t, j].item())
                # 1) gate_out = hidden_states[t] @ expert_gate_weights[e] -> [M], fp32
                x = hidden_states[t]  # [H_g], bfloat16
                w_gate = expert_gate_weights[e]  # [H_g, M], bfloat16
                gate_out = torch.empty((M,), dtype=torch.float32, device=hidden_states.device)  # [M], fp32
                grid = (triton.cdiv(M, 128),)
                bmm_row_kernel[grid](
                    x, w_gate, gate_out,
                    H_g, M,
                    x.stride(0), w_gate.stride(0), w_gate.stride(1), gate_out.stride(0),
                    BLOCK_M=128, BLOCK_H=64
                )

                # 2) up_out = hidden_states[t] @ expert_up_weights[e] -> [M], fp32
                w_up = expert_up_weights[e]  # [H_u, M], bfloat16
                up_out = torch.empty((M,), dtype=torch.float32, device=hidden_states.device)  # [M], fp32
                bmm_row_kernel[grid](
                    x, w_up, up_out,
                    H_u, M,
                    x.stride(0), w_up.stride(0), w_up.stride(1), up_out.stride(0),
                    BLOCK_M=128, BLOCK_H=64
                )

                # 3) activated = silu(gate_out) * up_out -> [M], fp32
                activated = torch.empty((M,), dtype=torch.float32, device=hidden_states.device)
                silu_mul_kernel[(triton.cdiv(M, 128),)](
                    gate_out, up_out, activated,
                    M, BLOCK=128
                )

                # 4) final_out = activated @ expert_down_weights[e] -> [H_out], fp32
                w_down = expert_down_weights[e]  # [M_d, H_out], bfloat16
                x2 = activated  # [M], fp32
                final_out = torch.empty((H_out,), dtype=torch.float32, device=hidden_states.device)  # [H_out], fp32
                grid2 = (triton.cdiv(H_out, 128),)
                bmm_row_kernel[grid2](
                    x2, w_down, final_out,
                    M, H_out,
                    x2.stride(0), w_down.stride(0), w_down.stride(1), final_out.stride(0),
                    BLOCK_M=128, BLOCK_H=64
                )

                # 5) Accumulate: result[t] += routing_weights[t, e] * final_out
                # routing_weights[t, j] is fp16 (bfloat16). Cast to fp32 for compute.
                w_scalar = float(routing_weights[t, j].item())
                w_scalar_t = torch.tensor(w_scalar, dtype=torch.float32, device=hidden_states.device)
                atomic_accum_kernel[(1,)](
                    final_out, w_scalar_t, result[t], H_out, BLOCK=128
                )

        # Cast result to bfloat16 to match original model's output dtype
        result = result.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
