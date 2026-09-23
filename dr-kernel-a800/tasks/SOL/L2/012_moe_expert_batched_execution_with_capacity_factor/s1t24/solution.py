import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_kernel(X_ptr, W_ptr, Y_ptr,
                      H: tl.constexpr, M: tl.constexpr,
                      stride_x0, stride_x1,
                      stride_w0, stride_w1,
                      stride_y0, stride_y1,
                      BLOCK_M: tl.constexpr):
    # Compute Y = X @ W where X: [1, H], W: [H, M], Y: [1, M]
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over H rows
    for h in range(0, H):
        x_val = tl.load(X_ptr + h * stride_x0)  # X is 1D vector
        x_val = x_val.to(tl.float32)
        w_vals = tl.load(W_ptr + h * stride_w0 + offs_m * stride_w1, mask=mask_m, other=0.0)
        w_vals = w_vals.to(tl.float32)
        acc += x_val * w_vals

    tl.store(Y_ptr + offs_m * stride_y1, acc, mask=mask_m)


@triton.jit
def silu_mul_triton_kernel(Z_ptr, U_ptr, Y_ptr,
                           M: tl.constexpr,
                           stride_z0, stride_z1,
                           stride_u0, stride_u1,
                           stride_y0, stride_y1,
                           BLOCK_M: tl.constexpr):
    # Elementwise: Y = SiLU(Z) * U, where Z and U are 1D vectors of length M
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    z = tl.load(Z_ptr + offs * stride_z1, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + offs * stride_u1, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    y = (z * sigmoid) * u

    tl.store(Y_ptr + offs * stride_y1, y, mask=mask)


@triton.jit
def atomic_accum_triton_kernel(result_ptr, final_ptr, weights_ptr,
                               stride_r0, stride_r1,
                               H_out: tl.constexpr,
                               stride_f0, stride_f1,
                               stride_w0, stride_w1,
                               BLOCK_H: tl.constexpr):
    # Accumulate per token: result[t] += weights[t, e] * final[t, :]
    t = tl.program_id(axis=0)
    # Loop over H_out in tiles
    for h_off in range(0, H_out, BLOCK_H):
        offs = h_off + tl.arange(0, BLOCK_H)
        mask = offs < H_out

        final_vals = tl.load(final_ptr + t * stride_f0 + offs * stride_f1, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weights_ptr + t * stride_w0 + 0 * stride_w1)  # [t, e] flattened
        weight = weight.to(tl.float32)

        # Atomic add into result
        tl.atomic_add(result_ptr + t * stride_r0 + offs * stride_r1, final_vals * weight, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tile sizes for Triton kernels
        self.block_m = 128
        self.block_h = 128

    def forward(self,
                hidden_states: torch.Tensor,       # [num_tokens, hidden_size], bfloat16
                selected_experts: torch.Tensor,    # [num_tokens, num_experts_per_tok], int64
                routing_weights: torch.Tensor,     # [num_tokens, num_experts_per_tok], bfloat16
                expert_gate_weights: torch.Tensor, # [num_experts, hidden_size, H_out], bfloat16
                expert_up_weights: torch.Tensor,   # [num_experts, hidden_size, H_out], bfloat16
                expert_down_weights: torch.Tensor  # [num_experts, H_out, hidden_size], bfloat16
                ) -> torch.Tensor:
        # Ensure device is CUDA for Triton
        assert hidden_states.is_cuda, "Triton kernels require CUDA tensors"
        device = hidden_states.device

        num_tokens, hidden_size = hidden_states.shape
        num_experts = expert_gate_weights.shape[0]
        H_out = expert_gate_weights.shape[2]

        # Prepare output result as fp32 for accumulation
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)

        # Iterate tokens and selected_experts deterministically
        for t in range(num_tokens):
            for j in range(selected_experts.shape[1]):
                e = int(selected_experts[t, j].item())
                # Hidden vector for this token
                x = hidden_states[t]  # [hidden_size], bfloat16

                # Compute gate_out: x @ expert_gate_weights[e]
                gate_out = torch.empty(H_out, dtype=torch.float32, device=device)
                bmm_triton_kernel[(triton.cdiv(H_out, self.block_m),)](
                    x, expert_gate_weights[e], gate_out,
                    hidden_size, H_out,
                    0, 1,
                    0, 1,
                    0, 1,
                    BLOCK_M=self.block_m,
                )

                # Compute up_out: x @ expert_up_weights[e]
                up_out = torch.empty(H_out, dtype=torch.float32, device=device)
                bmm_triton_kernel[(triton.cdiv(H_out, self.block_m),)](
                    x, expert_up_weights[e], up_out,
                    hidden_size, H_out,
                    0, 1,
                    0, 1,
                    0, 1,
                    BLOCK_M=self.block_m,
                )

                # Elementwise activated = silu(gate_out) * up_out
                activated = torch.empty(H_out, dtype=torch.float32, device=device)
                silu_mul_triton_kernel[(triton.cdiv(H_out, self.block_m),)](
                    gate_out, up_out, activated,
                    H_out,
                    0, 1,
                    0, 1,
                    0, 1,
                    BLOCK_M=self.block_m,
                )

                # Compute final_out: activated @ expert_down_weights[e]
                final_out = torch.empty(hidden_size, dtype=torch.float32, device=device)
                W_down = expert_down_weights[e].to(torch.float32)  # [H_out, hidden_size]
                bmm_triton_kernel[(triton.cdiv(hidden_size, self.block_m),)](
                    activated, W_down, final_out,
                    H_out, hidden_size,
                    0, 1,
                    0, 1,
                    0, 1,
                    BLOCK_M=self.block_m,
                )

                # Load routing weight for this (t, e)
                w = float(routing_weights[t, j].item())
                # Atomic add into result[t, :]
                atomic_accum_triton_kernel[(1,)](
                    result, final_out, torch.tensor([w], dtype=torch.float32, device=device),
                    hidden_size,  # stride_r0 for result
                    1,             # stride_r1 for result
                    H_out,
                    1,             # stride_f0 for final_out
                    1,             # stride_f1 for final_out
                    1,             # stride_w0 for weights (1x1)
                    1,             # stride_w1 for weights (unused)
                    BLOCK_H=self.block_h,
                )

        # Cast result to bfloat16 to match original return dtype
        result = result.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
