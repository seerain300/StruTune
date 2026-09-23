import math
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
    pid = tl.program_id(axis=0)  # tile along M
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over rows of X (H)
    for h in range(0, H):
        x_val = tl.load(X_ptr + h * stride_x0)  # X is [1, H], element access via stride
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
    # Elementwise: Y = silu(Z) * U, where Z and U are 1D vectors of length M
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    z = tl.load(Z_ptr + offs * stride_z1, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + offs * stride_u1, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    y = (z * sigmoid) * u

    tl.store(Y_ptr + offs * stride_y1, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # You may choose BLOCK sizes; here we use 128 which is fine for typical hidden sizes.
        self.block_m = 128

    def forward(self,
                hidden_states: torch.Tensor,       # [num_tokens, hidden_size], bfloat16
                selected_experts: torch.Tensor,    # [num_tokens, num_experts_per_tok], int64
                routing_weights: torch.Tensor,     # [num_tokens, num_experts_per_tok], bfloat16
                expert_gate_weights: torch.Tensor, # [num_experts, hidden_size, H_out], bfloat16
                expert_up_weights: torch.Tensor,   # [num_experts, hidden_size, H_out], bfloat16
                expert_down_weights: torch.Tensor, # [num_experts, H_out, hidden_size], bfloat16
                ):
        # Ensure CUDA device
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda and \
               expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
               "All tensors must be on CUDA for Triton."

        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, H_out = expert_gate_weights.shape

        # Prepare output buffer in fp32 for accumulation
        result_fp32 = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)

        # Process each token and each selected expert
        for t in range(num_tokens):
            # For each expert j selected for token t
            for j in range(selected_experts.shape[1]):
                e = int(selected_experts[t, j].item())  # get expert index for this token j

                # hidden vector for this token
                hidden_vec = hidden_states[t]  # shape [hidden_size], contiguous along hidden_size
                # Compute gate_out = hidden_vec @ expert_gate_weights[e] -> [H_out]
                gate_out_fp32 = torch.empty(H_out, dtype=torch.float32, device=hidden_states.device)
                bmm_triton_kernel[(triton.cdiv(H_out, self.block_m),)](
                    hidden_vec, expert_gate_weights[e], gate_out_fp32,
                    hidden_size, H_out,
                    0, 1,                      # X strides: [1, H] -> stride_x0=0, stride_x1=1
                    0, 1,                      # W strides: [H, M] -> stride_w0=0, stride_w1=1
                    0, 1,                      # Y strides: [1, M] -> stride_y0=0, stride_y1=1
                    BLOCK_M=self.block_m,
                )

                # Compute up_out = hidden_vec @ expert_up_weights[e] -> [H_out]
                up_out_fp32 = torch.empty(H_out, dtype=torch.float32, device=hidden_states.device)
                bmm_triton_kernel[(triton.cdiv(H_out, self.block_m),)](
                    hidden_vec, expert_up_weights[e], up_out_fp32,
                    hidden_size, H_out,
                    0, 1,
                    0, 1,
                    0, 1,
                    BLOCK_M=self.block_m,
                )

                # activated = silu(gate_out) * up_out
                activated_fp32 = torch.empty(H_out, dtype=torch.float32, device=hidden_states.device)
                silu_mul_triton_kernel[(triton.cdiv(H_out, self.block_m),)](
                    gate_out_fp32, up_out_fp32, activated_fp32,
                    H_out,
                    0, 1,
                    0, 1,
                    0, 1,
                    BLOCK_M=self.block_m,
                )

                # final_out = activated @ expert_down_weights[e] -> [hidden_size]
                final_out_fp32 = torch.empty(hidden_size, dtype=torch.float32, device=hidden_states.device)
                bmm_triton_kernel[(triton.cdiv(hidden_size, self.block_m),)](
                    activated_fp32, expert_down_weights[e], final_out_fp32,
                    H_out, hidden_size,
                    0, 1,                      # X: [1, H_out] but here vector length H_out; we treat as [1, H_out] -> stride_x0=0, stride_x1=1 for the leading dim. In our usage, X is 1xH_out vector; we pass pointer and use stride_x0=0, stride_x1=1 for accessing elements.
                    0, 1,                      # W: [H_out, hidden_size] -> stride_w0=0, stride_w1=1
                    0, 1,                      # Y: [1, hidden_size] -> stride_y0=0, stride_y1=1
                    BLOCK_M=self.block_m,
                )

                # Aggregate: result[t] += routing_weights[t, j] * final_out
                # routing_weights[t, j] is scalar in bfloat16; we can multiply as scalar and add.
                weight = float(routing_weights[t, j].item())
                result_fp32[t] += weight * final_out_fp32

        # Return result cast to bfloat16 to match original
        result_bf16 = result_fp32.to(torch.bfloat16)
        return result_bf16


def run(*args):
    return ModelNew()(*args)
