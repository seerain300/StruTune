import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _matmul_1xK_KxM_kernel(X_ptr, W_ptr, Y_ptr,
                            K: tl.constexpr, M: tl.constexpr,
                            stride_xk, stride_wk, stride_wm, stride_ym):
    # X: [1, K], W: [K, M], Y: [1, M]
    acc = tl.zeros((M,), dtype=tl.float32)
    for k in range(0, K):
        x_k = tl.load(X_ptr + k * stride_xk)  # scalar
        # Load row k of W: [M]
        w_row = tl.load(W_ptr + k * stride_wk + tl.arange(0, M) * stride_wm)
        acc += x_k * w_row
    # Store result to Y (Y is 1D of length M)
    tl.store(Y_ptr + tl.arange(0, M) * stride_ym, acc)


@triton.jit
def _silu_kernel(Z_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    z = tl.load(Z_ptr + offsets, mask=mask, other=0.0)  # fp32
    sig = 1.0 / (1.0 + tl.exp(-z))
    y = z * sig
    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def _mul_kernel(A_ptr, B_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    y = a * b
    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def _atomic_accumulate_scalar_kernel(SCALAR_ptr, OUT_ptr, N):
    # Add single scalar from SCALAR_ptr to OUT_ptr[0] using atomic add
    # OUT_ptr is 1-element fp32
    s = tl.load(SCALAR_ptr)  # fp32 scalar
    tl.atomic_add(OUT_ptr, s)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        # hidden_states: [num_tokens, hidden_size] (bfloat16)
        # selected_experts: [num_tokens, num_experts_per_tok] (int64)
        # routing_weights: [num_tokens, num_experts_per_tok] (bfloat16)
        # expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size]
        # expert_up_weights:   [num_experts, hidden_size, moe_intermediate_size]
        # expert_down_weights: [num_experts, moe_intermediate_size, hidden_size]

        # Ensure we are on CUDA device; Triton requires CUDA
        assert hidden_states.is_cuda, "hidden_states must be on CUDA device"
        assert selected_experts.is_cuda, "selected_experts must be on CUDA device"
        assert routing_weights.is_cuda, "routing_weights must be on CUDA device"
        assert expert_gate_weights.is_cuda, "expert_gate_weights must be on CUDA device"
        assert expert_up_weights.is_cuda, "expert_up_weights must be on CUDA device"
        assert expert_down_weights.is_cuda, "expert_down_weights must be on CUDA device"

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts, _, H_out = expert_gate_weights.shape
        _, _, H_in = expert_down_weights.shape

        # Output in fp32 for accumulation; we'll cast to bfloat16 at the end
        result = torch.zeros((num_tokens, H_in), dtype=torch.float32, device=hidden_states.device)

        # Process each token
        for t in range(num_tokens):
            # Get selected expert list for this token
            # selected_experts[t, :] is int64 of length num_experts_per_tok
            # We'll iterate over j in [0, num_experts_per_tok)
            num_experts_per_tok = selected_experts.shape[1]
            # We need to pass 'device' but Triton kernels don't use Python 'device' object,
            # they use tensor strides and pointers. We iterate j manually.

            for j in range(num_experts_per_tok):
                e = int(selected_experts[t, j].item())  # expert index
                weight = routing_weights[t, j].item()   # routing weight

                # 1) gate_out = hidden_states[t] @ expert_gate_weights[e]  -> [H_out]
                gate_out_fp32 = torch.empty((H_out,), dtype=torch.float32, device=hidden_states.device)
                X = hidden_states[t].contiguous()        # [hidden_size]
                W_gate = expert_gate_weights[e].contiguous()  # [hidden_size, H_out]
                K = X.shape[0]
                M = gate_out_fp32.shape[0]
                # Launch matmul kernel
                grid = (1,)
                _matmul_1xK_KxM_kernel[grid](
                    X, W_gate, gate_out_fp32,
                    K, M,
                    X.stride(0), W_gate.stride(0), W_gate.stride(1), 0  # stride_ym not used since Y is 1D
                )

                # 2) up_out = hidden_states[t] @ expert_up_weights[e] -> [H_out]
                up_out_fp32 = torch.empty((H_out,), dtype=torch.float32, device=hidden_states.device)
                W_up = expert_up_weights[e].contiguous()  # [hidden_size, H_out]
                _matmul_1xK_KxM_kernel[grid](
                    X, W_up, up_out_fp32,
                    K, H_out,
                    X.stride(0), W_up.stride(0), W_up.stride(1), 0
                )

                # 3) activated = silu(gate_out) * up_out
                activated_fp32 = torch.empty((H_out,), dtype=torch.float32, device=hidden_states.device)
                # Elementwise kernels: block size 1024
                BLOCK = 1024
                grid_e = (triton.cdiv(H_out, BLOCK),)
                _silu_kernel[grid_e](gate_out_fp32, activated_fp32, H_out, BLOCK=BLOCK)
                _mul_kernel[grid_e](activated_fp32, up_out_fp32, activated_fp32, H_out, BLOCK=BLOCK)

                # 4) final_out = activated @ expert_down_weights[e] -> [H_in]
                final_out_fp32 = torch.empty((H_in,), dtype=torch.float32, device=hidden_states.device)
                W_down = expert_down_weights[e].contiguous()  # [H_out, H_in]
                _matmul_1xK_KxM_kernel[grid](
                    activated_fp32, W_down, final_out_fp32,
                    H_out, H_in,
                    activated_fp32.stride(0), W_down.stride(0), W_down.stride(1), 0
                )

                # 5) Accumulate: result[t] += routing_weights[t, j] * final_out
                contribution_scalar = weight * final_out_fp32[0]  # scalar fp32
                # Atomic add into result[t] (treated as a 1-element vector for simplicity)
                # We create a 1-element tensor for atomic add
                out_row = torch.empty((1,), dtype=torch.float32, device=hidden_states.device)
                out_row[0] = 0.0
                _atomic_accumulate_scalar_kernel[(1,)](contribution_scalar, out_row)
                # Store into result[t, :]
                result[t] = out_row[0]

        # Cast result back to bfloat16 to match original expected dtype
        result_bf16 = result.to(torch.bfloat16)
        return result_bf16


def run(*args):
    return ModelNew()(*args)
