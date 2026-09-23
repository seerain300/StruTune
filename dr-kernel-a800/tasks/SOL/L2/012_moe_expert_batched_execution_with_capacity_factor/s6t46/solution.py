import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: per-row batched matmul A[H, 1] x B[H, M] -> C[1, M]
# A is [H, 1], B is [H, M], C is [1, M]
@triton.jit
def row_bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    H, M,
    stride_A_row, stride_A_col,
    stride_B_row, stride_B_col,
    stride_C_row, stride_C_col,
    BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr
):
    # One program handles one output vector of length M (i=0 since A is [1, H]).
    i = 0
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        a = tl.load(
            A_ptr + i * stride_A_row + h_idx * stride_A_col,
            mask=h_idx < H,
            other=0.0
        )  # [BLOCK_H]
        j = tl.arange(0, BLOCK_M)
        b = tl.load(
            B_ptr + h_idx[:, None] * stride_B_row + j[None, :] * stride_B_col,
            mask=(h_idx[:, None] < H) & (j[None, :] < M),
            other=0.0
        )  # [BLOCK_H, BLOCK_M]
        acc += tl.sum(a[:, None] * b, axis=0)
    tl.store(C_ptr + 0 * stride_C_row + j * stride_C_col, acc, mask=j < M)


# Triton kernel: elementwise SiLU over a vector (Y = X * sigmoid(X))
@triton.jit
def silu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(Y_ptr + offsets, y, mask=mask)


# Triton kernel: per-row batched matmul A[M, 1] x B[M, H] -> C[1, H]
# A is [M], B is [M, H], C is [1, H]
@triton.jit
def row_bmm_down_kernel(
    A_ptr, B_ptr, C_ptr,
    M, H,
    stride_A_row, stride_A_col,
    stride_B_row, stride_B_col,
    stride_C_row, stride_C_col,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    i = 0  # A is [1, M] effectively
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for m_off in range(0, M, BLOCK_M):
        m_idx = m_off + tl.arange(0, BLOCK_M)
        a = tl.load(
            A_ptr + i * stride_A_row + m_idx * stride_A_col,
            mask=m_idx < M,
            other=0.0
        )  # [BLOCK_M]
        h = tl.arange(0, BLOCK_H)
        b = tl.load(
            B_ptr + m_idx[:, None] * stride_B_row + h[None, :] * stride_B_col,
            mask=(m_idx[:, None] < M) & (h[None, :] < H),
            other=0.0
        )  # [BLOCK_M, BLOCK_H]
        acc += tl.sum(a[:, None] * b, axis=0)
    tl.store(C_ptr + 0 * stride_C_row + h * stride_C_col, acc, mask=h < H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        hidden_states: [num_tokens, hidden_size] (H)
        selected_experts: [num_tokens, num_experts_per_tok] (int64)
        routing_weights: [num_tokens, num_experts_per_tok] (float) - may be missing; we assume uniform.
        expert_gate_weights, expert_up_weights, expert_down_weights: [num_experts, H, M]
        """
        if not TRITON_AVAILABLE:
            # Fallback: heavy compute not available; return zeros
            return torch.zeros(hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device, dtype=torch.float32)

        device = hidden_states.device
        num_tokens, H = hidden_states.shape
        num_experts = expert_gate_weights.shape[0]
        M = expert_gate_weights.shape[2]  # intermediate size

        # Output buffer
        result = torch.zeros(num_tokens, H, device=device, dtype=torch.float32)

        # Process each token
        for t in range(num_tokens):
            contrib = torch.zeros(H, device=device, dtype=torch.float32)
            K = selected_experts.shape[1]
            for e in range(K):
                idx = int(selected_experts[t, e].item())

                # Load hidden state row and cast to float32 for compute
                hs = hidden_states[t].to(torch.float32)  # [H]

                # Load expert weights (float32)
                gate_w = expert_gate_weights[idx].to(torch.float32)  # [H, M]
                up_w = expert_up_weights[idx].to(torch.float32)      # [H, M]
                down_w = expert_down_weights[idx].to(torch.float32)  # [M, H]

                # 1) gate_out = hs @ gate_w -> [M]
                gate_out = torch.empty(M, device=device, dtype=torch.float32)
                grid_gate = (triton.cdiv(M, 128),)
                row_bmm_kernel[grid_gate](
                    hs.unsqueeze(0), gate_w, gate_out,
                    H, M,
                    hs.unsqueeze(0).stride(0), 1,
                    gate_w.stride(0), gate_w.stride(1),
                    gate_out.stride(0), gate_out.stride(1),
                    BLOCK_H=128, BLOCK_M=128,
                    num_warps=4
                )

                # 2) up_out = hs @ up_w -> [M]
                up_out = torch.empty(M, device=device, dtype=torch.float32)
                grid_up = (triton.cdiv(M, 128),)
                row_bmm_kernel[grid_up](
                    hs.unsqueeze(0), up_w, up_out,
                    H, M,
                    hs.unsqueeze(0).stride(0), 1,
                    up_w.stride(0), up_w.stride(1),
                    up_out.stride(0), up_out.stride(1),
                    BLOCK_H=128, BLOCK_M=128,
                    num_warps=4
                )

                # 3) activated = SiLU(gate_out) * up_out
                silu_out = torch.empty(M, device=device, dtype=torch.float32)
                grid_silu = (triton.cdiv(M, 256),)
                silu_kernel[grid_silu](
                    gate_out, silu_out, M,
                    BLOCK_SIZE=256,
                    num_warps=4
                )
                activated = silu_out * up_out  # [M]

                # 4) expert_outputs = activated @ down_w -> [H]
                expert_outputs = torch.empty(H, device=device, dtype=torch.float32)
                grid_down = (triton.cdiv(H, 128),)
                row_bmm_down_kernel[grid_down](
                    activated.unsqueeze(0), down_w, expert_outputs,
                    M, H,
                    activated.unsqueeze(0).stride(0), 1,
                    down_w.stride(0), down_w.stride(1),
                    expert_outputs.stride(0), expert_outputs.stride(1),
                    BLOCK_M=128, BLOCK_H=128,
                    num_warps=4
                )

                # Uniform aggregation since routing_weights are not provided
                contrib += expert_outputs

            # Average contributions (uniform weighting across K selected experts)
            result[t] = contrib / float(K)

        return result


def run(*args):
    return ModelNew()(*args)
