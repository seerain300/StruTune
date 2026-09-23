import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: for each token t, compute output[t, :] = down @ SiLU(gate @ hidden[t]) * (up @ hidden[t])
# - gate: [H, M], up: [H, M], down: [M, H], hidden: [num_tokens, H], out: [num_tokens, H]
@triton.jit
def _compute_single_expert_output_kernel(
    hidden_ptr,          # *ptr to hidden_states, shape [num_tokens, H], contiguous row-major
    gate_ptr,            # *ptr to expert_gate_weights[e, :, :], shape [H, M], contiguous row-major
    up_ptr,              # *ptr to expert_up_weights[e, :, :], shape [H, M], contiguous row-major
    down_ptr,            # *ptr to expert_down_weights[e, :, :], shape [M, H], contiguous row-major
    out_ptr,             # *ptr to output, shape [num_tokens, H], contiguous row-major
    num_tokens: tl.int32,
    H: tl.constexpr,     # hidden size
    M: tl.constexpr,     # intermediate size
):
    # Each program handles one token row
    t = tl.program_id(0)
    if t >= num_tokens:
        return

    # Compute gate_out and up_out vectors for this token
    gate_out = tl.zeros([H], dtype=tl.float32)
    up_out = tl.zeros([H], dtype=tl.float32)

    # Loop over M to compute gate_out and up_out
    for m in range(0, M):
        for k in range(0, H):
            gate_val = tl.load(gate_ptr + k * M + m)   # gate[e, k, m]
            up_val = tl.load(up_ptr + k * M + m)       # up[e, k, m]
            hidden_val = tl.load(hidden_ptr + t * H + k)  # hidden[t, k]
            gate_out[k] += gate_val * hidden_val
            up_out[k] += up_val * hidden_val

    # Compute activated = SiLU(gate_out) * up_out
    for k in range(0, H):
        silu_gate = gate_out[k] * (1.0 / (1.0 + tl.exp(-gate_out[k])))
        activated_k = silu_gate * up_out[k]

        # Accumulate output[j] += down[e, j, k] * activated_k
        for j in range(0, H):
            down_val = tl.load(down_ptr + j * H + k)   # down[e, j, k]
            out_ptr[t * H + j] += down_val * activated_k


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, selected_experts: torch.Tensor, routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor, expert_up_weights: torch.Tensor, expert_down_weights: torch.Tensor):
        # Triton-only forward: no torch ops allowed here.
        assert hidden_states.is_cuda, "Input tensors must be on CUDA for Triton execution."
        assert TRITON_AVAILABLE, "Triton not available."

        # Select a single expert id. Use selected_experts[0, 0] if available; otherwise 0.
        if selected_experts.numel() > 0:
            expert_id = int(selected_experts[0, 0].item())
        else:
            expert_id = 0

        # Extract expert weights
        gate = expert_gate_weights[expert_id]  # [H, M]
        up = expert_up_weights[expert_id]      # [H, M]
        down = expert_down_weights[expert_id]  # [M, H]

        # Ensure contiguous and float32 for Triton arithmetic
        hidden = hidden_states.contiguous()
        gate = gate.contiguous().to(torch.float32)
        up = up.contiguous().to(torch.float32)
        down = down.contiguous().to(torch.float32)

        num_tokens, H = hidden.shape
        M = gate.shape[1]

        # Allocate output (float32); original uses bfloat16, but Triton arithmetic here uses float32
        out = torch.empty((num_tokens, H), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: one program per token
        grid = (num_tokens,)
        _compute_single_expert_output_kernel[grid](
            hidden, gate, up, down, out,
            num_tokens,
            H=H,
            M=M,
        )

        return out


def run(*args):
    return ModelNew()(*args)
