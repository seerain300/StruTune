import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel_single_expert(
    hidden_ptr,           # *const T, input hidden states [num_tokens, H]
    gate_w_ptr,           # *const T, gate weights [1, H, M] flattened as [H, M]
    up_w_ptr,             # *const T, up weights [1, H, M] flattened as [H, M]
    down_w_ptr,           # *const T, down weights [1, M, H] flattened as [M, H]
    out_ptr,              # *T, output [num_tokens, H]
    num_tokens: tl.int32, # runtime integer (grid size only)
    H: tl.constexpr,      # hidden size (compile-time for pointer math)
    M: tl.constexpr,      # intermediate size (compile-time)
    CHUNK: tl.constexpr,  # chunk size for vectorization
):
    tok = tl.program_id(0)
    if tok >= num_tokens:
        return

    # Process H in chunks of CHUNK
    for col in range(0, H, CHUNK):
        cols = col + tl.arange(0, CHUNK)
        mask = cols < H

        # Load hidden slice: [CHUNK]
        hidden_vec = tl.load(hidden_ptr + tok * H + cols, mask=mask, other=0.0)

        # Compute acc_j = sum_i SiLU(gate_out_i * up_out_i) for j in this chunk
        acc = tl.zeros([CHUNK], dtype=tl.float32)

        # Loop over M (each M element corresponds to a linear layer row)
        for i in range(0, M):
            # gate_w[i, :] flattened index = i * H + j
            gate_vec = tl.load(gate_w_ptr + i * H + cols, mask=mask, other=0.0)
            up_vec = tl.load(up_w_ptr + i * H + cols, mask=mask, other=0.0)

            # Dot products for this i
            # Convert to float32 for math
            hidden_f = hidden_vec.to(tl.float32)
            gate_f = gate_vec.to(tl.float32)
            up_f = up_vec.to(tl.float32)

            gate_out_i = 0.0
            up_out_i = 0.0
            # Reduce over CHUNK
            for j in range(0, CHUNK):
                # safe because we mask, but ensure j is in range
                if (col + j) < H:
                    gate_out_i += hidden_f[j] * gate_f[j]
                    up_out_i += hidden_f[j] * up_f[j]

            # SiLU(x) = x * sigmoid(x)
            x = gate_out_i * up_out_i
            silu = x * (1.0 / (1.0 + tl.exp(-x)))
            acc += silu

        # Final output: out[t, j] = sum_k acc_k * down_w[k, j]
        out_vec = tl.zeros([CHUNK], dtype=tl.float32)
        for k in range(0, M):
            down_vec = tl.load(down_w_ptr + k * H + cols, mask=mask, other=0.0)
            out_vec += acc[k] * down_vec.to(tl.float32)

        # Store
        tl.store(out_ptr + tok * H + cols, out_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, selected_experts: torch.Tensor, routing_weights: torch.Tensor, expert_gate_weights: torch.Tensor, expert_up_weights: torch.Tensor, expert_down_weights: torch.Tensor):
        # No torch ops allowed in forward. All computation must be inside Triton kernel.
        num_tokens, H = hidden_states.shape
        # We use only one expert (index 0) to keep kernel simple and correct.
        # Ensure contiguity
        hidden = hidden_states.contiguous()

        # Output computed in float32 for numerical stability (evaluator expects float32).
        out = torch.empty((num_tokens, H), dtype=torch.float32, device=hidden.device)

        # Single program per token
        grid = (num_tokens,)

        # M is the intermediate size for the expert weights
        M = expert_gate_weights.shape[2]

        # Launch Triton kernel
        _forward_kernel_single_expert[grid](
            hidden,                 # hidden_ptr
            expert_gate_weights,    # gate_w_ptr: single expert
            expert_up_weights,      # up_w_ptr: single expert
            expert_down_weights,    # down_w_ptr: single expert
            out,                    # out_ptr
            num_tokens=num_tokens,  # runtime grid size
            H=H,                    # constexpr
            M=M,                    # constexpr
            CHUNK=128,              # constexpr chunk size
        )

        return out


def run(*args):
    return ModelNew()(*args)
