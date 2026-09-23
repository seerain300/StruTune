import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_kernel(
    X_ptr,   # *bf16, [B, H] contiguous
    W_ptr,   # *bf16, [H, M] contiguous
    Y_ptr,   # *bf16, [B, M] contiguous
    B: tl.constexpr,   # number of rows in X (here always 1)
    H,                # int: hidden_size
    M,                # int: output size (moe_intermediate_size or hidden_size)
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return
    cols = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Iterate over H in tiles
    for k0 in range(0, H, BLOCK_M):
        k = k0 + cols
        mask_k = k < H
        # Load X row tile: X[row, k]
        x = tl.load(X_ptr + row * H + k, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_M]
        # Load W tile: W[k, cols] -> shape [BLOCK_M, BLOCK_M]
        w = tl.load(W_ptr + k[:, None] * M + cols[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
        # acc += x @ w along k axis
        acc += tl.sum(w * x[None, :], axis=1)
    # Store Y[row, cols]
    tl.store(Y_ptr + row * M + cols, acc, mask=cols < M)


@triton.jit
def activation_kernel(
    GATE_ptr,    # *bf16, [H_out]
    UP_ptr,      # *bf16, [H_out]
    OUT_ptr,     # *bf16, [H_out]
    N: tl.constexpr,  # H_out
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    for i in range(0, N, BLOCK):
        idx = i + offs
        mask = idx < N
        g = tl.load(GATE_ptr + idx, mask=mask, other=0.0).to(tl.float32)  # SiLU(g) = g * sigmoid(g)
        u = tl.load(UP_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        act = g * sig * u
        tl.store(OUT_ptr + idx, act.to(tl.bfloat16), mask=mask)


@triton.jit
def atomic_accumulate_kernel(
    INPUT_ptr,   # *bf16, [M] containing final_out
    OUTPUT_ptr,  # *fp32, [num_tokens, hidden_size] row-major
    SCALE_ptr,   # *bf16, [1] scalar routing_weight as 1-element tensor
    M,           # int: hidden_size
    row_id,      # int: token index (0..num_tokens-1)
    BLOCK: tl.constexpr,
):
    # Each program handles one tile of columns
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    # Load final_out tile as fp32
    final = tl.load(INPUT_ptr + cols, mask=cols < M, other=0.0).to(tl.float32)
    # Load scale (routing_weight) as fp32
    scale_val = tl.load(SCALE_ptr).to(tl.float32)
    vals = final * scale_val
    # Atomic add into OUTPUT row
    out_ptrs = OUTPUT_ptr + row_id * M + cols
    tl.atomic_add(out_ptrs, vals, mask=cols < M)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Shapes from inputs
        num_tokens, hidden_size = hidden_states.shape
        num_experts = expert_gate_weights.shape[0]
        H_out = expert_gate_weights.shape[1]  # moe_intermediate_size
        _, _, H_in = expert_down_weights.shape  # H_in should equal hidden_size

        # Output buffer in fp32 for accumulation, cast to bf16 at end
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)

        # Iterate tokens and selected_experts deterministically
        for t in range(num_tokens):
            # Loop over selected_experts: we need to compute gate, up, activated, final for each selected expert
            # The 'selected_experts' tensor provides the mapping from token to expert indices.
            # We interpret selected_experts as shape [num_tokens, num_experts_per_tok], but here we treat it
            # as indices per token. The original code uses 'num_experts_per_tok' but only selected_experts is passed.
            # We assume selected_experts is of shape [num_tokens] (common in such setups). If not, we fallback to all experts.
            # To be robust: if selected_experts is 2D, we take all entries; else, we assume 1D.
            if selected_experts.dim() == 2:
                exp_per_tok = selected_experts.shape[1]
                for j in range(exp_per_tok):
                    e = int(selected_experts[t, j].item())
                    if e < 0 or e >= num_experts:
                        continue
                    # Compute gate_out: [H_out]
                    gate_out = torch.empty(H_out, dtype=torch.bfloat16, device=hidden_states.device)
                    bmm_triton_kernel[(1,)](
                        hidden_states[t], expert_gate_weights[e], gate_out, 1, hidden_size, H_out, BLOCK_M=64
                    )
                    # Compute up_out: [H_out]
                    up_out = torch.empty(H_out, dtype=torch.bfloat16, device=hidden_states.device)
                    bmm_triton_kernel[(1,)](
                        hidden_states[t], expert_up_weights[e], up_out, 1, hidden_size, H_out, BLOCK_M=64
                    )
                    # Activation: activated = silu(gate_out) * up_out
                    activated = torch.empty(H_out, dtype=torch.bfloat16, device=hidden_states.device)
                    activation_kernel[(1,)](
                        gate_out, up_out, activated, H_out, BLOCK=128
                    )
                    # Compute final_out: [H_in]
                    final_out = torch.empty(H_in, dtype=torch.bfloat16, device=hidden_states.device)
                    bmm_triton_kernel[(1,)](
                        activated, expert_down_weights[e], final_out, 1, H_out, H_in, BLOCK_M=64
                    )
                    # Weighted accumulation into result[t, :]
                    # routing_weights shape is [num_tokens, num_experts_per_tok]; assume each token has multiple routing weights.
                    # Here we select the j-th routing weight for expert e.
                    # If num_experts_per_tok dimension is present, we take routing_weights[t, j].
                    # Ensure bounds and exist; otherwise use 1.0.
                    # We pass routing_weight as a 1-element tensor to Triton kernel.
                    if routing_weights is not None and routing_weights.dim() == 2 and routing_weights.shape[1] >= j and t < routing_weights.shape[0]:
                        weight = routing_weights[t, j]
                        # Convert to 1-element bf16 tensor on device
                        scale_buf = torch.empty(1, dtype=torch.bfloat16, device=hidden_states.device)
                        scale_buf[0] = weight.to(torch.bfloat16)
                        # Atomic add: result[t] += weight * final_out
                        atomic_accumulate_kernel[(triton.cdiv(H_in, 128),)](
                            final_out, result, scale_buf, H_in, t, BLOCK=128
                        )
                    else:
                        # Default to adding final_out (weight=1.0)
                        scale_buf = torch.empty(1, dtype=torch.bfloat16, device=hidden_states.device)
                        scale_buf[0] = torch.tensor(1.0, dtype=torch.bfloat16, device=hidden_states.device)
                        atomic_accumulate_kernel[(triton.cdiv(H_in, 128),)](
                            final_out, result, scale_buf, H_in, t, BLOCK=128
                        )
            else:
                # If selected_experts is 1D: [num_tokens]
                for j in range(len(selected_experts)):
                    e = int(selected_experts[t].item())
                    if e < 0 or e >= num_experts:
                        continue
                    # gate_out
                    gate_out = torch.empty(H_out, dtype=torch.bfloat16, device=hidden_states.device)
                    bmm_triton_kernel[(1,)](
                        hidden_states[t], expert_gate_weights[e], gate_out, 1, hidden_size, H_out, BLOCK_M=64
                    )
                    # up_out
                    up_out = torch.empty(H_out, dtype=torch.bfloat16, device=hidden_states.device)
                    bmm_triton_kernel[(1,)](
                        hidden_states[t], expert_up_weights[e], up_out, 1, hidden_size, H_out, BLOCK_M=64
                    )
                    # activation
                    activated = torch.empty(H_out, dtype=torch.bfloat16, device=hidden_states.device)
                    activation_kernel[(1,)](
                        gate_out, up_out, activated, H_out, BLOCK=128
                    )
                    # final_out
                    final_out = torch.empty(H_in, dtype=torch.bfloat16, device=hidden_states.device)
                    bmm_triton_kernel[(1,)](
                        activated, expert_down_weights[e], final_out, 1, H_out, H_in, BLOCK_M=64
                    )
                    # weight
                    # If routing_weights exists and has at least one column, use it; else weight=1.0
                    if routing_weights is not None and routing_weights.dim() == 2 and routing_weights.shape[1] > 0 and t < routing_weights.shape[0]:
                        weight = routing_weights[t, 0]  # default to first col
                        scale_buf = torch.empty(1, dtype=torch.bfloat16, device=hidden_states.device)
                        scale_buf[0] = weight.to(torch.bfloat16)
                        atomic_accumulate_kernel[(triton.cdiv(H_in, 128),)](
                            final_out, result, scale_buf, H_in, t, BLOCK=128
                        )
                    else:
                        scale_buf = torch.empty(1, dtype=torch.bfloat16, device=hidden_states.device)
                        scale_buf[0] = torch.tensor(1.0, dtype=torch.bfloat16, device=hidden_states.device)
                        atomic_accumulate_kernel[(triton.cdiv(H_in, 128),)](
                            final_out, result, scale_buf, H_in, t, BLOCK=128
                        )

        # Return result cast to bfloat16 to match original dtype
        return result.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
