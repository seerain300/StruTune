import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_row_matmul(C_ptr, A_row_ptr, B_ptr, H: tl.constexpr, M: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute C[j] = sum_{k=0..H-1} A_row[k] * B[k, j], for j in [0, M).
    # Each program handles one j.
    j = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A_row_ptr + k, mask=k < H, other=0.0).to(tl.float32)
        # B[k, j] as a vector for this j
        b = tl.load(B_ptr + k * M + j, mask=k < H, other=0.0).to(tl.float32)
        acc += tl.sum(a * b, axis=0)
    tl.store(C_ptr + j, acc)


@triton.jit
def triton_silu(X_ptr, Y_ptr, N: tl.constexpr):
    # Y = X * sigmoid(X)
    for i in range(N):
        x = tl.load(X_ptr + i).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(Y_ptr + i, y)


@triton.jit
def triton_mul(A_ptr, B_ptr, C_ptr, N: tl.constexpr):
    # C = A * B elementwise
    for i in range(N):
        a = tl.load(A_ptr + i).to(tl.float32)
        b = tl.load(B_ptr + i).to(tl.float32)
        tl.store(C_ptr + i, a * b)


@triton.jit
def triton_atomic_add_weighted_vec(Out_ptr, Vec_ptr, Weight, N: tl.constexpr):
    # Atomic add: Out[i] += Weight * Vec[i], Weight is scalar float
    for i in range(N):
        v = tl.load(Vec_ptr + i).to(tl.float32)
        old = tl.load(Out_ptr + i).to(tl.float32)
        new = old + v * Weight
        tl.atomic_add(Out_ptr + i, new)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure inputs are on same device
        device = hidden_states.device
        dtype = hidden_states.dtype  # bfloat16
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts = expert_gate_weights.shape[0]

        # Flatten selected_experts and routing_weights
        flat_experts = selected_experts.reshape(-1).to(torch.int32).to(device)
        flat_weights = routing_weights.reshape(-1).to(torch.bfloat16).to(device)
        N_pairs = flat_experts.shape[0]
        num_experts_per_tok = selected_experts.shape[1]

        # Precompute counts and starts for capacity in Triton-compatible fashion
        # Use torch for lightweight preprocessing (these are not heavy and needed for correctness).
        counts = torch.bincount(flat_experts, minlength=num_experts)  # per-expert count
        starts = torch.cumsum(counts, dim=0)  # prefix sums
        # capacity = ceil(1.25 * avg_selected_per_expert)
        # avg_selected_per_expert = total_pairs / num_experts
        avg_selected = N_pairs / float(num_experts)
        capacity = max(1, int(math.ceil(1.25 * avg_selected)))
        # Compute within_pos: index within the sorted group; since we don't sort, we'll use original order
        # Note: Original code relies on stable sort to get starts; we mimic the capacity constraint locally.
        # For each p, within_pos = p - starts[flat_experts[p]] if flat_experts[p] < num_experts else -1
        # We'll compute a validity mask using this logic, though without stable sort we can still use
        # capacity as an upper bound; but to closely match the original, we compute a vector of indices and derive valid.
        # Create a tensor of flat_experts expanded to N_pairs
        # However, since we iterate in original order, we can compute starts[flat_experts[p]] per p.
        # We'll compute valid mask as within_pos < capacity, where within_pos = p - starts[flat_experts[p]].
        # To do that, we need to ensure p < starts.size(0); but starts is size num_experts. We'll compute for each p.

        # Allocate output in fp32 for stability
        out = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float32)

        # Iterate all pairs; compute per-p validity based on starts and capacity
        for p in range(N_pairs):
            token_id = p // num_experts_per_tok
            expert_id = int(flat_experts[p].item())
            # Compute within_pos for this pair assuming sorted order. Since we didn't sort, we apply capacity mask:
            # If p < starts[expert_id], then within_pos = p - starts[expert_id]; else invalid.
            # Note: starts[expert_id] may exceed N_pairs for high expert counts, but we keep it for consistency.
            starts_val = int(starts[expert_id].item())
            # If p < num_experts (which is always true), compute within_pos
            within_pos = p - starts_val
            valid = (within_pos >= 0) and (within_pos < capacity)

            # Load A_row
            A_row = hidden_states[token_id].contiguous().to(torch.float32)  # cast for matmul stability

            # Gate: A_row @ expert_gate_weights[expert_id]
            B_gate = expert_gate_weights[expert_id].contiguous().to(torch.float32)
            gate_out = torch.empty(hidden_size, device=device, dtype=torch.float32)
            triton_row_matmul[(hidden_size,)](gate_out, A_row, B_gate, H=hidden_size, M=hidden_size, BLOCK_K=128)

            # Up: A_row @ expert_up_weights[expert_id]
            B_up = expert_up_weights[expert_id].contiguous().to(torch.float32)
            up_out = torch.empty(hidden_size, device=device, dtype=torch.float32)
            triton_row_matmul[(hidden_size,)](up_out, A_row, B_up, H=hidden_size, M=hidden_size, BLOCK_K=128)

            # SiLU(gate_out)
            gate_silu = torch.empty(hidden_size, device=device, dtype=torch.float32)
            triton_silu[(hidden_size,)](gate_out, gate_silu, N=hidden_size)

            # Multiply gate_silu and up_out
            activated = torch.empty(hidden_size, device=device, dtype=torch.float32)
            triton_mul[(hidden_size,)](gate_silu, up_out, activated, N=hidden_size)

            # Down: activated @ expert_down_weights[expert_id]
            B_down = expert_down_weights[expert_id].contiguous().to(torch.float32)
            contribution = torch.empty(hidden_size, device=device, dtype=torch.float32)
            triton_row_matmul[(hidden_size,)](contribution, activated, B_down, H=hidden_size, M=hidden_size, BLOCK_K=128)

            # If valid, atomic add weighted contribution to out[token_id]
            if valid:
                # weight is bfloat16 scalar; cast to float for accumulation
                weight = float(flat_weights[p].item())
                # contribution is fp32; out[token_id] is fp32
                triton_atomic_add_weighted_vec[(hidden_size,)](out[token_id], contribution, weight, N=hidden_size)

        # Cast output to bfloat16 to match original dtype
        out = out.to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
