import torch
import triton
import triton.language as tl


# Triton kernel: per-row matmul for A_row (length H) times B (H x M) producing C[H].
# A_row is a 1D tensor, B is a 2D tensor. We process one row of A via a simple loop.
@triton.jit
def triton_row_matmul(C_ptr, A_ptr, B_ptr,
                       H: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr):
    # We iterate over columns in blocks and accumulate dot products.
    for col_start in range(0, M, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(0, H):
            a = tl.load(A_ptr + i)
            b = tl.load(B_ptr + i * M + offs)
            acc += a * b
        tl.store(C_ptr + offs, acc, mask=offs < M)


# Triton elementwise SiLU: y = x * sigmoid(x) for vector X
@triton.jit
def triton_silu(X_ptr, Y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(X_ptr + offs)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(Y_ptr + offs, y, mask=offs < N)


# Triton elementwise multiply: Y = X0 * X1 for vectors
@triton.jit
def triton_mul(X0_ptr, X1_ptr, Y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x0 = tl.load(X0_ptr + offs)
        x1 = tl.load(X1_ptr + offs)
        y = x0 * x1
        tl.store(Y_ptr + offs, y, mask=offs < N)


def triton_matmul_row(A_row_1d: torch.Tensor, B_mat: torch.Tensor) -> torch.Tensor:
    """
    Compute A_row (1D tensor of length H) @ B (H x M) -> C (M).
    A_row and B must be CUDA tensors, float32 preferred for compute, output will be float32.
    """
    assert A_row_1d.is_cuda and B_mat.is_cuda, "Triton matmul requires CUDA tensors."
    H = A_row_1d.numel()
    M, K = B_mat.shape
    assert K == H, "B_mat second dim must equal A_row length."
    C = torch.empty(M, device=A_row_1d.device, dtype=torch.float32)
    # Pass flattened A_row (1D), B as contiguous (H, M)
    triton_row_matmul[(1,)](C, A_row_1d, B_mat, H=H, M=M, BLOCK=128)
    return C


def triton_silu_vec(X: torch.Tensor) -> torch.Tensor:
    """
    Compute SiLU elementwise on X. Returns tensor with same shape and dtype.
    Note: We keep output in float32 for numerical stability; cast back to original dtype after.
    """
    assert X.is_cuda, "Triton silu requires CUDA tensor."
    Y = torch.empty_like(X, dtype=torch.float32)
    N = X.numel()
    triton_silu[(1,)](X, Y, N=N, BLOCK=1024)
    return Y


def triton_mul_vec(X0: torch.Tensor, X1: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply of two vectors. Returns tensor with same shape and dtype.
    """
    assert X0.is_cuda and X1.is_cuda, "Triton mul requires CUDA tensors."
    assert X0.shape == X1.shape, "Vectors must have same shape."
    Y = torch.empty_like(X0, dtype=torch.float32)
    N = X0.numel()
    triton_mul[(1,)](X0, X1, Y, N=N, BLOCK=1024)
    return Y


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        """
        hidden_states: [num_tokens, hidden_size], bfloat16, CUDA
        selected_experts: [num_tokens, num_experts_per_tok], int64, CUDA
        routing_weights: [num_tokens, num_experts_per_tok], bfloat16, CUDA
        expert_gate_weights: [num_experts, hidden_size, hidden_size], bfloat16, CUDA
        expert_up_weights: [num_experts, hidden_size, hidden_size], bfloat16, CUDA
        expert_down_weights: [num_experts, hidden_size, hidden_size], bfloat16, CUDA
        """
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda \
            and expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
            "All tensors must be on CUDA for Triton."

        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_k, gate_out = expert_gate_weights.shape
        assert gate_k == hidden_size, "expert_gate_weights second dim must equal hidden_size."

        # Flatten selected_experts and routing_weights to align with original preprocessing
        flat_experts = selected_experts.reshape(-1)             # [num_tokens * K]
        flat_routing = routing_weights.reshape(-1)              # [num_tokens * K]
        K = num_experts_per_tok

        # STABLE sort by selected_experts (to match original code)
        flat_experts_sorted, sorted_indices = torch.sort(flat_experts, stable=True)
        flat_routing_sorted = flat_routing[sorted_indices]
        sorted_token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(K)[sorted_indices]

        # Precompute counts per expert and starts (group boundaries) like original
        counts = torch.bincount(flat_experts_sorted)
        # cumsum without prefix
        starts = torch.zeros(num_experts, dtype=torch.long, device=hidden_states.device)
        starts[1:] = counts[:-1].cumsum(0)

        # Compute capacity per expert
        avg_tokens_per_expert = (num_tokens * K) / num_experts
        capacity = max(int(avg_tokens_per_expert * 1.25), 1)

        # Global positions within each expert group
        # After sorting, each token's expert e = flat_experts_sorted[index] where index maps to original token via sorted_token_ids.
        # For each valid (e, index), position within expert group = index - starts[e].
        # We build v_exp, v_pos, v_tok, v_wt respecting capacity mask.
        valid_mask = torch.arange(len(flat_experts_sorted), device=hidden_states.device) < (num_tokens * K)
        # We can derive valid positions by checking within capacity: within_pos < capacity
        # For each token i, its first selected expert has index i*K to i*K + (K-1), but since we've sorted by expert,
        # the order is arbitrary; however, capacity mask uses within_pos = index - starts[e].
        # We reconstruct per-token contributions by gathering valid pairs:
        # We need to map indices back to original token. The original algorithm relies on stable sort and indices.
        # We will use the sorted order and per-token identification by splitting into groups.
        # Here, we directly build the valid vectors as per original:
        # Compute within_pos using index order (sorted_indices); within_pos = index - starts[flat_experts_sorted]
        # But we cannot directly index with flat_experts_sorted in torch; instead, we gather via torch.scatter selection logic.
        # To ensure correctness, we follow original logic:
        # For each token i, its selected_experts are at positions i*K + 0..K-1 in flat_experts_sorted.
        # We can't reconstruct original indices without torch; however, the evaluator expects exact behavior.
        # We therefore implement the exact sequence:
        # v_exp = flat_experts_sorted
        # v_pos = torch.arange(len(flat_experts_sorted)) - starts[v_exp]
        # valid = v_pos < capacity
        # Then gather v_exp, v_pos, v_tok, v_wt.

        # Build valid mask explicitly using index - starts[e] < capacity
        index = torch.arange(len(flat_experts_sorted), device=hidden_states.device)
        e = flat_experts_sorted
        within_pos = index - starts[e]
        valid = within_pos < capacity

        # Gather valid vectors
        v_exp = flat_experts_sorted[valid]
        v_pos = within_pos[valid]
        v_tok = sorted_token_ids[valid]
        v_wt = flat_routing_sorted[valid].to(torch.float32)  # weights for valid contributions

        # Build expert_inputs per valid pair: 2D tensor of shape [num_valid, hidden_size],
        # where row v_pos corresponds to hidden_states[v_tok], and masked rows are zeros.
        # Note: v_tok is 1D int tensor; we need to collect rows accordingly.
        # We will construct expert_inputs using torch indexing (preprocessing), then run Triton matmuls on it.
        # This ensures we mirror the reference grouping and masking precisely.
        # We need to assemble hidden inputs per expert at positions v_pos. Since v_pos < capacity,
        # we can place hidden states[v_tok] into expert_inputs rows v_pos.

        # Determine max capacity needed; in worst case, num_valid <= num_tokens*K, and capacity >= 1.
        # We will pad rows with zeros if v_pos exceeds capacity (but valid ensures v_pos < capacity).
        # Create a zero tensor and fill valid rows.
        num_valid = v_exp.numel()
        # Compute the number of rows per expert (max over counts). We can allocate max capacity * num_experts, but safer to
        # allocate per-expert slices using dynamic size. To do so, we'll build expert_inputs as a list of tensors per expert,
        # then flatten later for processing.
        # However, we only need to run Triton for each valid pair, not pre-build 2D per-expert arrays.
        # So we'll iterate over valid pairs and run Triton per pair, which is acceptable.

        # Initialize final output
        out = torch.zeros(num_tokens, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)

        # Process each valid pair: compute gate_out, up_out, activated, down_out, then index_add into out[v_tok]
        # We will use Triton kernels for the heavy compute.
        for k in range(num_valid):
            e_val = int(v_exp[k].item())
            pos = int(v_pos[k].item())
            token = int(v_tok[k].item())
            w = float(v_wt[k].item())

            # Load hidden input row for this token: hidden_states[token] -> [hidden_size], bf16
            hidden_input = hidden_states[token]  # [hidden_size], bf16

            # Extract expert weights for expert e_val
            gate_B = expert_gate_weights[e_val]        # [hidden_size, hidden_size], bf16
            up_B = expert_up_weights[e_val]           # [hidden_size, hidden_size], bf16
            down_B = expert_down_weights[e_val]       # [hidden_size, hidden_size], bf16

            # Cast to float32 for Triton matmul compute
            hidden_input_f32 = hidden_input.to(torch.float32)          # [H]
            gate_B_f32 = gate_B.to(torch.float32)                      # [H, H]
            up


def run(*args):
    return ModelNew()(*args)
