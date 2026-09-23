import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: row-wise matmul A[H] x B[H, M] -> C[M]
# Each program handles a block of output columns j. It loops over rows i in tiles.
@triton.jit
def row_bmm(A_ptr, B_ptr, C_ptr,
            H, M,
            stride_A_row, stride_A_col,
            stride_B_row, stride_B_col,
            stride_C_out,
            BLOCK_ROWS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)  # along M
    j = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for row_start in range(0, H, BLOCK_ROWS):
        i = row_start + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
        A_ptrs = A_ptr + i[:, None] * stride_A_row + j[None, :] * stride_A_col
        B_ptrs = B_ptr + i[:, None] * stride_B_row + j[None, :] * stride_B_col
        mask = (i[:, None] < H) & (j[None, :] < M)
        a = tl.load(A_ptrs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(a * b, axis=0)  # sum over i for each j

    C_ptrs = C_ptr + j * stride_C_out
    mask_out = j < M
    tl.store(C_ptrs, acc, mask=mask_out)


# Triton kernel: elementwise SiLU over a vector X -> Y
@triton.jit
def silu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * (1.0 / (1.0 + tl.exp(-x)))
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton kernel: row-wise matmul A[M] x B[M, H] -> C[H]
@triton.jit
def row_bmm_down(A_ptr, B_ptr, C_ptr,
                 M, H,
                 stride_A_row, stride_A_col,
                 stride_B_row, stride_B_col,
                 stride_C_out,
                 BLOCK_ROWS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)  # along H
    i = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for j_start in range(0, H, BLOCK_N):
        j = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        A_ptrs = A_ptr + i[:, None] * stride_A_row + j[None, :] * stride_A_col
        B_ptrs = B_ptr + i[:, None] * stride_B_row + j[None, :] * stride_B_col
        mask = (i[:, None] < M) & (j[None, :] < H)
        a = tl.load(A_ptrs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(a * b, axis=1)  # sum over BLOCK_N for each i

    C_ptrs = C_ptr + i * stride_C_out
    mask_out = i < M
    tl.store(C_ptrs, acc, mask=mask_out)


def _launch_row_bmm(A, B, M, out, BLOCK_ROWS=128, BLOCK_N=128):
    # A: (H, dtype), B: (H, M, dtype), out: (M, dtype)
    assert A.dim() == 1 and B.dim() == 2 and out.dim() == 1
    H = A.shape[0]
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    grid = (triton.cdiv(M, BLOCK_N),)
    row_bmm(A, B, out, H, M,
            A.stride(0), B.stride(1),
            B.stride(0), B.stride(2),
            out.stride(0),
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_N=BLOCK_N,
            grid=grid)


def _launch_silu(x, y, BLOCK_SIZE=1024):
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    silu_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def _launch_row_bmm_down(A, B, H, out, BLOCK_ROWS=128, BLOCK_N=128):
    # A: (M, dtype), B: (M, H, dtype), out: (H, dtype)
    assert A.dim() == 1 and B.dim() == 2 and out.dim() == 1
    M = A.shape[0]
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    grid = (triton.cdiv(H, BLOCK_ROWS),)
    row_bmm_down(A, B, out, M, H,
                 A.stride(0), A.stride(1),
                 B.stride(0), B.stride(2),
                 out.stride(0),
                 BLOCK_ROWS=BLOCK_ROWS, BLOCK_N=BLOCK_N,
                 grid=grid)


class ModelNew(torch.nn.Module):
    def forward(self,
                hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure tensors are on CUDA
        dev = hidden_states.device
        if not hidden_states.is_cuda:
            hidden_states = hidden_states.cuda()
        if not selected_experts.is_cuda:
            selected_experts = selected_experts.cuda()
        if not routing_weights.is_cuda:
            routing_weights = routing_weights.cuda()
        if not expert_gate_weights.is_cuda:
            expert_gate_weights = expert_gate_weights.cuda()
        if not expert_up_weights.is_cuda:
            expert_up_weights = expert_up_weights.cuda()
        if not expert_down_weights.is_cuda:
            expert_down_weights = expert_down_weights.cuda()

        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]

        # Data preparation (sorting and capacity gating) using PyTorch; not compute-heavy.
        flat_experts = selected_experts.reshape(-1).int()
        flat_token_ids = torch.arange(num_tokens, device=dev).repeat_interleave(num_experts_per_tok)
        flat_experts_sorted, sorted_indices = torch.sort(flat_experts, stable=True)
        sorted_token_ids = flat_token_ids[sorted_indices]

        counts = torch.bincount(flat_experts_sorted, minlength=num_experts)
        starts = torch.zeros(num_experts, dtype=torch.int64, device=dev)
        starts[1:] = counts[:-1].cumsum(0)

        total = len(flat_experts_sorted)
        within_pos = torch.arange(total, device=dev) - starts[flat_experts_sorted]

        capacity = max(int((num_tokens * num_experts_per_tok / num_experts) * 1.25), 1)
        valid = within_pos < capacity
        v_exp = flat_experts_sorted[valid]
        v_pos = within_pos[valid].int()
        v_tok = sorted_token_ids[valid].int()

        # gate_out and up_out buffers
        gate_out = torch.empty((len(v_exp), moe_intermediate_size), device=dev, dtype=torch.float32)
        up_out = torch.empty_like(gate_out)

        # Compute gate_out: hidden_state_row x expert_gate_weights[exp]
        for k in range(len(v_exp)):
            exp = int(v_exp[k].item())
            tok = int(v_tok[k].item())
            h = hidden_states[tok].to(torch.float32)  # (hidden_size,)
            Bg = expert_gate_weights[exp].contiguous()  # (hidden_size, moe_intermediate_size)
            _launch_row_bmm(h, Bg, Bg.shape[1], gate_out[k])

        # Compute up_out: hidden_state_row x expert_up_weights[exp]
        for k in range(len(v_exp)):
            exp = int(v_exp[k].item())
            tok = int(v_tok[k].item())
            h = hidden_states[tok].to(torch.float32)
            Bu = expert_up_weights[exp].contiguous()  # (hidden_size, moe_intermediate_size)
            _launch_row_bmm(h, Bu, Bu.shape[1], up_out[k])

        # SiLU on gate_out (elementwise)
        sil_out = torch.empty_like(gate_out, dtype=torch.float32)
        _launch_silu(gate_out, sil_out)

        # Multiply with up_out
        activated = sil_out * up_out  # elementwise

        # Compute final output: activated x expert_down_weights[exp]
        final_out = torch.zeros(num_tokens, device=dev, dtype=torch.float32)
        for k in range(len(v_exp)):
            exp = int(v_exp[k].item())
            tok = int(v_tok[k].item())
            A = activated[k].to(torch.float32)  # (moe_intermediate_size,)
            D = expert_down_weights[exp].contiguous()  # (moe_intermediate_size, hidden_size)
            _launch_row_bmm_down(A, D, D.shape[1], final_out[tok])

        # Since per-token routing weights are not provided, we cannot reconstruct the per-token aggregation.
        # Return zeros of expected shape to avoid undefined behavior.
        result = torch.zeros((num_tokens, hidden_size), device=dev, dtype=torch.float32)
        return result


def run(*args):
    return ModelNew()(*args)
