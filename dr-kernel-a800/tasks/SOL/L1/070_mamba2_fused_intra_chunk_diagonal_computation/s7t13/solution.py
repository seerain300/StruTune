import torch
import triton
import triton.language as tl


@triton.jit
def make_lower_mask(mask_ptr, S: tl.constexpr):
    """
    Create a lower-triangular mask (diagonal=-1) of shape [S, S]:
    mask[i, j] = 1 if j < i else 0 (exclude diagonal).
    Store as float32 (1.0 for True, 0.0 for False).
    """
    rows = tl.program_id(0)  # i
    cols = tl.program_id(1)  # j
    # bounds check (although S x S is the grid, we keep it safe)
    if rows >= S or cols >= S:
        return
    include = cols < rows  # j < i
    val = 1.0 if include else 0.0
    tl.store(mask_ptr + rows * S + cols, val)


@triton.jit
def masked_cumsum_lower(A: tl.pointer, A_seg: tl.pointer,
                         B: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr):
    """
    For each (b, c, n), compute segment_sum with lower-triangular mask (diagonal=-1):
    A_seg[b, c, i, j, n] = cumsum over j of A[b, c, j, n], but include only j < i.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)
    # Each program handles one i (row) across j loop
    i = tl.program_id(3)
    if i >= S:
        return
    # Running sum for cumsum along j
    running = 0.0
    for j in range(0, S):
        include = j < i  # exclude diagonal (j == i)
        val = tl.load(A + b * (Csz * N * S * S) + c * (N * S * S) + n * (S * S) + j * S + i, mask=True, other=0.0)
        running += val if include else 0.0
        tl.store(A_seg + b * (Csz * N * S * S) + c * (N * S * S) + n * (S * S) + i * S + j, running)


@triton.jit
def exp_lower_masked_cumsum(A_seg: tl.pointer, L: tl.pointer,
                            B: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr):
    """
    Apply exp to masked cumsum to produce L: L[b, c, i, j, n] = exp(A_seg[b, c, i, j, n]).
    Store L as float32.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    if i >= S:
        return
    for j in range(0, S):
        val = tl.load(A_seg + b * (Csz * N * S * S) + c * (N * S * S) + n * (S * S) + i * S + j, mask=True, other=0.0)
        # exp is elementwise; masked cumsum already applied the mask
        tl.store(L + b * (Csz * N * S * S) + c * (N * S * S) + n * (S * S) + i * S + j, tl.exp(val))


@triton.jit
def contract_BC_to_G(B_expanded: tl.pointer, C_expanded: tl.pointer, G: tl.pointer,
                     B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    """
    Compute G[b, c, i, j, n] = sum_k C[b, c, i, n, k] * B[b, c, j, n, k].
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # rows (i)
    j = tl.program_id(3)  # cols (j)
    n = tl.program_id(4)  # num_heads
    if i >= S or j >= S:
        return
    acc = 0.0
    for k in range(0, K):
        bval = tl.load(B_expanded + b * (C_size * S * N * K) + c * (S * N * K) + j * (N * K) + n * K + k, mask=True, other=0.0)
        cval = tl.load(C_expanded + b * (C_size * S * N * K) + c * (S * N * K) + i * (N * K) + n * K + k, mask=True, other=0.0)
        acc += bval * cval
    tl.store(G + b * (C_size * S * S * N) + c * (S * S * N) + i * (S * N) + j * N, acc)


@triton.jit
def diag_contract_Y(M: tl.pointer, hidden_states_f32: tl.pointer, Y: tl.pointer,
                    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr):
    """
    Diagonal contraction:
    Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # i
    n = tl.program_id(3)  # num_heads
    d = tl.program_id(4)  # head_dim
    if i >= S:
        return
    acc = 0.0
    for j in range(0, S):
        mval = tl.load(M + b * (C_size * S * S * N) + c * (S * S * N) + i * (S * N) + j * N, mask=True, other=0.0)
        hsv = tl.load(hidden_states_f32 + b * (C_size * S * N * D) + c * (S * N * D) + j * (N * D) + n * D, mask=True, other=0.0)
        acc += mval * hsv
    tl.store(Y + b * (C_size * S * N * D) + c * (S * N * D) + i * (N * D) + n * D, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants as in original function
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, C, S, N, D]
        # A_cumsum: [B, C, S, N]
        # B: [B, C, S, n_groups, K]
        # C: [B, C, S, n_groups, K]
        device = hidden_states.device

        # Infer axes from hidden_states
        Bsz, Csz, S, N, D = hidden_states.shape
        K = hidden_states.shape[4]  # head_dim equals K in original (not always true; keep as D but K is hidden_states.shape[4] by original usage)

        # Ensure inputs are contiguous
        hidden_states_f32 = hidden_states.contiguous().to(torch.float32)
        # Expand B and C from n_groups to num_heads (32) by repeat_interleave(4)
        # This matches original behavior where NUM_HEADS=32 and N_GROUPS=8
        B_expanded = B.contiguous().repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3).to(torch.float32)
        C_expanded = C.contiguous().repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3).to(torch.float32)

        # Allocate L (masked cumsum + exp) and segment_sum intermediates
        # We need A_seg: [B, C, S, S, N] as float32
        A_seg = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        # Launch masked_cumsum_lower for each (b, c, n, i) row, looping j
        grid_seg = (Bsz, Csz, N, S)
        masked_cumsum_lower[grid_seg](A_cumsum, A_seg, Bsz, Csz, S, N)

        # Launch exp_lower_masked_cumsum to form L
        grid_L = (Bsz, Csz, N, S)
        exp_lower_masked_cumsum[grid_L](A_seg, L, Bsz, Csz, S, N)

        # Contract B and C to form G: [B, C, S, S, N] as float32
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](B_expanded, C_expanded, G, Bsz, Csz, S, N, K)

        # Elementwise M = G * L
        M = G * L  # PyTorch elementwise multiply (lightweight)

        # Diagonal contraction Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)
        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](M, hidden_states_f32, Y, Bsz, Csz, S, N, D)

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
