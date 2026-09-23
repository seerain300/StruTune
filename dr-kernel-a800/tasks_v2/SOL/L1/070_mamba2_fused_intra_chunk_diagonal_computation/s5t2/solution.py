import torch
import triton
import triton.language as tl

# Triton kernel to compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# Grid: (B, C, L, H)
@triton.jit
def y_diag_reduce_kernel(
    M_ptr,          # *float32, [B, C, L, L, H]
    HS_ptr,         # *float32, [B, C, L, H, D] = hidden_states
    Y_ptr,          # *float32, [B, C, L, H, D]
    B_size, C_size, L, H, D,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    # Vector of d offsets for this program (we only use contiguous D)
    d_offsets = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop over j to accumulate
    j = 0
    while j < L:
        # Load M[b, c, i, j, h]
        M_offset = b * (C_size * L * L * H) + c * (L * L * H) + i * (L * H) + j * H + h
        M_val = tl.load(M_ptr + M_offset)

        # Load hidden_states[b, c, j, h, d] vector for d in [0..BLOCK_D-1]
        HS_offset_base = b * (C_size * L * H * D) + c * (L * H * D) + j * (H * D) + h * D
        HS_vec = tl.load(HS_ptr + HS_offset_base + d_offsets, mask=d_offsets < D, other=0.0)

        acc += M_val * HS_vec
        j += 1

    # Store acc into Y[b, c, i, h, d]
    Y_offset_base = b * (C_size * L * H * D) + c * (L * H * D) + i * (H * D) + h * D
    tl.store(Y_ptr + Y_offset_base + d_offsets, acc, mask=d_offsets < D)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, C, L, H, D]
        A_cumsum:      [B, C, L, H]
        B:             [B, C, L, groups, S]
        C:             [B, C, L, groups, S]
        Returns:       [B, C, L, H, D] in bfloat16
        """

        B_size, C_size, L, H, D = hidden_states.shape
        groups = 8  # N_GROUPS
        GROUP_EXPAND = H // groups  # original uses NUM_HEADS=32, N_GROUPS=8 -> 4

        # Compute L (causal mask) using PyTorch to match original semantics exactly:
        # Original code expands A_cumsum to [B, H, C, L, L], applies lower-triangular mask (diagonal = -1),
        # then cumsum along -2 (internal chunk dim) and exp. We reconstruct this precisely.
        # Create an expanded view and apply mask/cumsum/exp.
        # Note: expand does not allocate storage; .to(torch.float32) is safe.
        A_expanded = A_cumsum.unsqueeze(1).unsqueeze(2)  # [B, 1, C, L, L]
        # Lower-triangular mask [L, L], diagonal = -1, then broadcast
        mask_ones = torch.ones((L, L), device=hidden_states.device, dtype=torch.float32)
        mask = torch.tril(mask_ones, diagonal=-1)  # [L, L]
        # Apply mask: zero out upper triangle
        A_masked = A_expanded * mask  # broadcast over B and C
        # Cumsum along internal chunk dimension (dim=-2): L
        cumsum = torch.cumsum(A_masked, dim=-2)  # [B, 1, C, L, L]
        # Exp for causal decay
        L_mat = torch.exp(cumsum)  # [B, 1, C, L, L], float32
        # We need L_mat shape [B, C, H, L, L] to multiply with M = [B, C, L, L, H]
        # Broadcast over H dimension: L_mat has no H dim; since M uses L_mat for each h, we can keep [B, 1, C, L, L] and broadcast in multiply.
        #


def run(*args):
    return ModelNew()(*args)
