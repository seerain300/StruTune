import torch

# Original run logic (corrected and self-contained)
@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A_cumsum: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> torch.Tensor:
    """
    Compute intra-chunk diagonal output Y_diag for Mamba2 SSD.

    Steps:
    1) Compute segment_sum(A) with lower-triangular masking for causality, then exp() -> L.
    2) Contract B and C matrices to form attention-like weights G.
    3) Apply mask: M = G * L element-wise.
    4) Compute output: Y_diag = sum(M * hidden_states) over sequence dimension.
    """
    # Constants as in the original code
    CHUNK_SIZE = 128
    NUM_HEADS = 32
    N_GROUPS = 8
    REPEAT = NUM_HEADS // N_GROUPS  # 4
    STATE_SIZE = 64
    HEAD_DIM = 64

    # Ensure input dimensions match expected shapes
    batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape
    assert chunk_size == CHUNK_SIZE, f"chunk_size must be {CHUNK_SIZE}, got {chunk_size}"
    assert num_heads == NUM_HEADS, f"num_heads must be {NUM_HEADS}, got {num_heads}"
    assert head_dim == HEAD_DIM, f"head_dim must be {HEAD_DIM}, got {head_dim}"

    # Step 1: Compute segment_sum(A) with lower-triangular mask (diagonal = -1)
    # A_cumsum: [B, H, N, K, K]
    device = hidden_states.device
    dtype = torch.float32

    # Build mask [K, K], lower-triangular including diagonal
    mask = torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float32),
        diagonal=0
    )

    # Expand A_cumsum to [B, H, N, K, K] (already this shape)
    A_cumsum_exp = A_cumsum.to(dtype)

    # Mask upper triangle to zero
    A_masked = A_cumsum_exp.masked_fill(~mask, 0.0)

    # Cumulative sum along the sequence dimension j (last axis -2)
    # After masking, only lower triangle contributes; upper is zero.
    A_cumsum_seg = torch.cumsum(A_masked, dim=-2)  # [B, H, N, K, K]

    # Apply exp to get L: [B, H, N, K, K]
    L = torch.exp(A_cumsum_seg)

    # Step 2: Contract B and C to form G: G[i, j, h] = sum_s C[b, n, i, g, s] * B[b, n, j, g, s]
    # B: [B, N, K, n_groups, STATE_SIZE], C: [B, N, K, n_groups, STATE_SIZE]
    Bt = B.to(torch.float32)
    Ct = C.to(torch.float32)

    # Expand n_groups -> heads: g = h // REPEAT (REPEAT=4)
    G = torch.empty((batch_size, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS), device=device, dtype=torch.float32)

    # Loop over (b, n) and compute G for all (i, j, h)
    for b in range(batch_size):
        for n in range(num_chunks):
            # Accumulate over groups
            for h in range(NUM_HEADS):
                g = h // REPEAT  # 0..7 -> heads 0..31
                acc = 0.0
                for i in range(CHUNK_SIZE):
                    for j in range(CHUNK_SIZE):
                        # sum over STATE_SIZE in blocks
                        for s_start in range(0, STATE_SIZE, 64):
                            s = s_start + torch.arange(64, device=device)
                            mask_s = s < STATE_SIZE
                            B_vals = Bt[b, n, j, g, s].to(torch.float32)  # [64]
                            C_vals = Ct[b, n, i, g, s].to(torch.float32)  # [64]
                            acc += torch.sum(B_vals * C_vals, dim=0)
                G[b, n, :, :, h] = acc  # same scalar for all (i,j), but original code multiplies G by L per (i,j)
    # Note: The original PyTorch code sets G[i,j,h] per (i,j) based on B and C. Here we use the expanded heads and scalar accumulation,
    # but to match exact semantics, we should compute per (i,j,h). The above is a simplification; for correctness, we keep the original G formation in torch.
    # To fully mirror original, we recompute G with per-(i,j,h) accumulation:
    # Recompute G correctly by iterating i,j,h and s:
    # Initialize G to zeros
    G = torch.zeros((batch_size, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS), device=device, dtype=torch.float32)
    for b in range(batch_size):
        for n in range(num_chunks):
            for i in range(CHUNK_SIZE):
                for j in range(CHUNK_SIZE):
                    for h in range(NUM_HEADS):
                        g = h // REPEAT
                        acc = 0.0
                        for s_start in range(0, STATE_SIZE, 64):
                            s = s_start + torch.arange(64, device=device)
                            mask_s = s < STATE_SIZE
                            B_vals = Bt[b, n, j, g, s].to(torch.float32)
                            C_vals = Ct[b, n, i, g, s].to(torch.float32)
                            acc += torch.sum(B_vals * C_vals, dim=0)
                        G[b, n, i, j, h] = acc

    # Step 3: Apply causal mask L to G element-wise: M = G * L
    M = G * L  # shapes match: [B, N, K, K, H]

    # Step 4: Compute Y_diag by applying M to hidden_states
    # hidden_states: [B, N, K, H, D]
    hidden = hidden_states.to(torch.float32)

    # Output: [B, N, K, H, D]
    Y_diag = torch.empty((batch_size, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM), device=device, dtype=torch.float32)

    for b in range(batch_size):
        for n in range(num_chunks):
            for i in range(CHUNK_SIZE):
                for h in range(NUM_HEADS):
                    # sum over j
                    acc = torch.zeros((HEAD_DIM,), device=device, dtype=torch.float32)
                    for j in range(CHUNK_SIZE):
                        m_vec = M[b, n, i, j, h]  # scalar
                        h_vec = hidden[b, n, j, h, :]  # [D]
                        acc += m_vec * h_vec
                    Y_diag[b, n, i, h, :] = acc

    return Y_diag.to(torch.bfloat16)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Entry point as requested. Uses the original run logic for correctness across varied axes.
        """
        return run(hidden_states, A_cumsum, B, C)


def run(*args):
    return ModelNew()(*args)
