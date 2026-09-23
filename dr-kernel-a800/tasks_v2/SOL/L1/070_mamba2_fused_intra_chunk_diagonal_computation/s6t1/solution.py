import torch
import triton
import triton.language as tl


# Triton kernel for Y_diag reduction:
# Computes Y[b, n, i, h, :] = sum_j M[b, n, i, j, h] * hidden_states[b, n, j, h, :]
# Inputs:
#   M_ptr: [B, N, S, S, H] float32
#   HS_ptr: [B, N, S, H, D] float32
# Output:
#   Y_ptr: [B, N, S, H, D] float32
# Grid: (B, N, S, H)
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, HS_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_size,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    HS_stride_b, HS_stride_n, HS_stride_s, HS_stride_h, HS_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d,
    BLOCK_D: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    if (b >= B_size) or (n >= N_size) or (i >= S_size) or (h >= H_size):
        return

    # Accumulator vector across D
    acc_vec = tl.zeros([D_size], dtype=tl.float32)

    # Loop over j in [0..S_size-1], accumulate M[i,j,h] * hidden_states[b,n,j,h,:]
    for j in range(0, S_size):
        # Load scalar M[b, n, i, j, h]
        m_ptr = M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h
        m_val = tl.load(m_ptr).to(tl.float32)

        # Load hidden_states[b, n, j, h, :]
        hs_ptrs = HS_ptr + b * HS_stride_b + n * HS_stride_n + j * HS_stride_s + h * HS_stride_h + tl.arange(0, D_size) * HS_stride_d
        hs_mask = tl.arange(0, D_size) < D_size
        hs_vals = tl.load(hs_ptrs, mask=hs_mask, other=0.0).to(tl.float32)

        acc_vec += m_val * hs_vals

    # Store result Y[b, n, i, h, :]
    y_ptrs = Y_ptr + b * Y_stride_b + n * Y_stride_n + i * Y_stride_s + h * Y_stride_h + tl.arange(0, D_size) * Y_stride_d
    y_mask = tl.arange(0, D_size) < D_size
    tl.store(y_ptrs, acc_vec, mask=y_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguity
        device = hidden_states.device
        if hidden_states.device.type != 'cuda':
            hidden_states = hidden_states.cuda()
        if A_cumsum.device.type != 'cuda':
            A_cumsum = A_cumsum.cuda()
        if B.device.type != 'cuda':
            B = B.cuda()
        if C.device.type != 'cuda':
            C = C.cuda()

        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Shapes
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        assert A_cumsum.shape == (B_size, H_size, N_size, S_size), "A_cumsum must have shape [B, H, N, S]"

        # Step 1: Compute L using exact PyTorch logic (lower-triangular mask with diagonal=-1, cumsum, then exp)
        # A_cumsum: [B, H, N, S]
        # We need L: [B, H, N, S, S] where L[i,j] = exp(cumsum over source dim with lower-triangular mask).
        # Lower-triangular mask (exclude diagonal): j < i
        lower_mask = torch.tril(torch.ones(S_size, S_size, device=device, dtype=torch.float32), diagonal=-1)

        # Expand A_cumsum to [B, H, N, S, S] along last dim
        A_expanded = A_cumsum.unsqueeze(-1).expand(B_size, H_size, N_size, S_size, S_size).to(torch.float32)

        # Apply lower-triangular mask: set upper triangle (j >= i) to 0
        A_masked = A_expanded.masked_fill(~lower_mask.bool(), 0.0)

        # Cumsum along the source dimension (last axis = S)
        # Note: A_masked has shape [B, H, N, S, S]; cumsum along axis=-1 (the S axis) means for each (b,h,n,i, j), we sum along j?
        # This is not correct; cumsum should be along the original S dimension, not the last expanded S.
        # Instead, compute cumsum along axis=3 (original S axis) for each (b,h,n):
        A_cumsum_cum = torch.cumsum(A_cumsum, dim=3)  # shape [B, H, N, S]

        # Now compute L[i,j] = exp(sum_{k=0..i} A[b,h,n,k]) for j < i, else 0
        L = torch.zeros((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)
        for b_idx in range(B_size):
            for h_idx in range(H_size):
                for n_idx in range(N_size):
                    a_vec = A_cumsum_cum[b_idx, h_idx, n_idx, :]  # shape [S_size]
                    cumsum_seg = torch.cumsum(a_vec, dim=0)       # shape [S_size]
                    for j in range(S_size):
                        if j < S_size:  # always true
                            # For lower-triangular with diagonal=-1, we use j < i
                            for i_idx in range(S_size):
                                if j < i_idx:
                                    L[b_idx, h_idx, n_idx, i_idx, j] = torch.exp(cumsum_seg[i_idx])

        # Step 2: Compute G = sum over state of C * B (B/C expanded to H). Original code expands B/C by repeat_interleave(NUM_HEADS // N_GROUPS = 4).
        # We need to ensure B/C have last dim H_size. If not, expand.
        if B.shape[3] != H_size:
            raise RuntimeError("B must have last dimension equal to NUM_HEADS (hidden_states' H).")
        if C.shape[3] != H_size:
            raise RuntimeError("C must have last dimension equal to NUM_HEADS (hidden_states' H).")

        # Compute G[i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)

        # Strides for C and B
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C.stride()
        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        # Grid (B, N, S, S, H)
        grid_G = (B_size, N_size, S_size, S_size, H_size)
        g_contract_kernel[grid_G](
            C, B, G,
            B_size, N_size, S_size, H_size, D_size,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            BLOCK_D=64,  # D_size is typically 64
            num_warps=2, num_stages=2
        )

        # Step 3: Compute Y_diag via Triton reduction: Y[b,n,i,h,:] = sum_j G[b,n,i,j,h] * hidden_states[b,n,j,h,:]
        Y = torch.empty((B_size, N_size, S_size, H_size, D_size), dtype=torch.float32, device=device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = G.stride()
        HS_stride_b, HS_stride_n, HS_stride_s, HS_stride_h, HS_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d = Y.stride()

        grid_Y = (B_size, N_size, S_size, H_size)
        y_diag_reduce_kernel[grid_Y](
            G, hidden_states, Y,
            B_size, N_size, S_size, H_size, D_size,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            HS_stride_b, HS_stride_n, HS_stride_s, HS_stride_h, HS_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d,
            BLOCK_D=64,
            num_warps=2, num_stages=2
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
