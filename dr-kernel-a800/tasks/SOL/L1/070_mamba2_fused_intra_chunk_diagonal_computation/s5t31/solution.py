import torch
import triton
import triton.language as tl


@triton.jit
def compute_Y_diag_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    M_b, M_c, M_i, M_j, M_h,
    H_b, H_c, H_l, H_h, H_d,
    Y_b, Y_c, Y_i, Y_h, Y_d,
    L, D
):
    """
    Compute Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
    Grid: (B, C, I, H, D)
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    # Loop over j dimension (chunk length L)
    for j in range(0, L):
        m_val = tl.load(M_ptr + b * M_b + c * M_c + i * M_i + j * M_j + h * M_h)
        h_val = tl.load(hidden_ptr + b * H_b + c * H_c + j * H_l + h * H_h + d * H_d)
        acc += m_val * h_val

    tl.store(Y_ptr + b * Y_b + c * Y_c + i * Y_i + h * Y_h + d * Y_d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute intra-chunk diagonal output Y_diag for Mamba2 SSD, with Triton kernel for reduction.
        """
        # Extract shapes
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape

        # Cast to float32 for computation
        hidden_f32 = hidden_states.to(torch.float32)
        A_cumsum_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # 1) Build L (causal mask) using PyTorch to ensure correctness (avoid Triton errors for now).
        # Original behavior: for each (b, c, h), compute cumsum over L, then set L[i, j] = exp(cumsum) if j <= i, else 0.
        # We use 128x128 mask (as in original code), but A_cumsum has length 'chunk_size', so we pad to 128 internally.
        L = torch.empty((batch_size, num_chunks, 128, 128, num_heads), dtype=torch.float32, device=hidden_f32.device)
        for b in range(batch_size):
            for c in range(num_chunks):
                for h in range(num_heads):
                    # Cumulative sum over chunk length
                    cumsum = torch.cumsum(A_cumsum_f32[b, c, :chunk_size, h], dim=0).float()  # shape [chunk_size]
                    # Build 128x128 lower-triangular matrix
                    i = torch.arange(128, device=A_cumsum_f32.device).float().view(128, 1)  # rows
                    j = torch.arange(128, device=A_cumsum_f32.device).float().view(1, 128)  # cols
                    # j <= i gives lower triangle (diagonal=-1)
                    mask = (j <= i).float()
                    # exp of cumsum at valid positions: since cumsum has length chunk_size, we broadcast
                    # For simplicity, we set L[i, j] = exp(cumsum[i]) if j <= i, else 0.
                    # But cumsum only has length chunk_size, so for i >= chunk_size, we need zeros. We pad with zeros.
                    # Better: compute cumsum vector for length chunk_size, then pad zeros for i >= chunk_size.
                    # However, we need 128x128 matrix. We can set cumsum to zeros beyond chunk_size.
                    cumvec = torch.cat([cumsum, torch.zeros(128 - chunk_size, device=A_cumsum_f32.device, dtype=torch.float32)])
                    L_vals = torch.exp(cumvec[:128])  # exp(cumsum) for i in 0..127
                    L[b, c, :, :, h] = torch.where((j <= i), L_vals.view(128, 1), torch.zeros(128, 128, device=A_cumsum_f32.device, dtype=torch.float32)).float()

        # 2) Expand B and C from groups to heads using PyTorch repeat_interleave (GROUP_EXPAND=4).
        S_size = B_f32.shape[-1]
        B_exp = B_f32.repeat_interleave(num_heads // 8, dim=3)  # from 8 groups to 32 heads
        C_exp = C_f32.repeat_interleave(num_heads // 8, dim=3)

        # 3) Compute G via PyTorch contraction
        # G[b, c, i, j, h] = sum_s B_exp[b, c, i, h, s] * C_exp[b, c, j, h, s]
        G = torch.einsum('bciHs,bcjHs->bciHj', B_exp, C_exp)

        # 4) Apply mask M = G * L
        M = G * L

        # 5) Compute Y_diag via Triton kernel (reduction over j)
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=hidden_f32.device)

        # Launch Triton kernel with grid (B, C, L, H, D)
        grid_Y = (batch_size, num_chunks, chunk_size, num_heads, head_dim)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_f32, Y,
            *M.stride(), *hidden_f32.stride(), *Y.stride(),
            chunk_size, head_dim
        )

        # Return in bfloat16 as original function returns
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
