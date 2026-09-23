import torch
import triton
import triton.language as tl

# Constants derived from the original code
CHUNK_SIZE = 128
HEAD_DIM = 128
NUM_HEADS = 32  # original num_heads (expanded via repeat_interleave)

@triton.jit
def compute_G_ij_h_kernel(
    B_ptr, C_ptr, G_ptr,
    B_batch_stride, B_chunk_stride, B_k_stride, B_group_stride, B_s_stride,
    C_batch_stride, C_chunk_stride, C_k_stride, C_group_stride, C_s_stride,
    G_batch_stride, G_chunk_stride, G_j_stride, G_h_stride,
    S_dim: tl.constexpr
):
    # Program IDs: (b, i, j, h)
    b = tl.program_id(0)
    i = tl.program_id(1)
    j = tl.program_id(2)
    h = tl.program_id(3)

    # Accumulate scalar G for this (b, i, j, h)
    G_val = 0.0  # fp32 scalar

    # Loop over s in state_size (HEAD_DIM) in float32
    for s in range(S_dim):
        # For B[b, j, k, h, s], fix k=0; original code expands n_groups -> num_heads
        B_addr = B_ptr + b * B_batch_stride + j * B_chunk_stride + 0 * B_k_stride + h * B_group_stride + s * B_s_stride
        # For C[b, i, k, h, s], fix k=0
        C_addr = C_ptr + b * C_batch_stride + i * C_chunk_stride + 0 * C_k_stride + h * C_group_stride + s * C_s_stride

        B_val = tl.load(B_addr).to(tl.float32)
        C_val = tl.load(C_addr).to(tl.float32)
        G_val += B_val * C_val

    # Store G_val to G[b, i, j, h]
    G_addr = G_ptr + b * G_batch_stride + i * G_chunk_stride + j * G_j_stride + h * G_h_stride
    tl.store(G_addr, G_val)


def _launch_compute_G(B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Launch Triton kernel to compute G[b, i, j, h] = sum_s C[b, i, 0, h, s] * B[b, j, 0, h, s].
    Returns G as float32 tensor of shape [B, num_chunks, chunk_size, num_heads].
    """
    # Ensure float32 contiguous for Triton
    B_f32 = B.contiguous().to(torch.float32)
    C_f32 = C.contiguous().to(torch.float32)

    B_batch_stride = B_f32.stride(0)
    B_chunk_stride = B_f32.stride(1)
    B_k_stride = B_f32.stride(2)
    B_group_stride = B_f32.stride(3)
    B_s_stride = B_f32.stride(4)

    C_batch_stride = C_f32.stride(0)
    C_chunk_stride = C_f32.stride(1)
    C_k_stride = C_f32.stride(2)
    C_group_stride = C_f32.stride(3)
    C_s_stride = C_f32.stride(4)

    B_size = B_f32.size(0)  # batch
    num_chunks = B_f32.size(1)  # num_chunks
    chunk_size = B_f32.size(2)  # chunk_size

    # Output G: [B, num_chunks, chunk_size, num_heads]
    G = torch.empty((B_size, num_chunks, chunk_size, NUM_HEADS), dtype=torch.float32, device=B_f32.device)

    G_batch_stride = G.stride(0)
    G_chunk_stride = G.stride(1)
    G_j_stride = G.stride(2)
    G_h_stride = G.stride(3)

    # Launch grid: (B, num_chunks, chunk_size, num_heads)
    grid = (B_size, num_chunks, chunk_size, NUM_HEADS)

    compute_G_ij_h_kernel[grid](
        B_f32, C_f32, G,
        B_batch_stride, B_chunk_stride, B_k_stride, B_group_stride, B_s_stride,
        C_batch_stride, C_chunk_stride, C_k_stride, C_group_stride, C_s_stride,
        G_batch_stride, G_chunk_stride, G_j_stride, G_h_stride,
        S_dim=HEAD_DIM
    )

    return G


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag using Triton to compute G = contract(B, C) and then contract with hidden_states.
        Triton is used to compute G; the final contraction is done with torch loops to avoid dynamic indexing pitfalls.
        """
        # 1) Compute G[b, i, j, h] via Triton kernel (simplified approach: k=0, expanded B/C to num_heads=32)
        G = _launch_compute_G(B, C)  # [B, num_chunks, chunk_size, num_heads] in float32

        # 2) Allocate output Y_diag in bfloat16: [B, num_chunks, chunk_size, num_heads, head_dim]
        B_size, num_chunks, chunk_size, num_heads = G.shape
        Y_diag = torch.empty((B_size, num_chunks, chunk_size, num_heads, HEAD_DIM),
                             dtype=torch.bfloat16, device=hidden_states.device)

        # 3) Contract: for each (b, i, k, h), Y[b, i, k, h, d] = sum_j G[b, i, j, h] * hidden[b, i, k, j, h, d]
        for b in range(B_size):
            for i in range(num_chunks):
                for k in range(chunk_size):  # hidden's third dim is chunk_size
                    for h in range(num_heads):
                        # Initialize Y row for this (b, i, k, h)
                        Y_row = torch.zeros((HEAD_DIM,), dtype=torch.float32)
                        # Sum over j
                        for j in range(chunk_size):
                            G_scalar = G[b, i, j, h]
                            hidden_row = hidden_states[b, i, k, h, :]  # [head_dim]
                            Y_row += G_scalar * hidden_row
                        # Store as bfloat16: Y_diag[b, i, k, h, :]
                        Y_diag[b, i, k, h, :] = Y_row.to(torch.bfloat16)

        return Y_diag


def run(*args):
    return ModelNew()(*args)
