import torch
import triton
import triton.language as tl

# Triton kernel: compute G[i, j, h] = sum_s C[i, k, h, s] * B[j, k, h, s] for all (i, j, h)
# Grid: (B, num_chunks, NUM_HEADS)
@triton.jit
def compute_G_per_i_h(
    B_ptr, C_ptr, G_ptr,
    B_batch_stride, B_chunk_stride, B_k_stride, B_group_stride, B_s_stride,
    C_batch_stride, C_chunk_stride, C_k_stride, C_group_stride, C_s_stride,
    G_batch_stride, G_chunk_stride, G_j_stride, G_head_stride,
    NUM_HEADS: tl.constexpr, CHUNK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)

    # Vector G over j dimension
    G_vec = tl.zeros([CHUNK_SIZE], dtype=tl.float32)

    # Loop over k and s (compile-time constants)
    for k in range(CHUNK_SIZE):
        for s in range(HEAD_DIM):
            # For each j, compute B_val = B[j, k, h, s] and C_val = C[i, k, h, s], and update G_vec[j]
            # Note: Triton does not support dynamic indexing into tensors; we keep j in compile-time loops.
            for j in range(CHUNK_SIZE):
                # Addresses for B[j, k, h, s] and C[i, k, h, s]
                # B_ptr indexing: batch, chunk, k, group(h), s
                B_addr = B_ptr + b * B_batch_stride + i * B_chunk_stride + k * B_k_stride + h * B_group_stride + s * B_s_stride
                C_addr = C_ptr + b * C_batch_stride + i * C_chunk_stride + k * C_k_stride + h * C_group_stride + s * C_s_stride

                # Load values (assumes float32 tensors)
                B_val = tl.load(B_addr)  # scalar
                C_val = tl.load(C_addr)  # scalar
                G_vec[j] += B_val * C_val

    # Store G_vec to G[b, i, :, h]
    for j in range(CHUNK_SIZE):
        addr = G_ptr + b * G_batch_stride + i * G_chunk_stride + j * G_j_stride + h * G_head_stride
        tl.store(addr, G_vec[j])


def _launch_compute_G(B: torch.Tensor, C: torch.Tensor, G: torch.Tensor):
    """
    Launch compute_G_per_i_h Triton kernel.
    B: [B, num_chunks, chunk_size, n_groups, state_size] (n_groups=8, state_size=128)
    C: same shape
    G: [B, num_chunks, chunk_size, num_heads] float32
    """
    # Ensure tensors are contiguous
    B = B.contiguous()
    C = C.contiguous()

    B_batch_stride = B.stride(0)
    B_chunk_stride = B.stride(1)
    B_k_stride = B.stride(2)
    B_group_stride = B.stride(3)
    B_s_stride = B.stride(4)

    C_batch_stride = C.stride(0)
    C_chunk_stride = C.stride(1)
    C_k_stride = C.stride(2)
    C_group_stride = C.stride(3)
    C_s_stride = C.stride(4)

    G_batch_stride = G.stride(0)
    G_chunk_stride = G.stride(1)
    G_j_stride = G.stride(2)  # j dimension in G is second dim
    G_head_stride = G.stride(3)

    grid = (B.shape[0], B.shape[1], 32)
    compute_G_per_i_h[grid](
        B, C, G,
        B_batch_stride, B_chunk_stride, B_k_stride, B_group_stride, B_s_stride,
        C_batch_stride, C_chunk_stride, C_k_stride, C_group_stride, C_s_stride,
        G_batch_stride, G_chunk_stride, G_j_stride, G_head_stride,
        NUM_HEADS=32, CHUNK_SIZE=128, HEAD_DIM=128,
        num_warps=4
    )


def run(hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Compute Y_diag as per original logic, but implemented robustly.
    We use Triton to compute the reduction G and assemble Y_diag using a small torch loop.
    """
    # Ensure dtypes and contiguity
    hidden_states = hidden_states.contiguous().to(torch.float32)
    B = B.contiguous().to(torch.float32)
    C = C.contiguous().to(torch.float32)

    # Expand groups to heads (as in original)
    B_expanded = B.repeat_interleave(4, dim=3)  # 32 heads from 8 groups
    C_expanded = C.repeat_interleave(4, dim=3)

    # Allocate G: [B, num_chunks, chunk_size, num_heads] float32
    batch_size, num_chunks, chunk_size, num_groups_exp, state_size = B_expanded.shape
    assert num_groups_exp == 32, "Expansion to 32 heads required"
    G = torch.empty((batch_size, num_chunks, chunk_size, 32), dtype=torch.float32, device=B_expanded.device)

    # Launch Triton reduction to compute G[i, j, h]
    _launch_compute_G(B_expanded, C_expanded, G)

    # Now contract G with hidden_states to get Y_diag: [B, num_chunks, chunk_size, num_heads, head_dim]
    # hidden_states: [B, num_chunks, chunk_size, num_heads, head_dim]
    B_in, num_chunks_in, chunk_size_in, num_heads_in, head_dim = hidden_states.shape
    assert num_chunks == num_chunks_in and chunk_size == chunk_size_in, "Shape mismatch"
    Y_diag = torch.empty((B_in, num_chunks_in, chunk_size_in, num_heads_in, head_dim), dtype=torch.float32, device=hidden_states.device)

    # Contract: Y_diag[b, i, k, h, d] = sum_j G[b, i, j, h] * hidden[b, i, k, j, h, d]
    for b in range(B_in):
        for i in range(num_chunks_in):
            for k in range(chunk_size_in):
                for h in range(num_heads_in):
                    Y_diag[b, i, k, h, :] = torch.zeros(head_dim, dtype=torch.float32, device=hidden_states.device)
                    for j in range(chunk_size_in):
                        g = G[b, i, j, h]  # scalar float32
                        hs = hidden_states[b, i, k, j, h, :]  # [head_dim] float32
                        Y_diag[b, i, k, h, :] += g * hs

    # Cast to bfloat16 to match original return dtype
    Y_diag = Y_diag.to(torch.bfloat16)
    return Y_diag


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Launch Triton in forward; no torch ops here except minimal allocation.
        # The heavy reduction G is computed via Triton; contraction uses small torch loops.
        return run(hidden_states, A_cumsum, B, C)


def run(*args):
    return ModelNew()(*args)
