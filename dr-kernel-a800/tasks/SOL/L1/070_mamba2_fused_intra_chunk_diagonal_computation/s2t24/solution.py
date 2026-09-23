import torch
import triton
import triton.language as tl

# Constants
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
HEAD_DIM = 128  # state_size and head_dim are 128 in the original model

@triton.jit
def compute_G_per_i_h_kernel(
    B_ptr, C_ptr, G_ptr,
    B_bs, B_cs, B_cd, B_w, B_sd,  # strides for B: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM]
    C_bs, C_cs, C_cd, C_w, C_sd,  # strides for C: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM]
    G_bs, G_hs, G_is,  # strides for G: [B, NUM_HEADS, num_chunks]
    num_chunks: tl.constexpr,
    num_heads: tl.constexpr,
):
    """
    For each (b, h), compute G[i, h] = sum over k, s of C[b, i, k, h, s] * B[b, j, k, h, s]
    for all i in [0..num_chunks-1], j in [0..CHUNK_SIZE-1], s in [0..HEAD_DIM-1].
    Store G[b, h, i] at G_ptr + b*G_bs + h*G_hs + i*G_is.
    """
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)

    # Accumulator for G[i, h]
    # We need a vector G of length num_chunks; Triton allows per-iteration scalar accumulation.
    G_vec = tl.zeros((num_chunks,), dtype=tl.float32)

    # Loop over j, k, s with compile-time bounds
    for i in range(num_chunks):
        G_vec[i] = 0.0
        for j in range(CHUNK_SIZE):
            for s in range(HEAD_DIM):
                # sum over k
                for k in range(CHUNK_SIZE):
                    # Load B[b, j, k, h, s]
                    B_ptr_ijks = B_ptr + b_id * B_bs + j * B_cs + k * B_cd + h_id * B_w + s * B_sd
                    B_val = tl.load(B_ptr_ijks)
                    # Load C[b, i, k, h, s]
                    C_ptr_icks = C_ptr + b_id * C_bs + i * C_cs + k * C_cd + h_id * C_w + s * C_sd
                    C_val = tl.load(C_ptr_icks)
                    G_vec[i] += (B_val * C_val)

    # Store G_vec to output G[b, h, :]
    for i in range(num_chunks):
        tl.store(G_ptr + b_id * G_bs + h_id * G_hs + i * G_is, G_vec[i])

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag using Triton:
        1) Compute G[i, h] in Triton.
        2) Assemble Y_diag = G[i, h] * hidden[b, i, k, 0, h, d] (since hidden has one j=0 slice).
        Returns tensor of shape [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM], dtype bfloat16.
        """
        # Shapes
        B_b, B_nc, B_cs, B_nh, B_hd = hidden_states.shape
        # Expand B and C to NUM_HEADS by repeating groups
        # num_heads = 32, n_groups = 8, so repeat_interleave 4
        B_exp = B.repeat_interleave(4, dim=3)
        C_exp = C.repeat_interleave(4, dim=3)

        # Allocate G: [B, NUM_HEADS, num_chunks] in float32
        G = torch.empty((B_b, NUM_HEADS, B_nc), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel
        grid = (B_b, NUM_HEADS)
        compute_G_per_i_h_kernel[grid](
            B_exp, C_exp, G,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2),
            num_chunks=B_nc, num_heads=NUM_HEADS,
        )

        # Prepare output Y_diag: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM], dtype bfloat16
        Y_diag = torch.empty((B_b, B_nc, CHUNK_SIZE, NUM_HEADS, HEAD_DIM), device=hidden_states.device, dtype=torch.bfloat16)

        # Fill Y_diag using simplified relation (hidden has one j=0 slice):
        # Y[b, i, k, h, d] = G[b, i, h] * hidden[b, i, k, 0, h, d]
        for b in range(B_b):
            for i in range(B_nc):
                g_val = G[b, :, i]  # shape [NUM_HEADS]
                # Broadcast g_val over k, h, d
                # hidden: [B, num_chunks, chunk_size, num_heads, head_dim] -> we only need j=0
                hidden_j0 = hidden_states[b, i, 0, :, :, :]  # shape [NUM_HEADS, HEAD_DIM]
                # Outer product and cast to bfloat16
                # g_val[:, None] has shape [NUM_HEADS, 1], broadcast over HEAD_DIM
                y_block = (g_val[:, None] * hidden_j0).to(torch.bfloat16)  # shape [NUM_HEADS, HEAD_DIM]
                # Assign to Y_diag[b, i, k, h, d] for all k in [0..CHUNK_SIZE-1]
                # We iterate k to fill slices; since hidden has only one j, value is constant across k.
                for k in range(CHUNK_SIZE):
                    Y_diag[b, i, k, :, :] = y_block

        return Y_diag


def run(*args):
    return ModelNew()(*args)
