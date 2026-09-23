import torch
import triton
import triton.language as tl

# Triton kernel: compute G[b, i, j, h] = sum over s of C[b, i, 0, h, s] * B[b, j, 0, h, s]
@triton.jit
def compute_G_ij_h(
    B_ptr, C_ptr, G_ptr,
    B_batch_stride, B_chunk_stride, B_k_stride, B_group_stride, B_s_stride,
    C_batch_stride, C_chunk_stride, C_k_stride, C_group_stride, C_s_stride,
    G_batch_stride, G_chunk_stride, G_j_stride, G_head_stride,
    CHUNK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    j = tl.program_id(2)
    h = tl.program_id(3)

    # Accumulate scalar G_val
    G_val = 0.0  # float32 accumulation
    for s in range(HEAD_DIM):
        # Address for B[j, 0, h, s]
        B_addr = B_ptr + b * B_batch_stride + i * B_chunk_stride + 0 * B_k_stride + h * B_group_stride + s * B_s_stride
        B_val = tl.load(B_addr).to(tl.float32)
        # Address for C[i, 0, h, s]
        C_addr = C_ptr + b * C_batch_stride + i * C_chunk_stride + 0 * C_k_stride + h * C_group_stride + s * C_s_stride
        C_val = tl.load(C_addr).to(tl.float32)
        G_val += B_val * C_val

    # Store G_val to G[b, i, j, h]
    G_addr = G_ptr + b * G_batch_stride + i * G_chunk_stride + j * G_j_stride + h * G_head_stride
    tl.store(G_addr, G_val)


# Triton kernel: compute Y_diag[b, i, k, h, d] = sum_j G[b, i, j, h] * hidden[b, i, k, j, h, d]
# Grid over (B, num_chunks, chunk_size, num_heads, head_dim). Each program computes one output element.
@triton.jit
def contract_hidden_into_Ydiag(
    hidden_ptr, G_ptr, Y_ptr,
    hidden_batch_stride, hidden_chunk_stride, hidden_k_stride, hidden_j_stride, hidden_d_stride,
    G_batch_stride, G_chunk_stride, G_j_stride, G_head_stride,
    Y_batch_stride, Y_chunk_stride, Y_k_stride, Y_head_stride, Y_d_stride,
    CHUNK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    j0 = tl.program_id(2)  # k index in hidden (we use k=0 as in original code)
    h = tl.program_id(3)
    d = tl.program_id(4)   # head_dim index

    # Accumulate sum over j of G[b, i, j, h] * hidden[b, i, j0, j, h, d]
    sum_val = 0.0
    for j in range(CHUNK_SIZE):
        # G[b, i, j, h]
        G_addr = G_ptr + b * G_batch_stride + i * G_chunk_stride + j * G_j_stride + h * G_head_stride
        G_val = tl.load(G_addr).to(tl.float32)
        # hidden[b, i, j0, j, h, d]
        hidden_addr = hidden_ptr + b * hidden_batch_stride + i * hidden_chunk_stride + j0 * hidden_k_stride + j * hidden_j_stride + d * hidden_d_stride
        hid_val = tl.load(hidden_addr).to(tl.float32)
        sum_val += G_val * hid_val

    # Store into Y[b, i, j0, h, d]
    Y_addr = Y_ptr + b * Y_batch_stride + i * Y_chunk_stride + j0 * Y_k_stride + h * Y_head_stride + d * Y_d_stride
    tl.store(Y_addr, sum_val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag using Triton:
          1) Triton kernel to compute G[b, i, j, h] = sum over s of C[b, i, 0, h, s] * B[b, j, 0, h, s]
          2) Triton kernel to contract Y[b, i, k, h, d] = sum_j G[b, i, j, h] * hidden[b, i, k, j, h, d]
        Returns Y_diag of shape [B, num_chunks, chunk_size, num_heads, head_dim], dtype bfloat16.
        """
        # Ensure tensors are contiguous
        hidden = hidden_states.contiguous().to(torch.float32)
        B = B.contiguous().to(torch.float32)
        C = C.contiguous().to(torch.float32)

        # Compute G with Triton: G[b, i, j, h] = sum over s of C[b, i, 0, h, s] * B[b, j, 0, h, s]
        B_dim = B.shape  # (B, num_chunks, chunk_size, n_groups, state_size)
        C_dim = C.shape
        B_batch, B_chunks, B_chunk_size, B_groups, B_state = B_dim
        C_batch, C_chunks, C_chunk_size, C_groups, C_state = C_dim
        # num_heads = 4 * n_groups (as per original code)
        num_heads = 4 * B_groups
        assert C_batch == B_batch and C_chunks == B_chunks and C_chunk_size == B_chunk_size and C_groups == B_groups and C_state == B_state

        G = torch.empty((B_batch, B_chunks, B_chunk_size, num_heads), dtype=torch.float32, device=B.device)

        # Strides for B, C, G
        B_batch_stride = B.stride(0); B_chunk_stride = B.stride(1); B_k_stride = B.stride(2); B_group_stride = B.stride(3); B_s_stride = B.stride(4)
        C_batch_stride = C.stride(0); C_chunk_stride = C.stride(1); C_k_stride = C.stride(2); C_group_stride = C.stride(3); C_s_stride = C.stride(4)
        G_batch_stride = G.stride(0); G_chunk_stride = G.stride(1); G_j_stride = G.stride(2); G_head_stride = G.stride(3)

        # Launch Triton kernel to compute G
        grid_G = (B_batch, B_chunks, B_chunk_size, num_heads)
        compute_G_ij_h[grid_G](
            B, C, G,
            B_batch_stride, B_chunk_stride, B_k_stride, B_group_stride, B_s_stride,
            C_batch_stride, C_chunk_stride, C_k_stride, C_group_stride, C_s_stride,
            G_batch_stride, G_chunk_stride, G_j_stride, G_head_stride,
            CHUNK_SIZE=B_chunk_size, HEAD_DIM=B_state,  # head_dim equals state_size (128)
            num_warps=1, num_stages=1,
        )

        # Allocate Y_diag in float32
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden.shape
        Y_diag = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=hidden.device)

        # Launch Triton contraction kernel
        # hidden strides
        hidden_batch_stride = hidden.stride(0)
        hidden_chunk_stride = hidden.stride(1)
        hidden_k_stride = hidden.stride(2)   # k (chunk_size) stride
        hidden_j_stride = hidden.stride(3)   # j stride
        hidden_d_stride = hidden.stride(4)   # head_dim stride

        # Y strides
        Y_batch_stride = Y_diag.stride(0)
        Y_chunk_stride = Y_diag.stride(1)
        Y_k_stride = Y_diag.stride(2)
        Y_head_stride = Y_diag.stride(3)
        Y_d_stride = Y_diag.stride(4)

        # Grid over (B, num_chunks, chunk_size, num_heads, head_dim); we use k=0 as per original code's expansion
        grid = (batch_size, num_chunks, chunk_size, num_heads, head_dim)

        contract_hidden_into_Ydiag[grid](
            hidden, G, Y_diag,
            hidden_batch_stride, hidden_chunk_stride, hidden_k_stride, hidden_j_stride, hidden_d_stride,
            G_batch_stride, G_chunk_stride, G_j_stride, G_head_stride,
            Y_batch_stride, Y_chunk_stride, Y_k_stride, Y_head_stride, Y_d_stride,
            CHUNK_SIZE=chunk_size,  # 128
            num_warps=1, num_stages=1,
        )

        # Cast to bfloat16 to match original output dtype
        Y_diag = Y_diag


def run(*args):
    return ModelNew()(*args)
