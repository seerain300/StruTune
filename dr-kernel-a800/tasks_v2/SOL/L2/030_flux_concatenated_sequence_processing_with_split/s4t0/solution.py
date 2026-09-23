import torch
import triton
import triton.language as tl

# Triton kernel: batched matrix multiply
# Computes C[M, H] = A[M, H] @ B[H, H]
# A: [M, H], B: [H, H], C: [M, H]
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, H,
    stride_am, stride_ah,
    stride_bh, stride_bk,
    stride_cm, stride_ch,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # compute row/col offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # masks for boundary conditions
    m_mask = offs_m < M
    n_mask = offs_n < H

    # accumulator for partial results
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension (hidden_dim)
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < H

        # load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ah)
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # load B tile: [BLOCK_K, BLOCK_N], B is [H, H] with strides (stride_bh, stride_bk)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bh + offs_n[None, :] * stride_bk)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # accumulate
        acc += tl.dot(a, b)

    # store results to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_ch)
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies linear projection via Triton batched matmul.
        - Splits back into separate encoder and image streams.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]  # text_seq_len
        I = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]          # hidden_dim

        # Concatenate sequences along sequence dimension: [B, T+I, H]
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        # Flatten to [M, H], where M = B * (T + I)
        M = B * (T + I)
        A = concatenated.reshape(M, H).contiguous()

        # Weight: process_weight is [H, H], pass as B for matmul with A [M, H] -> [M, H]
        # Ensure process_weight.T is contiguous [H, H]
        B_mat = process_weight.t().contiguous()  # [H, H]

        # Allocate output C [M, H]
        C = torch.empty((M, H), dtype=torch.float32, device=A.device)

        # Launch Triton kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        batched_matmul_kernel[grid](
            A, B_mat, C,
            M, H,
            A.stride(0), A.stride(1),
            B_mat.stride(0), B_mat.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Reshape back to [B, T+I, H]
        processed = C.reshape(B, T + I, H)

        # Split into encoder and hidden parts
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
