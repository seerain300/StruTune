import torch
import triton
import triton.language as tl

# Triton kernel for batched matrix multiply without bias:
# A: [M, K], B: [K, N], C: [M, N]
@triton.jit
def batched_matmul_no_bias(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the tile this program instance will compute
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k

        # Pointers to A and B tiles
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak)
        B_tile_ptr = B_ptr + (k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; assume inputs are float16/float32; Triton will cast appropriately
        A = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        B = tl.load(B_tile_ptr, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A, B)

    # Write back to C
    C_tile_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates sequences along sequence dimension.
        - Performs the dense linear projection using a Triton GEMM.
        - Splits the processed result back into encoder and hidden streams.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]  # text_seq_len
        I = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]          # hidden_dim

        # Step 1: Concatenate along sequence dimension (on host)
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T + I, H]

        # Build A: [M, K] where M = B * (T + I), K = H
        # Flatten rows over batch and sequence
        M = B * (T + I)
        K = H
        # Reshape concatenated to [M, K]
        # Note: We want rows to be contiguous in sequence-major followed by batch-major flattening.
        # Because concatenation puts [B, T, H] then [B, I, H], flatten along the last dim, then along the sequence dim for each batch.
        # torch.reshape will do it in row-major; we can just view as 2D by stacking:
        # Since concatenated is [B, L, H], we can reshape to 2D by using reshape(-1, H):
        A = concatenated.reshape(M, K).contiguous()

        # B is process_weight.T with shape [K, N] where N = K = H (no bias)
        # Ensure process_weight is on the same device and dtype acceptable; we'll compute in fp32 and store to output dtype
        B = process_weight.t().contiguous()  # [H, H]

        # Output C: [M, N] = [B*(T+I), H], same dtype as process_weight (for consistency)
        C = torch.empty((M, K), device=A.device, dtype=process_weight.dtype)

        # Strides for Triton kernel
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Kernel launch configuration
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))

        # Launch Triton kernel
        batched_matmul_no_bias[grid](
            A, B, C,
            M, K, K,  # N == K because process_weight is [H, H]
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape C back to [B, T+I, H]
        processed = C.reshape(B, T + I, H)

        # Step 3: Split along sequence axis
        processed_encoder = processed[:, :T, :]  # [B, T, H]
        processed_hidden = processed[:, T:, :]   # [B, I, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
