import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr,         # *ptr to A, shape [M, H], M = B*(T+I)
    B_ptr,         # *ptr to B, shape [H, H] (process_weight.T)
    C_ptr,         # *ptr to C, shape [M, H]
    M, H,          # dimensions: rows in A/C and hidden_dim
    stride_am, stride_an,   # A strides: row and col
    stride_bh, stride_bk,   # B strides: row (hidden) and col (reduction dim)
    stride_cm, stride_cn,   # C strides: row and col
    BLOCK_M: tl.constexpr,  # tile in M dimension
    BLOCK_N: tl.constexpr,  # tile in N (hidden) dimension
    BLOCK_K: tl.constexpr,  # tile in reduction dim
):
    # 2D launch grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K (hidden_dim)
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A tile: A[offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_an
        # B tile: B[offs_n, offs_k] -> [BLOCK_N, BLOCK_K]
        B_ptrs = B_ptr + offs_n[:, None] * stride_bh + offs_k[None, :] * stride_bk

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < H)
        b_mask = (offs_n[:, None] < H) & (offs_k[None, :] < H)

        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate (B_tile is [BLOCK_N, BLOCK_K], transpose for dot)
        acc += tl.dot(A_tile, tl.trans(B_tile))

    # Store to C: C[offs_m, offs_n]
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < H)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only for heavy compute:
        - Concatenate along sequence dimension using torch.cat (data movement).
        - Compute matmul using Triton kernel: C = A @ process_weight.T
        - Split C into processed_encoder and processed_hidden.
        Returns (processed_encoder, processed_hidden) with shapes [B, T, H] and [B, I, H].
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, L, H]"
        assert hidden_states.shape[2] == encoder_hidden_states.shape[2] == process_weight.shape[0], "Hidden dim mismatch"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = B * (T + I)

        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        # Concatenate along sequence dimension: [B, T+I, H]
        A = torch.cat([enc, img], dim=1).contiguous()  # shape [B, T+I, H]
        # Linear projection weight transpose: [H, H]
        weight_T = process_weight.t().contiguous()  # [H, H]

        device = enc.device

        # Allocate output C: [M, H], compute in fp32 for stability
        C = torch.empty((M, H), dtype=torch.float32, device=device)

        # Launch Triton matmul kernel
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        batched_matmul_kernel[grid](
            A, weight_T, C,
            M, H,
            A.stride(0), A.stride(2),           # A strides: row (M) and hidden (H)
            weight_T.stride(0), weight_T.stride(1),  # B strides: row (H) and col (H)
            C.stride(0), C.stride(1),           # C strides
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split: processed = [B*(T+I), H]
        processed_encoder = C[:B * T].reshape(B, T, H)
        processed_hidden = C[B * T:].reshape(B, I, H)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
