import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, H, K,
    stride_am, stride_ak,   # strides for A [M, H]
    stride_bh, stride_bk,   # strides for B [H, K]
    stride_cm, stride_ck,   # strides for C [M, K]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B, where:
      - A: [M, H], strides (stride_am, stride_ak)
      - B: [H, K], strides (stride_bh, stride_bk)
      - C: [M, K], strides (stride_cm, stride_ck)
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension H in tiles of BLOCK_K
    for k0 in range(0, H, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < H

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_range[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_range[:, None] * stride_bh + offs_n[None, :] * stride_bk)

        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_ck)
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        1) Concatenate sequences along the sequence dimension using torch.cat.
        2) Reshape concatenated tensor to [M, H] where M = batch * (T + I), H = hidden_dim.
        3) Compute C = A @ process_weight.T via Triton batched matmul (no torch operations).
        4) Reshape C back to [batch, T+I, H] and split into processed_encoder and processed_hidden.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton"

        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]   # text_seq_len
        I = hidden_states.shape[1]           # img_seq_len
        H = hidden_states.shape[2]           # hidden_dim
        K = process_weight.shape[0]          # should equal H

        # Concatenate sequences along sequence dimension: [B, T+I, H]
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1).contiguous()  # [B, T+I, H]
        T_out = T + I

        # Create A [M, H] where M = B * (T + I)
        M = B * T_out
        A = concatenated.view(M, H).contiguous()  # [M, H]

        # B = process_weight.T [H, H]
        weight_T = process_weight.t().contiguous()  # [H, H]

        # Allocate C [M, H]
        C = torch.empty((M, H), dtype=A.dtype, device=A.device)

        # Launch Triton matmul kernel: C = A @ weight_T
        # 1D grid over tiles of M
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        batched_matmul_kernel[grid](
            A, weight_T, C,
            M, H, K,
            A.stride(0), A.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Reshape back to [B, T+I, H]
        processed = C.view(B, T_out, H)

        # Split into encoder and hidden parts
        processed_encoder = processed[:, :T, :]  # [B, T, H]
        processed_hidden = processed[:, T:, :]   # [B, I, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
