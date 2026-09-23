import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides: (row=M, col=K)
    stride_bk, stride_bn,   # B strides: (row=K, col=N)
    stride_cm, stride_cn,   # C strides: (row=M, col=N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)          # batch id
    pid_m = tl.program_id(1)          # tile id along M
    pid_n = tl.program_id(2)          # tile id along N

    # compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # we will loop over K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # pointers for A and B tiles
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # load with masks
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # accumulate
        acc += tl.dot(a, b)

    # write back
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        returns (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        S = T + I

        # 1) Concatenate along sequence dim using torch (data movement)
        # Ensure contiguity
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, S, H]

        # 2) Allocate output for GEMM: [B*S, H]
        C_flat = torch.empty((B * S, H), device=concatenated.device, dtype=torch.float32)

        # 3) Triton batched GEMM: A = concatenated viewed as [B*S, H], B = process_weight.T
        # Ensure process_weight is contiguous and float32
        B_weight = process_weight.t().contiguous()  # [H, H]

        # 3D grid over (batch, tiles over M=B*S, tiles over N=H)
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (B, triton.cdiv(B * S, BLOCK_M), triton.cdiv(H, BLOCK_N))

        batched_matmul_kernel[grid](
            concatenated, B_weight, C_flat,
            B * S, H, H,
            concatenated.stride(0), concatenated.stride(2),  # A strides: (M=B*S, K=H)
            B_weight.stride(0), B_weight.stride(1),         # B strides: (K=H, N=H)
            C_flat.stride(0), C_flat.stride(1),             # C strides: (M=B*S, N=H)
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 4) Reshape back to [B, S, H]
        C = C_flat.view(B, S, H)

        # 5) Split using torch slicing (simple and correct)
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
