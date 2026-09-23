import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (batch, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute offsets for this program instance
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first block
    A_block_ptr = A_ptr + pid_b * stride_am + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_block_ptr = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k_iter = 0
    while k_iter < K:
        k_mask = (offs_k[None, :] + k_iter) < K
        a = tl.load(A_block_ptr, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        b = tl.load(B_block_ptr, mask=k_mask & (offs_n[None, :] < N), other=0.0)
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)
        # advance by BLOCK_K
        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # Write back with masks
    C_block_ptr = C_ptr + pid_b * stride_cm + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_block_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function:
        Concatenates encoder_hidden_states and hidden_states along the sequence dimension,
        applies linear projection via matmul with process_weight.T, and splits back.
        The core computation (matmul) is performed by Triton kernels. No torch.matmul is used.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA for Triton kernels."

        # Shapes
        batch = hidden_states.shape[0]
        text_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]
        hidden_dim = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == hidden_dim, "Mismatched hidden_dim"
        assert process_weight.shape == (hidden_dim, hidden_dim), "process_weight must be [hidden_dim, hidden_dim]"

        # Make sure inputs are contiguous for Triton
        A_enc = encoder_hidden_states.contiguous()           # [batch, text_seq_len, hidden_dim]
        A_img = hidden_states.contiguous()                   # [batch, img_seq_len, hidden_dim]
        B = process_weight.t().contiguous()                  # [hidden_dim, hidden_dim]

        # We'll compute outputs for each slice and concatenate along sequence dim.
        # Choose tiling parameters. These are robust for hidden_dim ~ 1024 and typical seq lengths.
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        # Helper to compute grid: (batch, tiles over M, tiles over N)
        def grid_fn(M_slice, N_out):
            grid_m = triton.cdiv(M_slice, BLOCK_M)
            grid_n = triton.cdiv(N_out, BLOCK_N)
            return (batch, grid_m, grid_n)

        # Output buffers for each slice (per batch)
        out_encoder_list = []
        out_img_list = []

        # Kernel launch for encoder slice
        M_encoder = text_seq_len
        N_out = hidden_dim
        # For each batch item
        for b in range(batch):
            # A: [M, K] contiguous from [batch, seq, hidden]
            A = A_enc[b]  # [text_seq_len, hidden_dim]
            # Allocate output [M, N]
            C = torch.empty((M_encoder, N_out), device=A.device, dtype=torch.float32)
            grid = grid_fn(M_encoder, N_out)
            _batched_matmul_kernel[grid](
                A, B, C,
                M_encoder, N_out, hidden_dim,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )
            out_encoder_list.append(C)

        # Kernel launch for image slice
        M_img = img_seq_len
        for b in range(batch):
            A = A_img[b]  # [img_seq_len, hidden_dim]
            C = torch.empty((M_img, N_out), device=A.device, dtype=torch.float32)
            grid = grid_fn(M_img, N_out)
            _batched_matmul_kernel[grid](
                A, B, C,
                M_img, N_out, hidden_dim,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )
            out_img_list.append(C)

        # Stack per-batch outputs along sequence dimension: [batch, seq_len, hidden_dim]
        processed_encoder = torch.stack(out_encoder_list, dim=0)  # [batch, text_seq_len, hidden_dim]
        processed_hidden = torch.stack(out_img_list, dim=0)       # [batch, img_seq_len, hidden_dim]

        # Note: original code returns (processed_encoder, processed_hidden). We match that.
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
