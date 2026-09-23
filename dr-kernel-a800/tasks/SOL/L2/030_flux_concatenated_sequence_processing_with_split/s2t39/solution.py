import torch
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr,        # * [B, M=L, K=H]
    B_ptr,        # * [K=H, N=H] (process_weight.T)
    C_ptr,        # * [B, M=L, N=H]
    B, M, N, K,
    stride_a_b, stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,   # B_ptr strides for [K, N]
    stride_c_b, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(M / BLOCK_M), ceil(N / BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initial pointers for tiles
    a_ptrs = A_ptr + pid_b * stride_a_b + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
    b_ptrs = B_ptr + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n

    # Masks for edges
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    # Accumulate in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)

        # advance A along K
        a_ptrs += BLOCK_K * stride_a_k
        # advance B along K
        b_ptrs += BLOCK_K * stride_b_k

    # Store result
    c_ptrs = C_ptr + pid_b * stride_c_b + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: 画像潜伏列, [B, I, H]
        encoder_hidden_states: テキスト条件付け列, [B, T, H]
        process_weight: [H, H], これは本モデルの重みで、与えられたrun()と一致するはず。
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        # 引数の順序を意図せず保持するために、内側でスワップを避けて、直接使用する。
        # つまりhidden_statesは画像列(後ろ), encoder_hidden_statesはテキスト列(手前に)と捉える。
        # では、concatenationを画像列を最初、テキスト列を次に配置する形で作るためには...
        # concat([hidden, encoder], dim=1) → [B, L, H], L=I+T
        # そしてGEMMを行う。結果を分割する。

        # 各次元
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D [B, dim, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]  # テキストの長さ
        I = hidden_states.shape[1]          # 画像の長さ
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        # dtypeの一貫性
        assert hidden_states.dtype == encoder_hidden_states.dtype == process_weight.dtype, "All tensors must have the same dtype"
        # CUDAチェック
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton"

        L = I + T

        # Concatenate along sequence: [B, L, H], using PyTorch to ensure robustness and avoid shape surprises
        out_cat = torch.cat([hidden_states, encoder_hidden_states], dim=1)  # [B, L, H] with L=I+T

        # Prepare W_T = process_weight.T [H, H], contiguous for Triton
        W_T = process_weight.transpose(0, 1).contiguous()  # [H, H]

        # Allocate output
        processed = torch.empty((B, L, H), dtype=out_cat.dtype, device=out_cat.device)

        # GEMM parameters
        BLOCK_M = 64   # tiles over M=L
        BLOCK_N = 64   # tiles over N=H
        BLOCK_K = 64   # reduction tiles over K=H

        # 3D grid over (batch, tiles over M=L, tiles over N=H)
        grid = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        bmm_kernel[grid](
            out_cat, W_T, processed,
            B, L, H, H,  # M=L, N=H, K=H
            *out_cat.stride(),           # A strides: (along B, along M, along K)
            W_T.stride(1), W_T.stride(0),  # B strides: (along K, along N)
            *processed.stride(),         # C strides: (along B, along M, along N)
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split into encoder and hidden outputs
        processed_encoder = processed[:, :T, :]    # from original last T rows (original encoder positions)
        processed_hidden = processed[:, T:, :]     # from original first I rows (original hidden positions)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
