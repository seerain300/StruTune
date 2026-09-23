import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def batched_matmul_kernel(
    A_ptr,        # *ptr to A: [B, M, K]
    W_ptr,        # *ptr to W: [K, N]  (process_weight; we will use as W^T by indexing as [k, n])
    C_ptr,        # *ptr to C: [B, M, N]
    B: tl.int32,  # batch size
    M: tl.int32,  # sequence length after concat (L_txt + L_img)
    N: tl.int32,  # hidden_dim
    K: tl.int32,  # hidden_dim
    stride_ab: tl.int32,  # stride for batch dim in A
    stride_am: tl.int32,  # stride for M dim in A
    stride_ak: tl.int32,  # stride for K dim in A
    stride_wk: tl.int32,  # stride for K dim in W (row stride)
    stride_wn: tl.int32,  # stride for N dim in W (col stride)
    stride_cb: tl.int32,  # stride for batch dim in C
    stride_cm: tl.int32,  # stride for M dim in C
    stride_cn: tl.int32,  # stride for N dim in C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute offsets for the current tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A_tile: shape [BLOCK_M, BLOCK_K]
        # A[b, m, k] => offsets: b*stride_ab + m*stride_am + k*stride_ak
        a_ptrs = A_ptr + pid_b * stride_ab + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        A_tile = A_tile.to(tl.float32)

        # Load W^T tile: shape [BLOCK_K, BLOCK_N]
        # We want B[k, n] = W[k, n], but tl.dot expects A: [M, K], B: [K, N].
        # Indexing W_ptr as (k, n) with strides (stride_wk, stride_wn).
        w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        Wt_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        Wt_tile = Wt_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, Wt_tile)

    # Write back to C[b, m, n]
    c_ptrs = C_ptr + pid_b * stride_cb + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [batch, img_seq_len, hidden_dim]
        encoder_hidden_states: [batch, text_seq_len, hidden_dim]
        process_weight: [hidden_dim, hidden_dim]
        Returns: (processed_encoder_hidden_states, processed_hidden_states)
        """
        # Ensure we are on CUDA device; Triton requires CUDA
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        # Shapes
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert process_weight.shape == (D, D), f"process_weight must have shape [hidden_dim, hidden_dim], got {process_weight.shape}"

        # Concatenate along sequence dimension
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, L_txt + L_img, D]
        # Ensure contiguity for better performance
        concatenated = concatenated.contiguous()
        process_weight = process_weight.contiguous()

        # Allocate output processed: [B, L_txt + L_img, D]
        M = L_txt + L_img
        processed = torch.empty((B, M, D), device=concatenated.device, dtype=concatenated.dtype)

        # Compute grid based on tile sizes; Triton will pick a config among autotuned ones.
        # We can choose a conservative default for grid's last two dims using typical tile 64x64.
        # However, Triton's autotune configs will override block sizes; grid must be in terms of those blocks.
        # We'll set grid with lambda that uses meta['BLOCK_M'] and meta['BLOCK_N'].
        def grid(meta):
            return (
                B,
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(D, meta["BLOCK_N"]),
            )

        # Launch the Triton kernel. Note: We pass W_ptr directly; we use it as W^T by indexing as [k, n].
        batched_matmul_kernel[grid](
            concatenated,  # A: [B, M, D]
            process_weight,  # W: [D, D], we index as W[k, n] which corresponds to W^T[n, k] layout
            processed,      # C: [B, M, D]
            B, M, D, D,     # sizes: batch, M, N, K
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),  # A strides
            process_weight.stride(0), process_weight.stride(1),                      # W strides
            processed.stride(0), processed.stride(1), processed.stride(2),           # C strides
        )

        # Split back into separate streams
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
