import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,       # *T: [M, K]
    W_ptr,       # *T: [K, N]  (note: process_weight.T here)
    C_ptr,       # *T: [M, N]
    M: tl.int32, K: tl.int32, N: tl.int32,
    A_s0: tl.int32, A_s1: tl.int32,
    W_s0: tl.int32, W_s1: tl.int32,
    C_s0: tl.int32, C_s1: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * A_s0 + k_offsets[None, :] * A_s1
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # W tile: [BLOCK_K, BLOCK_N], W has shape [K, N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store results to C
    c_ptrs = C_ptr + m_offsets[:, None] * C_s0 + n_offsets[None, :] * C_s1
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    # Store as float32; if input is float32, this is fine. For half/bf16, consider casting.
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_kernel(
    src_ptr,     # *T: [M, N] (processed)
    dst_ptr,     # *T: [B, I_or_T, N] (output)
    B: tl.int32, L: tl.int32, N: tl.int32,
    src_s0: tl.int32, src_s1: tl.int32,
    dst_s0: tl.int32, dst_s1: tl.int32, dst_s2: tl.int32,
    start_row: tl.int32,  # starting row to copy
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l_offsets = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # rows to copy [0..L)
    h_offsets = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # columns [0..N)

    # mask: valid batch, valid row in [start_row, start_row+L), valid columns
    mask = (l_offsets < L) & (h_offsets < N) & (pid_b == 0)  # pid_b is 0..B-1

    # src row indices: l_offsets + start_row
    src_rows = start_row + l_offsets

    # Compute pointers
    src_ptrs = src_ptr + src_rows[:, None] * src_s0 + h_offsets[None, :] * src_s1

    # Load
    vals = tl.load(src_ptrs, mask=mask, other=0.0)

    # dst base for this batch
    dst_base = dst_ptr + pid_b * dst_s0
    dst_ptrs = dst_base + (start_row + l_offsets)[:, None] * dst_s1 + h_offsets[None, :] * dst_s2

    # Store
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, I, H] (image latents)
        encoder_hidden_states: [B, T, H] (text conditioning)
        process_weight: [H, H]
        returns (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H
        assert process_weight.shape[0] == H and process_weight.shape[1] == H

        # 1) Concatenate along sequence dimension: [B, T+I, H]
        # Using torch.cat here (on device) to simplify host-side logic; Triton kernel will handle matmul.
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 2) Compute processed = concatenated @ process_weight.T
        # Shapes: A [M,K] where M=B*(T+I), K=H; W [K,N]=[H,H]; C [M,N]=[B*(T+I), H]
        M = B * (T + I)
        K = H
        N = H
        A = concatenated.reshape(M, K).contiguous()  # [M,K]
        W_t = process_weight.t().contiguous()        # [K,N] where N=H

        # Allocate output C [M, N]
        C = torch.empty((M, N), dtype=torch.float32, device=concatenated.device)  # accumulate/store in float32

        # Launch matmul kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_kernel[grid](
            A, W_t, C,
            M, K, N,
            A.stride(0), A.stride(1),
            W_t.stride(0), W_t.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Reshape back to [B, T+I, H]
        processed = C.reshape(B, T + I, H)

        # 3) Split streams
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=processed.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=processed.device)

        # Triton kernels to copy rows:
        # Copy first T rows: rows 0..T-1
        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=0,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows: rows T..T+I-1
        grid_copy_image = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_image](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row=T,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
