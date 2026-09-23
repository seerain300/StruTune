import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,            # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,            # *ptr to hidden_states [B, I, H]
    out_ptr,            # *ptr to concatenated [B, S, H], S = T + I
    B, T, I, H, S,
    stride_enc_b, stride_enc_s, stride_enc_h,
    stride_hid_b, stride_hid_s, stride_hid_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr,
):
    # Grid: (B, ceil(T / BLOCK_T), ceil(I / BLOCK_I))
    b = tl.program_id(0)
    t_tile = tl.program_id(1)
    i_tile = tl.program_id(2)

    t_offsets = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    i_offsets = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_t = t_offsets < T
    mask_i = i_offsets < I

    # Copy encoder rows into out[:, :T, :]
    for tt in range(0, BLOCK_T):
        t_idx = t_offsets[tt]
        if mask_t[tt]:
            src_row_ptr = enc_ptr + b * stride_enc_b + t_idx * stride_enc_s
            dst_row_ptr = out_ptr + b * stride_out_b + t_idx * stride_out_s
            x = tl.load(src_row_ptr + tl.arange(0, H) * stride_enc_h, mask=True, other=0.0)
            tl.store(dst_row_ptr + tl.arange(0, H) * stride_out_h, x, mask=True)

    # Copy hidden rows into out[:, T:, :]
    for ii in range(0, BLOCK_I):
        i_idx = i_offsets[ii]
        if mask_i[ii]:
            src_row_ptr = hid_ptr + b * stride_hid_b + i_idx * stride_hid_s
            dst_row_ptr = out_ptr + b * stride_out_b + (T + i_idx) * stride_out_s
            x = tl.load(src_row_ptr + tl.arange(0, H) * stride_hid_h, mask=True, other=0.0)
            tl.store(dst_row_ptr + tl.arange(0, H) * stride_out_h, x, mask=True)


@triton.jit
def matmul_seqs_kernel(
    A_ptr,      # *ptr to concatenated A [S, H], view as [B*S, H]
    B_ptr,      # *ptr to process_weight.T [H, H]
    C_ptr,      # *ptr to output [B*S, H]
    M,          # total rows = B*S
    N,          # cols = H
    K,          # reduction dim = H
    stride_A_m, stride_A_k,
    stride_B_k, stride_B_n,
    stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (ceil(M / BLOCK_M), ceil(N / BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # A submatrix: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_A_m + k_offsets[None, :] * stride_A_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B submatrix: [BLOCK_K, BLOCK_N], note B is [H, H] (K, N)
        b_ptrs = B_ptr + k_offsets[:, None] * stride_B_k + n_offsets[None, :] * stride_B_n
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # acc += a @ b
        acc += tl.dot(a, b)

    # Write back to C
    c_ptrs = C_ptr + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def split_rows_kernel(
    C_ptr,            # *ptr to processed [B*S, H]
    out_ptr,          # *ptr to output [B, T_or_I, H]
    B, S, T_or_I, H,
    stride_C_m, stride_C_n,
    stride_out_b, stride_out_s, stride_out_n,
    BLOCK_S: tl.constexpr,
):
    # Grid: (B, ceil(T_or_I / BLOCK_S))
    b = tl.program_id(0)
    s_tile = tl.program_id(1)
    s_offsets = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offsets < T_or_I

    # For each offset in this tile, write row s_offsets[idx] of C[b, :, :] into out[b, s_offsets[idx], :]
    for idx in range(0, BLOCK_S):
        s_idx = s_offsets[idx]
        if mask[idx]:
            c_row_ptr = C_ptr + (b * S + s_idx) * stride_C_m
            out_row_ptr = out_ptr + b * stride_out_b + s_idx * stride_out_s
            x = tl.load(c_row_ptr + tl.arange(0, H) * stride_C_n, mask=True, other=0.0)
            tl.store(out_row_ptr + tl.arange(0, H) * stride_out_n, x, mask=True)


# ModelNew: entry point required by the evaluation
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        S = T + I

        # 1) Concatenate in Triton: [B, S, H]
        out_concat = torch.empty((B, S, H), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_concat = (B, triton.cdiv(T, 128), triton.cdiv(I, 128))
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out_concat,
            B, T, I, H, S,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out_concat.stride(0), out_concat.stride(1), out_concat.stride(2),
            BLOCK_T=128, BLOCK_I=128,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul in Triton: out = out_concat @ process_weight.T
        # out_concat is [B, S, H]; view as A [M, K] with M = B*S, K = H
        M = B * S
        out = torch.empty((M, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # B_ptr = process_weight.T [H, H]
        Bw_T = process_weight.t().contiguous()

        grid_matmul = (triton.cdiv(M, 128), triton.cdiv(H, 128))
        matmul_seqs_kernel[grid_matmul](
            out_concat, Bw_T, out,
            M, H, H,  # K == H
            out_concat.stride(0), out_concat.stride(2),     # stride_A_m, stride_A_k
            Bw_T.stride(0), Bw_T.stride(1),                 # stride_B_k, stride_B_n
            out.stride(0), out.stride(1),                   # stride_C_m, stride_C_n
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # 3) Split in Triton into processed_encoder [B, T, H] and processed_hidden [B, I, H]
        # processed_encoder: rows 0..T-1
        encoder_out = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_split_encoder = (B, triton.cdiv(T, 128))
        split_rows_kernel[grid_split_encoder](
            out, encoder_out,
            B, S, T, H,
            out.stride(0), out.stride(1),
            encoder_out.stride(0), encoder_out.stride(1), encoder_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2,
        )

        # processed_hidden: rows T..T+I-1
        hidden_out = torch.empty((B, I, H), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_split_hidden = (B, triton.cdiv(I, 128))
        split_rows_kernel[grid_split_hidden](
            out, hidden_out,
            B, S, I, H,
            out.stride(0), out.stride(1),
            hidden_out.stride(0), hidden_out.stride(1), hidden_out.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2,
        )

        return encoder_out, hidden_out


def run(*args):
    return ModelNew()(*args)
