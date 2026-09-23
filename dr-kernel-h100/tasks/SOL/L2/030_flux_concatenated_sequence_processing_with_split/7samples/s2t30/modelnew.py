import torch
import triton
import triton.language as tl

# Kernel 1: Concatenate encoder_hidden_states [B, T, H] and hidden_states [B, I, H] into
# out[B, S, H], where S = T + I. Copy rows: out[:, :T, :] = encoder, out[:, T:, :] = hidden.
@triton.jit
def concat_seqs_kernel(
    out_ptr, enc_ptr, hid_ptr,
    B, T, I, H,
    out_s0, out_s1, out_s2,
    enc_s0, enc_s1, enc_s2,
    hid_s0, hid_s1, hid_s2,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    mask_t = offs_t < T
    mask_i = offs_i < I

    # Base pointers for this batch
    out_b = out_ptr + pid_b * out_s0
    enc_b = enc_ptr + pid_b * enc_s0
    hid_b = hid_ptr + pid_b * hid_s0

    # Copy encoder rows into out[:, :T, :]
    for h in range(0, H):
        src = enc_b + offs_t * enc_s1 + h * enc_s2
        dst = out_b + offs_t * out_s1 + h * out_s2  # first T rows
        tl.store(dst, tl.load(src, mask=mask_t))

    # Copy hidden rows into out[:, T:, :]
    seq_start = T
    for h in range(0, H):
        src = hid_b + offs_i * hid_s1 + h * hid_s2
        dst = out_b + (offs_i + seq_start) * out_s1 + h * out_s2  # next I rows
        tl.store(dst, tl.load(src, mask=mask_i))


# Kernel 2: Per-batch matrix multiply: out[b, :, :] = A[b, :, :] @ Bw_T, where
# A is concatenated [S, H] per batch, Bw_T is process_weight.T [H, H], out is [S, H] per batch.
@triton.jit
def matmul_batch_kernel(
    out_ptr,                 # [B, S, H] output per batch
    A_ptr,                   # [S, H] concatenated input per batch
    Bw_T_ptr,                # [H, H] process_weight.T
    B, S, H,
    out_bs0, out_bs1, out_bs2,
    A_s0, A_s1, A_s2,
    Bw_s0, Bw_s1, Bw_s2,
    BLOCK_M: tl.constexpr,   # tile size over S (rows)
    BLOCK_N: tl.constexpr,   # tile size over H (cols)
    BLOCK_K: tl.constexpr,   # tile size over K (hidden dim)
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # over S
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # over H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K (hidden dimension)
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A[b, m, k] -> [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * A_s0 + offs_m[:, None] * A_s1 + offs_k[None, :] * A_s2
        a_mask = (offs_m[:, None] < S) & (offs_k[None, :] < H)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load Bw_T[k, n] -> [BLOCK_K, BLOCK_N]
        b_ptrs = Bw_T_ptr + offs_k[:, None] * Bw_s0 + offs_n[None, :] * Bw_s1
        b_mask = (offs_k[:, None] < H) & (offs_n[None, :] < H)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back to out[b, offs_m, offs_n]
    out_ptrs = out_ptr + pid_b * out_bs0 + offs_m[:, None] * out_bs1 + offs_n[None, :] * out_bs2
    out_mask = (offs_m[:, None] < S) & (offs_n[None, :] < H)
    tl.store(out_ptrs, acc, mask=out_mask)


# Kernel 3: Split out[B, S, H] into processed_encoder[B, T, H] and processed_hidden[B, I, H].
@triton.jit
def split_seqs_kernel(
    in_ptr, enc_ptr, hid_ptr,
    B, T, I, H,
    in_s0, in_s1, in_s2,
    enc_s0, enc_s1, enc_s2,
    hid_s0, hid_s1, hid_s2,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    mask_t = offs_t < T
    mask_i = offs_i < I

    in_b = in_ptr + pid_b * in_s0
    enc_b = enc_ptr + pid_b * enc_s0
    hid_b = hid_ptr + pid_b * hid_s0

    # Copy first T rows to encoder
    for h in range(0, H):
        src = in_b + offs_t * in_s1 + h * in_s2
        dst = enc_b + offs_t * enc_s1 + h * enc_s2
        tl.store(dst, tl.load(src, mask=mask_t))

    # Copy remaining I rows to hidden
    seq_start = T
    for h in range(0, H):
        src = in_b + (offs_i + seq_start) * in_s1 + h * in_s2
        dst = hid_b + offs_i * hid_s1 + h * hid_s2
        tl.store(dst, tl.load(src, mask=mask_i))


def _run(
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-optimized version:
    - Concatenate along sequence dimension in Triton.
    - Matrix multiply with process_weight.T in Triton per batch.
    - Split back into encoder and image streams in Triton.
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
    assert encoder_hidden_states.dtype == hidden_states.dtype == process_weight.dtype, "All tensors must have the same dtype"
    assert encoder_hidden_states.dim() == 3 and hidden_states.dim() == 3, "Inputs must be [B, S, H]"
    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]
    assert hidden_states.shape[2] == H, "hidden_dim must match"
    assert process_weight.shape == (H, H), "process_weight must be [H, H]"

    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype

    S = T + I

    # Allocate output concatenated [B, S, H] and processed [B, S, H]
    out = torch.empty((B, S, H), device=device, dtype=dtype)
    processed = torch.empty((B, S, H), device=device, dtype=dtype)

    # 1) Concatenate encoder and hidden into out
    grid_concat = (B, triton.cdiv(T, 128), triton.cdiv(I, 128))
    concat_seqs_kernel[grid_concat](
        out, encoder_hidden_states, hidden_states,
        B, T, I, H,
        out.stride(0), out.stride(1), out.stride(2),
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        BLOCK_T=128, BLOCK_I=128,
        num_warps=4, num_stages=2,
    )

    # 2) Per-batch matmul: processed[b] = out[b] @ process_weight.T
    # Use process_weight.T [H, H]
    Bw_T = process_weight.transpose(0, 1).contiguous()
    grid_matmul = (B, triton.cdiv(S, 128), triton.cdiv(H, 128))
    matmul_batch_kernel[grid_matmul](
        processed, out, Bw_T,
        B, S, H,
        processed.stride(0), processed.stride(1), processed.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        Bw_T.stride(0), Bw_T.stride(1), Bw_T.stride(2),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )

    # 3) Split processed back into encoder and hidden parts
    processed_encoder = torch.empty((B, T, H), device=device, dtype=dtype)
    processed_hidden = torch.empty((B, I, H), device=device, dtype=dtype)
    grid_split = (B, triton.cdiv(T, 128), triton.cdiv(I, 128))
    split_seqs_kernel[grid_split](
        processed, processed_encoder, processed_hidden,
        B, T, I, H,
        processed.stride(0), processed.stride(1), processed.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        BLOCK_T=128, BLOCK_I=128,
        num_warps=2, num_stages=2,
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return _run(*args)