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
    pid_t = tl.program_id(1)  # tiles over T
    pid_i = tl.program_id(2)  # tiles over I

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
        dst = out_b + (offs_t) * out_s1 + h * out_s2  # sequence dim is offs_t
        tl.store(dst, tl.load(src, mask=mask_t))

    # Copy hidden rows into out[:, T:, :]
    seq_start = T
    for h in range(0, H):
        src = hid_b + offs_i * hid_s1 + h * hid_s2
        dst = out_b + (offs_i + seq_start) * out_s1 + h * out_s2
        tl.store(dst, tl.load(src, mask=mask_i))


# Kernel 2: Per-batch matrix multiply: out[b, :, :] = out[b, :, :] @ process_weight.T
# We pass out as both A (concatenated input per batch) and output pointer.
@triton.jit
def matmul_batch_kernel(
    out_ptr,                 # [B, S, H] output per batch (also used as A if in-place)
    Bw_T_ptr,                # [H, H] process_weight.T
    B, S, H,
    out_bs0, out_bs1, out_bs2,
    Bw_s0, Bw_s1, Bw_s2,
    BLOCK_M: tl.constexpr,   # tile size over S (rows)
    BLOCK_N: tl.constexpr,   # tile size over H (cols)
    BLOCK_K: tl.constexpr,   # reduction tile over H (K)
):
    # Grid: (B, tiles over M=S, tiles over N=H)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices in [0, S)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # col indices in [0, H)

    mask_m = m_offsets < S
    mask_n = n_offsets < H

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K=H in chunks of BLOCK_K
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load A tile: out[m, k] where out is concatenated for this batch
        A_ptrs = out_ptr + pid_b * out_bs0 + m_offsets[:, None] * out_bs1 + k_offsets[None, :] * out_bs2
        A = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load Bw_T tile: B[k, n]
        Bw_ptrs = Bw_T_ptr + k_offsets[:, None] * Bw_s0 + n_offsets[None, :] * Bw_s1
        Bw = tl.load(Bw_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(A, Bw)

    # Store result into out[b, m, n]
    Out_ptrs = out_ptr + pid_b * out_bs0 + m_offsets[:, None] * out_bs1 + n_offsets[None, :] * out_bs2
    tl.store(Out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# Kernel 3: Split processed [B, S, H] into encoder [B, T, H] and hidden [B, I, H]
@triton.jit
def split_seqs_kernel(
    in_ptr,                  # [B, S, H]
    out_enc_ptr,             # [B, T, H]
    out_hid_ptr,             # [B, I, H]
    B, T, I, H,
    in_s0, in_s1, in_s2,
    out_e_s0, out_e_s1, out_e_s2,
    out_h_s0, out_h_s1, out_h_s2,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)  # tiles over T
    pid_i = tl.program_id(2)  # tiles over I

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    mask_t = offs_t < T
    mask_i = offs_i < I

    in_b = in_ptr + pid_b * in_s0
    out_e_b = out_enc_ptr + pid_b * out_e_s0
    out_h_b = out_hid_ptr + pid_b * out_h_s0

    # Copy first T rows to encoder
    for h in range(0, H):
        src = in_b + offs_t * in_s1 + h * in_s2
        dst = out_e_b + offs_t * out_e_s1 + h * out_e_s2
        tl.store(dst, tl.load(src, mask=mask_t))

    # Copy remaining I rows to hidden
    # src rows start at T
    src_rows = offs_i + T
    for h in range(0, H):
        src = in_b + src_rows * in_s1 + h * in_s2
        dst = out_h_b + offs_i * out_h_s1 + h * out_h_s2
        tl.store(dst, tl.load(src, mask=mask_i))


def _run(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure CUDA tensors and contiguity
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
    assert encoder_hidden_states.dtype == hidden_states.dtype == process_weight.dtype, "All tensors must have the same dtype"
    assert encoder_hidden_states.ndim == 3 and hidden_states.ndim == 3 and process_weight.ndim == 2, "Invalid input shapes"
    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]

    # Allocate concatenated out [B, S, H]
    S = T + I
    out = torch.empty((B, S, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype).contiguous()

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
    Bw_T = process_weight.transpose(0, 1).contiguous()  # [H, H]
    grid_matmul = (B, triton.cdiv(S, 128), triton.cdiv(H, 128))
    matmul_batch_kernel[grid_matmul](
        out, Bw_T,
        B, S, H,
        out.stride(0), out.stride(1), out.stride(2),
        Bw_T.stride(0), Bw_T.stride(1), Bw_T.stride(2),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )

    # 3) Split processed back into encoder and hidden parts
    processed_encoder = torch.empty((B, T, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
    processed_hidden = torch.empty((B, I, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
    grid_split = (B, triton.cdiv(T, 128), triton.cdiv(I, 128))
    split_seqs_kernel[grid_split](
        out, processed_encoder, processed_hidden,
        B, T, I, H,
        out.stride(0), out.stride(1), out.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        BLOCK_T=128, BLOCK_I=128,
        num_warps=2, num_stages=2,
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Assumes args are: hidden_states [B, I, H], encoder_hidden_states [B, T, H], process_weight [H, H]
        hidden_states = args[1]
        encoder_hidden_states = args[0]
        process_weight = args[2]
        return _run(encoder_hidden_states, hidden_states, process_weight)