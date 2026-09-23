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
        dst = out_b + (offs_t + pid_b * 0) * out_s1 + h * out_s2  # sequence dim is offs_t
        # tl.store expects a vector; Triton will handle the broadcast. Ensure shape compatibility.
        tl.store(dst, tl.load(src, mask=mask_t))

    # Copy hidden rows into out[:, T:, :]
    seq_start = T
    for h in range(0, H):
        src = hid_b + offs_i * hid_s1 + h * hid_s2
        dst = out_b + (offs_i + seq_start) * out_s1 + h * out_s2
        tl.store(dst, tl.load(src, mask=mask_i))

# Kernel 2: Per-batch matrix multiply: out[b, :, :] = concatenated[b, :, :] @ process_weight.T
# concatenated is [S, H], process_weight.T is [H, H], output is [S, H]
@triton.jit
def matmul_batch_kernel(
    out_ptr,                 # [B, S, H] output per batch
    A_ptr,                   # [S, H] concatenated input per batch
    Bw_T_ptr,                # [H, H] process_weight.T
    B, S, H,
    out_bs0, out_bs1, out_bs2,
    A_s0, A_s1, A_s2,
    Bw_s0, Bw_s1, Bw_s2,
    BLOCK_M: tl.constexpr,   # tile size over S
    BLOCK_N: tl.constexpr,   # tile size over H
    BLOCK_K: tl.constexpr,   # tile size over H (reduction)
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in [0, S)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in [0, H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (H)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A[b, m, k] with broadcasting over k
        A_ptrs = A_ptr + pid_b * A_s0 + offs_m[:, None] * A_s1 + offs_k[None, :] * A_s2
        A_mask = (offs_m[:, None] < S) & (offs_k[None, :] < H)
        A_vals = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Bw_T[k, n] = process_weight.T[k, n]
        Bw_ptrs = Bw_T_ptr + offs_k[:, None] * Bw_s0 + offs_n[None, :] * Bw_s1
        Bw_mask = (offs_k[:, None] < H) & (offs_n[None, :] < H)
        Bw_vals = tl.load(Bw_ptrs, mask=Bw_mask, other=0.0)

        acc += tl.dot(A_vals, Bw_vals)

    # Store acc to out[b, offs_m, offs_n]
    out_ptrs = out_ptr + pid_b * out_bs0 + offs_m[:, None] * out_bs1 + offs_n[None, :] * out_bs2
    out_mask = (offs_m[:, None] < S) & (offs_n[None, :] < H)
    tl.store(out_ptrs, acc, mask=out_mask)

# Kernel 3: Split out[B, S, H] into processed_encoder [B, T, H] and processed_hidden [B, I, H]
@triton.jit
def split_seqs_kernel(
    in_ptr, out_e_ptr, out_i_ptr,
    B, T, I, H,
    in_s0, in_s1, in_s2,
    out_e_s0, out_e_s1, out_e_s2,
    out_i_s0, out_i_s1, out_i_s2,
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
    out_e_b = out_e_ptr + pid_b * out_e_s0
    out_i_b = out_i_ptr + pid_b * out_i_s0

    # Copy first T rows to encoder output
    for h in range(0, H):
        src = in_b + offs_t * in_s1 + h * in_s2
        dst_e = out_e_b + offs_t * out_e_s1 + h * out_e_s2
        tl.store(dst_e, tl.load(src, mask=mask_t))

    # Copy remaining I rows to hidden output (starting at T in input)
    for h in range(0, H):
        src = in_b + (offs_i + T) * in_s1 + h * in_s2
        dst_i = out_i_b + offs_i * out_i_s1 + h * out_i_s2
        tl.store(dst_i, tl.load(src, mask=mask_i))

def _run(
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-only implementation of the original run:
      - Concatenate along sequence dimension: [B, T+I, H]
      - Apply linear projection: concatenated @ process_weight.T
      - Split back into encoder and hidden streams
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
    assert encoder_hidden_states.dtype == hidden_states.dtype == process_weight.dtype == torch.float32, "Use float32 for correctness"
    B, T, H = encoder_hidden_states.shape
    I, H2 = hidden_states.shape
    assert H == H2 and process_weight.shape[1] == H and process_weight.shape[0] == H, "Dimension mismatch"

    # Ensure contiguous
    enc = encoder_hidden_states.contiguous()
    hid = hidden_states.contiguous()
    Bw = process_weight.contiguous()  # [H, H]
    Bw_T = Bw.t().contiguous()        # [H, H]

    S = T + I

    # 1) Concatenate sequences into out [B, S, H]
    out = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)
    grid_concat = (B, triton.cdiv(T, 128), triton.cdiv(I, 128))  # tiles over T and I
    concat_seqs_kernel[grid_concat](
        out, enc, hid,
        B, T, I, H,
        out.stride(0), out.stride(1), out.stride(2),
        enc.stride(0), enc.stride(1), enc.stride(2),
        hid.stride(0), hid.stride(1), hid.stride(2),
        BLOCK_T=128, BLOCK_I=128,
        num_warps=4, num_stages=2,
    )

    # 2) Per-batch matmul: out[b, :, :] = out[b, :, :] @ Bw_T
    # We'll do one program per (b, tile_m, tile_n). For S and H up to a few thousand, this is fine.
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    grid_matmul = (B, triton.cdiv(S, BLOCK_M), triton.cdiv(H, BLOCK_N))
    matmul_batch_kernel[grid_matmul](
        out, out, Bw_T,          # note: we compute out[b, :, :] into out itself
        B, S, H,
        out.stride(0), out.stride(1), out.stride(2),
        out.stride(0), out.stride(1), out.stride(2),  # A_ptr is out itself as concatenated per batch
        Bw_T.stride(0), Bw_T.stride(1), Bw_T.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # 3) Split out into processed_encoder [B, T, H] and processed_hidden [B, I, H]
    processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
    processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)
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
        return _run(*args)


def run(*args):
    return ModelNew()(*args)
