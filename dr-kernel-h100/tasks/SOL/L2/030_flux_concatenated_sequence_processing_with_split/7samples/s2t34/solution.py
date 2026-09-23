import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder_hidden_states and hidden_states into a single [B, S, H] tensor.
# We assume inputs are contiguous, dtype float32, on CUDA.
@triton.jit
def concat_seqs_kernel(
    enc_ptr, hid_ptr, out_ptr,
    B, T, I, H,
    out_stride0, out_stride1, out_stride2,
    enc_stride0, enc_stride1, enc_stride2,
    hid_stride0, hid_stride1, hid_stride2,
):
    # program_id(0) enumerates batches
    b = tl.program_id(0)
    # Each row m in the concatenated output corresponds to either encoder or hidden rows
    # We process rows serially within each program to avoid cross-program writes
    # Outer loop over T rows from encoder
    for m in range(0, T):
        # Compute output row index
        s = m
        # Compute pointers
        out_off = b * out_stride0 + s * out_stride1
        enc_row_off = b * enc_stride0
        # Vector of column indices
        cols = tl.arange(0, H)
        # Load from encoder and store to output
        enc_vals = tl.load(enc_ptr + enc_row_off + m * enc_stride1 + cols * enc_stride2, mask=(cols < H), other=0.0)
        tl.store(out_ptr + out_off + cols * out_stride2, enc_vals, mask=(cols < H))
    # Inner loop over I rows from hidden
    for m2 in range(0, I):
        s = T + m2
        out_off2 = b * out_stride0 + s * out_stride1
        hid_row_off = b * hid_stride0
        hid_vals = tl.load(hid_ptr + hid_row_off + m2 * hid_stride1 + cols * hid_stride2, mask=(cols < H), other=0.0)
        tl.store(out_ptr + out_off2 + cols * out_stride2, hid_vals, mask=(cols < H))


# Triton kernel: per-row GEMM: C[m, :] = A[m, :] @ Bw_T, where A is [S, H], Bw_T is [H, H], C is [S, H].
# We use a 2D grid over (m rows, tiles over N=H). For H=1024, we set BLOCK_N=H to avoid partial tiles.
@triton.jit
def matmul_row_kernel(
    A_ptr, Bw_ptr, C_ptr,
    M, N, K,
    A_stride0, A_stride1,
    Bw_stride0, Bw_stride1,
    C_stride0, C_stride1,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)  # row index in A (concatenated)
    n_block = tl.program_id(1)  # tile index along N
    # Compute column offsets for this tile
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator for this row-tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Loop over K dimension in chunks
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        # Load A[m, k_offsets] -> vector of size BLOCK_K
        A_row_off = m * A_stride0
        a = tl.load(A_ptr + A_row_off + k_offsets * A_stride1, mask=(k_offsets < K), other=0.0)  # [BLOCK_K]
        # Load Bw[k_offsets, n_offsets] -> matrix [BLOCK_K, BLOCK_N]
        Bw_row_off = k_offsets * Bw_stride0
        b = tl.load(Bw_ptr + Bw_row_off[:, None] + n_offsets[None, :] * Bw_stride1,
                    mask=((k_offsets < K)[:, None] & (n_offsets < N)[None, :]),
                    other=0.0)  # [BLOCK_K, BLOCK_N]
        # Accumulate
        acc += tl.sum(a[:, None] * b, axis=0)
    # Store results back to C[m, n_offsets]
    C_row_off = m * C_stride0
    tl.store(C_ptr + C_row_off + n_offsets * C_stride1, acc, mask=(n_offsets < N))


# Triton kernel: split C [B*S, H] into processed_encoder [B, T, H] and processed_hidden [B, I, H].
@triton.jit
def split_seqs_kernel(
    C_ptr, out_e_ptr, out_i_ptr,
    B, T, I, H, S,
    C_stride0, C_stride1, C_stride2,
    out_e_stride0, out_e_stride1, out_e_stride2,
    out_i_stride0, out_i_stride1, out_i_stride2,
):
    b = tl.program_id(0)
    # Copy first T rows to encoder output
    for m in range(0, T):
        s = m
        c_off = b * C_stride0 + s * C_stride1
        e_off = b * out_e_stride0 + m * out_e_stride1
        cols = tl.arange(0, H)
        vals = tl.load(C_ptr + c_off + cols * C_stride2, mask=(cols < H), other=0.0)
        tl.store(out_e_ptr + e_off + cols * out_e_stride2, vals, mask=(cols < H))
    # Copy remaining I rows to hidden output
    for m2 in range(0, I):
        s = T + m2
        c_off2 = b * C_stride0 + s * C_stride1
        i_off = b * out_i_stride0 + m2 * out_i_stride1
        vals2 = tl.load(C_ptr + c_off2 + cols * C_stride2, mask=(cols < H), other=0.0)
        tl.store(out_i_ptr + i_off + cols * out_i_stride2, vals2, mask=(cols < H))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run:
        - Concatenate along sequence dimension: [B, T, H] + [B, I, H] -> [B, S, H], S=T+I
        - Apply linear projection: [B, S, H] @ process_weight.T  -> [B, S, H]
        - Split back: [B, T, H] and [B, I, H]
        All computations are done by Triton kernels.
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA."
        # Enforce dtype and contiguity
        dtype = encoder_hidden_states.dtype
        B, T, H = encoder_hidden_states.shape
        Bi, I, Hi = hidden_states.shape
        assert B == Bi and H == Hi, "encoder_hidden_states and hidden_states must have same batch and hidden_dim."
        # We will compute in float32 for robustness
        if dtype != torch.float32:
            enc = encoder_hidden_states.float().contiguous()
            hid = hidden_states.float().contiguous()
        else:
            enc = encoder_hidden_states.contiguous()
            hid = hidden_states.contiguous()

        # process_weight should be [H, H], ensure contiguous and float32
        Bw = process_weight.contiguous()
        if Bw.dtype != torch.float32:
            Bw = Bw.float()
        # If process_weight isn't [H, H], transpose on-the-fly
        # The original code expects process_weight [H, H], as used in torch.matmul(concatenated, process_weight.t())
        assert Bw.shape[0] == H and Bw.shape[1] == H, "process_weight must be of shape [hidden_dim, hidden_dim]."
        Bw_T = Bw  # we will use Bw as [H, H] without transpose

        # 1) Allocate concatenated tensor [B, S, H]
        S = T + I
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=torch.float32)

        # 2) Concatenate in Triton
        grid_concat = (B,)
        concat_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            num_warps=4, num_stages=2,
        )

        # 3) Compute processed = concatenated @ process_weight.T using Triton GEMM
        # Flatten concatenated rows: A is [B*S, H]
        A = concatenated.reshape(B * S, H).contiguous()
        C = torch.empty((B * S, H), device=enc.device, dtype=torch.float32)

        # 2D grid: (rows, tiles over N). Since BLOCK_N=H, tiles=1.
        grid_matmul = (B * S, 1)
        matmul_row_kernel[grid_matmul](
            A, Bw_T, C,
            B * S, H, H,
            A.stride(0), A.stride(1),
            Bw_T.stride(0), Bw_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_N=H, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 4) Split into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=torch.float32)

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=1, num_stages=1,
        )

        # Cast back to original dtype if needed
        if dtype != torch.float32:
            processed_encoder = processed_encoder.to(dtype)
            processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
