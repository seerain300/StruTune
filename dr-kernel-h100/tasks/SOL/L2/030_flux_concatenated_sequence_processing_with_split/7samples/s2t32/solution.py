import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder and hidden tensors along sequence dimension into [B, S, H]
# We assume all inputs are contiguous and float32.
@triton.jit
def concat_seqs_kernel(
    enc_ptr,            # *f32 [B, T, H]
    hid_ptr,            # *f32 [B, I, H]
    out_ptr,            # *f32 [B, S, H], where S = T + I
    B: tl.constexpr,    # int
    T: tl.constexpr,    # int
    I: tl.constexpr,    # int
    H: tl.constexpr,    # int
    BLOCK_N: tl.constexpr,  # tile size for H (set to H to avoid partial tiles)
):
    # One program per (batch, row) where row in [0, S)
    pid = tl.program_id(0)
    # batch index
    b = pid // S
    row = pid % S

    # Compute source and destination indices
    if row < T:
        src_idx = row
    else:
        src_idx = row - T  # map hidden rows into second half

    # Base pointers for this batch
    enc_base = enc_ptr + b * T * H
    hid_base = hid_ptr + b * I * H
    out_base = out_ptr + b * S * H

    # Vector over N=H
    n = tl.arange(0, BLOCK_N)
    # Load from source and store to out
    # For out: out[b, row, n] -> out_ptr + b*S*H + row*H + n
    # For src: enc[row, n] -> enc_ptr + b*T*H + row*H + n
    #         hid[src_idx, n] -> hid_ptr + b*I*H + src_idx*H + n
    if row < T:
        vals = tl.load(enc_base + src_idx * H + n)
    else:
        vals = tl.load(hid_base + src_idx * H + n)
    tl.store(out_base + row * H + n, vals)


# Triton kernel: GEMM per-row, C[m, :] = sum_k A[m, k] * Bw_T[k, :], where
# A is concatenated flattened rows: A[m, k] = out_ptr[m, k], m in [0, B*S), k in [0, H)
# Bw_T is process_weight.T: [H, H], input provided as [H, H] and we pass its transpose via strides
# Output C: [B*S, H]
@triton.jit
def matmul_row_kernel(
    A_ptr,              # *f32 [M, H], where M = B*S
    Bw_T_ptr,           # *f32 [H, H] (we will use strides to read as Bw_T[k, n])
    C_ptr,              # *f32 [M, H]
    M: tl.constexpr,    # int (B*S)
    H: tl.constexpr,    # int (hidden_dim)
    BLOCK_K: tl.constexpr,  # int, tile over K
    BLOCK_N: tl.constexpr,  # int, tile over N (set to H to avoid partial tiles)
):
    # 2D grid: (pid_m over rows, pid_n over N tiles). With BLOCK_N=H, pid_n will be 0.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # compute n indices
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # loop over K dimension
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        # load A row segment: A[pid_m, k]
        a_ptrs = A_ptr + pid_m * H + k
        a = tl.load(a_ptrs, mask=k < H, other=0.0)
        # load Bw_T segment: Bw_T[k, n] -> since Bw_T is [H, H], index as [k, n]
        b_ptrs = Bw_T_ptr + k * H + n  # row-major [H, H]
        b = tl.load(b_ptrs, mask=(k < H) & (n < H), other=0.0)
        # accumulate
        acc += tl.sum(a[:, None] * b[None, :], axis=0)

    # store result to C[pid_m, n]
    c_ptrs = C_ptr + pid_m * H + n
    tl.store(c_ptrs, acc, mask=n < H)


# Triton kernel: split C [B, S, H] into encoder [B, T, H] and hidden [B, I, H]
@triton.jit
def split_seqs_kernel(
    C_ptr,              # *f32 [B, S, H]
    enc_out_ptr,        # *f32 [B, T, H]
    hid_out_ptr,        # *f32 [B, I, H]
    B: tl.constexpr,    # int
    T: tl.constexpr,    # int
    I: tl.constexpr,    # int
    H: tl.constexpr,    # int
    S: tl.constexpr,    # int = T + I
    BLOCK_N: tl.constexpr,  # tile size for H (set to H to avoid partial tiles)
):
    # One program per batch
    pid_b = tl.program_id(0)
    b = pid_b

    # First T rows -> encoder
    for t in range(0, T):
        n = tl.arange(0, BLOCK_N)
        vals = tl.load(C_ptr + b * S * H + t * H + n)
        tl.store(enc_out_ptr + b * T * H + t * H + n, vals)

    # Remaining I rows -> hidden
    for i in range(0, I):
        src_row = i + T
        n = tl.arange(0, BLOCK_N)
        vals = tl.load(C_ptr + b * S * H + src_row * H + n)
        tl.store(hid_out_ptr + b * I * H + i * H + n, vals)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run:
        - Concatenate along sequence dimension: [B, T, H] + [B, I, H] -> [B, S, H], S=T+I
        - Linear projection: @ process_weight.T
        - Split back into [B, T, H] and [B, I, H]
        """
        # Ensure CUDA, contiguous, and float32
        if not (encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda):
            raise RuntimeError("All inputs must be CUDA tensors.")
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        Bw_T = process_weight.transpose(0, 1).contiguous()  # [H, H]

        B, T, H = enc.shape
        I = hid.shape[1]
        S = T + I

        # 1) Allocate concatenated [B, S, H] and fill with Triton
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)

        # Launch concat kernel: one program per (batch, row) up to S
        # Note: we pass S as constexpr in grid; Triton allows dynamic values for program_id args.
        grid_concat = (B * S,)
        concat_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H,
            BLOCK_N=H,
            num_warps=4, num_stages=2,
        )

        # 2) Allocate C [B*S, H]
        C = torch.empty((B * S, H), device=enc.device, dtype=enc.dtype)

        # Launch matmul kernel: 2D grid over (rows, N tiles). With BLOCK_N=H, N tiles = 1.
        grid_matmul = (B * S, 1)
        matmul_row_kernel[grid_matmul](
            concatenated, Bw_T, C,
            B * S, H,
            BLOCK_K=64, BLOCK_N=H,
            num_warps=4, num_stages=2,
        )

        # 3) Allocate outputs and split
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            BLOCK_N=H,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
