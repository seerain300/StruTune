import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,            # *const float, [B, T, H]
    hid_ptr,            # *const float, [B, I, H]
    out_ptr,            # *float,        [B, S, H]
    B, T, I, H, S,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    # Grid: (B, S)
    b = tl.program_id(0)
    m = tl.program_id(1)
    if b >= 0 and m >= 0 and b < B and m < S:
        # Determine source tensor and offset
        # If m < T: source = enc[b, m, :]
        # Else:     source = hid[b, m - T, :]
        if m < T:
            row_in = b * T * H + m * H
        else:
            row_in = b * I * H + (m - T) * H

        # Each row has H elements
        # Write to out[b, m, :]
        row_out = b * S * H + m * H
        # Copy H elements with simple loop to avoid complex masks
        # This is robust for H=1024 and reduces error chances
        for n in range(0, H):
            val = tl.load(enc_ptr + row_in + n) if m < T else tl.load(hid_ptr + row_in + n)
            tl.store(out_ptr + row_out + n, val)


@triton.jit
def matmul_row_kernel(
    A_ptr,              # *const float, [S, H]
    BwT_ptr,            # *const float, [H, H]
    C_ptr,              # *float,        [S, H]
    S, H,               # int
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (S, tiles over N=H), here N tiles = 1 when BLOCK_N == H
    m = tl.program_id(0)  # row index in [0, S)
    n_block = tl.program_id(1)
    n_start = n_block * BLOCK_N
    # Vector of N indices for this block
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    # Accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Loop over K dimension
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        # A[m, k_offsets]
        a = tl.load(A_ptr + m * H + k_offsets, mask=k_offsets < H, other=0.0)
        # BwT[k_offsets, n_offsets]
        b = tl.load(BwT_ptr + k_offsets[:, None] * H + n_offsets[None, :], mask=(k_offsets[:, None] < H) & (n_offsets[None, :] < H), other=0.0)
        # Multiply-accumulate
        acc += tl.sum(a[:, None] * b, axis=0)
    # Store results for this row
    tl.store(C_ptr + m * H + n_offsets, acc, mask=n_offsets < H)


@triton.jit
def split_seqs_kernel(
    C_ptr,              # *const float, [B*S, H]
    out_e_ptr,          # *float,        [B, T, H]
    out_h_ptr,          # *float,        [B, I, H]
    B, T, I, H, S,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    # Grid: (B,)
    b = tl.program_id(0)
    if b >= 0 and b < B:
        # Copy first T rows to out_e
        for t in range(0, T):
            row_in = b * S * H + t * H
            row_out = b * T * H + t * H
            for n in range(0, H):
                val = tl.load(C_ptr + row_in + n)
                tl.store(out_e_ptr + row_out + n, val)
        # Copy next I rows to out_h
        for i in range(0, I):
            row_in = b * S * H + (i + T) * H
            row_out = b * I * H + i * H
            for n in range(0, H):
                val = tl.load(C_ptr + row_in + n)
                tl.store(out_h_ptr + row_out + n, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, process_weight):
        # Ensure CUDA, contiguous, float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        dtype = hidden_states.dtype
        if dtype != torch.float32:
            hidden_states = hidden_states.float()
        if dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        S = T + I

        # Allocate concatenated [B, S, H]
        concatenated = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)

        # 1) Concatenate encoder and hidden states
        grid_concat = (B, S)
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H, S,
            num_warps=1, num_stages=1,
        )

        # 2) Prepare Bw_T = process_weight.T [H, H]
        Bw_T = process_weight.t().contiguous()

        # Allocate C [B*S, H]
        C = torch.empty((B * S, H), device=hidden_states.device, dtype=torch.float32)

        # Launch matmul kernel: 2D grid over (rows, N tiles). With BLOCK_N=H, N tiles = 1.
        grid_matmul = (B * S, 1)
        matmul_row_kernel[grid_matmul](
            concatenated, Bw_T, C,
            S, H,
            BLOCK_K=64, BLOCK_N=H,
            num_warps=4, num_stages=2,
        )

        # 3) Allocate outputs and split
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=torch.float32)

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            num_warps=1, num_stages=1,
        )

        # If original dtype was not float32, cast outputs back
        if dtype != torch.float32:
            processed_encoder = processed_encoder.to(dtype)
            processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
