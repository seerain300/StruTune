import torch
import triton
import triton.language as tl

@triton.jit
def concat_seqs_kernel(
    enc_ptr,           # [B, T, H]
    hidden_ptr,        # [B, I, H]
    out_ptr,           # [B, S, H], S = T + I
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr,
):
    b = tl.program_id(0)
    # Copy encoder rows into out[:, :T, :]
    t_start = 0
    while t_start < T:
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T
        h_offsets = tl.arange(0, H)
        for h in range(0, H):
            enc_ptr_offset = b * T * H + t_offsets * H + h
            out_ptr_offset = b * S * H + t_offsets * H + h
            val = tl.load(enc_ptr + enc_ptr_offset, mask=mask_t, other=0.0)
            tl.store(out_ptr + out_ptr_offset, val, mask=mask_t)
        t_start += BLOCK_T

    # Copy hidden rows into out[:, T:, :]
    i_start = 0
    while i_start < I:
        i_offsets = i_start + tl.arange(0, BLOCK_I)
        mask_i = i_offsets < I
        h_offsets = tl.arange(0, H)
        for h in range(0, H):
            hidden_ptr_offset = b * I * H + i_offsets * H + h
            out_ptr_offset = b * S * H + (T + i_offsets) * H + h
            val = tl.load(hidden_ptr + hidden_ptr_offset, mask=mask_i, other=0.0)
            tl.store(out_ptr + out_ptr_offset, val, mask=mask_i)
        i_start += BLOCK_I

@triton.jit
def matmul_row_kernel(
    A_ptr,            # [S, H] i.e., concatenated rows
    B_ptr,            # [H, H] i.e., process_weight.T
    C_ptr,            # [S, H] output
    S: tl.constexpr, H: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)  # row index in [0, S)
    n_tile = tl.program_id(1)  # tile index over N (H)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    k = 0
    while k < H:
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H
        # A[m, k_offsets] -> scalar vector along k for this row m
        A_vals = tl.load(A_ptr + m * H + k_offsets, mask=mask_k, other=0.0)  # [BLOCK_K]
        # B[k_offsets, n_offsets] -> [BLOCK_K, BLOCK_N]
        B_vals = tl.load(B_ptr + k_offsets[:, None] * H + n_offsets[None, :], mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.sum(A_vals[:, None] * B_vals, axis=0)
        k += BLOCK_K

    # Store acc to C[m, n_offsets]
    C_out_ptr = C_ptr + m * H + n_offsets
    tl.store(C_out_ptr, acc, mask=mask_n)

@triton.jit
def split_seqs_kernel(
    C_ptr,            # [B, S, H]
    out_encoder_ptr,  # [B, T, H]
    out_hidden_ptr,   # [B, I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    S: tl.constexpr,
):
    b = tl.program_id(0)
    # Copy first T rows to encoder output
    t_start = 0
    while t_start < T:
        t_offsets = t_start + tl.arange(0, 64)  # BLOCK_T=64 for simplicity
        mask_t = t_offsets < T
        for h in range(0, H):
            C_src = b * S * H + t_offsets * H + h
            out_e = b * T * H + t_offsets * H + h
            val = tl.load(C_ptr + C_src, mask=mask_t, other=0.0)
            tl.store(out_encoder_ptr + out_e, val, mask=mask_t)
        t_start += 64
    # Copy remaining I rows to hidden output starting at index T in concatenated
    i_start = 0
    while i_start < I:
        i_offsets = i_start + tl.arange(0, 64)
        mask_i = i_offsets < I
        for h in range(0, H):
            C_src = b * S * H + (T + i_offsets) * H + h
            out_h = b * I * H + i_offsets * H + h
            val = tl.load(C_ptr + C_src, mask=mask_i, other=0.0)
            tl.store(out_hidden_ptr + out_h, val, mask=mask_i)
        i_start += 64

class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA and float32; make contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA"
        assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        B = enc.shape[0]
        T = enc.shape[1]
        I = hid.shape[1]
        H = enc.shape[2]
        S = T + I

        # Allocate outputs
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)
        # 1) Concatenate encoder and hidden sequences
        grid_concat = (B,)
        # Use BLOCK_T=128, BLOCK_I=128 for vectorized copy
        concat_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H,
            BLOCK_T=128, BLOCK_I=128,
            num_warps=4, num_stages=2,
        )

        # 2) Compute matmul: C = concatenated @ process_weight.T, shape [B, S, H]
        # Treat concatenated as [S, H] per batch (we use S directly). Create B_ptr [H, H]
        Bw_T = process_weight.transpose(0, 1).contiguous()  # [H, H]
        C = torch.empty((S, H), device=enc.device, dtype=enc.dtype)

        grid_m = (S,)
        grid_n = (triton.cdiv(H, 1024),)  # BLOCK_N=H=1024 for provided workloads
        matmul_row_kernel[grid_m * grid_n](
            concatenated, Bw_T, C,
            S, H,
            BLOCK_K=64, BLOCK_N=1024,
            num_warps=4, num_stages=2,
        )

        # 3) Split C into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden