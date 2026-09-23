import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,  # *f32 [B, T, H]
    hid_ptr,  # *f32 [B, I, H]
    out_ptr,  # *f32 [B, S, H]
    B, T, I, H,
    BLOCK_T: tl.constexpr,  # tile size for T
    BLOCK_I: tl.constexpr,  # tile size for I
):
    b = tl.program_id(0)
    # base pointers for this batch
    enc_base = enc_ptr + b * T * H
    hid_base = hid_ptr + b * I * H
    out_base = out_ptr + b * S * H  # S = T + I

    # copy encoder rows into out[:, :T, :]
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        for h0 in range(0, H):
            src = enc_base + offs_t * H + h0
            dst = out_base + offs_t * H + h0
            tl.store(dst, tl.load(src, mask=mask_t, other=0.0), mask=mask_t)

    # copy hidden rows into out[:, T:, :]
    for i0 in range(0, I, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        mask_i = offs_i < I
        for h0 in range(0, H):
            src = hid_base + offs_i * H + h0
            dst = out_base + (T + offs_i) * H + h0
            tl.store(dst, tl.load(src, mask=mask_i, other=0.0), mask=mask_i)


@triton.jit
def matmul_row_kernel(
    A_ptr,   # *f32 [S, H]
    B_ptr,   # *f32 [H, H] (process_weight.T)
    C_ptr,   # *f32 [S, H] output
    S, H,
    BLOCK_K: tl.constexpr,  # loop chunk for K
    BLOCK_N: tl.constexpr,  # tile for N (set to H=1024 for provided workloads)
):
    # one program per row
    m = tl.program_id(0)
    if m >= S:
        return

    # initialize output row
    for n0 in range(0, BLOCK_N, BLOCK_N):
        n_offs = n0 + tl.arange(0, BLOCK_N)
        mask_n = n_offs < H
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # loop over K
        for k0 in range(0, H, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_offs < H
            a = tl.load(A_ptr + m * H + k_offs, mask=mask_k, other=0.0)  # [BLOCK_K]
            b = tl.load(B_ptr + k_offs[:, None] * H + n_offs[None, :], mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]
            acc += tl.sum(a[:, None] * b, axis=0)
        tl.store(C_ptr + m * H + n_offs, acc, mask=mask_n)


@triton.jit
def split_seqs_kernel(
    C_ptr,         # *f32 [S, H]
    out_enc_ptr,   # *f32 [B, T, H]
    out_hid_ptr,   # *f32 [B, I, H]
    B, T, I, H, S,
    BLOCK_T: tl.constexpr,  # tile size for T
    BLOCK_I: tl.constexpr,  # tile size for I
):
    b = tl.program_id(0)
    C_batch_base = C_ptr + b * S * H
    enc_base = out_enc_ptr + b * T * H
    hid_base = out_hid_ptr + b * I * H

    # copy first T rows to encoder
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        for h0 in range(0, H):
            src = C_batch_base + offs_t * H + h0
            dst = enc_base + offs_t * H + h0
            tl.store(dst, tl.load(src, mask=mask_t, other=0.0), mask=mask_t)

    # copy remaining I rows to hidden
    for i0 in range(0, I, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        mask_i = offs_i < I
        for h0 in range(0, H):
            src = C_batch_base + (T + offs_i) * H + h0
            dst = hid_base + offs_i * H + h0
            tl.store(dst, tl.load(src, mask=mask_i, other=0.0), mask=mask_i)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA and contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        S = T + I

        # Preallocate concatenated [B, S, H] as float32
        concatenated = torch.empty((B, S, H), device=encoder_hidden_states.device, dtype=torch.float32)

        # Launch concat kernel
        grid_concat = (B,)
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            BLOCK_T=128, BLOCK_I=128,
            num_warps=4, num_stages=2,
        )

        # Matmul: C = concatenated @ process_weight.T, treat concatenated as [S, H]
        A = concatenated.view(S, H).contiguous()
        C = torch.empty((S, H), device=encoder_hidden_states.device, dtype=torch.float32)

        Bw_T = process_weight.transpose(0, 1).contiguous()  # [H, H]
        grid_m = (S,)
        grid_n = (1,)  # BLOCK_N = H for provided workloads
        matmul_row_kernel[grid_m * grid_n](
            A, Bw_T, C,
            S, H,
            BLOCK_K=64, BLOCK_N=1024,  # BLOCK_N must equal H for provided workloads
            num_warps=4, num_stages=2,
        )

        # Split into encoder and hidden
        processed_encoder = torch.empty((B, T, H), device=encoder_hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=encoder_hidden_states.device, dtype=torch.float32)

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            BLOCK_T=128, BLOCK_I=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
