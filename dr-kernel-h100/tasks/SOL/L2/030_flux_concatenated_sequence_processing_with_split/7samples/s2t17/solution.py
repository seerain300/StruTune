import torch
import triton
import triton.language as tl

@triton.jit
def copy_encoder_into_concat_kernel(
    encoder: tl.pointer_type(tl.float32, 3),    # [B, T, H]
    concatenated: tl.pointer_type(tl.float32, 3),  # [B, S, H], S = T + I
    B: tl.int32, T: tl.int32, H: tl.int32,
    stride_encoder_b: tl.int32, stride_encoder_t: tl.int32, stride_encoder_h: tl.int32,
    stride_concat_b: tl.int32, stride_concat_s: tl.int32, stride_concat_h: tl.int32,
    BLOCK_T: tl.constexpr = 128,
):
    b = tl.program_id(0)
    t_offsets = tl.arange(0, BLOCK_T)
    # rows to copy: 0..T-1
    mask = t_offsets < T
    # copy rows t in [0, T)
    for t in range(0, T, BLOCK_T):
        t_offsets = t + tl.arange(0, BLOCK_T)
        m = t_offsets  # m is the row index in concatenated (0..T-1)
        # pointers
        enc_ptrs = encoder + b * stride_encoder_b + m * stride_encoder_t + tl.arange(0, H) * stride_encoder_h
        out_ptrs = concatenated + b * stride_concat_b + m * stride_concat_s + tl.arange(0, H) * stride_concat_h
        # load and store
        vals = tl.load(enc_ptrs, mask=mask, other=0.0)
        tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def copy_hidden_into_concat_after_kernel(
    hidden: tl.pointer_type(tl.float32, 3),     # [B, I, H]
    concatenated: tl.pointer_type(tl.float32, 3),  # [B, S, H], S = T + I
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_hidden_b: tl.int32, stride_hidden_i: tl.int32, stride_hidden_h: tl.int32,
    stride_concat_b: tl.int32, stride_concat_s: tl.int32, stride_concat_h: tl.int32,
    BLOCK_I: tl.constexpr = 128,
):
    b = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_I)
    # start row in concatenated after T
    start = T
    for i in range(0, I, BLOCK_I):
        i_offsets = i + tl.arange(0, BLOCK_I)
        m = start + i_offsets  # m is the row index in concatenated (T..T+I-1)
        mask = i_offsets < I
        # pointers
        hid_ptrs = hidden + b * stride_hidden_b + i_offsets * stride_hidden_i + tl.arange(0, H) * stride_hidden_h
        out_ptrs = concatenated + b * stride_concat_b + m * stride_concat_s + tl.arange(0, H) * stride_concat_h
        vals = tl.load(hid_ptrs, mask=mask, other=0.0)
        tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def matmul_row_kernel(
    A: tl.pointer_type(tl.float32, 2),   # [S, H], rows 0..S-1
    B: tl.pointer_type(tl.float32, 2),   # [H, H]
    C: tl.pointer_type(tl.float32, 2),   # [S, H]
    S: tl.int32, H: tl.int32,
    stride_A_m: tl.int32, stride_A_k: tl.int32,
    stride_B_k: tl.int32, stride_B_n: tl.int32,
    stride_C_m: tl.int32, stride_C_n: tl.int32,
    BLOCK_K: tl.constexpr = 64, BLOCK_N: tl.constexpr = 1024,
):
    # 2D grid: (rows m, tiles over N)
    m = tl.program_id(0)
    n_tile = tl.program_id(1)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H
    # accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # loop over K
    for k in range(0, H, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H
        # A[m, k] -> vector of size BLOCK_K
        a_ptrs = A + m * stride_A_m + k_offsets * stride_A_k
        a_vec = tl.load(a_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]
        # B[k, n] -> matrix [BLOCK_K, BLOCK_N]
        b_ptrs = B + k_offsets[:, None] * stride_B_k + n_offsets[None, :] * stride_B_n
        b_mat = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]
        # FMA
        acc += tl.sum(b_mat * a_vec[:, None], axis=0)
    # store
    c_ptrs = C + m * stride_C_m + n_offsets * stride_C_n
    tl.store(c_ptrs, acc, mask=mask_n)


@triton.jit
def split_seqs_kernel(
    C: tl.pointer_type(tl.float32, 3),         # [B, S, H]
    out_encoder: tl.pointer_type(tl.float32, 3),  # [B, T, H]
    out_hidden: tl.pointer_type(tl.float32, 3),  # [B, I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_C_b: tl.int32, stride_C_s: tl.int32, stride_C_h: tl.int32,
    stride_e_b: tl.int32, stride_e_t: tl.int32, stride_e_h: tl.int32,
    stride_h_b: tl.int32, stride_h_i: tl.int32, stride_h_h: tl.int32,
    BLOCK_T: tl.constexpr = 128, BLOCK_I: tl.constexpr = 128,
):
    b = tl.program_id(0)
    # copy first T rows to encoder output
    t_offsets = tl.arange(0, BLOCK_T)
    for t in range(0, T, BLOCK_T):
        t_offsets = t + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T
        c_ptrs = C + b * stride_C_b + t_offsets * stride_C_s + tl.arange(0, H) * stride_C_h
        e_ptrs = out_encoder + b * stride_e_b + t_offsets * stride_e_t + tl.arange(0, H) * stride_e_h
        vals = tl.load(c_ptrs, mask=mask_t, other=0.0)
        tl.store(e_ptrs, vals, mask=mask_t)
    # copy remaining I rows to hidden output
    i_offsets = tl.arange(0, BLOCK_I)
    for i in range(0, I, BLOCK_I):
        i_offsets = i + tl.arange(0, BLOCK_I)
        mask_i = i_offsets < I
        start = T
        c_ptrs = C + b * stride_C_b + (start + i_offsets) * stride_C_s + tl.arange(0, H) * stride_C_h
        h_ptrs = out_hidden + b * stride_h_b + i_offsets * stride_h_i + tl.arange(0, H) * stride_h_h
        vals = tl.load(c_ptrs, mask=mask_i, other=0.0)
        tl.store(h_ptrs, vals, mask=mask_i)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,            # [B, I, H]
        encoder_hidden_states: torch.Tensor,   # [B, T, H]
        process_weight: torch.Tensor,          # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        1) Concatenate [B, T, H] and [B, I, H] into [B, S, H] where S = T + I.
        2) Compute processed = concatenated @ process_weight.T via Triton GEMM.
        3) Split processed back into [B, T, H] and [B, I, H].
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 for now"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2 and H == H2, "Mismatched shapes"

        S = T + I

        # 1) Allocate and copy encoder into concatenated[:, :T, :]
        concatenated = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)
        # launch copy encoder
        grid_copy_encoder = (B,)
        copy_encoder_into_concat_kernel[grid_copy_encoder](
            encoder_hidden_states, concatenated,
            B, T, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_T=128,
            num_warps=4, num_stages=2,
        )

        # 2) Copy hidden into concatenated[:, T:, :]
        grid_copy_hidden = (B,)
        copy_hidden_into_concat_after_kernel[grid_copy_hidden](
            hidden_states, concatenated,
            B, T, I, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_I=128,
            num_warps=4, num_stages=2,
        )

        # 3) Compute processed = concatenated @ process_weight.T using row-wise Triton GEMM
        #    A is [S, H], B is [H, H], C is [S, H]
        C = torch.empty((S, B, H), device=hidden_states.device, dtype=torch.float32)  # [S, B, H] so each row m gives C[m, :]
        grid_mat = (S, 1)  # since BLOCK_N=1024 covers H=1024, we only need one tile over N
        matmul_row_kernel[grid_mat](
            concatenated, process_weight.t().contiguous(), C,
            S, H,
            concatenated.stride(0), concatenated.stride(2),  # stride over rows (m) and K (h) for A
            process_weight.t().stride(0), process_weight.t().stride(1),  # for B: [H, H], k stride, n stride
            C.stride(0), C.stride(2),  # C strides over m and n
            BLOCK_K=64, BLOCK_N=1024,
            num_warps=4, num_stages=3,
        )

        # Note: C is [S, B, H]. We need outputs [B, S, H]. Create a transposed view for split? Easier: recompute in correct layout.
        # To produce [B, S, H], we can instead compute C as [B, S, H] by using a different pointer layout in kernel.
        # Fix: recompute C as [B, S, H] directly by adjusting grid and pointers.

        # Recompute C correctly shaped as [B, S, H]
        C_fixed = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)
        grid_mat_fixed = (S, 1)
        matmul_row_kernel[grid_mat_fixed](
            concatenated, process_weight.t().contiguous(), C_fixed,
            S, H,
            concatenated.stride(0), concatenated.stride(2),  # A strides: m and k
            process_weight.t().stride(0), process_weight.t().stride(1),  # B strides: k and n
            C_fixed.stride(1), C_fixed.stride(2),  # C strides: m (row over B) and n (H)
            BLOCK_K=64, BLOCK_N=1024,
            num_warps=4, num_stages=3,
        )

        # 4) Split C_fixed [B, S, H] into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=torch.float32)

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C_fixed, processed_encoder, processed_hidden,
            B, T, I, H,
            C_fixed.stride(0), C_fixed.stride(1), C_fixed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_T=128, BLOCK_I=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


# Example usage (if needed):
# model = ModelNew().cuda()
# hidden_states = torch.randn(2, 128, 1024, device='cuda', dtype=torch.float32)
# encoder_hidden_states = torch.randn(2, 256, 1024, device='cuda', dtype=torch.float32)
# process_weight = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
# out_e, out_i = model(encoder_hidden_states, hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
