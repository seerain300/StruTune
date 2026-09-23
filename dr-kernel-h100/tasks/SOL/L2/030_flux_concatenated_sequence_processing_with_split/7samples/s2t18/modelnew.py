import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder [B, T, H] and hidden [B, I, H] into [B, S, H], S = T + I.
@triton.jit
def concat_seqs_kernel(
    enc_ptr,          # *const float, [B, T, H]
    hid_ptr,          # *const float, [B, I, H]
    out_ptr,          # *float, [B, S, H]
    B: tl.constexpr,  # int
    T: tl.constexpr,  # int
    I: tl.constexpr,  # int
    H: tl.constexpr,  # int (hidden dim)
    S: tl.constexpr,  # int (T + I)
    stride_enc_b, stride_enc_t, stride_enc_h,
    stride_hid_b, stride_hid_i, stride_hid_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_H: tl.constexpr,  # tile along H
):
    b = tl.program_id(0)  # batch id
    # copy encoder rows 0..T-1 into out[b, 0..T-1, :]
    for t in range(0, T):
        # We process the full H dimension in one tile to avoid masks
        for h_off in range(0, H, BLOCK_H):
            h_offsets = h_off + tl.arange(0, BLOCK_H)
            mask = h_offsets < H
            enc_row_ptr = enc_ptr + b * stride_enc_b + t * stride_enc_t + h_offsets * stride_enc_h
            out_row_ptr = out_ptr + b * stride_out_b + t * stride_out_s + h_offsets * stride_out_h
            val = tl.load(enc_row_ptr, mask=mask, other=0.0)
            tl.store(out_row_ptr, val, mask=mask)
    # copy hidden rows 0..I-1 into out[b, T..T+I-1, :]
    for i in range(0, I):
        for h_off in range(0, H, BLOCK_H):
            h_offsets = h_off + tl.arange(0, BLOCK_H)
            mask = h_offsets < H
            hid_row_ptr = hid_ptr + b * stride_hid_b + i * stride_hid_i + h_offsets * stride_hid_h
            out_row_ptr = out_ptr + b * stride_out_b + (T + i) * stride_out_s + h_offsets * stride_out_h
            val = tl.load(hid_row_ptr, mask=mask, other=0.0)
            tl.store(out_row_ptr, val, mask=mask)

# Triton kernel: for each row m of A [S, H], compute C[m, :] = A[m, :] @ B [H, H]
@triton.jit
def row_matmul_kernel(
    A_ptr,            # *const float, [S, H], row-major: A[m, k]
    B_ptr,            # *const float, [H, H], row-major: B[k, n]
    C_ptr,            # *float, [S, H], row-major: C[m, n]
    S: tl.constexpr,  # int
    H: tl.constexpr,  # int
    stride_A_m, stride_A_k,
    stride_B_k, stride_B_n,
    stride_C_m, stride_C_n,
    BLOCK_K: tl.constexpr,  # tile over K
    BLOCK_N: tl.constexpr,  # tile over N (set to H to avoid partial tiles)
):
    m = tl.program_id(0)  # row id in [0, S)
    n_tile = tl.program_id(1)  # tile id over N (since BLOCK_N=H, there is only one tile)
    # Accumulator for the row
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Loop over K dimension
    for k_off in range(0, H, BLOCK_K):
        k_offsets = k_off + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N], equals [0..H-1] since BLOCK_N=H
        # Load A[m, k] as vector [BLOCK_K]
        A_row_ptr = A_ptr + m * stride_A_m + k_offsets * stride_A_k
        a = tl.load(A_row_ptr)  # [BLOCK_K]
        # Load B[k, n] as matrix [BLOCK_K, BLOCK_N]
        B_mat_ptr = B_ptr + k_offsets[:, None] * stride_B_k + n_offsets[None, :] * stride_B_n
        b = tl.load(B_mat_ptr)  # [BLOCK_K, BLOCK_N]
        # acc += a @ b
        acc += tl.sum(a[:, None] * b, axis=0)  # [BLOCK_N]
    # Store the result row into C[m, :]
    C_row_ptr = C_ptr + m * stride_C_m + n_offsets * stride_C_n
    # mask is trivial since BLOCK_N=H; but keep it for safety
    mask = n_offsets < H
    tl.store(C_row_ptr, acc, mask=mask)

# Triton kernel: split C [B, S, H] into processed_encoder [B, T, H] and processed_hidden [B, I, H]
@triton.jit
def split_seqs_kernel(
    C_ptr,            # *const float, [B, S, H]
    out_e_ptr,        # *float, [B, T, H]
    out_i_ptr,        # *float, [B, I, H]
    B: tl.constexpr,  # int
    T: tl.constexpr,  # int
    I: tl.constexpr,  # int
    H: tl.constexpr,  # int
    S: tl.constexpr,  # int (T + I)
    stride_C_b, stride_C_s, stride_C_h,
    stride_e_b, stride_e_t, stride_e_h,
    stride_i_b, stride_i_i, stride_i_h,
):
    b = tl.program_id(0)
    # copy first T rows to encoder
    for t in range(0, T):
        for h_off in range(0, H):
            src_ptr = C_ptr + b * stride_C_b + t * stride_C_s + h_off * stride_C_h
            dst_ptr_e = out_e_ptr + b * stride_e_b + t * stride_e_t + h_off * stride_e_h
            val = tl.load(src_ptr)
            tl.store(dst_ptr_e, val)
    # copy remaining I rows to hidden
    for i in range(0, I):
        for h_off in range(0, H):
            src_ptr = C_ptr + b * stride_C_b + (T + i) * stride_C_s + h_off * stride_C_h
            dst_ptr_i = out_i_ptr + b * stride_i_b + i * stride_i_i + h_off * stride_i_h
            val = tl.load(src_ptr)
            tl.store(dst_ptr_i, val)

class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        Concatenates along sequence dim, applies linear projection, splits back.
        All heavy ops are done in Triton. Forward only allocates tensors and launches Triton kernels.
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"
        # Shapes
        B, T, H = encoder_hidden_states.shape
        Bi, I, Hi = hidden_states.shape
        assert B == Bi, "Batch sizes must match"
        assert H == Hi, "Hidden dims must match"
        S = T + I

        # Ensure contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        Bw = process_weight.contiguous()  # [H, H]
        # 1) Concatenate into [B, S, H]
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)
        grid_concat = (B,)
        concat_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H, S,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_H=H,  # process full H dimension in one tile
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: C = concatenated @ process_weight.T, i.e. A [S, H] @ B [H, H] -> C [S, H]
        # A is [S, H]; we pass row m = concatenated[m, :] against B = process_weight.T
        Bw_T = Bw.t().contiguous()  # [H, H]
        C = torch.empty((S, H), device=enc.device, dtype=enc.dtype)
        grid_matmul = (S, 1)  # single tile over N since BLOCK_N=H
        row_matmul_kernel[grid_matmul](
            concatenated, Bw_T, C,
            S, H,
            concatenated.stride(0), concatenated.stride(2),  # A[m, k] with m=row, k=hidden
            Bw_T.stride(0), Bw_T.stride(1),                 # B[k, n]
            C.stride(0), C.stride(1),
            BLOCK_K=64, BLOCK_N=H,
            num_warps=4, num_stages=2,
        )

        # 3) Split C into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)
        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden