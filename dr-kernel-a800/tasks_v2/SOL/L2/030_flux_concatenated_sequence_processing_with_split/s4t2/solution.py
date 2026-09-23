import torch
import triton
import triton.language as tl

# Kernel 1: Concatenate encoder_hidden_states and hidden_states into A[M, H]
@triton.jit
def cat_to_A_kernel(
    enc_ptr, img_ptr, A_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_am, stride_ah,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile over rows
    pid_n = tl.program_id(1)  # tile over columns
    M_total = B * (T + I)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M_total
    n_mask = offs_n < H

    # Compute batch and sequence index for each row in concatenated space
    b = offs_m // (T + I)
    s = offs_m % (T + I)

    # Masks: which rows come from encoder vs image
    mask_e = (s < T) & m_mask
    mask_i = (s >= T) & m_mask

    # Compute pointers for encoder and image rows
    enc_ptrs = enc_ptr + (b[:, None] * stride_eb + s[:, None] * stride_et + offs_n[None, :] * stride_eh)
    img_ptrs = img_ptr + (b[:, None] * stride_ib + (s - T)[:, None] * stride_it + offs_n[None, :] * stride_ih)

    # Load with masks; invalid rows contribute zeros
    vals_e = tl.load(enc_ptrs, mask=(mask_e[:, None] & n_mask[None, :]), other=0.0)
    vals_i = tl.load(img_ptrs, mask=(mask_i[:, None] & n_mask[None, :]), other=0.0)

    # Sum gives the correct row; disjoint masks ensure only one contributes per row
    A_vals = vals_e + vals_i

    # Store into A [M, H]
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_ah)
    tl.store(A_ptrs, A_vals, mask=(m_mask[:, None] & n_mask[None, :]))


# Kernel 2: Batched matmul C[M, H] = A[M, H] @ B[H, H] (B = process_weight.T)
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, H,
    stride_am, stride_ah,
    stride_bh, stride_bk,   # B is [H, H], so stride_bh along rows (H), stride_bk along cols (H)
    stride_cm, stride_ch,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tiles over M
    pid_n = tl.program_id(1)  # tiles over H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < H

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ah)
        a = tl.load(a_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N], B[H, H]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bh + offs_n[None, :] * stride_bk)
        b = tl.load(b_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)

        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_ch)
    tl.store(c_ptrs, acc, mask=(m_mask[:, None] & n_mask[None, :]))


# Kernel 3: Per-batch copy for encoder rows [b*T : (b+1)*T, :] from C into processed_encoder[b]
@triton.jit
def copy_rows_kernel(
    src_ptr, dest_ptr,
    B, T, I, H,
    stride_srcm, stride_srch,
    stride_destb, stride_destt, stride_desth,
    start_row,  # starting row in src (e.g., b*T)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < T
    n_mask = offs_n < H

    src_rows = start_row + offs_m
    src_ptrs = src_ptr + (src_rows[:, None] * stride_srcm + offs_n[None, :] * stride_srch)
    # Destination tensor is [B, T, H] for processed_encoder[b]
    dest_ptrs = dest_ptr + (offs_m[:, None] * stride_destt + offs_n[None, :] * stride_desth)

    vals = tl.load(src_ptrs, mask=(m_mask[:, None] & n_mask[None, :]), other=0.0)
    tl.store(dest_ptrs, vals, mask=(m_mask[:, None] & n_mask[None, :]))


# Kernel 4: Per-batch copy for hidden rows [(b+1)*T + b*I : (b+1)*(T+I), :] from C into processed_hidden[b]
@triton.jit
def copy_rows_kernel_hidden(
    src_ptr, dest_ptr,
    B, T, I, H,
    stride_srcm, stride_srch,
    stride_destb, stride_destt, stride_desth,
    start_row,  # starting row in src (e.g., (b+1)*T + b*I)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < I
    n_mask = offs_n < H

    src_rows = start_row + offs_m
    src_ptrs = src_ptr + (src_rows[:, None] * stride_srcm + offs_n[None, :] * stride_srch)
    dest_ptrs = dest_ptr + (offs_m[:, None] * stride_destt + offs_n[None, :] * stride_desth)

    vals = tl.load(src_ptrs, mask=(m_mask[:, None] & n_mask[None, :]), other=0.0)
    tl.store(dest_ptrs, vals, mask=(m_mask[:, None] & n_mask[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states into A[M, H] via Triton (no torch.cat).
        - Performs batched matmul C[M, H] = A @ process_weight.T via Triton (no torch.matmul).
        - Copies rows per batch to produce processed_encoder and processed_hidden (no torch slicing).
        """
        # Validate shapes
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, S, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = B * (T + I)

        # Ensure contiguity
        enc = encoder_hidden_states.contiguous()     # [B, T, H]
        img = hidden_states.contiguous()            # [B, I, H]
        weight_T = process_weight.t().contiguous()  # [H, H]

        # 1) Build A [M, H] via Triton concatenation
        A = torch.empty((M, H), dtype=enc.dtype, device=enc.device)
        BLOCK_M = 128
        BLOCK_N = 128
        grid_cat = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        cat_to_A_kernel[grid_cat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            num_warps=4, num_stages=2,
        )

        # 2) Batched matmul C[M, H] = A[M, H] @ weight_T[H, H] via Triton
        C = torch.empty((M, H), dtype=enc.dtype, device=enc.device)
        BLOCK_Mm = 64
        BLOCK_Nn = 64
        BLOCK_Kk = 32
        grid_mm = (triton.cdiv(M, BLOCK_Mm), triton.cdiv(H, BLOCK_Nn))
        batched_matmul_kernel[grid_mm](
            A, weight_T, C,
            M, H,
            A.stride(0), A.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_Mm, BLOCK_N=BLOCK_Nn, BLOCK_K=BLOCK_Kk,
            num_warps=4, num_stages=2,
        )

        # 3) Per-batch row copies for splitting
        processed_encoder = torch.empty((B, T, H), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, I, H), dtype=enc.dtype, device=enc.device)

        BLOCK_COPY_M = 128
        BLOCK_COPY_N = 128

        for b in range(B):
            # Copy encoder rows [b*T : (b+1)*T, :] to processed_encoder[b]
            grid_e = (triton.cdiv(T, BLOCK_COPY_M), triton.cdiv(H, BLOCK_COPY_N))
            copy_rows_kernel[grid_e](
                C, processed_encoder[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_encoder[b].stride(0), processed_encoder[b].stride(1), processed_encoder[b].stride(2),
                start_row=b * T,
                num_warps=4, num_stages=2,
            )
            # Copy hidden rows [(b+1)*T + b*I : (b+1)*(T+I), :] to processed_hidden[b]
            start_row_hidden = (b + 1) * T + b * I
            grid_h = (triton.cdiv(I, BLOCK_COPY_M), triton.cdiv(H, BLOCK_COPY_N))
            copy_rows_kernel_hidden[grid_h](
                C, processed_hidden[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_hidden[b].stride(0), processed_hidden[b].stride(1), processed_hidden[b].stride(2),
                start_row=start_row_hidden,
                num_warps=4, num_stages=2,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
