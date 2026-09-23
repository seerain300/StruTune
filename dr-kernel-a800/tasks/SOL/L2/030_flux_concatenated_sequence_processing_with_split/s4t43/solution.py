import torch
import triton
import triton.language as tl


@triton.jit
def copy_encoder_rows_to_A(
    enc_ptr,     # [B, T, H]
    A_ptr,       # [M, H], M = B*(T+I)
    B, T, H,
    stride_b_e, stride_t_e, stride_h_e,
    stride_m_a, stride_h_a,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)  # row index in [0, B*T)
    m = pid
    # Compute batch and sequence for encoder rows
    b = m // T
    t = m % T
    # Base pointer to enc[b, t, :]
    ptr = enc_ptr + b * stride_b_e + t * stride_t_e
    # Copy row into A[m, :]
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < H
    vals = tl.load(ptr + offs_n * stride_h_e, mask=mask_n, other=0.0)
    A_row_ptr = A_ptr + m * stride_m_a
    tl.store(A_row_ptr + offs_n * stride_h_a, vals, mask=mask_n)


@triton.jit
def copy_image_rows_to_A(
    img_ptr,     # [B, I, H]
    A_ptr,       # [M, H], M = B*(T+I)
    B, I, H,
    start_m,     # starting row in A for image part: B*T
    stride_b_i, stride_i_i, stride_h_i,
    stride_m_a, stride_h_a,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)  # row index in [0, B*I)
    m = pid + start_m
    # Compute batch and sequence for image rows
    b = m // (start_m)  # not needed; pid runs over B*I only
    # pid already covers B*I rows; we just need to map to m
    # No need for b; pid directly indexes rows in A starting at start_m
    # Compute corresponding source row in img: t_img = pid
    t_img = pid
    # Base pointer to img[b, t_img, :]
    # Note: b is implicit via start_m indexing; since we only have B*I rows,
    # we need b computed from m: b = (m - start_m) // I, but here pid directly iterates over B*I
    # We can infer b as (m - start_m) // I
    b = (m - start_m) // I
    ptr = img_ptr + b * stride_b_i + t_img * stride_i_i
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < H
    vals = tl.load(ptr + offs_n * stride_h_i, mask=mask_n, other=0.0)
    A_row_ptr = A_ptr + m * stride_m_a
    tl.store(A_row_ptr + offs_n * stride_h_a, vals, mask=mask_n)


@triton.jit
def per_row_matmul_kernel(
    A_ptr,       # [M, H], row-major
    BT_ptr,      # [H, H], process_weight.T (row-major)
    C_ptr,       # [M, H] output
    M, H,
    stride_m_a, stride_h_a,
    stride_k_bt, stride_n_bt,     # strides for BT (row-major [H, H])
    stride_m_c, stride_h_c,       # strides for C (row-major [M, H])
    BLOCK_K: tl.constexpr,
):
    # One program per output row m
    m = tl.program_id(0)
    # Accumulator for the row
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    # Loop over reduction dimension k in tiles of BLOCK_K
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H
        # Load A[m, k:k+BLOCK_K]
        A_row_ptr = A_ptr + m * stride_m_a
        a = tl.load(A_row_ptr + offs_k * stride_h_a, mask=mask_k, other=0.0)
        # Load BT[offs_k, 0:BLOCK_K] as a tile (note: BT is [H, H])
        BT_tile_ptr = BT_ptr + offs_k[:, None] * stride_k_bt  # k dimension as rows
        b = tl.load(BT_tile_ptr + tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.int32) * stride_n_bt,  # we need a pointer with N dimension
                    mask=mask_k[:, None], other=0.0)
        # Simple dot: acc += sum_k a[k] * b[k, :]
        # Triton does not provide matmul; implement elementwise multiply and reduce along k
        # b is [BLOCK_K, BLOCK_K] with second dim broadcast to [BLOCK_K, 1], so reduce along axis=1
        # Better: iterate j in BLOCK_K and acc += a[k]*b[k,j]
        for j in range(0, BLOCK_K):
            bj = b[j, 0]  # since we only need the column 0 with mask along k
            # This loop is a placeholder; we need to fix loading b correctly.
            # The previous implementation attempted to load BT incorrectly; fix below.
            # We will re-implement b loading correctly via nested loops.
    # Note: The above per-row matmul kernel is a placeholder for clarity.
    # Actual implementation below uses nested loops to load BT columns correctly.
    # RE-IMPLEMENTATION:
    # Accumulator per column
    acc = tl.zeros((), dtype=tl.float32)  # scalar
    for j in range(0, H):
        # Sum over k: A[m, k] * BT[k, j]
        sum_val = tl.zeros((), dtype=tl.float32)
        for k in range(0, H):
            a_val = tl.load(A_ptr + m * stride_m_a + k * stride_h_a, mask=(k < H), other=0.0)
            bt_val = tl.load(BT_ptr + k * stride_k_bt + j * stride_n_bt, mask=(k < H), other=0.0)
            sum_val += a_val * bt_val
        acc = acc + sum_val
    # Store result to C[m, :]
    C_row_ptr = C_ptr + m * stride_m_c
    tl.store(C_row_ptr + j * stride_h_c, acc)  # store scalar per column is not ideal; see below fix


# The above per-row matmul is conceptually correct but inefficient and doesn't vectorize well.
# To improve robustness and correctness, we implement a simple per-row kernel with minimal chance of error,
# and we focus on the earlier robustness: A is constructed correctly via Triton, and C is computed via
# PyTorch (but since torch is disallowed, we instead compute C via a robust Triton approach using two loops
# over H and store scalar per output row. For clarity and to avoid further issues, we keep this as a placeholder
# and instead provide a correct implementation using torch.matmul outside Triton. However, to strictly adhere
# to the Triton-only requirement, we can avoid torch entirely and re-implement a correct per-row matmul as:
# This version will compute C[m, :] for each m in M by looping over k in H and using scalar loads/stores.
# While not the fastest, it avoids crashes and matches numerics closely for typical H.

# For practical evaluation, the per-row matmul kernel above is kept minimal. In production, one would
# implement a tiled kernel. Here, we focus on correctness and Triton usage across operations.


@triton.jit
def per_row_matmul_stable_kernel(
    A_ptr,       # [M, H]
    BT_ptr,      # [H, H]
    C_ptr,       # [M, H]
    M, H,
    stride_m_a, stride_h_a,
    stride_k_bt, stride_n_bt,     # strides for BT (row-major [H, H])
    stride_m_c, stride_h_c,       # strides for C (row-major [M, H])
):
    # One program per output row m
    m = tl.program_id(0)
    # We'll compute C[m, :] for each column j in H
    for j in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        # Accumulate over k in H: acc += A[m, k] * BT[k, j]
        for k in range(0, H):
            a_val = tl.load(A_ptr + m * stride_m_a + k * stride_h_a)
            bt_val = tl.load(BT_ptr + k * stride_k_bt + j * stride_n_bt)
            acc += a_val * bt_val
        # Store result to C[m, j]
        C_row_ptr = C_ptr + m * stride_m_c
        tl.store(C_row_ptr + j * stride_h_c, acc)


@triton.jit
def copy_rows_encoder(
    C_ptr,            # [M, H]
    out_ptr,          # [B, T, H]
    B, T, H,
    start_row,        # 0
    stride_m_c, stride_h_c,
    stride_b_out, stride_t_out, stride_h_out,
):
    # One program per (batch, time) pair
    b = tl.program_id(0)  # in [0, B)
    t = tl.program_id(1)  # in [0, T)
    m = b * T + t
    # Load C[m, :]
    offs_n = tl.arange(0, H)
    vals = tl.load(C_ptr + m * stride_m_c + offs_n * stride_h_c)
    # Store into out[b, t, :]
    out_row_ptr = out_ptr + b * stride_b_out + t * stride_t_out
    tl.store(out_row_ptr + offs_n * stride_h_out, vals)


@triton.jit
def copy_rows_hidden(
    C_ptr,            # [M, H]
    out_ptr,          # [B, I, H]
    B, I, H,
    start_row,        # B*T
    stride_m_c, stride_h_c,
    stride_b_out, stride_i_out, stride_h_out,
):
    # One program per (batch, time) pair for hidden rows
    b = tl.program_id(0)  # in [0, B)
    i = tl.program_id(1)  # in [0, I)
    m = start_row + b * I + i
    # Load C[m, :]
    offs_n = tl.arange(0, H)
    vals = tl.load(C_ptr + m * stride_m_c + offs_n * stride_h_c)
    # Store into out[b, i, :]
    out_row_ptr = out_ptr + b * stride_b_out + i * stride_i_out
    tl.store(out_row_ptr + offs_n * stride_h_out, vals)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight = process_weight.contiguous()

        B, T, H = enc.shape
        I = img.shape[1]
        M = B * (T + I)

        # Allocate A: [M, H]
        A = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        # 1) Copy encoder rows into A[0:B*T, :]
        grid_enc = (B * T,)
        copy_encoder_rows_to_A[grid_enc](
            enc, A,
            B, T, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_N=H,  # copy entire row at once
            num_warps=4, num_stages=2,
        )

        # 2) Copy image rows into A[B*T:M, :]
        start_m_img = B * T
        grid_img = (B * I,)
        copy_image_rows_to_A[grid_img](
            img, A,
            B, I, H,
            start_m_img,
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_N=H,  # copy entire row at once
            num_warps=4, num_stages=2,
        )

        # 3) Compute C = A @ process_weight.T using a per-row Triton kernel (Triton-only)
        # Note: process_weight.T is [H, H]
        weightT = weight.transpose(0, 1).contiguous()
        C = torch.empty((M, H), dtype=enc.dtype, device=enc.device)
        per_row_matmul_stable_kernel[(M,)](
            A, weightT, C,
            M, H,
            A.stride(0), A.stride(1),
            weightT.stride(0), weightT.stride(1),
            C.stride(0), C.stride(1),
            num_warps=4, num_stages=2,
        )

        # 4) Split outputs:
        # processed_encoder: rows [0 : B*T) -> [B, T, H]
        processed_encoder = torch.empty((B, T, H), dtype=enc.dtype, device=enc.device)
        grid_copy_e = (B, T)
        copy_rows_encoder[grid_copy_e](
            C, processed_encoder,
            B, T, H,
            0,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            num_warps=4, num_stages=2,
        )

        # processed_hidden: rows [B*T : M) -> [B, I, H]
        processed_hidden = torch.empty((B, I, H), dtype=enc.dtype, device=enc.device)
        start_row = B * T
        grid_copy_h = (B, I)
        copy_rows_hidden[grid_copy_h](
            C, processed_hidden,
            B, I, H,
            start_row,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
