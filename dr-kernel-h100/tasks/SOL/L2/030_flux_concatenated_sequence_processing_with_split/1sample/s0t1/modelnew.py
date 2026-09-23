import torch
import triton
import triton.language as tl


# Kernel 1: Concatenate along sequence dimension for each batch.
# Input:
#   - encoder_hidden: [B, T, H], row-major
#   - hidden_states: [B, I, H], row-major
#   - out_cat: [B, T+I, H], row-major
# Each program handles one batch b; it writes out rows 0..T-1 and T..T+I-1 sequentially.
@triton.jit
def _concat_seq_dim_kernel(
    encoder_hidden_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_hb, stride_hi, stride_hh,
    stride_ob, stride_ot, stride_oh,
    BLOCK: tl.constexpr
):
    b = tl.program_id(0)
    total = T + I
    # We will iterate over rows in [0, T+I). For each row i, decide source: i < T -> encoder; else -> hidden.
    for i in range(0, total, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < total
        # Compute source pointers for each element in offs
        # If offs < T -> encoder; else -> hidden
        src_is_encoder = offs < T
        # Select pointers
        src_ptr = tl.where(src_is_encoder, encoder_hidden_ptr + b * stride_eb + offs * stride_et,
                                   hidden_ptr   + b * stride_hb + (offs - T) * stride_hi)
        # Destination pointer
        dst_ptr = out_ptr + b * stride_ob + offs * stride_ot
        # Load and store
        vals = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, vals, mask=mask)


# Kernel 2: Batched GEMM on concatenated rows: C = A @ W, where
#   - A: [M, K] with M = B*(T+I), K = H, A laid out as rows in out_cat: A[b, row] = out_cat[b, row, :]
#   - W: [H, H]
#   - C: [M, H]
# Grid: one program per output row (i.e., per (b, seq) pair). M is passed as a scalar argument.
@triton.jit
def _batched_gemm_right_kernel(
    A_ptr, W_ptr, C_ptr,
    M, K, N,                 # all equal to H in this case
    stride_am, stride_ak,    # strides for A: row m and col k
    stride_wk, stride_wn,    # strides for W: col k and row n
    stride_cm, stride_cn,    # strides for C: row m and col n
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # One program computes one output row m in [0, M)
    m = tl.program_id(0)
    # We iterate over N (columns of output) in blocks and accumulate over K
    for n0 in range(0, N, BLOCK_N):
        acc = tl.zeros((), dtype=tl.float32)  # scalar accumulator for this row
        for k0 in range(0, K, BLOCK_K):
            # Compute block pointers
            a_ptrs = A_ptr + m * stride_am + (k0 + tl.arange(0, BLOCK_K)) * stride_ak
            w_ptrs = W_ptr + (k0 + tl.arange(0, BLOCK_K))[:, None] * stride_wk + (n0 + tl.arange(0, BLOCK_N)) * stride_wn
            a = tl.load(a_ptrs, mask=(k0 + tl.arange(0, BLOCK_K)) < K, other=0.0)          # shape [BLOCK_K]
            w = tl.load(w_ptrs, mask=((k0 + tl.arange(0, BLOCK_K))[:, None] < K) & ((n0 + tl.arange(0, BLOCK_N))[None, :] < N), other=0.0)  # shape [BLOCK_K, BLOCK_N]
            # Dot: sum over K-block of a[j] * w[j, :]
            # Triton will handle broadcasting and compute the dot per n element in the block.
            acc += tl.sum(a[:, None] * w, axis=0)
        # Store result to C[m, n0 + offs_n]
        c_ptrs = C_ptr + m * stride_cm + (n0 + tl.arange(0, BLOCK_N)) * stride_cn
        tl.store(c_ptrs, acc, mask=(n0 + tl.arange(0, BLOCK_N)) < N)


# Kernel 3: Split C [B*(T+I), H] into processed_encoder [B, T, H] and processed_hidden [B, I, H].
# We map m -> (b, s) where s is sequence index in [0, T+I).
@triton.jit
def _split_into_encoder_hidden_kernel(
    C_ptr, out_e_ptr, out_i_ptr,
    B, T, I, H,
    stride_cm, stride_cn,       # C strides: row m and col n
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_ii, stride_ih,
    BLOCK: tl.constexpr
):
    b = tl.program_id(0)
    for s in range(0, T, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        mask = offs < T
        m = b * (T + I) + offs
        src_ptr = C_ptr + m * stride_cm + tl.arange(0, BLOCK) * stride_cn
        dst_e_ptr = out_e_ptr + b * stride_eb + offs * stride_et
        vals = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_e_ptr, vals, mask=mask)
    for s in range(0, I, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        mask = offs < I
        m = b * (T + I) + (T + offs)
        src_ptr = C_ptr + m * stride_cm + tl.arange(0, BLOCK) * stride_cn
        dst_i_ptr = out_i_ptr + b * stride_ib + offs * stride_ii
        vals = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_i_ptr, vals, mask=mask)


# Kernel 4: Optional kernel to show how to avoid cat by writing directly into out_cat,
# but we keep cat as a separate kernel for clarity. Not used in ModelNew.

# Define ModelNew with Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Nothing to initialize; we use Triton kernels.

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate sequences (Triton).
        - Apply linear projection (Triton GEMM).
        - Split results into encoder and hidden streams (Triton).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA tensors"
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H, "Shape mismatch"
        assert process_weight.shape == (H, H), "process_weight must be [H, H]"

        # 1) Concatenate along sequence dimension: out_cat [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Strides
        se_b, se_t, se_h = encoder_hidden_states.stride()
        sh_b, sh_i, sh_h = hidden_states.stride()
        so_b, so_t, so_h = out_cat.stride()

        # Launch concat kernel: one program per batch
        _concat_seq_dim_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            se_b, se_t, se_h,
            sh_b, sh_i, sh_h,
            so_b, so_t, so_h,
            BLOCK=1,  # we write one row at a time in a loop
            num_warps=1
        )

        # 2) Compute C = out_cat @ process_weight.T, i.e., A @ W
        # A has shape [M, K] where M = B*(T+I), K = H; we can materialize A by indexing out_cat:
        # For m in [0, M), row m is out_cat[b, s, :] where b = m // (T+I), s = m % (T+I)
        M = B * (T + I)
        K = H  # hidden_dim
        N = H  # square weight
        # Create C [M, H]
        C = torch.empty((M, H), dtype=process_weight.dtype, device=process_weight.device)

        # Strides for A (we read rows from out_cat)
        # A[m, k] = out_cat[b, s, k] -> pointer = out_cat_ptr + b*so_b + s*so_t + k*so_h
        # But we pass A_ptr as a flat buffer; we compute addresses inside kernel via b and s.
        # We will pass A_ptr as out_cat_ptr, and compute strides via b and s on the fly in kernel.
        # For the kernel, we need a pointer to A rows. We can construct A_ptr as out_cat and then pass
        # A strides via pointers computed on-the-fly (we set stride_am = so_b, stride_ak = so_h).
        # However Triton expects explicit tensors; simpler: we pass out_cat as A_ptr, and for A strides,
        # we set stride_am = 0 (row index computed via m), and stride_ak = H (column index k). Not feasible.
        # Instead, we'll materialize A by indexing out_cat in Python, but since Triton kernel requires tensors,
        # we'll pass out_cat as A_ptr, and in kernel, we read A[m, k] directly from out_cat: row = m, col = k.
        # Triton doesn't support dynamic indexing like x[m] in kernel easily; therefore we compute A row in Python
        # by flattening out_cat into a temporary A tensor. Since this is cheap compared to GEMM, we do it:
        # A = out_cat.view(M, H). But to avoid extra copy, we can keep out_cat as-is and in kernel we read A rows
        # by pointer arithmetic: A_ptr + m*stride_am + k*stride_ak. The correct way is to create a contiguous 2D view.
        # To avoid complexity, we materialize A as a contiguous 2D tensor [M, H] from out_cat:
        # A_mat = out_cat.reshape(M, H).contiguous()
        A_mat = out_cat.reshape(M, H).contiguous()

        # Now launch GEMM kernel. We set strides appropriately.
        # A_mat strides: row stride = H, col stride = 1
        a_stride_m = H
        a_stride_k = 1

        # Weight W is [H, H], contiguous. We will pass it as is.
        W = process_weight.contiguous()
        w_stride_k = W.stride(0)  # typically 1
        w_stride_n = W.stride(1)  # typically H

        # Output C strides: row m, col n
        c_stride_m = C.stride(0)  # typically H
        c_stride_n = C.stride(1)  # typically 1

        # Launch GEMM: one program per output row m
        # We pick block sizes; H is usually 64/128/256. Defaults work for many cases.
        BLOCK_M = 1
        BLOCK_N = 64
        BLOCK_K = 32

        _batched_gemm_right_kernel[(M,)](
            A_mat, W, C,
            M, K, N,
            a_stride_m, a_stride_k,
            w_stride_k, w_stride_n,
            c_stride_m, c_stride_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split C into encoder and hidden outputs
        processed_encoder = torch.empty((B, T, H), dtype=process_weight.dtype, device=process_weight.device)
        processed_hidden = torch.empty((B, I, H), dtype=process_weight.dtype, device=process_weight.device)

        # Strides for outputs
        pe_b, pe_t, pe_h = processed_encoder.stride()
        ph_b, ph_i, ph_h = processed_hidden.stride()

        # Strides for C
        c_stride_cm = C.stride(0)  # typically N (H)
        c_stride_cn = C.stride(1)  # typically 1

        # Launch split kernels: one program per batch
        _split_into_encoder_hidden_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            c_stride_cm, c_stride_cn,
            pe_b, pe_t, pe_h,
            ph_b, ph_i, ph_h,
            BLOCK=1,
            num_warps=1
        )

        return processed_encoder, processed_hidden