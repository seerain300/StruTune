import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    enc_ptr,       # *ptr to encoder_hidden_states: [N, L_txt, K]
    hid_ptr,       # *ptr to hidden_states: [N, L_img, K]
    out_ptr,       # *ptr to output concatenated: [N, L_total, K]
    N, L_txt, L_img, K,                # ints
    grid_m, grid_n,                     # ints, strides for pointer arithmetic inside the kernel
    BLOCK_K: tl.constexpr,             # tile along K
):
    # 2D launch: (n, which) where which = 0 means encoder part, which = 1 means hidden part
    n = tl.program_id(0)
    which = tl.program_id(1)
    # Compute t or i index based on which
    # We'll use a 1D range along K for copying
    # Each program copies a full K-length vector (row) for a given (n, t) or (n, i)
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        if which == 0:
            # copy from enc_ptr[n, t, offs_k] to out_ptr[n, t, offs_k]
            # We need t here; since which==0, t = grid_m index (not used here because grid_m is N? No: grid is (N, 2); but we cannot access grid_m directly. Instead, pass t via program_id grid mapping.)
            # Correction: We should have a separate launch for t loop. Triton doesn't provide dynamic program_id mapping across grid sizes easily. Better: use a 3D grid (N, L_txt, ceil_div(K, BLOCK_K)) and a separate grid for img (N, L_img, ceil_div(K, BLOCK_K)).
            # To keep code concise, we implement two separate kernels for encoder and image parts below.
            # Placeholder logic: we need to read t from a second program_id. Triton doesn't allow accessing program_id arguments beyond those provided. Hence, we split into two kernels. This kernel will not be used in final code.
            pass
    # Note: Above is a placeholder. We will define two actual kernels below for encoder and image parts.


# Actual Triton kernel to concatenate encoder and hidden parts.
@triton.jit
def _concat_sequences_kernel_actual(
    enc_ptr,       # *ptr to encoder_hidden_states: [N, L_txt, K]
    hid_ptr,       # *ptr to hidden_states: [N, L_img, K]
    out_ptr,       # *ptr to output concatenated: [N, L_total, K]
    N, L_txt, L_img, K,                 # ints
    enc_stride_n, enc_stride_l, enc_stride_k,
    hid_stride_n, hid_stride_l, hid_stride_k,
    out_stride_n, out_stride_l, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (N, L_txt) for encoder part and (N, L_img) for hidden part, but we can use masks and pass which via program_id(2) to select source.
    # Simpler approach: launch two separate Triton calls from host. For completeness, we implement a single kernel with an extra program_id dimension to select which part.
    which = tl.program_id(2)  # 0 -> encoder part, 1 -> hidden part
    n = tl.program_id(0)
    idx = tl.program_id(1)
    if which == 0:
        t = idx  # index along text sequence
        # Copy encoder_hidden_states[n, t, :] -> out[n, t, :]
        # Loop over K in tiles
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask = offs_k < K
            enc_row_ptr = enc_ptr + n * enc_stride_n + t * enc_stride_l + offs_k * enc_stride_k
            out_row_ptr = out_ptr + n * out_stride_n + t * out_stride_l + offs_k * out_stride_k
            vals = tl.load(enc_row_ptr, mask=mask, other=0.0)
            tl.store(out_row_ptr, vals, mask=mask)
    else:
        i = idx  # index along image sequence
        # Copy hidden_states[n, i, :] -> out[n, L_txt + i, :]
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask = offs_k < K
            hid_row_ptr = hid_ptr + n * hid_stride_n + i * hid_stride_l + offs_k * hid_stride_k
            # out index along sequence is L_txt + i
            out_row_ptr = out_ptr + n * out_stride_n + (L_txt + i) * out_stride_l + offs_k * out_stride_k
            vals = tl.load(hid_row_ptr, mask=mask, other=0.0)
            tl.store(out_row_ptr, vals, mask=mask)


@triton.jit
def _matmul_batched_rows_kernel(
    A_ptr, B_ptr, C_ptr,
    N_rows, K,                          # ints
    A_stride_m, A_stride_k,            # A is [N_rows, K]
    B_stride_k, B_stride_n,            # B is [K, K]
    C_stride_m, C_stride_n,            # C is [N_rows, K]
    BLOCK_M: tl.constexpr,             # not used since we have only one M=N_rows; included for signature consistency
    BLOCK_N: tl.constexpr,             # output rows N (here N_rows)
    BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B, A is [N_rows, K], B is [K, K], C is [N_rows, K]
    # We flatten N_rows as the "row" dimension of A. Here we treat each row independently:
    # grid_m = N_rows, grid_n = 1. This kernel would be more efficient as a tiled matmul over a 2D grid. To keep simple and correct, we implement a loop over K and write a vector output. In practice, we should use a 2D grid over rows and K tiles.
    # However, Triton expects a grid. Since we need output [N_rows, K], we can instead write a kernel that takes grid over rows and loops over K. For robustness and to match typical Triton matmul examples, we implement a 2D grid over (rows, tiles along N), but here N is 1. So we adjust and provide a proper tiled kernel below.

    # Proper tiled matmul kernel over [N_rows, K] x [K, K] -> [N_rows, K]
    # Grid: (N_rows, ceil_div(K, BLOCK_N)), but since output is [N_rows, K], we can use BLOCK_N = K for simplicity. Better: use a 2D tiling with BLOCK_M, BLOCK_N, BLOCK_K and compute over multiple rows per program if needed. For clarity and correctness, we implement a straightforward approach with a 2D grid where each program handles one output row and reduces over K in tiles.
    # But Triton matmul patterns typically use a 2D grid over (M_tiles, N_tiles). Here M=N_rows, N=K. We implement that:

    # We need to compute C[m, n] = sum_k A[m, k] * B[k, n] for m in [0, N_rows), n in [0, K)
    # However, Triton expects us to write a kernel that produces a tile of C. To handle general N_rows, we launch grid_m over rows and grid_n over output columns. Since here N is the same as K, we can do:

    # Re-implement a correct tiled kernel:
    # Grid is (N_rows, ceil_div(K, BLOCK_N)). Each program computes one row m and a block of columns n0:n0+BLOCK_N, reducing over K in BLOCK_K chunks.

    # Note: Triton requires a grid. We will set grid as (N_rows, 1) but implement internal loops. For simplicity and correctness, we provide a complete matmul kernel with proper 2D grid over (rows, column tiles). Triton doesn’t expose M/N dims beyond strides. To avoid confusion, we define the grid and use while loops.

    # Triton doesn’t support dynamic grid_m in Python; we pass grid sizes as (N_rows, 1). Inside the kernel, we treat grid_n as 1 and loop over K in tiles. This ensures correctness, but performance is not ideal. For a better approach, we provide a correct matmul kernel with proper grid mapping below.

    # We will now define a correct matmul kernel with grid mapping over tiles. Triton’s example matmul uses three loops over K in tiles and writes tiles. We’ll mirror that.

    # Kernel structure: 2D grid over (row_tiles, col_tiles), each program handles a tile of size (BLOCK_M, BLOCK_N), reduces over K in BLOCK_K chunks, and stores into C.
    # We need to compute which tile this program belongs to:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile coordinates
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # Create row/col indices for the tile
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = offs_m < N_rows
    mask_n = offs_n < K  # Since N is K in our case

    # Initialize accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    k_iter = 0
    while k_iter < K:
        k0 = k_iter
        # Pointer arithmetic for A and B
        # A is [N_rows, K], row index vector offs_m, col index vector k0 + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * A_stride_m + (k0 + offs_k[None, :]) * A_stride_k
        # B is [K, K], row index vector k0 + offs_k, col index vector offs_n
        b_ptrs = B_ptr + (k0 + offs_k[:, None]) * B_stride_k + offs_n[None, :] * B_stride_n

        # Masks for loads
        a_mask = (offs_m[:, None] < N_rows) & (offs_k[None, :] + k0 < K)
        b_mask = (offs_k[:, None] + k0 < K) & (offs_n[None, :] < K)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)
        k_iter += BLOCK_K

    # Store results into C at [offs_m, offs_n]
    c_ptrs = C_ptr + offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


# Final implementation: Triton kernels for concatenation and GEMM. We will use the proper grid mapping for GEMM.

@triton.jit
def _matmul_batched_rows_tiled_kernel(
    A_ptr, B_ptr, C_ptr,
    N_rows, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: programs over tiles of output rows and columns
    pid_m = tl.program_id(0)  # tile index along rows
    pid_n = tl.program_id(1)  # tile index along columns

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    offs_m = m0 + tl.arange(0, BLOCK_M)          # rows in A and C
    offs_n = n0 + tl.arange(0, BLOCK_N)          # columns in B and C
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < N_rows
    mask_n = offs_n < K

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K
    k_iter = 0
    while k_iter < K:
        k0 = k_iter
        a_ptrs = A_ptr + offs_m[:, None] * A_stride_m + (k0 + offs_k[None, :]) * A_stride_k
        b_ptrs = B_ptr + (k0 + offs_k[:, None]) * B_stride_k + offs_n[None, :] * B_stride_n

        a_mask = mask_m[:, None] & (offs_k[None, :] + k0 < K)
        b_mask = (offs_k[:, None] + k0 < K) & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)
        k_iter += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


# Helper to launch the tiled GEMM
def triton_matmul_concat(
    hidden_states: torch.Tensor,            # [N, L_img, K]
    encoder_hidden_states: torch.Tensor,    # [N, L_txt, K]
    process_weight_T: torch.Tensor,         # [K, K], process_weight transposed
) -> torch.Tensor:                         # [N, L_total, K]
    """
    Compute (cat(encoder_hidden_states, hidden_states) @ process_weight_T) using Triton.
    Returns a tensor of shape [N, L_total, K] which we will then split into encoder and image streams.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight_T.is_cuda, "Tensors must be on CUDA for Triton."
    assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight_T.dtype == torch.float32, "Use float32 for Triton kernels."

    N = hidden_states.shape[0]
    L_txt = encoder_hidden_states.shape[1]
    L_img = hidden_states.shape[1]
    K = hidden_states.shape[2]
    L_total = L_txt + L_img

    # We will create A_cat in PyTorch, but populate it with Triton via two kernels: one for encoder part, one for hidden part. To keep a single allocation, we directly write into output A_cat using two kernel launches.
    # However, Triton kernels read from inputs and write to output. We can first allocate A_cat as a tensor, then launch two kernels to copy encoder and hidden parts into A_cat accordingly.

    # Allocate A_cat (input for matmul) and directly fill via Triton by calling two separate copy kernels below.
    # Simpler: allocate A_cat and run two kernels to fill it. We will write a Triton kernel for concatenation.

    # Allocate A_cat [N, L_total, K]
    A_cat = torch.empty((N, L_total, K), device=hidden_states.device, dtype=torch.float32)

    # Prepare strides for the concatenation kernel
    # We need to pass strides for both encoder and hidden inputs and output.
    # Since we will fill A_cat directly via kernels, we derive strides by making tensors contiguous.
    # But we can pass strides of the existing tensors; we'll call .contiguous() just in case.
    # For simplicity, we assume contiguous inputs. We can make them contiguous before passing to kernel.
    enc = encoder_hidden_states.contiguous()
    hid = hidden_states.contiguous()

    # Strides
    enc_stride_n, enc_stride_l, enc_stride_k = enc.stride()
    hid_stride_n, hid_stride_l, hid_stride_k = hid.stride()
    out_stride_n, out_stride_l, out_stride_k = A_cat.stride()

    # Launch kernel for encoder part: (N, L_txt) tiles along K
    BLOCK_K_concat = 128  # tile size along K
    # We'll use a 3D grid: (N, L_txt, ceil_div(K, BLOCK_K))
    grid_encoder = (N, L_txt, triton.cdiv(K, BLOCK_K_concat))
    _concat_sequences_kernel_actual[grid_encoder](
        enc, hid, A_cat,
        N, L_txt, L_img, K,
        enc_stride_n, enc_stride_l, enc_stride_k,
        hid_stride_n, hid_stride_l, hid_stride_k,
        out_stride_n, out_stride_l, out_stride_k,
        BLOCK_K=BLOCK_K_concat,
        num_warps=1, num_stages=2,
    )

    # Launch kernel for hidden part: (N, L_img, ceil_div(K, BLOCK_K))
    grid_hidden = (N, L_img, triton.cdiv(K, BLOCK_K_concat))
    _concat_sequences_kernel_actual[grid_hidden](
        enc, hid, A_cat,
        N, L_txt, L_img, K,
        enc_stride_n, enc_stride_l, enc_stride_k,
        hid_stride_n, hid_stride_l, hid_stride_k,
        out_stride_n, out_stride_l, out_stride_k,
        BLOCK_K=BLOCK_K_concat,
        num_warps=1, num_stages=2,
    )

    # Now A_cat should be filled. Note: we double-launched with overlapping logic is unnecessary; but Triton does not support selecting source via grid dimensions without combining. A cleaner approach is to implement two kernels with separate program_id(2) values, but Triton grid size must be known. So we keep two launches with careful index mapping. To avoid confusion, we can instead implement a single kernel that selects which via a combined launch. However, Triton grid size is fixed; a simple and robust way is to call two kernels with masks (but masks cannot change which source). Therefore, we perform a single kernel call that we can't condition on which. To ensure correctness, we will instead write a single kernel that takes the source tensor and index range and write into A_cat at appropriate positions. We’ll replace the previous two launches with a single kernel that computes t and i ranges.

    # Re-implement a single kernel that fills A_cat correctly:
    # We need a single kernel that writes into A_cat based on which part (encoder or hidden) and the index. Triton doesn’t support a separate program_id(2) grid dimension for selecting source without pre-splitting. Hence, we revert to two launches using the same kernel with different program_id(2) values.

    # Launch encoder part with which=0
    grid0 = (N, L_txt, triton.cdiv(K, BLOCK_K_concat))
    _concat_sequences_kernel_actual[grid0](
        enc, hid, A_cat,
        N, L_txt, L_img, K,
        enc_stride_n, enc_stride_l, enc_stride_k,
        hid_stride_n, hid_stride_l, hid_stride_k,
        out_stride_n, out_stride_l, out_stride_k,
        BLOCK_K=BLOCK_K_concat,
        num_warps=1, num_stages=2,
    )
    # Launch hidden part with which=1
    grid1 = (N, L_img, triton.cdiv(K, BLOCK_K_concat))
    _concat_sequences_kernel_actual[grid1](
        enc, hid, A_cat,
        N, L_txt, L_img, K,
        enc_stride_n, enc_stride_l, enc_stride_k,
        hid_stride_n, hid_stride_l, hid_stride_k,
        out_stride_n, out_stride_l, out_stride_k,
        BLOCK_K=BLOCK_K_concat,
        num_warps=1, num_stages=2,
    )

    # Now A_cat contains concatenation. Next, perform GEMM in Triton.
    # We need A_rows = A_cat flattened as [N_rows, K], where N_rows = N * L_total.
    # Create A_rows from A_cat by view without actual data copy; but Triton kernels read pointers, so we pass A_cat pointer and let kernel compute addresses. Our matmul kernel expects A_ptr of shape [N_rows, K]. We'll allocate a temporary C_rows and compute C_rows = A_cat @ process_weight_T.

    # Allocate C_rows [N_rows, K]
    N_rows = N * L_total
    C_rows = torch.empty((N_rows, K), device=hidden_states.device, dtype=torch.float32)

    # Strides for A_rows: A is [N_rows, K], contiguous
    A_stride_m = K
    A_stride_k = 1

    # B is [K, K]
    B_stride_k = K
    B_stride_n = 1

    # C is [N_rows, K]
    C_stride_m = K
    C_stride_n = 1

    # Launch tiled GEMM kernel. We use a 2D grid over tiles of rows and columns.
    # Choose BLOCK sizes. For robustness, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32.
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid_m = triton.cdiv(N_rows, BLOCK_M)  # tiles along rows
    grid_n = triton.cdiv(K, BLOCK_N)       # tiles along columns (here N is K)

    _matmul_batched_rows_tiled_kernel[(grid_m, grid_n)](
        A_cat, process_weight_T, C_rows,
        N_rows, K,
        A_stride_m, A_stride_k,
        B_stride_k, B_stride_n,
        C_stride_m, C_stride_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # Reshape C_rows back to [N, L_total, K]
    processed = C_rows.view(N, L_total, K)

    return processed


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenation of encoder_hidden_states and hidden_states done in Triton.
        - Linear projection (matmul with process_weight.T) done in Triton.
        - Returns (processed_encoder_hidden_states, processed_hidden_states) split along sequence dimension.
        """
        # Ensure CUDA and float32 for Triton
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            raise RuntimeError("ModelNew.forward requires CUDA tensors for Triton kernels.")

        # Make inputs contiguous and float32 (if not already)
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        # Transpose process_weight to [K, K] for matmul: B = process_weight.T
        process_weight_T = process_weight.t().contiguous()

        # Perform Triton computation
        processed = triton_matmul_concat(hidden_states, encoder_hidden_states, process_weight_T)

        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        L_total = L_txt + L_img

        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden

# The original run function can be used to verify correctness; the new ModelNew uses Triton kernels for both concatenation and matmul.
# If you want to quickly check, compare outputs:
# model = ModelNew().cuda()
# hidden_states = torch.randn(1, 256, 1024, device='cuda', dtype=torch.float32)
# encoder_hidden_states = torch.randn(1, 128, 1024, device='cuda', dtype=torch.float32)
# process_weight = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
# out_triton = model(hidden_states, encoder_hidden_states, process_weight)
# out_ref = run(hidden_states, encoder_hidden_states, process_weight)
# print(torch.allclose(out_triton[0], out_ref[0], atol=1e-6))
# print(torch.allclose(out_triton[1], out_ref[1], atol=1e-6))