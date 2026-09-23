import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_tile_kernel(
    X_ptr,        # *ptr to concatenated input: [B, P, D]
    WT_ptr,       # *ptr to process_weight.T: [D, D]
    Y_ptr,        # *ptr to output processed: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile size along M = P
    BLOCK_N: tl.constexpr,  # tile size along N = D
    BLOCK_K: tl.constexpr,  # reduction tile size along K = D
):
    # Grid: (B, ceil(P / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < P
    mask_n = n_offsets < D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load X tile: [BLOCK_M, BLOCK_K], X[b, m, k]
        x_ptrs = X_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load WT tile: [BLOCK_K, BLOCK_N], WT[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(x_tile.to(tl.float32), wt_tile.to(tl.float32))

    # Store result Y[b, m, n] = acc
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_encoder(
    y_ptr,        # *ptr to processed Y: [B, P, D]
    enc_out_ptr,  # *ptr to output encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile along T
    BLOCK_N: tl.constexpr,  # tile along D
):
    # Grid: (B, ceil(T / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < T
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(y_ptr + b * P * D + m * D + n, mask=True, other=0.0)
                    tl.store(enc_out_ptr + b * T * D + m * D + n, val, mask=True)


@triton.jit
def _copy_slice_to_hidden(
    y_ptr,        # *ptr to processed Y: [B, P, D]
    hid_out_ptr,  # *ptr to output hidden: [B, I, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile along I
    BLOCK_N: tl.constexpr,  # tile along D
):
    # Grid: (B, ceil(I / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < I
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        i = m_start + mi
        if i < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(y_ptr + b * P * D + (T + i) * D + n, mask=True, other=0.0)
                    tl.store(hid_out_ptr + b * I * D + i * D + n, val, mask=True)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tile sizes; chosen conservatively for robustness
        self.BLOCK_M = 64  # along P (sequence length after concat)
        self.BLOCK_N = 64  # along D (hidden feature)
        self.BLOCK_K = 32  # reduction chunk along K=D

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Compute Y = (concatenated) @ process_weight.T
        - Split Y into processed_encoder (first T rows) and processed_hidden (remaining I rows).
        Returns (processed_encoder, processed_hidden).
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3
        B, T, D = encoder_hidden_states.shape
        B2, I, D2 = hidden_states.shape
        assert B == B2 and D == D2, "Mismatched batch or hidden_dim"
        P = T + I

        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        wt_t = process_weight.t().contiguous()  # [D, D]

        # Allocate output for GEMM
        y = torch.empty((B, P, D), dtype=torch.float32, device=enc.device)

        # 1) GEMM: y = enc @ wt_t + hst @ wt_t (fused by concatenating inputs into X then matmul)
        # We don't need to explicitly construct X here; the Triton kernel loads from enc/hst as needed.
        # To do that, we will temporarily construct X via two kernels; but since forward must use Triton,
        # we'll build X using Triton copy kernels, then run the GEMM kernel over X.
        # However, constructing X via Triton adds extra overhead. Instead, we can simulate concatenation
        # by offsetting loads: the GEMM kernel reads from enc (for m < T) and from hst (for m >= T).
        # We'll construct X as an empty [B, P, D] and fill via Triton kernels. But that would double work.
        #
        # Simplification: since we must perform matmul via Triton and not torch.matmul, we'll directly
        # compute the matmul using the original enc and hst by leveraging the fact that our GEMM kernel
        # operates on a single tensor X which we construct via two copy kernels. For correctness and robustness,
        # we do that. Note: in practice, torch.cat is forbidden, so we build X via Triton.

        # Construct X = [B, P, D] with enc in [:T, :] and hst in [T:, :]
        X = torch.empty((B, P, D), dtype=torch.float32, device=enc.device)

        # Copy encoder into X[:T, :]
        grid_copy_enc = (B, triton.cdiv(T, self.BLOCK_M), triton.cdiv(D, self.BLOCK_N))
        _copy_slice_to_encoder[grid_copy_enc](
            X, X,  # dummy: we'll implement copy from enc in Triton below
            B, T, D,
            self.BLOCK_M, self.BLOCK_N
        )
        # Above placeholder; actual copy is done by a dedicated Triton kernel in code below.

        # Implement actual copies in Triton:
        # We need separate Triton kernels to fill X from enc and hst. To keep code compact, we define them inline.

        # Define Triton copy kernels here (inline) to avoid torch.cat usage:
        # Kernel 1: copy enc -> X[:T, :]
        @triton.jit
        def _fill_X_from_enc(X_ptr, enc_ptr, B: tl.constexpr, T: tl.constexpr, D: tl.constexpr):
            # We'll rely on slicing done by torch for X; Triton kernels can't directly slice. Instead,
            # we'll call a simple kernel that copies enc into X using provided base pointers.
            pass  # Not used; see inline kernel below

        # Inline Triton kernels (names defined at top level) are not accessible here. So we implement as functions.

        # Function to copy enc into X[:T, :] using Triton
        def _triton_copy_encoder_to_X(enc_ptr, X_ptr, B, T, D, BLOCK_M, BLOCK_N):
            grid = (B, triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_N))
            _copy_slice_to_encoder[grid](enc_ptr, X_ptr, B, T, D, BLOCK_M, BLOCK_N)

        # Function to copy hst into X[T:, :]
        def _triton_copy_img_to_X(hst_ptr, X_ptr, B, I, T, D, BLOCK_P, BLOCK_N):
            grid = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
            # We need a kernel that writes to X[b, T + i, n]
            @triton.jit
            def _copy_img_to_X(hst_ptr, X_ptr, B, I, T, D, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr):
                b = tl.program_id(0)
                pid_p = tl.program_id(1)
                pid_n = tl.program_id(2)

                p_start = pid_p * BLOCK_P  # within I
                n_start = pid_n * BLOCK_N

                p_offsets = p_start + tl.arange(0, BLOCK_P)
                n_offsets = n_start + tl.arange(0, BLOCK_N)

                mask_p = p_offsets < I
                mask_n = n_offsets < D

                for pi in range(BLOCK_P):
                    i = p_start + pi
                    if i < I:
                        for ni in range(BLOCK_N):
                            n = n_start + ni
                            if n < D:
                                val = tl.load(hst_ptr + b * I * D + i * D + n, mask=True, other=0.0)
                                tl.store(X_ptr + b * D * (T + I) + (T + i) * D + n, val, mask=True)

            _copy_img_to_X[grid](hst_ptr, X_ptr, B, I, T, D, self.BLOCK_P, self.BLOCK_N)

        # Build X using Triton copy functions
        # Note: Triton kernels require pointers; we'll launch them with tensors.
        _triton_copy_encoder_to_X(enc, X, B, T, D, self.BLOCK_M, self.BLOCK_N)
        _triton_copy_img_to_X(hst, X, B, I, T, D, self.BLOCK_P, self.BLOCK_N)

        # 2) GEMM: y = X @ wt_t
        grid_gemm = (B, triton.cdiv(P, self.BLOCK_M), triton.cdiv(D, self.BLOCK_N))
        _gemm_tile_kernel[grid_gemm](
            X, wt_t, y,
            B, P, D,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split outputs
        processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=enc.device)
        processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=enc.device)

        grid_slice_enc = (B, triton.cdiv(T, self.BLOCK_M), triton.cdiv(D, self.BLOCK_N))
        _copy_slice_to_encoder[grid_slice_enc](
            y, processed_encoder,
            B, T, D,
            self.BLOCK_M, self.BLOCK_N
        )

        grid_slice_hid = (B, triton.cdiv(I, self.BLOCK_M), triton.cdiv(D, self.BLOCK_N))
        _copy_slice_to_hidden[grid_slice_hid](
            y, processed_hidden,
            B, T, I, D,
            self.BLOCK_M, self.BLOCK_N
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
