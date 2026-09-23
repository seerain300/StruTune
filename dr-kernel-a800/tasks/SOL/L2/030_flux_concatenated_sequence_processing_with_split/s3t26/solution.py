import torch
import triton
import triton.language as tl


# 1) Build concatenated input without torch.cat:
#    - Copy encoder_hidden_states into out[:, :T, :]
#    - Copy hidden_states into out[:, T:, :]
# These are elementwise copy kernels; simple and robust.

@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to out: [B, P, D] (we only write first T rows)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,  # total sequence length (P = T + I) passed for clarity
    BLOCK_P: tl.constexpr,  # tile size along T
    BLOCK_N: tl.constexpr,  # tile size along D
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < T
    mask_n = n_offsets < D

    # For safety, guard batch
    if b >= 0:  # always true
        pass

    # Compute base offsets using strides: row-major => offset = b*stride_b + p*stride_p + n*stride_n
    # Here stride_b = T*D, stride_p = D, stride_n = 1
    # But since we pass base pointers per b, we can compute as:
    enc_base = enc_ptr + b * T * D
    out_base = out_ptr + b * P * D

    # Create 2D pointers for tile loads/stores
    # Load tile from encoder: enc_ptr + b*T*D + p*stride_p + n*stride_n
    # Store tile to out: out_ptr + b*P*D + p*stride_p + n*stride_n
    # Using mask for edge handling
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(enc_base + p * D + n)
                    tl.store(out_base + p * D + n, val)


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to out: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,  # total sequence length (P = T + I)
    BLOCK_P: tl.constexpr,  # tile size along I
    BLOCK_N: tl.constexpr,  # tile size along D
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P  # we map p_start to I (img part)
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    if b >= 0:
        pass

    hst_base = hst_ptr + b * I * D
    out_base = out_ptr + b * P * D

    # For each (p, n) in tile, copy from hst to out at position (p_start + p, :)
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_base + p * D + n)
                    # Store into out at index (T + p, n)
                    tl.store(out_base + (T + p) * D + n, val)


# 2) Triton GEMM: compute Y = X @ WT, where
#    - X: [B, P, D] built in previous step
#    - WT: [D, D] = process_weight.T (we pass process_weight.T)
#    - Y: [B, P, D] output

@triton.jit
def _gemm_bpn_xwd(
    X_ptr,        # *ptr to X: [B, P, D] (concatenated input)
    WT_ptr,       # *ptr to WT: [D, D] (process_weight.T)
    Y_ptr,        # *ptr to Y: [B, P, D] result
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile size along N (output features)
    BLOCK_K: tl.constexpr,  # reduction chunk along K
):
    # One program per (batch, tile over P, tile over N)
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_N  # we tile over P with size BLOCK_N (we set P == N == D here? Not exactly...)
    # Correction: We actually tile over P (rows) and N (cols) using the same BLOCK_N for both.
    # To cover P, we need a different parameter for P tile size. We'll set BLOCK_P explicitly in launch.
    # However, this kernel assumes a 3D grid with enough programs to cover P. To keep simple, we launch with grid (B, ceil(P/BLOCK_P), ceil(D/BLOCK_N)).
    # The below code assumes that pid_p and pid_n refer to tiles over P and N respectively. We fix grid in host.

    # We cannot declare BLOCK_P here; instead, we handle via launch grid. So we keep pid_p and pid_n as indices.
    # Load masks for tiles
    # Note: We don't have BLOCK_P at this scope, so we cannot compute p_offsets. We will launch with a grid that provides them.

    # Placeholder: The correct kernel below handles tiling over P and N explicitly with BLOCK_P and BLOCK_N.
    # To avoid confusion, replace this function with a well-tested matmul kernel.


# This placeholder indicates where the robust matmul kernel should be defined. Due to space limits and to keep the code concise,
# we provide a simple, safe matmul below that iterates over K in a loop and accumulates in fp32. It's slower but robust.

@triton.jit
def _gemm_loop_k(
    X_ptr,        # *ptr to X: [B, P, D]
    WT_ptr,       # *ptr to WT: [D, D] (process_weight.T)
    Y_ptr,        # *ptr to Y: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile size over P
    BLOCK_N: tl.constexpr,  # tile size over N
):
    # Grid: (B, ceil(P/BLOCK_P), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < P
    mask_n = n_offsets < D

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    # Loop over K = D (hidden dimension), load X tile [BP, 1], WT tile [1, BN], accumulate
    # Note: This is a simple K-loop. For performance, use BLOCK_K > 1, but correctness is paramount here.
    for k in range(0, D):
        # X tile: [BP, 1]
        x_ptrs = X_ptr + b * P * D + p_offsets[:, None] * D + k
        # WT tile: [1, BN]
        wt_ptrs = WT_ptr + k * D + n_offsets[None, :]

        # Mask for X: any p in tile
        x_vals = tl.load(x_ptrs, mask=mask_p[:, None], other=0.0)
        wt_vals = tl.load(wt_ptrs, mask=mask_n[None, :], other=0.0)

        # Cast to fp32 for accumulation
        x_vals = x_vals.to(tl.float32)
        wt_vals = wt_vals.to(tl.float32)

        acc += x_vals @ wt_vals  # [BP, 1] @ [1, BN] -> [BP, BN]

    # Store result
    y_ptrs = Y_ptr + b * P * D + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_p[:, None] & mask_n[None, :])


# 3) Triton copy kernels to produce the two outputs (splitting)
#    - Copy Y[:, :T, :] -> processed_encoder
#    - Copy Y[:, T:, :] -> processed_hidden

@triton.jit
def _copy_slice_to_encoder(
    Y_ptr,        # *ptr to Y: [B, P, D]
    out_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile over T
    BLOCK_N: tl.constexpr,  # tile over D
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < T
    mask_n = n_offsets < D

    y_base = Y_ptr + b * P * D
    out_base = out_ptr + b * T * D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(y_base + p * D + n)
                    tl.store(out_base + p * D + n, val)


@triton.jit
def _copy_slice_to_hidden(
    Y_ptr,        # *ptr to Y: [B, P, D]
    out_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile over I (since we copy from Y[:, T:, :], P = T + I)
    BLOCK_N: tl.constexpr,  # tile over D
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P  # this p indexes the img part [T, T+I]
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    y_base = Y_ptr + b * P * D
    out_base = out_ptr + b * I * D

    # For each (p, n) in tile, copy from Y at (T + p, n) to out at (p, n)
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(y_base + (T + p) * D + n)
                    tl.store(out_base + p * D + n, val)


# Host-side forward (ModelNew): Triton-only, no torch.cat/linear
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        B, T, D = encoder_hidden_states.shape
        B2, I, D2 = hidden_states.shape
        assert B == B2, "Batch size must match"
        assert D == D2, "Hidden dim must match"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # Build concatenated X: [B, P, D], P = T + I
        P = T + I
        X = torch.empty((B, P, D), dtype=torch.float32, device=hidden_states.device)

        # Launch concat kernels: copy encoder -> X[:, :T, :], copy hidden -> X[:, T:, :]
        BLOCK_P = 128  # tile over sequence
        BLOCK_N = 64   # tile over features

        grid_encoder = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_encoder_to_out[grid_encoder](
            encoder_hidden_states,
            X,
            B, T, D, P,
            BLOCK_P, BLOCK_N,
        )

        grid_img = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_img_to_out[grid_img](
            hidden_states,
            X,
            B, I, D, P,
            BLOCK_P, BLOCK_N,
        )

        # Prepare WT = process_weight.T
        WT = process_weight.t().contiguous().to(torch.float32)

        # Output Y: [B, P, D]
        Y = torch.empty((B, P, D), dtype=torch.float32, device=hidden_states.device)

        # GEMM: Y = X @ WT
        # Use simple robust K-loop matmul kernel (accumulates in fp32)
        grid_gemm = (B, triton.cdiv(P, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _gemm_loop_k[grid_gemm](
            X, WT, Y,
            B, P, D,
            BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
        )

        # Split into two outputs
        processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=hidden_states.device)

        grid_split_encoder = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_slice_to_encoder[grid_split_encoder](
            Y, processed_encoder,
            B, T, D,
            BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
        )

        grid_split_hidden = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_slice_to_hidden[grid_split_hidden](
            Y, processed_hidden,
            B, I, D,
            BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
        )

        return processed_encoder, processed_hidden


# The provided run helper can remain as-is; ModelNew is the entry point the evaluator expects.
def run(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ModelNew()(hidden_states, encoder_hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
