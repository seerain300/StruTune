import torch
import triton
import triton.language as tl

@triton.jit
def batched_gemm_two_outputs(
    E0, E1,  # inputs: [B, T, D], [B, I, D]
    W,       # weight: [D, D]
    Y0, Y1,  # outputs: [B, T, D], [B, I, D]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, D: tl.constexpr,
    E0_b_stride, E0_t_stride, E0_d_stride,
    E1_b_stride, E1_i_stride, E1_d_stride,
    W0_stride, W1_stride,
    Y0_b_stride, Y0_t_stride, Y0_d_stride,
    Y1_b_stride, Y1_i_stride, Y1_d_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    # Grid:
    # axis=0: tiles along sequence length (T or I)
    # axis=1: batch index
    pid_m = tl.program_id(axis=0)  # tile id along T/I
    b = tl.program_id(axis=1)      # batch index

    # Output row indices for this tile
    # For E0 (encoder): rows 0..T-1
    # For E1 (image): rows 0..I-1
    # We compute offsets relative to the start of each stream.
    # Since pid_m is reused across E0 and E1, we choose which output based on axis=1.
    # We will write to Y0 for encoder rows and Y1 for image rows by masking.
    # Compute row start index for this tile
    # Note: we can't branch by output here; instead, we let the caller launch two grids
    #       with appropriate strides and outputs. We'll implement it by detecting b==0 for Y0, b==1 for Y1.
    # However, Triton kernels are launched with fixed grid, so we assume caller invokes once for each output.
    # To support two outputs in one kernel, we re-launch the kernel twice: once for Y0 and once for Y1.
    # Here we only compute one output per launch, so pid_m and b are valid for that output.

    # Compute M (rows) per output: we'll specialize the kernel for each output by relaunching.
    # Placeholder logic not executed; this kernel is intended to be relaunched twice.
    # For clarity, we keep it as a simple GEMM for one output (encoder or image).

    # We need to know which output to compute. Triton doesn't allow dynamic output selection here,
    # so we define two separate launches in the forward function below.
    # This kernel will be called once for Y0 (encoder) and once for Y1 (image), with corresponding strides.
    # For now, implement the core GEMM logic for one output; the forward function will pass correct shapes.

    # The following lines are a template for the actual GEMM. We will omit them in the final launch.
    # We'll restructure forward to launch this kernel twice with appropriate parameters.

# Note: The above placeholder is to keep the code structure; in practice, we will define two separate kernels or
#       two launches of this kernel, relabeling outputs. Below, we define two kernels that mirror the GEMM logic
#       specialized for Y0 and Y1 respectively.

@triton.jit
def matmul_encoder_stream(
    E0, W, Y0,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    E0_b_stride, E0_t_stride, E0_d_stride,
    W0_stride, W1_stride,
    Y0_b_stride, Y0_t_stride, Y0_d_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    # Grid: axis=0 over tiles along T, axis=1 over batch
    pid_m = tl.program_id(axis=0)
    b = tl.program_id(axis=1)

    # Output rows for this tile: rows = pid_m * BLOCK_M + [0..BLOCK_M-1]
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    # We will handle rows < T via mask

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (hidden_dim) in tiles
    for k0 in range(0, D, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load E0[b, rows, k] as vector for each row (shape [BLOCK_M])
        # E0[b, t, d] with strides
        # Compute pointers: E0 + b*E0_b_stride + t*E0_t_stride + d*E0_d_stride
        E_ptrs = E0 + b * E0_b_stride + rows[:, None] * E0_t_stride + k_idx[None, :] * E0_d_stride
        mask_E = (rows[:, None] < T) & (k_idx[None, :] < D)
        E = tl.load(E_ptrs, mask=mask_E, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load W[k, :] as matrix for the current K tile across all N (hidden_dim) columns
        # W is [D, D], so load W[k, n] where n in 0..D-1
        W_ptrs = W + k_idx[:, None] * W0_stride + tl.arange(0, BLOCK_N)[None, :] * W1_stride
        mask_W = (k_idx[:, None] < D) & (tl.arange(0, BLOCK_N)[None, :] < D)
        W_sub = tl.load(W_ptrs, mask=mask_W, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += E @ W_sub  -> E: [BM, BK], W_sub: [BK, BN]
        acc += tl.dot(E, W_sub)

    # Store results to Y0[b, rows, :]
    Y_ptrs = Y0 + b * Y0_b_stride + rows[:, None] * Y0_t_stride + tl.arange(0, BLOCK_N)[None, :] * Y0_d_stride
    mask_Y = (rows[:, None] < T) & (tl.arange(0, BLOCK_N)[None, :] < D)
    tl.store(Y_ptrs, acc, mask=mask_Y)

@triton.jit
def matmul_image_stream(
    E1, W, Y1,
    B: tl.constexpr, I: tl.constexpr, D: tl.constexpr,
    E1_b_stride, E1_i_stride, E1_d_stride,
    W0_stride, W1_stride,
    Y1_b_stride, Y1_i_stride, Y1_d_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    # Grid: axis=0 over tiles along I, axis=1 over batch
    pid_m = tl.program_id(axis=0)
    b = tl.program_id(axis=1)

    # Output rows for this tile: rows = pid_m * BLOCK_M + [0..BLOCK_M-1]
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    # We will handle rows < I via mask

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, D, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load E1[b, rows, k] as vector
        E_ptrs = E1 + b * E1_b_stride + rows[:, None] * E1_i_stride + k_idx[None, :] * E1_d_stride
        mask_E = (rows[:, None] < I) & (k_idx[None, :] < D)
        E = tl.load(E_ptrs, mask=mask_E, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load W[k, :] for current K tile
        W_ptrs = W + k_idx[:, None] * W0_stride + tl.arange(0, BLOCK_N)[None, :] * W1_stride
        mask_W = (k_idx[:, None] < D) & (tl.arange(0, BLOCK_N)[None, :] < D)
        W_sub = tl.load(W_ptrs, mask=mask_W, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(E, W_sub)

    # Store results to Y1[b, rows, :]
    Y_ptrs = Y1 + b * Y1_b_stride + rows[:, None] * Y1_i_stride + tl.arange(0, BLOCK_N)[None, :] * Y1_d_stride
    mask_Y = (rows[:, None] < I) & (tl.arange(0, BLOCK_N)[None, :] < D)
    tl.store(Y_ptrs, acc, mask=mask_Y)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Avoid concatenation; compute two batched matmuls in Triton:
          processed_encoder = encoder_hidden_states @ process_weight.T
          processed_hidden   = hidden_states @ process_weight.T
        - All computation happens inside Triton kernels (no torch.matmul).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D, "hidden_dim mismatch"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # Make inputs contiguous for predictable strides
        E0 = encoder_hidden_states.contiguous()
        E1 = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Output tensors
        Y0 = torch.empty((B, T, D), device=hidden_states.device, dtype=torch.float32)
        Y1 = torch.empty((B, I, D), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernels: one for encoder stream, one for image stream
        # Tile sizes: choose BLOCK_M=64, BLOCK_N=64, BLOCK_K=32. Masks handle boundaries.
        grid_encoder = (triton.cdiv(T, 64), B)
        matmul_encoder_stream[grid_encoder](
            E0, W, Y0,
            B=B, T=T, D=D,
            E0_b_stride=E0.stride(0), E0_t_stride=E0.stride(1), E0_d_stride=E0.stride(2),
            W0_stride=W.stride(0), W1_stride=W.stride(1),
            Y0_b_stride=Y0.stride(0), Y0_t_stride=Y0.stride(1), Y0_d_stride=Y0.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            NUM_WARPS=4, NUM_STAGES=2,
        )

        grid_image = (triton.cdiv(I, 64), B)
        matmul_image_stream[grid_image](
            E1, W, Y1,
            B=B, I=I, D=D,
            E1_b_stride=E1.stride(0), E1_i_stride=E1.stride(1), E1_d_stride=E1.stride(2),
            W0_stride=W.stride(0), W1_stride=W.stride(1),
            Y1_b_stride=Y1.stride(0), Y1_i_stride=Y1.stride(1), Y1_d_stride=Y1.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            NUM_WARPS=4, NUM_STAGES=2,
        )

        return Y0, Y1