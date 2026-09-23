import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,            # *bfloat16, shape [M, K]
    W_ptr,            # *bfloat16, shape [K, N]
    Y_ptr,            # *float32,  shape [M, N]
    M: tl.constexpr,  # int
    K: tl.constexpr,  # int
    N: tl.constexpr,  # int
    stride_xm,        # int
    stride_xk,        # int
    stride_wk,        # int
    stride_wn,        # int
    stride_ym,        # int
    stride_yn,        # int
    BLOCK_N: tl.constexpr,  # tile size over N
    BLOCK_K: tl.constexpr,  # tile size over K
):
    # Each program handles one row m
    m = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load x[m, k_offsets]
        x_vec = tl.load(
            X_ptr + m * stride_xm + k_offsets * stride_xk,
            mask=k_mask,
            other=0.0
        )  # bfloat16 vector
        x_vec_f32 = x_vec.to(tl.float32)

        # Load W[k_offsets, n_offsets] tile, shape [BLOCK_K, BLOCK_N]
        w_tile = tl.load(
            W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn,
            mask=k_mask[:, None] & (n_offsets[None, :] < N),
            other=0.0
        )  # bfloat16 tile
        w_tile_f32 = w_tile.to(tl.float32)

        # Accumulate: acc += sum over k of x_vec[k] * w_tile[k, :]
        acc += tl.sum(w_tile_f32 * x_vec_f32[:, None], axis=0)

    # Store acc into Y[m, :]
    y_row_ptr = Y_ptr + m * stride_ym + n_offsets * stride_yn
    store_mask = n_offsets < N
    tl.store(y_row_ptr, acc, mask=store_mask)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,  # *float32, shape [M, N]
    UpOut_ptr,    # *float32, shape [M, N]
    Y_ptr,        # *float32, shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_gm,
    stride_gn,
    stride_um,
    stride_un,
    stride_ym,
    stride_yn,
):
    # 2D grid: one program per (m, n)
    m = tl.program_id(0)
    n = tl.program_id(1)

    gate_val = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    up_val = tl.load(UpOut_ptr + m * stride_um + n * stride_un)

    # silu(x) = x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-gate_val))
    y = gate_val * sig * up_val

    tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Compute shared_activated = SiLU(hidden_states @ gate_weight.T) * (hidden_states @ up_weight.T)
        Args:
        - hidden_states: [M, K], bfloat16 (M = batch_seq_len, K = 4096)
        - shared_expert_gate_weight: [K, N], bfloat16 (K=4096, N=1408)
        - shared_expert_up_weight: [K, N], bfloat16 (K=4096, N=1408)
        Returns:
        - [M, N] tensor, bfloat16, matching original run's shared_activated
        """
        # Extract inputs (no torch ops on tensors)
        hidden_states = args[0]  # [M, K], bfloat16
        gate_weight = args[1]    # [K, N], bfloat16
        up_weight = args[2]      # [K, N], bfloat16

        # Ensure CUDA and contiguity
        assert hidden_states.is_cuda and gate_weight.is_cuda and up_weight.is_cuda, "Tensors must be on CUDA"
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        M, K = hidden_states.shape
        K_w, N = gate_weight.shape
        assert K == K_w, "hidden_states second dim must match gate_weight first dim"
        assert up_weight.shape == gate_weight.shape, "gate_weight and up_weight must have same shape"
        # Assert fixed constants to simplify and ensure robustness
        assert K == 4096, "hidden_size must be 4096"
        assert N == 1408, "moe_intermediate_size must be 1408"

        # Allocate outputs (float32 for accumulation)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch GEMV kernels: one program per row
        BLOCK_N = 128
        BLOCK_K = 128

        grid = (M,)

        # gate_out = hidden_states @ gate_weight.T
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # up_out = hidden_states @ up_weight.T
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Compute shared_activated = gate_out * SiLU(up_out)
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid2 = (M, N)
        silu_mul_kernel[grid2](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return bfloat16 tensor (cast via torch without torch ops on tensors)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
