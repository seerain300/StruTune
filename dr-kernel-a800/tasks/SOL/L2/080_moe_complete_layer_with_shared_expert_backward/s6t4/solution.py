import torch
import triton
import triton.language as tl


@triton.jit
def matmul_rowwise_bf16(
    X_ptr,           # *const bfloat16, shape [M, K]
    W_ptr,           # *const bfloat16, shape [K, N]  (W is [K, N])
    Y_ptr,           # *float32, shape [M, N]
    M,               # int
    K,               # int
    N,               # int
    stride_xm,       # int: X stride along M
    stride_xk,       # int: X stride along K
    stride_wk,       # int: W stride along K
    stride_wn,       # int: W stride along N
    stride_ym,       # int: Y stride along M
    stride_yn,       # int: Y stride along N
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row
    m = tl.program_id(0)
    # Accumulator for this row (float32)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension in tiles
    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        # Load this row slice of X for row m (bfloat16 -> float32)
        x_row = tl.load(
            X_ptr + m * stride_xm + k_offsets * stride_xk,
            mask=k_offsets < K,
            other=0.0,
        ).to(tl.float32)

        # For each tile of N, accumulate contributions
        for nn in range(0, N, BLOCK_N):
            n_offsets = nn + tl.arange(0, BLOCK_N)
            # Load a tile of W: shape [BLOCK_K, BLOCK_N]
            w_tile = tl.load(
                W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn,
                mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
                other=0.0,
            ).to(tl.float32)

            # Partial dot: [BLOCK_K] * [BLOCK_K, BLOCK_N] -> [BLOCK_N]
            acc[n_offsets] += tl.sum(x_row[:, None] * w_tile, axis=0)

    # Store the full row acc to output Y[m, :]
    out_ptr = Y_ptr + m * stride_ym
    tl.store(
        out_ptr + tl.arange(0, BLOCK_N) * stride_yn,
        acc,
        mask=tl.arange(0, BLOCK_N) < N,
    )


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,     # *const float32, shape [M, N]
    UpOut_ptr,       # *const float32, shape [M, N]
    Y_ptr,           # *float32, shape [M, N]
    M,               # int
    N,               # int
    stride_gm,       # int
    stride_gn,       # int
    stride_um,       # int
    stride_un,       # int
    stride_ym,       # int
    stride_yn,       # int
):
    # 1D grid over M*N elements
    idx = tl.program_id(0)
    m = idx // N
    n = idx % N
    if m < M and n < N:
        gate = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
        up = tl.load(UpOut_ptr + m * stride_um + n * stride_un)
        # SiLU(x) = x * sigmoid(x)
        sig = 1.0 / (1.0 + tl.exp(-gate))
        y = gate * sig * up
        tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def __init__(self, BLOCK_K: int = 256, BLOCK_N: int = 128):
        super().__init__()
        self.BLOCK_K = BLOCK_K
        self.BLOCK_N = BLOCK_N

    def forward(self, *args):
        # Forward returns: SiLU( F.linear(hidden, gate_weight) ) * F.linear(hidden, up_weight)
        hidden_states = args[0]   # [M, K]
        gate_weight = args[1]     # [K, N_gate]
        up_weight = args[2]       # [K, N_up]

        device = hidden_states.device
        assert device.type == "cuda", "ModelNew.forward requires CUDA tensors"
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N_gate = gate_weight.shape[1]
        N_up = up_weight.shape[1]

        # Outputs (float32 for stability)
        gate_out = torch.empty((M, N_gate), dtype=torch.float32, device=device)
        up_out = torch.empty((M, N_up), dtype=torch.float32, device=device)

        # Launch GEMV: one program per row
        grid = (M,)
        matmul_rowwise_bf16[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N_gate,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=self.BLOCK_K, BLOCK_N=self.BLOCK_N,
        )

        matmul_rowwise_bf16[grid](
            hidden_states, up_weight, up_out,
            M, K, N_up,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=self.BLOCK_K, BLOCK_N=self.BLOCK_N,
        )

        # Elementwise activation and multiply in Triton
        if N_gate != N_up:
            # Fallback for safety if shapes mismatch (shouldn't happen with provided inputs)
            activated = torch.nn.functional.silu(gate_out) * up_out
            return activated.to(torch.bfloat16)

        Y = torch.empty((M, N_gate), dtype=torch.float32, device=device)
        silu_mul_kernel[(M * N_gate,)](
            gate_out, up_out, Y,
            M, N_gate,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            Y.stride(0), Y.stride(1),
        )

        # Cast to bfloat16 to match original dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
