import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,          # *bfloat16, shape [M, K]
    W_ptr,          # *bfloat16, shape [K, N]
    Y_ptr,          # *float32,  shape [M, N]
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wk: tl.constexpr,  # stride along K for W
    stride_wn: tl.constexpr,  # stride along N for W
    stride_ym: tl.constexpr,  # stride along M for Y
    stride_yn: tl.constexpr,  # stride along N for Y
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    # Accumulator for output row m
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in tiles of BLOCK_K (K is 4096 in our setup)
    for k in range(0, K, BLOCK_K):
        # Load X[m, k:k+BLOCK_K] as bfloat16
        x_idx = k + tl.arange(0, BLOCK_K)
        x_ptrs = X_ptr + m * stride_xm + x_idx * stride_xk
        x_mask = x_idx < K
        x_vec = tl.load(x_ptrs, mask=x_mask, other=0.0)  # bfloat16 vector
        x_vec_f32 = x_vec.to(tl.float32)

        # Load W[k:k+BLOCK_K, :] as bfloat16; this is a vector of length BLOCK_N
        n_idx = tl.arange(0, BLOCK_N)
        w_ptrs = W_ptr + x_idx[:, None] * stride_wk + n_idx[None, :] * stride_wn
        k_mask = x_mask[:, None]
        w_mask = n_idx[None, :] < N
        w_tile = tl.load(w_ptrs, mask=k_mask & w_mask, other=0.0)  # [BLOCK_K, BLOCK_N] bfloat16
        w_tile_f32 = w_tile.to(tl.float32)

        # Accumulate: acc += x_vec * W_tile_row
        # x_vec_f32: [BLOCK_K], W_tile_f32: [BLOCK_K, BLOCK_N]
        # Multiply x_vec by each row of W_tile and accumulate along K
        for kk in range(BLOCK_K):
            # scalar x from this k position
            x_scalar = x_vec_f32[kk]
            # corresponding row of W_tile across N
            w_row = w_tile_f32[kk, :]  # [BLOCK_N]
            acc += x_scalar * w_row

    # Store acc into Y[m, :]
    y_ptrs = Y_ptr + m * stride_ym + tl.arange(0, BLOCK_N) * stride_yn
    tl.store(y_ptrs, acc, mask=tl.arange(0, BLOCK_N) < N)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,    # *float32, shape [M, N]
    UpOut_ptr,      # *float32, shape [M, N]
    Out_ptr,        # *float32, shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_um: tl.constexpr,
    stride_un: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    g = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
    u = tl.load(UpOut_ptr + m * stride_um + n * stride_un)
    # sigmoid(g) = 1 / (1 + exp(-g))
    sig = 1.0 / (1.0 + tl.exp(-g))
    y = g * sig * u
    tl.store(Out_ptr + m * stride_om + n * stride_on, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We expect: hidden_states, gate_weight, up_weight in args (others are ignored).
        hidden_states = args[0]
        gate_weight = args[1]
        up_weight = args[2]

        # Ensure CUDA and contiguity
        if hidden_states.device.type != 'cuda':
            hidden_states = hidden_states.cuda()
        if gate_weight.device.type != 'cuda':
            gate_weight = gate_weight.cuda()
        if up_weight.device.type != 'cuda':
            up_weight = up_weight.cuda()

        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        # Dimensions
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = gate_weight.shape[1]  # gate_weight is [K, N]

        # Cast hidden states to bfloat16 for kernels; gate_weight and up_weight are [K, N]
        # and will be loaded as bfloat16 in kernels (assuming inputs are bfloat16).
        # If they are not, converting here ensures consistent type. The original get_inputs uses bfloat16.
        hidden_states_bf16 = hidden_states.to(torch.bfloat16)

        # Allocate outputs (float32 for accumulation)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch GEMV kernels: one program per row
        grid = (M,)
        linear_rowwise_bf16_to_f32[grid](
            hidden_states_bf16, gate_weight, gate_out,
            M, K, N,
            hidden_states_bf16.stride(0), hidden_states_bf16.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=4096, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        linear_rowwise_bf16_to_f32[grid](
            hidden_states_bf16, up_weight, up_out,
            M, K, N,
            hidden_states_bf16.stride(0), hidden_states_bf16.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=4096, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # Elementwise y = GateOut * sigmoid(GateOut) * UpOut in float32
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

        # Return in bfloat16 (cast via dtype constructor, not elementwise op)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
