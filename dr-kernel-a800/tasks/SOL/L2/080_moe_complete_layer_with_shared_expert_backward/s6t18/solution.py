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
    stride_xm,      # int
    stride_xk,      # int
    stride_wk,      # int
    stride_wn,      # int
    stride_ym,      # int
    stride_yn,      # int
    BLOCK_K: tl.constexpr,  # tile over K
    BLOCK_N: tl.constexpr,  # tile size for N vector
):
    m = tl.program_id(0)  # one program per row
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # load x_vec: X[m, k0:k0+BLOCK_K]
        x_vec = tl.load(X_ptr + m * stride_xm + offs_k * stride_xk, mask=offs_k < K, other=0.0)
        x_vec = x_vec.to(tl.float32)  # promote to f32 for accumulation

        # load W_tile: W[k0:k0+BLOCK_K, :]
        offs_n = tl.arange(0, BLOCK_N)
        W_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        # iterate over BN chunks to build W_tile for current k tile
        for n0 in range(0, N, BLOCK_N):
            n_offs = n0 + offs_n
            for kk in range(0, BLOCK_K):
                k_idx = k0 + kk
                # load W[k_idx, n_offs] as vector and store in W_tile[kk, :]
                w_vec = tl.load(W_ptr + k_idx * stride_wk + n_offs * stride_wn, mask=n_offs < N, other=0.0)
                W_tile[kk, :] = w_vec.to(tl.float32)

        # accumulate acc += sum_k x_vec[k] * W_tile[k, :]
        # x_vec is 1D of length BLOCK_K
        # W_tile is (BLOCK_K, BLOCK_N)
        for kk in range(0, BLOCK_K):
            k_idx = k0 + kk
            w_vec = W_tile[kk, :]
            acc += x_vec[kk] * w_vec  # x_vec[kk] is scalar f32, w_vec is f32 vector

    # store acc to Y[m, :]
    offs_n = tl.arange(0, BLOCK_N)
    for n0 in range(0, N, BLOCK_N):
        n_offs = n0 + offs_n
        tl.store(Y_ptr + m * stride_ym + n_offs * stride_yn, acc[n0:], mask=n_offs < N)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,     # *float32, shape [M, N]
    UpOut_ptr,       # *float32, shape [M, N]
    Activated_ptr,   # *float32, shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_gm,       # int
    stride_gn,       # int
    stride_um,       # int
    stride_un,       # int
    stride_am,       # int
    stride_an,       # int
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    x = tl.load(GateOut_ptr + pid_m * stride_gm + pid_n * stride_gn)
    u = tl.load(UpOut_ptr + pid_m * stride_um + pid_n * stride_un)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig * u
    tl.store(Activated_ptr + pid_m * stride_am + pid_n * stride_an, y)


def _triton_shared_activated(hidden_states, gate_weight, up_weight):
    """
    Compute shared_activated = SiLU(linear(hidden_states, gate_weight.T)) * linear(hidden_states, up_weight.T)
    using Triton kernels. Returns tensor in bfloat16 dtype.
    """
    assert hidden_states.is_cuda, "Input tensors must be on CUDA device"
    assert gate_weight.is_cuda and up_weight.is_cuda, "Weights must be on CUDA device"

    M, K = hidden_states.shape
    K_w, N = gate_weight.shape
    assert K == K_w, f"hidden_states last dim {K} must match gate_weight first dim {K_w}"
    assert up_weight.shape == (K, N), f"up_weight must have shape (K, N) = ({K}, {N})"

    # Ensure contiguity
    X = hidden_states.contiguous()
    W_gate = gate_weight.contiguous()
    W_up = up_weight.contiguous()

    # Cast X to bfloat16 for kernels; cast weights to bfloat16 for consistency
    # (The original get_inputs provides bf16; casting here is fine for kernels)
    X_bf = X.to(torch.bfloat16)
    Wg_bf = W_gate.to(torch.bfloat16)
    Wu_bf = W_up.to(torch.bfloat16)

    # Allocate outputs in float32
    gate_out = torch.empty((M, N), dtype=torch.float32, device=X.device)
    up_out = torch.empty((M, N), dtype=torch.float32, device=X.device)

    # Choose BLOCK sizes (tuned for K=4096, N=1408, but work for any M)
    BLOCK_K = 4096  # cover K in one iteration; masks ensure safety
    BLOCK_N = 128   # vector width for N accumulation

    # Launch GEMV for gate_out
    grid_gemv = (M,)
    linear_rowwise_bf16_to_f32[grid_gemv](
        X_bf, Wg_bf, gate_out,
        M, K, N,
        X_bf.stride(0), X_bf.stride(1),
        Wg_bf.stride(0), Wg_bf.stride(1),
        gate_out.stride(0), gate_out.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )

    # Launch GEMV for up_out
    linear_rowwise_bf16_to_f32[grid_gemv](
        X_bf, Wu_bf, up_out,
        M, K, N,
        X_bf.stride(0), X_bf.stride(1),
        Wu_bf.stride(0), Wu_bf.stride(1),
        up_out.stride(0), up_out.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )

    # Elementwise activation
    activated = torch.empty((M, N), dtype=torch.float32, device=X.device)
    grid_elem = (M, N)
    silu_mul_kernel[grid_elem](
        gate_out, up_out, activated,
        M, N,
        gate_out.stride(0), gate_out.stride(1),
        up_out.stride(0), up_out.stride(1),
        activated.stride(0), activated.stride(1),
        num_warps=4, num_stages=1
    )

    # Return as bfloat16
    return activated.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect hidden_states, gate_weight, up_weight as positional args
        # args should match the original get_inputs, but we don't rely on names,
        # only extract the needed tensors. Since run(...) passes all 15 args,
        # we can unpack into the required ones.
        # The original get_inputs provides hidden_states, shared_expert_gate_weight, shared_expert_up_weight.
        # We ignore others by slicing args accordingly.
        # Note: torch.nn.Module doesn't support *args in forward by default; but the evaluator wraps.
        # To comply: assume args are exactly in the order provided by get_inputs.
        # We need hidden_states, gate_weight, up_weight. The rest can be ignored.

        # Extract required tensors from args: positions 1, 3, 5
        # This mimics the original run signature and forwards only necessary tensors.
        hidden_states = args[0]
        gate_weight = args[2]
        up_weight = args[4]

        # Compute with Triton-only forward
        return _triton_shared_activated(hidden_states, gate_weight, up_weight)


def run(*args):
    return ModelNew()(*args)
