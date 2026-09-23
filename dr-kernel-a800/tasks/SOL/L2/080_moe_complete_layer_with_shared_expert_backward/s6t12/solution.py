import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,   # *bfloat16, shape [M, K]
    W_ptr,   # *bfloat16, shape [K, N]
    Y_ptr,   # *float32,  shape [M, N]
    M,       # int
    K,       # int
    N,       # int
    stride_xm,  # int
    stride_xk,  # int
    stride_wk,  # int (row stride for W)
    stride_wn,  # int (col stride for W)
    stride_ym,  # int (row stride for Y)
    stride_yn,  # int (col stride for Y)
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row m
    m = tl.program_id(0)
    if m >= M:
        return

    # Accumulator for output row m
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in tiles of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Vector of column offsets for this tile
        n_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load X[m, n_offsets] as bfloat16, masked
        x_ptrs = X_ptr + m * stride_xm + n_offsets * stride_xk
        x_vals = tl.load(x_ptrs, mask=n_offsets < K, other=0.0)  # shape [BLOCK_K], bfloat16

        # Load corresponding W[n_offsets, 0:BLOCK_N] as bfloat16, masked, shape [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + n_offsets[:, None] * stride_wk + tl.arange(0, BLOCK_N)[None, :] * stride_wn
        w_vals = tl.load(w_ptrs, mask=(n_offsets[:, None] < K) & (tl.arange(0, BLOCK_N)[None, :] < N), other=0.0)

        # Cast to float32 for accumulation
        x_vals_f32 = x_vals.to(tl.float32)
        w_vals_f32 = w_vals.to(tl.float32)

        # Accumulate: dot product of x_vals with each column of W
        # We iterate over k within the tile to keep it simple and avoid tl.dot complexities
        for kk in range(BLOCK_K):
            # If k0 + kk >= K, mask x_vals_f32[kk] is zeroed above
            acc += x_vals_f32[kk] * w_vals_f32[kk, :]

    # Store acc into Y[m, :] with mask for n < N
    y_ptrs = Y_ptr + m * stride_ym + tl.arange(0, BLOCK_N) * stride_yn
    tl.store(y_ptrs, acc, mask=tl.arange(0, BLOCK_N) < N)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,  # *float32, shape [M, N]
    UpOut_ptr,    # *float32, shape [M, N]
    Out_ptr,      # *float32, shape [M, N]
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    g = tl.load(GateOut_ptr + pid_m * stride_gm + pid_n * stride_gn)
    u = tl.load(UpOut_ptr + pid_m * stride_um + pid_n * stride_un)
    s = tl.sigmoid(g)
    y = g * s * u
    tl.store(Out_ptr + pid_m * stride_om + pid_n * stride_on, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: hidden_states, shared_expert_gate_weight, shared_expert_up_weight
        # We ignore others and compute shared_activated = SiLU(gate) * up
        hidden_states, gate_weight, up_weight = args
        # Ensure contiguous and CUDA
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        # Extract shapes
        M = hidden_states.shape[0]  # batch_seq_len
        K = hidden_states.shape[1]  # hidden_size = 4096
        N = gate_weight.shape[1]    # moe_intermediate_size = 1408

        # Allocate outputs (float32 for accumulation)
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch kernels: one program per row
        grid = (M,)
        # Choose BLOCK sizes (multiples of 32, reasonable for K=4096, N=1408)
        BLOCK_K = 256
        BLOCK_N = 128

        # Compute gate_out = hidden_states @ gate_weight.T
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Compute up_out = hidden_states @ up_weight.T
        linear_rowwise_bf16_to_f32[grid](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Compute shared_activated = gate_out * sigmoid(gate_out) * up_out
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

        # Return bfloat16 (cast is allowed as dtype conversion, not torch op on tensor elements)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
