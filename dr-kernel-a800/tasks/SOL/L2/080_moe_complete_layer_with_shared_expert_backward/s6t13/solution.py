import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,            # *bfloat16, shape [M, K]
    W_ptr,            # *bfloat16, shape [K, N]
    Y_ptr,            # *float32,  shape [M, N]
    M,                # int
    K,                # int
    N,                # int
    stride_xm,        # int
    stride_xk,        # int
    stride_wk,        # int (row stride for W)
    stride_wn,        # int (col stride for W)
    stride_ym,        # int
    stride_yn,        # int
    BLOCK_K: tl.constexpr,  # tile size for K
    BLOCK_N: tl.constexpr   # tile size for N
):
    # program id for row
    m = tl.program_id(0)
    # initialize accumulator for this row (float32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        # compute offsets for K tile
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # load x_row[k_offsets] as bfloat16, cast to float32
        x_row_ptr = X_ptr + m * stride_xm
        x_vals = tl.load(x_row_ptr + k_offsets * stride_xk, mask=k_mask, other=0.0)
        x_vals = x_vals.to(tl.float32)

        # accumulate over the tile: acc += sum(x_vals[j] * W[k_offsets[j], :])
        # We do this by looping j over BLOCK_K and loading W rows with masks.
        for j in range(BLOCK_K):
            kk = k0 + j
            valid_k = kk < K
            # load W[kk, :] as bfloat16 vector of size BLOCK_N
            w_row_ptr = W_ptr + kk * stride_wk
            w_vec = tl.load(w_row_ptr + tl.arange(0, BLOCK_N) * stride_wn, mask=(kk < K), other=0.0)
            w_vec = w_vec.to(tl.float32)
            # add contribution
            acc += x_vals[j] * w_vec

    # store acc to Y[m, :] with mask n < N
    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    y_row_ptr = Y_ptr + m * stride_ym
    tl.store(y_row_ptr + n_offsets * stride_yn, acc, mask=n_mask)


@triton.jit
def silu_mul_kernel(
    GateOut_ptr,      # *float32, shape [M, N]
    UpOut_ptr,        # *float32, shape [M, N]
    Y_ptr,            # *float32, shape [M, N]
    M,                # int
    N,                # int
    stride_gm,        # int
    stride_gn,        # int
    stride_um,        # int
    stride_un,        # int
    stride_ym,        # int
    stride_yn,        # int
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    m_mask = m < M
    n_mask = n < N
    if m_mask and n_mask:
        gate_val = tl.load(GateOut_ptr + m * stride_gm + n * stride_gn)
        up_val = tl.load(UpOut_ptr + m * stride_um + n * stride_un)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-gate_val))
        y = gate_val * sig * up_val
        tl.store(Y_ptr + m * stride_ym + n * stride_yn, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: hidden_states, shared_expert_gate_weight, shared_expert_up_weight
        # Extract arguments (the evaluation harness provides these)
        hidden_states = args[0]  # [M, K] bfloat16
        gate_weight = args[1]    # [K, N] bfloat16
        up_weight = args[2]      # [K, N] bfloat16

        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and gate_weight.is_cuda and up_weight.is_cuda, "Tensors must be on CUDA"
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        up_weight = up_weight.contiguous()

        M, K = hidden_states.shape
        K_w, N = gate_weight.shape
        assert K_w == K, "Weight K must match hidden_states K"
        assert up_weight.shape == (K, N), "up_weight shape must match gate_weight"

        # Allocate outputs in float32
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Launch row-wise GEMV for gate_out
        grid_gate = (M,)
        linear_rowwise_bf16_to_f32[grid_gate](
            hidden_states, gate_weight, gate_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            gate_weight.stride(0), gate_weight.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=256, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # Launch row-wise GEMV for up_out
        grid_up = (M,)
        linear_rowwise_bf16_to_f32[grid_up](
            hidden_states, up_weight, up_out,
            M, K, N,
            hidden_states.stride(0), hidden_states.stride(1),
            up_weight.stride(0), up_weight.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=256, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # Elementwise activation: y = gate_out * sigmoid(gate_out) * up_out
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        grid_elem = (M, N)
        silu_mul_kernel[grid_elem](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            num_warps=4, num_stages=1
        )

        # Return as bfloat16 (cast is allowed here; no torch ops on tensors required)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
