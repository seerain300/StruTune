import torch
import triton
import triton.language as tl


@triton.jit
def linear_rowwise_bf16_to_f32(
    X_ptr,            # *bfloat16, [M, K]
    W_ptr,            # *bfloat16, [K, N]
    Y_ptr,            # *float32,  [M, N]
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row m
    m = tl.program_id(0)

    # Initialize accumulator for this row as float32
    # We'll compute it per column block
    # We will store to Y[m, n] with mask
    for n_start in range(0, N, BLOCK_N):
        # acc holds contributions for this block of N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        # Loop over K in chunks
        for k_start in range(0, K, BLOCK_K):
            # Load X scalar x[m, k]
            # offs_k over the chunk
            for i in range(0, BLOCK_K):
                k = k_start + i
                x_val = tl.load(X_ptr + m * stride_xm + k * stride_xk, mask=k < K, other=0.0)
                x_val_f32 = x_val.to(tl.float32)

                # Load W vector W[k, n_start:n_start+BLOCK_N]
                offs_n = n_start + tl.arange(0, BLOCK_N)
                n_mask = offs_n < N
                w_vec = tl.load(W_ptr + k * stride_wk + offs_n * stride_wn, mask=n_mask, other=0.0)
                w_vec_f32 = w_vec.to(tl.float32)

                # Accumulate outer product contribution
                acc += x_val_f32 * w_vec_f32

        # Store results to Y[m, n_start:n_start+BLOCK_N]
        n_store = n_start + tl.arange(0, BLOCK_N)
        n_mask_store = n_store < N
        tl.store(Y_ptr + m * stride_ym + n_store * stride_yn, acc, mask=n_mask_store)


@triton.jit
def silu_mul_kernel(
    Gate_ptr, Up_ptr, Y_ptr,
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Each program computes a BLOCK_M x BLOCK_N tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Load gate_out and up_out tiles (cast to float32 for computation)
    gate = tl.load(Gate_ptr + m_offsets[:, None] * stride_gm + n_offsets[None, :] * stride_gn,
                   mask=m_mask[:, None] & n_mask[None, :],
                   other=0.0)
    up = tl.load(Up_ptr + m_offsets[:, None] * stride_um + n_offsets[None, :] * stride_un,
                 mask=m_mask[:, None] & n_mask[None, :],
                 other=0.0)

    # Compute sigmoid(gate) = 1 / (1 + exp(-gate))
    # Cast to float32 for numerical stability if needed (gate is already float32 from GEMV)
    gate_f32 = gate  # already float32
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate_f32))

    y = gate_f32 * sigmoid_gate * up  # Up is float32

    tl.store(Y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn,
             y,
             mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states,            # [M, K], bfloat16
        shared_expert_gate_weight, # [K, N], bfloat16
        shared_expert_up_weight,   # [K, N], bfloat16
    ):
        # Ensure tensors are contiguous and on CUDA
        hidden = hidden_states.contiguous()
        gate_w = shared_expert_gate_weight.contiguous()
        up_w = shared_expert_up_weight.contiguous()

        M, K = hidden.shape
        K_w, N = gate_w.shape
        assert K_w == K, "shared_expert_gate_weight shape mismatch"
        assert up_w.shape[0] == K, "shared_expert_up_weight shape mismatch"

        # Allocate outputs in float32
        gate_out = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        up_out = torch.empty((M, N), dtype=torch.float32, device=hidden.device)

        # Launch gate GEMV: one program per row
        grid_m = (M,)
        # Choose conservative BLOCK sizes to reduce risk
        BLOCK_K = 128
        BLOCK_N = 128
        linear_rowwise_bf16_to_f32[grid_m](
            hidden, gate_w, gate_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            gate_w.stride(0), gate_w.stride(1),
            gate_out.stride(0), gate_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Launch up GEMV: one program per row
        linear_rowwise_bf16_to_f32[grid_m](
            hidden, up_w, up_out,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            up_w.stride(0), up_w.stride(1),
            up_out.stride(0), up_out.stride(1),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Compute shared_activated = SiLU(gate_out) * up_out using Triton elementwise kernel
        # Note: SiLU(x) = x * sigmoid(x)
        activated = torch.empty((M, N), dtype=torch.float32, device=hidden.device)

        BLOCK_M = 32
        BLOCK_N_elem = 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N_elem))
        silu_mul_kernel[grid](
            gate_out, up_out, activated,
            M, N,
            gate_out.stride(0), gate_out.stride(1),
            up_out.stride(0), up_out.stride(1),
            activated.stride(0), activated.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_elem,
            num_warps=4, num_stages=2,
        )

        # Return bfloat16 (dtype cast, not elementwise op)
        return activated.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
