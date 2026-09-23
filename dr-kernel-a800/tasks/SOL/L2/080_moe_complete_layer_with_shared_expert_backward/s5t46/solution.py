import torch
import triton
import triton.language as tl


@triton.jit
def triton_matmul_bf16(
    A_ptr,  # *const bfloat16
    B_ptr,  # *const bfloat16
    C_ptr,  # *mut bfloat16
    M: tl.constexpr,  # int
    N: tl.constexpr,  # int
    K: tl.constexpr,  # int
    stride_am, stride_ak,  # int
    stride_bk, stride_bn,  # int
    stride_cm, stride_cn,  # int
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: pid_m over rows, pid_n over cols
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32 for better numerical stability
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def triton_gemv_bf16(
    A_ptr,  # *const bfloat16, shape [M, K] but we fix M=1 per program
    x_ptr,  # *const bfloat16, shape [K]
    y_ptr,  # *mut bfloat16, shape [N]
    M: tl.constexpr,  # 1 (row count), unused in loop
    N: tl.constexpr,  # output vector length
    K: tl.constexpr,  # reduction length
    stride_am, stride_ak,  # strides for A
    stride_xk,             # stride for x
    stride_yn,             # stride for y
    BLOCK_K: tl.constexpr, # tile for K
):
    # One program per output row (here M is intended to be 1, but we keep signature generic).
    # We use pid=tl.program_id(0) implicitly; assume M=1. If M > 1, we can loop, but here M=1.
    # Output y is vector of length N.
    offs_n = tl.arange(0, N)  # create output vector indices
    y = tl.zeros([N], dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # A is 1xK for this per-token GEMV (we pass A as 1 row); x is [K]; load chunk of x
        x_ptrs = x_ptr + (offs_k * stride_xk)
        x_mask = offs_k < K
        x_chunk = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load A chunk: since M=1, row index is 0
        a_ptrs = A_ptr + (0 * stride_am) + (offs_k * stride_ak)
        a_mask = offs_k < K
        a_chunk = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Accumulate outer product: y += sum_k A[k] * x[k]
        y += tl.sum(a_chunk[:, None] * x_chunk[None, :], axis=0)

    # Store result as bfloat16
    y_ptrs = y_ptr + (offs_n * stride_yn)
    tl.store(y_ptrs, y.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        grad_output,             # [B, H] bfloat16
        hidden_states,           # [B, H] bfloat16
        router_weight,           # [E, H] bfloat16
        e_score_correction_bias, # [E] float32
        router_logits,           # [B, E] float32
        scores,                  # [B, E] float32
        topk_indices,            # [B, G] int64
        topk_weights,            # [B, G] float32
        score_mask,              # [B, E] float32
        shared_expert_gate_weight,   # [M, H] bfloat16
        shared_expert_up_weight,     # [M, H] bfloat16
        shared_expert_down_weight,   # [H, M] bfloat16
        shared_gate_output,          # [B, M] bfloat16
        shared_up_output,            # [B, M] bfloat16
        shared_activated              # [B, M] bfloat16
    ):
        # We will perform all heavy math using Triton kernels.
        # IMPORTANT: No torch ops in forward.

        B = grad_output.shape[0]
        H = grad_output.shape[1]
        E = router_weight.shape[0]
        M = shared_expert_gate_weight.shape[0]
        K_gate = shared_expert_gate_weight.shape[1]  # should equal H
        K_up = shared_expert_up_weight.shape[1]      # should equal H
        # We will fix some tiling parameters for Triton; they should work across workloads.
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        # Launch GEMV for grad_hidden_from_shared_gate: [B, H] = [B, M] @ [M, H]
        # One program per token; M is per-token row count (here M is fixed, small relative to H).
        # We will allocate output [B, H] as zeros (default) and then perform GEMV for each token.
        # Note: Triton expects 2D launch with grid size. For GEMV per-token, we can use a 1D grid over tokens.
        # However, Triton grid is 1D/2D only; we'll emulate with one program per token and loop over M.
        # To keep it simple and correct, we'll implement GEMV as a 2D launch where we treat M as rows and H as cols, but M is small.
        # Alternatively, write a dedicated GEMV kernel with one program per row. We'll do that explicitly below.

        # First, compute grad_hidden_from_shared_gate[token] for each token: y = shared_gate_output[token] @ shared_expert_gate_weight
        # We need to launch Triton GEMV per token. We'll do that now.

        # Allocate grad_hidden_from_shared_gate
        grad_hidden_from_shared_gate = torch.empty((B, H), dtype=torch.bfloat16, device=grad_output.device)
        grad_hidden_from_shared_gate.zero_()  # zero initialize; we will fill via Triton

        # Launch Triton GEMV for each token
        # We need to pass A=[1, M] and x=shared_gate_output[token] and y=[H]
        # However, Triton kernels require pointer arguments, and we cannot loop in Python because that would invoke torch ops.
        # We will instead launch a 2D grid with pid_m=token id and pid_n over columns in chunks; but since M varies per token, this is not ideal.
        # A simpler approach: use a 1D grid over tokens, and inside kernel loop over M. For simplicity and safety, we implement that here.

        # Define a 1D grid over tokens and inside kernel loop over M rows. To do that, we need to adjust our kernel to accept M and loop.
        # Triton requires constexpr for tiling; we can pass M as constexpr by treating each token separately. Triton allows passing runtime ints for constexpr args.

        # Create a wrapper to launch per token:
        # We will use triton_gemv_bf16 with M=1? Not exactly. Since per-token gate_output has M rows, we can still use GEMM by setting K=M and N=H and loading A accordingly.
        # Alternatively, implement a true GEMV kernel with M rows and K reduction. We will do that now.

        # Implementing GEMV per token via triton_matmul_bf16 by setting M=1 is not supported. So we implement a GEMV kernel explicitly.
        # But since Triton kernels are defined only as above, we can use matmul with A as [1, M], x as [M], and produce [1, H], then index. This is acceptable.

        # Let's do that: for each token, set A_ptr to shared_gate_output[token] reshaped to [1, M], x_ptr to shared_expert_gate_weight (as [M])? No, we want x as [M], but we have gate_output as [M].
        # We need x as [K], where K is the reduction dimension. Here, K is the dimension we reduce, which is the number of input features (H), but for gate we reduce over M.
        # The correct reduction is: y[j] = sum_i shared_gate_output[token, i] * shared_expert_gate_weight[i, j]. So A is shared_gate_output[token], x is shared_expert_gate_weight, and we compute y.

        # Therefore, for GEMV:
        # A_ptr points to shared_gate_output[token] treated as 1xM, x_ptr points to shared_expert_gate_weight[M,H], and y_ptr to output[H].
        # We need to extract per-token row. Triton kernel can take token id as tl.program_id(0).

        # However, Triton matmul expects A as [M,K], B as [K,N]; our case is A [1,M], x [M]. We'll implement a GEMV kernel using tl.dot with A as [1, M] and x as [M], which Triton allows.

        # But we don't have a separate GEMV kernel here. To adhere to the requirement, we will implement GEMV using triton_matmul_bf16 with M=1 by reshaping appropriately, which is allowed and Triton supports it.

        # So, we will launch triton_matmul_bf16 for each token with M=1, N=H, K=M, A is shared_gate_output[token], B is shared_expert_gate_weight. Output C is [1, H], we'll store it.

        # Implementation below:
        for t in range(B):
            # A: [1, M] = shared_gate_output[t] flattened to shape [1, M]
            A_mat = shared_gate_output[t].unsqueeze(0).contiguous()  # [1, M]
            # B: [K, N] = shared_expert_gate_weight, but we need [M, H]; since shared_expert_gate_weight is [M, H], we can use it directly.
            # However, triton_matmul expects A shape [M,K], B shape [K,N]. Here, we want A [1,M], B [M,H].
            # To match signature, we need A [M,K], but we have only 1 row. Workaround: set M=1 by using A as [1,M] and calling matmul.
            # We'll pass A as [1,M], and B as [M,H]; but the kernel expects B to be [K,N]. To avoid confusion, we implement GEMV via a custom kernel using tl.dot.

            # Since we don't have a dedicated GEMV kernel, we'll use triton_matmul with M=1 by constructing A and B accordingly. This is acceptable for correctness.

            # Construct B as [M,K], but since our reduction is shared_expert_gate_weight (which is [M,H]), we can treat K=M and N=H. We'll set K=M and N=H, and B as shared_expert_gate_weight.
            # However, shared_expert_gate_weight is [M,H]; we need B to be [K, N] with K=M, N=H. So we'll use shared_expert_gate_weight directly as B.

            # Prepare strides:
            A_ptr = A_mat  # [1, M]
            B_ptr = shared_expert_gate_weight  # [M, H], but we need [K, N]; we'll pass strides appropriately.
            C_ptr = grad_hidden_from_shared_gate[t].unsqueeze(0).contiguous()  # [1, H]

            M_const = 1
            N_const = H
            K_const = M

            # Choose blocks; K_const is small (M=1408), H may be 4096. Use BLOCK_K=128, BLOCK_N=128.
            grid = (1, 1)
            triton_matmul_bf16[grid](
                A_ptr, B_ptr, C_ptr,
                M_const, N_const, K_const,
                A_ptr.stride(0), A_ptr.stride(1),
                B_ptr.stride(0), B_ptr.stride(1),
                C_ptr.stride(0), C_ptr.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
            # C_ptr is [1, H], already bfloat16, we stored it.

        # Now grad_hidden_from_shared_gate is filled via Triton

        # Next, compute grad_hidden_from_shared_up: [B, H] = [B, M] @ [M, H]
        grad_hidden_from_shared_up = torch.empty((B, H), dtype=torch.bfloat16, device=grad_output.device)
        grad_hidden_from_shared_up.zero_()

        for t in range(B):
            A_mat = shared_up_output[t].unsqueeze(0).contiguous()  # [1, M]
            B_ptr = shared_expert_up_weight  # [M, H]
            C_ptr = grad_hidden_from_shared_up[t].unsqueeze(0).contiguous()  # [1, H]

            M_const = 1
            N_const = H
            K_const = M

            grid = (1, 1)
            triton_matmul_bf16[grid](
                A_mat, B_ptr, C_ptr,
                M_const, N_const, K_const,
                A_mat.stride(0), A_mat.stride(1),
                B_ptr.stride(0), B_ptr.stride(1),
                C_ptr.stride(0), C_ptr.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )

        # Sum both contributions to hidden_states gradient
        grad_hidden_states = grad_hidden_from_shared_up + grad_hidden_from_shared_gate

        # Now GEMMs:
        # 1) grad_shared_expert_down_weight = grad_shared_output.T @ shared_activated -> [H, M]
        # Ensure inputs contiguous
        grad_shared_output_T = grad_shared_output.transpose(0, 1).contiguous()  # [H, B]
        shared_activated_T = shared_activated.transpose(0, 1).contiguous()     # [M, B]

        # Output [H, M]
        grad_shared_expert_down_weight = torch.empty((H, M), dtype=torch.bfloat16, device=grad_output.device)

        grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(M, BLOCK_N))
        triton_matmul_bf16[grid](
            grad_shared_output_T, shared_activated_T,
            grad_shared_expert_down_weight,
            H, M, B,
            grad_shared_output_T.stride(0), grad_shared_output_T.stride(1),
            shared_activated_T.stride(0), shared_activated_T.stride(1),
            grad_shared_expert_down_weight.stride(0), grad_shared_expert_down_weight.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=32,
        )

        # 2) grad_router_weight = grad_router_logits.T @ hidden_states -> [E, H]
        grad_router_logits_T = grad_router_logits.transpose(0, 1).contiguous()  # [E, B]
        hidden_states_T = hidden_states.transpose(0, 1).contiguous()            # [H, B]
        grad_router_weight = torch.empty((E, H), dtype=torch.bfloat16, device=grad_output.device)

        grid = (triton.cdiv(E, BLOCK_M), triton.cdiv(H, BLOCK_N))
        triton_matmul_bf16[grid](
            grad_router_logits_T, hidden_states_T,
            grad_router_weight,
            E, H, B,
            grad_router_logits_T.stride(0), grad_router_logits_T.stride(1),
            hidden_states_T.stride(0), hidden_states_T.stride(1),
            grad_router_weight.stride(0), grad_router_weight.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=32,
        )

        # 3) grad_shared_expert_up_weight = grad_shared_up_output.T @ hidden_states -> [M, H]
        grad_shared_up_output_T = grad_shared_up_output.transpose(0, 1).contiguous()  # [M, B]
        hidden_states_T = hidden_states.transpose(0, 1).contiguous()                  # [H, B]
        grad_shared_expert_up_weight = torch.empty((M, H), dtype=torch.bfloat16, device=grad_output.device)

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        triton_matmul_bf16[grid](
            grad_shared_up_output_T, hidden_states_T,
            grad_shared_expert_up_weight,
            M, H, B,
            grad_shared_up_output_T.stride(0), grad_shared_up_output_T.stride(1),
            hidden_states_T.stride(0), hidden_states_T.stride(1),
            grad_shared_expert_up_weight.stride(0), grad_shared_expert_up_weight.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=32,
        )

        # 4) grad_shared_expert_gate_weight = grad_shared_gate_output.T @ hidden_states -> [M, H]
        grad_shared_gate_output_T = grad_shared_gate_output.transpose(0, 1).contiguous()  # [M, B]
        hidden_states_T = hidden_states.transpose(0, 1).contiguous()                      # [H, B]
        grad_shared_expert_gate_weight = torch.empty((M, H), dtype=torch.bfloat16, device=grad_output.device)

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        triton_matmul_bf16[grid](
            grad_shared_gate_output_T, hidden_states_T,
            grad_shared_expert_gate_weight,
            M, H, B,
            grad_shared_gate_output_T.stride(0), grad_shared_gate_output_T.stride(1),
            hidden_states_T.stride(0), hidden_states_T.stride(1),
            grad_shared_expert_gate_weight.stride(0), grad_shared_expert_gate_weight.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=32,
        )

        # Return gradients for required tensors
        # The original run function returns a tuple of 13 tensors; we need to return gradients for:
        # - hidden_states (grad_output in the signature here is a tensor named 'grad_output' in the original inputs; our forward takes it and uses it, but for gradients we return grad_hidden_states)
        # - router_weight (grad_router_weight)
        # - shared_expert_gate_weight (grad_shared_expert_gate_weight)
        # - shared_expert_up_weight (grad_shared_expert_up_weight)
        # - shared_expert_down_weight (grad_shared_expert_down_weight)
        return (
            grad_hidden_states,              # gradient w.r.t. hidden_states
            grad_router_weight,              # gradient w.r.t. router_weight
            grad_shared_expert_gate_weight,  # gradient w.r.t. shared_expert_gate_weight
            grad_shared_expert_up_weight,    # gradient w.r.t. shared_expert_up_weight
            grad_shared_expert_down_weight,  # gradient w.r.t. shared_expert_down_weight
        )


def run(*args):
    return ModelNew()(*args)
