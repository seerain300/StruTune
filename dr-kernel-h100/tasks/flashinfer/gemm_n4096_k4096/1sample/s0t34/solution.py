import torch
import triton
import triton.language as tl


@triton.jit
def _rowcol_matmul_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,          # strides for A[m, k]
    stride_bTk, stride_bTn,        # strides for BT[k, n]
    stride_cm, stride_cn,          # strides for C[m, n]
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid is (M, ceil(N / BLOCK_N))
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    n_start = n_block * BLOCK_N

    # Column indices this program handles
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    # Accumulator for this row m across BLOCK_N columns
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A[m, k_offsets] as a vector of length BLOCK_K
        a_ptrs = A + m * stride_am + k_offsets * stride_ak
        a_vec = tl.load(a_ptrs, mask=k_offsets < K, other=0.0)  # [BLOCK_K]

        # Load BT[k_offsets, n_offsets] as a vector of length BLOCK_N
        # BT is [K, N], so BT[k, n] = B[n, k]. We index with k and n.
        b_ptrs = BT + k_offsets[:, None] * stride_bTk + n_offsets[None, :] * stride_bTn
        b_vec = tl.load(b_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)  # [BLOCK_K, BLOCK_N]
        # Reduce this chunk: sum over k to get [BLOCK_N]
        acc += tl.sum(b_vec * a_vec[:, None], axis=0)

    # Store the accumulated result to C[m, n_offsets]
    c_ptrs = C + m * stride_cm + n_offsets * stride_cn
    tl.store(c_ptrs, acc, mask=n_offsets < N)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Validate dtypes (expect floating types)
        if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise AssertionError("A dtype must be a floating type.")
        if B.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise AssertionError("B dtype must be a floating type.")

        # Shapes: A [M, K], B [N, K] (per get_inputs). We need C = A @ B.T with shape [M, N]
        # In general, B could be any [N, K], but here it's [4096, 4096].
        M, K = A.shape
        N = B.shape[0]

        # Explicitly create BT with correct shape and contiguous strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Allocate output in fp32 for accumulation stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Choose tile sizes; masks handle edges. Tuned for N=4096, K=4096.
        BLOCK_N = 256
        BLOCK_K = 256

        # Grid: one program per (m, n_block)
        grid = (M, triton.cdiv(N, BLOCK_N))

        _rowcol_matmul_at_bt_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Cast to original dtype to match torch.matmul behavior (inputs are fp16 per get_inputs)
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
