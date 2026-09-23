import math
import torch
import triton
import triton.language as tl


@triton.jit
def qk_matmul_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute logits L = Q @ K_expanded^T, where:
      - Q: [Nq, Hq, D], Hq=32, D=128
      - K_expanded: [Nk, Hq, D] (we pass K with 8 heads and expand in host)
      - L: [Nq, Hq, Nk]
    """
    pid0 = tl.program_id(0)  # tile over Nq
    pid1 = tl.program_id(1)  # tile over Nk
    pid2 = tl.program_id(2)  # head index 0..Hq-1

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    j_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_j = j_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over D (compile-time 128) to compute Q @ K^T
    for d in tl.static_range(0, D):
        q_vec = tl.load(
            Q_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d,
            mask=mask_q[:, None],
            other=0.0  # [BLOCK_M, 1]
        )
        k_vec = tl.load(
            K_ptr + j_offsets[:, None] * (Hq * D) + pid2 * D + d,
            mask=mask_j[:, None],
            other=0.0  # [BLOCK_N, 1]
        )
        # acc += q @ k^T -> [BLOCK_M, 1] @ [1, BLOCK_N] = [BLOCK_M, BLOCK_N]
        acc += q_vec @ tl.trans(k_vec)

    # Apply scaling
    acc = acc * sm_scale

    # Store logits L: [Nq, Hq, Nk] => index = nq*(Hq*Nk) + h*Nk + j
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + j_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_j[None, :]
    )


@triton.jit
def attn_dot_v_kernel(
    Soft_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute Y = Soft @ V_expanded^T, where:
      - Soft: [Nq, Hq, Nk] float32 (attention weights)
      - V_expanded: [Nk, Hq, D] (we pass expanded to 32 heads; here D=128)
      - Y: [Nq, Hq, D] float32
    Grid: (Nq tiles, Nk tiles, Hq)
    """
    pid0 = tl.program_id(0)  # tile over Nq
    pid1 = tl.program_id(1)  # tile over Nk
    pid2 = tl.program_id(2)  # head index 0..Hq-1

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    j_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_j = j_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # D=128; single iteration
    d = 0
    Soft_tile = tl.load(
        Soft_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + j_offsets[None, :],
        mask=mask_q[:, None] & mask_j[None, :],
        other=0.0
    )  # [BLOCK_M, BLOCK_N]
    V_tile = tl.load(
        V_ptr + j_offsets[:, None] * (Hq * D) + pid2 * D + d,
        mask=mask_j[:, None],
        other=0.0  # [BLOCK_N, 1]
    )  # [BLOCK_N, 1]
    acc += Soft_tile @ tl.trans(V_tile)

    # For completeness, if D > 1, we would iterate; here D=128 and we handle all in next loop below.
    # However, we need a proper loop; Triton requires static unrolling over D dimension.
    # Since D is 128, we can do a single iteration using tiles:
    for d_start in tl.static_range(0, 128, 128):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask_d = d_offsets < 128
        # Load Soft tile [BLOCK_M, BLOCK_N]
        Soft_tile = tl.load(
            Soft_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + j_offsets[None, :],
            mask=mask_q[:, None] & mask_j[None, :],
            other=0.0
        )
        # Load V tile as [BLOCK_N, BLOCK_D]
        V_tile = tl.load(
            V_ptr + j_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_j[:, None] & mask_d[None, :],
            other=0.0
        )
        acc += Soft_tile @ V_tile  # [BLOCK_M, BLOCK_D]

    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_d[None, :]
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k: [total_kv, 8, 128], bfloat16
        v: [total_kv, 8, 128], bfloat16
        qo_indptr: [len_indptr], int32
        kv_indptr: [len_indptr], int32
        sm_scale: float32 scalar
        Returns (output: [total_q, 32, 128], lse: [total_q, 32], both float32)
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels require CUDA tensors"
        device = q.device

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Output and LSE tensors (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Iterate segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end]  # [num_kv_tokens, 8, 128]

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_batch.shape[0]
            delta = num_kv_tokens - num_q_tokens

            # Expand K and V


def run(*args):
    return ModelNew()(*args)
