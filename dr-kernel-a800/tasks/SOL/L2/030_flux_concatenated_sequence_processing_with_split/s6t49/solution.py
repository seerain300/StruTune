import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # text_seq_len
    N: tl.constexpr,    # img_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile size along seq
    BLOCK_H: tl.constexpr,  # tile size along hidden (here use H)
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    tile = tl.program_id(1)
    C = M + N

    # sequence offsets handled by this program
    seq_offsets = tile * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = seq_offsets < C

    # Determine if each seq index comes from A or B
    from_A = seq_offsets < M
    from_B = ~from_A

    # Hidden dimension offsets (use H as BLOCK_H)
    h_offsets = tl.arange(0, BLOCK_H)  # [BLOCK_H], BLOCK_H == H here
    mask_h = h_offsets < H  # always true since BLOCK_H == H

    # Compute pointers and loads
    a_ptrs = A_ptr + b * stride_ab + (seq_offsets[:, None] - 0) * stride_am + h_offsets[None, :] * stride_ah
    b_ptrs = B_ptr + b * stride_bb + (seq_offsets[:, None] - M) * stride_bn + h_offsets[None, :] * stride_bh

    a_mask = mask_m[:, None] & mask_h[None, :]
    b_mask = mask_m[:, None] & mask_h[None, :]
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, H]
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_M, H]

    # Select based on source
    vals = tl.where(from_A[:, None], a_vals, b_vals)  # [BLOCK_M, H]

    # Store to Out
    out_ptrs = Out_ptr + b * stride_ob + seq_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    tl.store(out_ptrs, vals, mask=(mask_m[:, None]))


@triton.jit
def _batched_matmul_tl_dot_kernel(
    X_ptr,  # *f32, [B, C, K]
    WT_ptr, # *f32, [K, K]  (process_weight transposed view)
    P_ptr,  # *f32, [B, C, K] (same as X)
    B: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    stride_wk,  # int (rows of WT: K)
    stride_wh,  # int (cols of WT: K)
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over C
    BLOCK_N: tl.constexpr,  # tile over K
    BLOCK_K: tl.constexpr,  # tile over reduction dim
):
    # 3D grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    c_block = tl.program_id(1)
    k_block = tl.program_id(2)

    c_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_c = c_offsets < C
    mask_k = k_offsets < K

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k_chunk = k_ids < K

        # Load X[b, c, k] tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + c_offsets[:, None] * stride_xc + k_ids[None, :] * stride_xk
        x_mask = mask_c[:, None] & mask_k_chunk[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load WT[k, k'] tile: [BLOCK_K, BLOCK_N]
        w_ptrs = WT_ptr + k_ids[:, None] * stride_wk + k_offsets[None, :] * stride_wh
        w_mask = mask_k_chunk[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_N]

    # Store acc to P[b, c, k]
    p_ptrs = P_ptr + b * stride_pb + c_offsets[:, None] * stride_pc + k_offsets[None, :] * stride_pk
    p_mask = mask_c[:, None] & mask_k[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized concat + GEMM:
        - Concatenate encoder_hidden_states [B, M, H] and hidden_states [B, N, H] along sequence to X [B, C, H] where C = M + N.
        - Compute processed = X @ process_weight.T (no bias), with process_weight [H, H].
        - Split processed into (processed_encoder [B, M, H], processed_hidden [B, N, H]).
        Forward only orchestrates allocation and Triton kernel launches; no torch GPU ops in forward (except allocation).
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3 and process_weight.ndim == 2, "Input shapes must be [B, M/H, H] and [H, H]"
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        C = M + N

        # Ensure inputs are contiguous (PyTorch side for safety; forward has no GPU torch ops after this)
        A = encoder_hidden_states  # [B, M, H]
        Bt = hidden_states          # [B, N, H]
        # Allocate output P [B, C, H] and write concat into it via Triton
        P = torch.empty((B, C, H), device=A.device, dtype=A.dtype)

        # Launch Triton concat kernel
        BLOCK_M = 64
        BLOCK_H = H  # use H as BLOCK_H for concat
        grid_concat = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid_concat](
            A, Bt, P,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bt.stride(0), stride_bn=Bt.stride(1), stride_bh=Bt.stride(2),
            stride_ob=P.stride(0), stride_oc=P.stride(1), stride_oh=P.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=3,
        )

        # GEMM in Triton: P @ process_weight.T -> P (in-place)
        # WT is a transposed view of process_weight for Triton's [K, K] expectation
        WT = process_weight.transpose(0, 1)  # [H, H] -> [H, H] view, no data copy

        # Strides for X=P [B, C, K]
        stride_xb, stride_xc, stride_xk = P.stride(0), P.stride(1), P.stride(2)
        # Strides for WT [K, K]
        stride_wk, stride_wh = WT.stride(0), WT.stride(1)
        # Strides for P output [B, C, K]
        stride_pb, stride_pc, stride_pk = P.stride(0), P.stride(1), P.stride(2)

        # GEMM tiling parameters
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_gemm), triton.cdiv(H, BLOCK_N_gemm))
        _batched_matmul_tl_dot_kernel[grid_gemm](
            P, WT, P,  # we write back to P
            B=B, C=C, K=H,
            stride_xb=stride_xb, stride_xc=stride_xc, stride_xk=stride_xk,
            stride_wk=stride_wk, stride_wh=stride_wh,
            stride_pb=stride_pb, stride_pc=stride_pc, stride_pk=stride_pk,
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=3,
        )

        # Split back into encoder and image streams
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
