import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _concatenate_seqs_kernel(
    enc_ptr, hid_ptr, out_ptr,
    B, T, I, H,
    enc_bs, enc_ts, enc_hs,
    hid_bs, hid_is, hid_hs,
    out_bs, out_ts, out_hs,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)

    # Base pointers for this batch
    enc_b = enc_ptr + b * enc_bs
    hid_b = hid_ptr + b * hid_bs
    out_b = out_ptr + b * out_bs

    # Iterate over total sequence length T + I
    # Use masks to avoid out-of-bounds
    for l in range(0, T + I):
        # Determine source tensor and index
        is_encoder = l < T
        if is_encoder:
            src_ptr = enc_b
            idx = l
        else:
            src_ptr = hid_b
            idx = l - T

        # Load row from source and store to output
        row_ptr = src_ptr + idx * enc_ts  # enc_ts/hid_is is the stride along sequence dim
        out_row_ptr = out_b + l * out_ts

        # Load H elements with mask for safety
        for j in range(0, H):
            # Simple per-element load/store; H is typically modest (e.g., 128/256), this is safe
            val = tl.load(row_ptr + j * enc_hs)  # enc_hs/hid_hs is stride along hidden dim
            tl.store(out_row_ptr + j * out_hs, val)


@triton.jit
def _gmm_batched_right_kernel(
    A_ptr, WT_ptr, C_ptr,
    M, K, N,  # M = B*(T+I), K = H, N = H
    A_stride_m, A_stride_k,
    WT_stride_k, WT_stride_n,  # WT is [K, N], here K=N=H, but keep generic
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over output [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # WT tile: [BLOCK_K, BLOCK_N]
        wt_ptrs = WT_ptr + k_offsets[:, None] * WT_stride_k + n_offsets[None, :] * WT_stride_n
        wt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, wt)

    # Write back to C
    c_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_streams_kernel(
    C_ptr, out_e_ptr, out_i_ptr,
    B, T, I, H,
    C_stride_m, C_stride_n,
    out_e_bs, out_e_ts, out_e_hs,
    out_i_bs, out_i_ts, out_i_hs,
    BLOCK_H: tl.constexpr,
):
    # One program per batch b, loop over sequence and copy
    b = tl.program_id(0)
    total = T + I

    for l in range(0, total):
        # Determine destination
        if l < T:
            dest_ptr = out_e_ptr + b * out_e_bs + l * out_e_ts
        else:
            dest_ptr = out_i_ptr + b * out_i_bs + (l - T) * out_i_ts

        # Copy H elements
        for j in range(0, H, BLOCK_H):
            offs = j + tl.arange(0, BLOCK_H)
            row_ptr = C_ptr + (b * total + l) * C_stride_m  # m index = b*(T+I) + l
            vals = tl.load(row_ptr + offs * C_stride_n, mask=offs < H, other=0.0)
            if l < T:
                tl.store(dest_ptr + offs * out_e_hs, vals, mask=offs < H)
            else:
                tl.store(dest_ptr + offs * out_i_hs, vals, mask=offs < H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H] (right-multiply)
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D: [B, L, H]"
        B, I, H = hidden_states.shape
        B2, T, H2 = encoder_hidden_states.shape
        assert B == B2 and H == H2, "Batch size and hidden_dim must match between inputs"

        # Ensure CUDA tensors (Triton requires CUDA)
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew.forward requires CUDA tensors"
        # Ensure contiguous for simple stride handling
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        W = process_weight.contiguous()  # [H, H]

        # 1) Concatenate along sequence dim to [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=torch.float32, device=device)

        _concatenate_seqs_kernel[(B,)](
            enc, hid, out_cat,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            BLOCK_T=128, BLOCK_I=128,
            num_warps=1, num_stages=1,
        )

        # 2) GEMM: C = out_cat @ W.T, where out_cat is treated as rows [M= B*(T+I), K=H], W.T is [K, N=H]
        # Make A row-major [M, K]
        A = out_cat.reshape(-1, H).contiguous()  # [M, H]
        M = A.shape[0]  # B*(T+I)

        # WT = W.T contiguous, shape [H, H]
        WT = W.t().contiguous()

        C = torch.empty((M, H), dtype=torch.float32, device=device)

        # Launch 2D grid over (M, H) tiles
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _gmm_batched_right_kernel[grid](
            A, WT, C,
            M, H, H,
            A.stride(0), A.stride(1),
            WT.stride(0), WT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split C back into encoder and image streams
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)

        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=128,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
