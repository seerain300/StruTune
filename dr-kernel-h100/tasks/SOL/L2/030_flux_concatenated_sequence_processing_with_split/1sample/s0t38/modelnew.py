import torch
import triton
import triton.language as tl

# Triton kernel: concatenate [B, T, H] and [B, I, H] into [B, T+I, H]
@triton.jit
def _concatenate_sequences_kernel(
    enc_ptr,           # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,           # *ptr to hidden_states [B, I, H]
    out_ptr,           # *ptr to output [B, T+I, H]
    B, T, I, H,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch
    # Iterate over concatenated sequence length in chunks
    for l in range(0, T + I, BLOCK_L):
        offs = l + tl.arange(0, BLOCK_L)
        mask = offs < (T + I)
        # Load from encoder or hidden depending on offs
        is_encoder = offs < T
        # Compute per-element base offsets
        enc_offsets = b * enc_stride_b + offs * enc_stride_t
        hid_offsets = b * hid_stride_b + (offs - T) * hid_stride_i
        # Select source based on mask
        src_offsets = tl.where(is_encoder, enc_offsets, hid_offsets)
        # Load values
        vals = tl.load(
            out_ptr + b * out_stride_b + offs * out_stride_l + tl.zeros([BLOCK_L], dtype=tl.int32) * out_stride_h,
            mask=mask,
            other=0.0
        )  # placeholder; will be overwritten
        # Since Triton lacks dynamic select for tensors easily, we implement by branching per element:
        # Create a zero tensor
        vals = tl.zeros([BLOCK_L], dtype=tl.float32)
        # For each i in chunk, load from enc if is_encoder[i], else from hid
        for i in range(0, BLOCK_L):
            m_i = i < (T + I)
            if m_i:
                if is_encoder[i]:
                    val_i = tl.load(enc_ptr + src_offsets[i], mask=True, other=0.0)
                else:
                    val_i = tl.load(hid_ptr + src_offsets[i], mask=True, other=0.0)
                vals[i] = val_i
        # Store into output
        tl.store(out_ptr + b * out_stride_b + offs * out_stride_l, vals, mask=mask)

# Triton kernel: batched GEMM C = A @ W^T, where
# A: [M, K], M = B*(T+I), K = H
# W: [H, H], we right-multiply by W^T
# C: [M, H]
@triton.jit
def _batched_gemm_right_kernel(
    A_ptr,           # *ptr to A [M, K]
    W_ptr,           # *ptr to W [H, H] (right-multiply: W^T)
    C_ptr,           # *ptr to C [M, H]
    M, K, H,
    A_stride_m, A_stride_k,
    W_stride_h, W_stride_k,
    C_stride_m, C_stride_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N (H)

    # Compute tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A block: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_block = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # W^T block: we load W[k, n] to form a [BLOCK_K, BLOCK_N] tile for dot
        WT_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_h
        WT_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < H)
        WT_block = tl.load(WT_ptrs, mask=WT_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_block, WT_block)

    # Write back to C
    C_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_k
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < H)
    tl.store(C_ptrs, acc, mask=C_mask)

# Triton kernel: split C [B*(T+I), H] into two outputs:
# processed_encoder [B, T, H] and processed_hidden [B, I, H]
@triton.jit
def _split_streams_kernel(
    C_ptr,            # *ptr to C [B*(T+I), H]
    out_enc_ptr,      # *ptr to processed_encoder [B, T, H]
    out_hid_ptr,      # *ptr to processed_hidden [B, I, H]
    B, T, I, H,
    C_stride_m, C_stride_k,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch
    # Encode rows: m in [0, B*T)
    for t in range(0, T, BLOCK_H):
        m_offs = t + tl.arange(0, BLOCK_H)
        mask_t = m_offs < T
        m = b * (T + I) + m_offs
        src_ptrs = C_ptr + m[:, None] * C_stride_m + tl.arange(0, H)[None, :] * C_stride_k
        vals = tl.load(src_ptrs, mask=mask_t[:, None], other=0.0)
        dst_ptrs = out_enc_ptr + b * enc_stride_b + m_offs[:, None] * enc_stride_t + tl.arange(0, H)[None, :] * enc_stride_h
        tl.store(dst_ptrs, vals, mask=mask_t[:, None])

    # Hidden rows: m in [B*T, B*(T+I))
    for i in range(0, I, BLOCK_H):
        m_offs = b * T + i + tl.arange(0, BLOCK_H)
        mask_i = m_offs < (b * T + I)
        m = m_offs
        src_ptrs = C_ptr + m[:, None] * C_stride_m + tl.arange(0, H)[None, :] * C_stride_k
        vals = tl.load(src_ptrs, mask=mask_i[:, None], other=0.0)
        dst_ptrs = out_hid_ptr + b * hid_stride_b + (m_offs[:, None] - b * T) * hid_stride_i + tl.arange(0, H)[None, :] * hid_stride_h
        tl.store(dst_ptrs, vals, mask=mask_i[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along the sequence dimension using Triton.
        - Compute processed = concatenated @ process_weight.T using a Triton matmul kernel.
        - Split processed back into encoder and hidden streams using Triton.
        Returns: (processed_encoder [B, T, H], processed_hidden [B, I, H])
        """
        # Ensure tensors are on CUDA and contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Triton requires CUDA tensors"
        B, T, H = encoder_hidden_states.shape
        _, I, H2 = hidden_states.shape
        assert H == H2, "hidden_dim must match for encoder_hidden_states and hidden_states"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # 1) Concatenate sequences into [B, T+I, H] using Triton
        out_cat = torch.empty((B, T + I, H), dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        _concatenate_sequences_kernel[(B,)](
            enc, hid, out_cat,
            B, T, I, H,
            *enc.stride(), *hid.stride(),
            *out_cat.stride(),
            BLOCK_L=1024,
            num_warps=1, num_stages=1,
        )

        # 2) Compute processed = out_cat @ process_weight.T using Triton GEMM
        M = B * (T + I)
        K = H
        # A is out_cat [M, K], W is process_weight [H, H], right-multiply by W^T
        A = out_cat.reshape(M, K).contiguous()
        W = process_weight  # [H, H]
        # Allocate output C [M, H]
        C = torch.empty((M, H), dtype=torch.float32, device=A.device)  # compute in fp32 for stability

        # Choose tile sizes; tune as needed
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid_m = triton.cdiv(M, BLOCK_M)
        grid_n = triton.cdiv(H, BLOCK_N)

        _batched_gemm_right_kernel[(grid_m, grid_n)](
            A, W, C,
            M, K, H,
            *A.stride(), *W.stride(),
            *C.stride(),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split C [B*(T+I), H] into encoder and hidden streams
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=C.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=C.device)

        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_H=256,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden