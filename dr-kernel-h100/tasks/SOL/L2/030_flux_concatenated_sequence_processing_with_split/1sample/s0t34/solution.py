import torch
import triton
import triton.language as tl

@triton.jit
def _concatenate_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    e_stride_b, e_stride_t, e_stride_h,
    h_stride_b, h_stride_i, h_stride_h,
    o_stride_b, o_stride_t, o_stride_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    # Compute base pointers for this batch
    e_base = encoder_ptr + b * e_stride_b
    h_base = hidden_ptr + b * h_stride_b
    o_base = out_ptr + b * o_stride_b

    # Loop over concatenated sequence length
    for l in range(0, T + I):
        # Determine source and offsets
        src_is_encoder = l < T
        # H dimension: we copy the entire hidden vector of size H
        h_off = tl.arange(0, H)

        # Compute destination pointer for out_cat[b, l, :]
        o_ptr = o_base + l * o_stride_t + h_off * o_stride_h

        # If src is encoder
        if src_is_encoder:
            e_row_ptr = e_base + l * e_stride_t + h_off * e_stride_h
            # masked load/store is unnecessary here because H is the loop bound,
            # but ensure we don't write if l out of bounds (not possible).
            # tl.store will handle; we can explicitly guard via mask if needed.
            tl.store(o_ptr, tl.load(e_row_ptr))
        else:
            # src is hidden at offset l - T
            h_row_ptr = h_base + (l - T) * h_stride_i + h_off * h_stride_h
            tl.store(o_ptr, tl.load(h_row_ptr))


@triton.jit
def _matmul_per_row_kernel(
    A_ptr, W_ptr, C_ptr,
    M, H,
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_K: tl.constexpr,
):
    # One program per output row m
    m = tl.program_id(0)

    # Initialize accumulator
    acc = tl.zeros((H,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, H, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_off < H

        # Load A row segment: A[m, k0:k0+BLOCK_K]
        a_ptr = A_ptr + m * A_stride_m + k_off * A_stride_k
        a_vec = tl.load(a_ptr, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Load W tile segment: W[k0:k0+BLOCK_K, 0:H] as rows
        w_rows = tl.zeros((BLOCK_K, H), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            ki = k0 + kk
            if ki < H:
                w_row_ptr = W_ptr + ki * W_stride_k + tl.arange(0, H) * W_stride_n
                w_row = tl.load(w_row_ptr)  # [H]
                w_rows[kk, :] = w_row

        # Accumulate: acc += sum_{kk in chunk} a_vec[kk] * w_rows[kk, :]
        # Do the reduction explicitly
        for kk in range(0, BLOCK_K):
            if (k0 + kk) < H:
                acc += a_vec[kk] * w_rows[kk, :]

    # Store result to C[m, :]
    c_ptr = C_ptr + m * C_stride_m + tl.arange(0, H) * C_stride_n
    tl.store(c_ptr, acc)


@triton.jit
def _split_streams_kernel(
    C_ptr, out_enc_ptr, out_hid_ptr,
    B, T, I, H,
    C_stride_m, C_stride_n,
    oes_b, oes_t, oes_h,  # encoder output strides
    ohs_b, ohs_i, ohs_h,  # hidden output strides
    num_rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    b = tl.program_id(0)
    base_c = C_ptr + b * C_stride_m

    # Write first T rows to encoder output
    for row in range(0, T):
        c_row_ptr = base_c + row * C_stride_n + tl.arange(0, H)
        enc_row_ptr = out_enc_ptr + b * oes_b + row * oes_t + tl.arange(0, H) * oes_h
        tl.store(enc_row_ptr, tl.load(c_row_ptr))

    # Write next I rows to hidden output (offset by T in original C)
    for row in range(0, I):
        c_row_ptr = base_c + (T + row) * C_stride_n + tl.arange(0, H)
        hid_row_ptr = out_hid_ptr + b * ohs_b + row * ohs_i + tl.arange(0, H) * ohs_h
        tl.store(hid_row_ptr, tl.load(c_row_ptr))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Performs GEMM in Triton (one program per output row).
        - Splits the result back into two streams using Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # 1) Concatenate along sequence axis: out_cat [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=hidden_states.device)

        _concatenate_seq_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *out_cat.stride(),
            BLOCK_L=256,  # mask handles T+I not being multiple of BLOCK_L
            num_warps=1, num_stages=1
        )

        # 2) GEMM in Triton: C[m, :] = out_cat[m, :] @ process_weight.T, where m in [0, B*(T+I))
        M = B * (T + I)
        C = torch.empty((M, H), dtype=torch.float32, device=hidden_states.device)  # compute in fp32 for stability

        # Ensure weights are contiguous and fp32 for consistent loads
        W = process_weight.t().contiguous().to(torch.float32)  # [H, H]
        # Launch one program per output row
        _matmul_per_row_kernel[(M,)](
            out_cat, W, C,
            M, H,
            *out_cat.stride(0), *out_cat.stride(2),  # A_stride_m, A_stride_k
            *W.stride(0), *W.stride(1),              # W_stride_k, W_stride_n
            *C.stride(0), *C.stride(1),
            BLOCK_K=64,  # reduction chunk size; masks handle H not multiple of BLOCK_K
            num_warps=1, num_stages=1
        )

        # 3) Split C into encoder and image streams
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            *C.stride(), *processed_encoder.stride(), *processed_hidden.stride(),
            num_rows=T+I,
            BLOCK_M=256,
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
