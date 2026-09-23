import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seqlen_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    e_stride_b, e_stride_t, e_stride_h,
    h_stride_b, h_stride_i, h_stride_h,
    o_stride_b, o_stride_l, o_stride_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    # bounds check on batch
    if b >= B:
        return
    # iterate over concatenated sequence length
    for l in range(0, T + I):
        mask_l = l < (T + I)
        # choose source
        if l < T:
            ptr = encoder_ptr + b * e_stride_b + l * e_stride_t
        else:
            src_l = l - T
            ptr = hidden_ptr + b * h_stride_b + src_l * h_stride_i
        # load row [H]
        offs_h = tl.arange(0, H)
        mask_h = offs_h < H
        row = tl.load(ptr + offs_h * e_stride_h, mask=mask_h & mask_l, other=0.0)
        # store to out_cat[b, l, :]
        out_row_ptr = out_ptr + b * o_stride_b + l * o_stride_l
        tl.store(out_row_ptr + offs_h * o_stride_h, row, mask=mask_h & mask_l)


@triton.jit
def _batched_gemm_right_kernel_row(
    A_ptr, W_ptr, C_ptr,
    M, H,
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_m, C_stride_n,
):
    m = tl.program_id(0)
    if m >= M:
        return
    # accumulate across K
    acc = tl.zeros((H,), dtype=tl.float32)
    for k in range(0, H):
        a_row_ptr = A_ptr + m * A_stride_m + k * A_stride_k
        a = tl.load(a_row_ptr)
        w_ptr = W_ptr + k * W_stride_k + tl.arange(0, H) * W_stride_n
        w = tl.load(w_ptr)
        acc += a * w
    # store acc to C[m, :]
    c_row_ptr = C_ptr + m * C_stride_m
    tl.store(c_row_ptr + tl.arange(0, H) * C_stride_n, acc)


@triton.jit
def _split_streams_kernel(
    C_ptr, enc_ptr, hid_ptr,
    M, T, I, H,
    C_stride_m, C_stride_n,
    e_stride_b, e_stride_t, e_stride_h,
    h_stride_b, h_stride_i, h_stride_h,
):
    b = tl.program_id(0)
    if b >= B:
        return
    # copy rows 0..T-1 to encoder
    for m in range(0, T):
        ptr = C_ptr + m * C_stride_m
        row = tl.load(ptr + tl.arange(0, H) * C_stride_n)
        out_ptr = enc_ptr + b * e_stride_b + m * e_stride_t
        tl.store(out_ptr + tl.arange(0, H) * e_stride_h, row)
    # copy rows T..T+I-1 to hidden
    for m in range(0, I):
        src_m = T + m
        ptr = C_ptr + src_m * C_stride_m
        row = tl.load(ptr + tl.arange(0, H) * C_stride_n)
        out_ptr = hid_ptr + b * h_stride_b + m * h_stride_i
        tl.store(out_ptr + tl.arange(0, H) * h_stride_h, row)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure dtype and device
        device = process_weight.device
        dtype = process_weight.dtype  # assume fp32; if not, cast
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.to(dtype)
        if encoder_hidden_states.dtype != dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype)

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # 1) Concatenate along sequence dimension using Triton
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)
        _concatenate_seqlen_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *out_cat.stride(),
            BLOCK_L=256,
            num_warps=1,
            num_stages=1,
        )
        # out_cat: [B, T+I, H], contiguous along H by construction

        # 2) GEMM in Triton: C = out_cat @ process_weight.T, shape [B*(T+I), H]
        M = B * (T + I)
        A = out_cat  # [M, H] via flattening implied by pointer strides
        W = process_weight  # [H, H]
        C = torch.empty((M, H), dtype=dtype, device=device)

        # One program per output row
        _batched_gemm_right_kernel_row[(M,)](
            A, W, C,
            M, H,
            *A.stride(), *W.stride(), *C.stride(),
            num_warps=1,
            num_stages=1,
        )

        # 3) Split into two outputs using Triton
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)

        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            M, T, I, H,
            *C.stride(), *processed_encoder.stride(), *processed_hidden.stride(),
            num_warps=1,
            num_stages=1,
        )

        return processed_encoder, processed_hidden