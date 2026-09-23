import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seqs_kernel(
    enc_ptr,  # *float or *half
    hid_ptr,  # *float or *half
    out_ptr,  # *float or *half
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_hb, stride_hi, stride_hh,
    stride_ob, stride_ot, stride_oh,
    BLOCK_L: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)
    # Loop over the concatenated sequence length
    for l in range(0, T + I, BLOCK_L):
        offs = l + tl.arange(0, BLOCK_L)
        valid = offs < (T + I)

        # Pointers for encoder or hidden depending on offs
        # enc_idx = offs[offs < T], hid_idx = offs[offs >= T]
        # We build two masks: mask_e and mask_h
        mask_e = valid & (offs < T)
        mask_h = valid & (offs >= T)

        # Compute pointer for encoder part
        enc_row_ptr = enc_ptr + b * stride_eb + offs * 0  # offs here is scalar indices
        # Note: offs is 1D; Triton will broadcast pointer arithmetic over vector offsets.
        # We need to load BLOCK_L elements, so we compute per-element pointers:
        # For simplicity, we iterate in Python loop and load/store per element.
        # However, Triton requires vectorized operations; use tl.load with mask:
        # We need to create pointer vectors for each element; since Triton doesn't support
        # vectorized pointer arithmetic in loop body cleanly, we use masked load with a base pointer.
        # Build a base pointer and then load with mask.
        base_enc = enc_ptr + b * stride_eb
        # Load with mask mask_e
        enc_vals = tl.load(base_enc + offs * stride_et, mask=mask_e, other=0)

        # Build base pointer for hidden
        base_hid = hid_ptr + b * stride_hb
        hid_vals = tl.load(base_hid + (offs - T) * stride_hi, mask=mask_h, other=0)

        # Combine encoder and hidden values: where offs<T use enc_vals, else use hid_vals
        # Triton doesn't support where on tensor types; we can use tl.where with boolean masks.
        vals = tl.where(mask_e, enc_vals, 0) + tl.where(mask_h, hid_vals, 0)

        # Store into out[b, l, :]
        out_row_ptr = out_ptr + b * stride_ob
        tl.store(out_row_ptr + (offs - 0) * stride_ot, vals, mask=valid)


@triton.jit
def _batched_gemm_kernel(
    A_ptr,  # *float or *half, shape [M, K], row-major contiguous
    W_ptr,  # *float or *half, shape [K, N] where N=K=H (right multiply), contiguous
    C_ptr,  # *float or *half, shape [M, N], row-major contiguous
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W tile as right-multiply: we want W[k, n] -> [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Cast to fp32 for accumulation
        a = a.to(tl.float32)
        w = w.to(tl.float32)

        # acc += a @ w
        acc += tl.dot(a, w)

    # Write back to C: [M, N]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store as original dtype by casting acc
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_streams_kernel(
    C_ptr,  # *float or *half, shape [M, H], row-major contiguous
    out_e_ptr,  # *float or *half, shape [B, T, H], row-major contiguous
    out_h_ptr,  # *float or *half, shape [B, I, H], row-major contiguous
    M, T, I, H,
    stride_cm, stride_cn,
    stride_eb, stride_et, stride_eh,
    stride_hb, stride_hi, stride_hh,
):
    # One program per batch
    b = tl.program_id(0)
    total = T + I
    for l in range(0, T):
        m = b * total + l
        src_row_ptr = C_ptr + m * stride_cm
        dst_e_ptr = out_e_ptr + b * stride_eb + l * stride_et
        vals = tl.load(src_row_ptr + tl.arange(0, H) * stride_cn, mask=True, other=0)
        tl.store(dst_e_ptr + tl.arange(0, H) * stride_eh, vals, mask=True)
    for l in range(0, I):
        m = b * total + (T + l)
        src_row_ptr = C_ptr + m * stride_cm
        dst_h_ptr = out_h_ptr + b * stride_hb + l * stride_hi
        vals = tl.load(src_row_ptr + tl.arange(0, H) * stride_cn, mask=True, other=0)
        tl.store(dst_h_ptr + tl.arange(0, H) * stride_hh, vals, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim using Triton.
        - Performs batched matmul (right multiply) using Triton.
        - Splits the result back into two streams using Triton.
        Returns (processed_encoder: [B, T, H], processed_hidden: [B, I, H]).
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3
        assert process_weight.dim() == 2 and process_weight.shape[1] == process_weight.shape[0], "process_weight must be square"
        B, I, H = hidden_states.shape
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H
        T = encoder_hidden_states.shape[1]
        assert process_weight.shape[1] == H, "process_weight second dim must match hidden_dim"

        device = hidden_states.device
        # Allocate concatenated tensor [B, T+I, H] and launch concat kernel
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=device)

        _concatenate_seqs_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *out_cat.stride(),
            BLOCK_L=256,
            num_warps=1,
            num_stages=1,
        )

        # Prepare A for GEMM: A = out_cat contiguous, shape [M, K] where M = B*(T+I), K=H
        A = out_cat.contiguous()
        M = B * (T + I)
        K = H
        N = H  # since process_weight is [H, H], right multiply

        # Allocate output C [M, H]
        C = torch.empty((M, H), dtype=out_cat.dtype, device=device)

        # W is process_weight [H, H], but we multiply from the right: need W_t = process_weight.t()
        # Using Triton, we pass W_t contiguous
        W_t = process_weight.t().contiguous()

        # Launch GEMM kernel with a 2D grid
        # Choose block sizes; 128x128x32 is a good default for many H up to 1024
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _batched_gemm_kernel[grid](
            A, W_t, C,
            M, N, K,
            A.stride(0), A.stride(1),
            W_t.stride(0), W_t.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Allocate outputs and split using Triton
        processed_encoder = torch.empty((B, T, H), dtype=out_cat.dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=out_cat.dtype, device=device)

        _split_streams_kernel[(B,)](
            C,
            processed_encoder, processed_hidden,
            M, T, I, H,
            C.stride(0), C.stride(1),
            *processed_encoder.stride(), *processed_hidden.stride(),
            num_warps=1,
            num_stages=1,
        )

        return processed_encoder, processed_hidden