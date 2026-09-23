import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(enc_ptr, hidden_ptr, concat_ptr,
                       B, T, I, H,
                       enc_stride_b, enc_stride_t, enc_stride_h,
                       hidden_stride_b, hidden_stride_i, hidden_stride_h,
                       concat_stride_b, concat_stride_s, concat_stride_h,
                       BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // (T + I)
    rem = pid % (T + I)
    if b >= B:
        return
    if rem < T:
        # write into concat[b, rem, :]
        # encoder row at (b, rem, :)
        for h in range(0, H, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            mask = offs < H
            enc_off = b * enc_stride_b + rem * enc_stride_t + offs * enc_stride_h
            c_off = b * concat_stride_b + rem * concat_stride_s + offs * concat_stride_h
            val = tl.load(enc_ptr + enc_off, mask=mask)
            tl.store(concat_ptr + c_off, val, mask=mask)
    else:
        i = rem - T
        # write into concat[b, T + i, :]
        # hidden row at (b, i, :)
        for h in range(0, H, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            mask = offs < H
            hid_off = b * hidden_stride_b + i * hidden_stride_i + offs * hidden_stride_h
            c_off = b * concat_stride_b + (T + i) * concat_stride_s + offs * concat_stride_h
            val = tl.load(hidden_ptr + hid_off, mask=mask)
            tl.store(concat_ptr + c_off, val, mask=mask)


@triton.jit
def matmul_row_kernel(A_ptr, BwT_ptr, C_ptr,
                      B, S, H,
                      A_stride_b, A_stride_s, A_stride_h,
                      BwT_stride_k, BwT_stride_h,
                      C_stride_s, C_stride_h,
                      BLOCK_K: tl.constexpr):
    # 2D grid: (row in S, 1)
    pid_s = tl.program_id(0)
    if pid_s >= S:
        return
    # Output row: C[pid_s, :]
    acc = tl.zeros((H,), dtype=tl.float32)
    for k in range(0, H, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        # A row: A[pid_s, kk] (kk enumerates H dimension)
        a_offs = pid_s * A_stride_s + kk * A_stride_h
        a_vals = tl.load(A_ptr + a_offs, mask=kk < H, other=0.0)  # [BLOCK_K]
        # BwT: [H, H], take columns kk and produce rows
        b_offs = kk[:, None] * BwT_stride_k + tl.arange(0, H)[None, :] * BwT_stride_h  # [BLOCK_K, H]
        b_vals = tl.load(BwT_ptr + b_offs, mask=(kk[:, None] < H), other=0.0)         # [BLOCK_K, H]
        # acc += sum_k a_vals[k] * b_vals[k, :]
        acc += tl.sum(a_vals[:, None] * b_vals, axis=0)
    c_offs = pid_s * C_stride_s + tl.arange(0, H) * C_stride_h
    tl.store(C_ptr + c_offs, acc, mask=tl.arange(0, H) < H)


@triton.jit
def split_seqs_kernel(C_ptr, out_encoder_ptr, out_hidden_ptr,
                      B, T, I, H, S,
                      C_stride_b, C_stride_s, C_stride_h,
                      out_encoder_stride_b, out_encoder_stride_t, out_encoder_stride_h,
                      out_hidden_stride_b, out_hidden_stride_i, out_hidden_stride_h,
                      BLOCK_H: tl.constexpr):
    b = tl.program_id(0)
    if b >= B:
        return
    # Copy first T rows into encoder output
    for t in range(0, T):
        for h in range(0, H, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            mask = offs < H
            c_off = b * C_stride_b + t * C_stride_s + offs * C_stride_h
            out_off = b * out_encoder_stride_b + t * out_encoder_stride_t + offs * out_encoder_stride_h
            val = tl.load(C_ptr + c_off, mask=mask, other=0.0)
            tl.store(out_encoder_ptr + out_off, val, mask=mask)
    # Copy remaining I rows into hidden output starting at s = T
    for i in range(0, I):
        s = T + i
        for h in range(0, H, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            mask = offs < H
            c_off = b * C_stride_b + s * C_stride_s + offs * C_stride_h
            out_off = b * out_hidden_stride_b + i * out_hidden_stride_i + offs * out_hidden_stride_h
            val = tl.load(C_ptr + c_off, mask=mask, other=0.0)
            tl.store(out_hidden_ptr + out_off, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies linear projection via matmul with process_weight.T.
        - Splits back into processed_encoder and processed_hidden.
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA device."
        assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32 and \
               process_weight.dtype == torch.float32, "Use float32 for consistency."
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        B, T, H = enc.shape
        _, I, H2 = hid.shape
        assert H == H2, "hidden_dim must match across tensors."
        S = T + I

        # 1) Concatenate along sequence dimension into [B, S, H]
        concat = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)
        # Grid: one program per (b, s)
        grid_concat = (B * S,)
        concat_seqs_kernel[grid_concat](
            enc, hid, concat,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            concat.stride(0), concat.stride(1), concat.stride(2),
            BLOCK_H=128,  # H is typically 1024, but we tile in the kernel loops
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: C[b, s, :] = concat[b, s, :] @ process_weight.T
        # We operate over rows s in 1D grid; since H=1024 in provided workloads, we can set a single tile.
        # To avoid partial tiles, we loop K over H in the kernel. We pass process_weight.T as [H, H].
        Bw_T = process_weight.t().contiguous()  # [H, H]
        C = torch.empty((B * S, H), device=enc.device, dtype=enc.dtype)
        grid_matmul = (B * S,)
        matmul_row_kernel[grid_matmul](
            concat, Bw_T, C,
            B, S, H,
            concat.stride(0), concat.stride(1), concat.stride(2),
            Bw_T.stride(0), Bw_T.stride(1),   # Bw_T is [H, H]
            C.stride(0), C.stride(1),
            BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into encoder and hidden streams
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)
        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
