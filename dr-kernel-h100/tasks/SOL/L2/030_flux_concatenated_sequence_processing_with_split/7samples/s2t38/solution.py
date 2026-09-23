import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,        # *const T, shape [B, T, H]
    hid_ptr,        # *const T, shape [B, I, H]
    out_ptr,        # *T, shape [B, S, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    out_stride_b, out_stride_s, out_stride_h,
):
    # Grid: (B, T+I) = (B, S)
    b = tl.program_id(0)
    m = tl.program_id(1)  # m in [0, S)

    # Compute source indices
    if m < T:
        t_idx = m
        src_ptr = enc_ptr + b * enc_stride_b + t_idx * enc_stride_t
    else:
        i_idx = m - T
        src_ptr = hid_ptr + b * hid_stride_b + i_idx * hid_stride_i

    # Compute destination pointer for out[b, m, :]
    dst_ptr = out_ptr + b * out_stride_b + m * out_stride_s

    # Copy H elements
    for n in range(0, H):
        val = tl.load(src_ptr + n * enc_stride_h)  # enc_stride_h is typically H, but we just index by n
        tl.store(dst_ptr + n * out_stride_h, val)


@triton.jit
def matmul_row_kernel(
    A_ptr,          # *const T, shape [B*S, H] represented as out_ptr with linear indexing
    BwT_ptr,        # *const T, shape [H, H] (process_weight.T)
    C_ptr,          # *T, shape [B, S, H]
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    A_stride_m, A_stride_k,          # A[m, k] strides (we'll pass 0 and H for simplicity; see forward)
    BwT_stride_k, BwT_stride_n,      # B[k, n] strides
    C_stride_b, C_stride_s, C_stride_h,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, S) => compute C[b, m, :]
    b = tl.program_id(0)
    m = tl.program_id(1)

    # Vector to hold output row [H]
    acc = tl.zeros((H,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, H, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        # Load A[m, k_ids] as a vector (m is row index in [0, B*S); here we map m -> b, m')
        # We pass A_stride_m and A_stride_k so that A_ptr + m*A_stride_m + k*A_stride_k works.
        # In forward, we'll pass appropriate strides for A viewed as [B*S, H].
        # For simplicity, we treat A as a flattened [B*S, H] tensor; BwT is [H, H].
        A_row_ptr = A_ptr + m * A_stride_m
        A_vec = tl.load(A_row_ptr + k_ids * A_stride_k, mask=k_ids < H, other=0.0)
        BwT_sub = tl.load(BwT_ptr + k_ids[:, None] * BwT_stride_k + tl.arange(0, H)[None, :] * BwT_stride_n,
                          mask=(k_ids[:, None] < H), other=0.0)
        # acc += A_vec[:, None] * BwT_sub[None, :]
        # Implement elementwise multiply and reduce over K:
        acc += tl.sum(A_vec[:, None] * BwT_sub, axis=0)

    # Store acc into C[b, m, :]
    C_row_ptr = C_ptr + b * C_stride_b + m * C_stride_s
    for n in range(0, H):
        tl.store(C_row_ptr + n * C_stride_h, acc[n])


@triton.jit
def split_seqs_kernel(
    C_ptr,          # *const T, shape [B, S, H]
    out_e_ptr,      # *T, shape [B, T, H]
    out_i_ptr,      # *T, shape [B, I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    C_stride_b, C_stride_s, C_stride_h,
    out_e_stride_b, out_e_stride_t, out_e_stride_h,
    out_i_stride_b, out_i_stride_i, out_i_stride_h,
):
    b = tl.program_id(0)

    # Copy first T rows to out_e
    for m in range(0, T):
        src_ptr = C_ptr + b * C_stride_b + m * C_stride_s
        dst_ptr = out_e_ptr + b * out_e_stride_b + m * out_e_stride_t
        for n in range(0, H):
            val = tl.load(src_ptr + n * C_stride_h)
            tl.store(dst_ptr + n * out_e_stride_h, val)

    # Copy remaining I rows to out_i
    for m in range(0, I):
        src_ptr = C_ptr + b * C_stride_b + (T + m) * C_stride_s
        dst_ptr = out_i_ptr + b * out_i_stride_b + m * out_i_stride_i
        for n in range(0, H):
            val = tl.load(src_ptr + n * C_stride_h)
            tl.store(dst_ptr + n * out_i_stride_h, val)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of:
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
        processed = concatenated @ process_weight.T
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]
        Returns: processed_encoder, processed_hidden
        """
        # Ensure on CUDA and contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton kernels."
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        Bw = process_weight.contiguous()  # [H, H]
        B, T, H = enc.shape
        B2, I, H2 = hid.shape
        assert B == B2, "Batch sizes must match"
        assert H == H2, "Hidden sizes must match"
        S = T + I

        # Allocate outputs
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)
        C = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)

        # 1) Concatenate in Triton
        grid_concat = (B, S)
        concat_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Compute matmul per row in Triton: C = concatenated @ process_weight.T
        # We need to pass strides for A viewed as [B*S, H]. Here, m in [0, B*S) where S=T+I.
        # To avoid complex mapping, we launch grid=(B, S) and treat each (b, m) row of concatenated.
        # We need A[m, :] pointer. For simplicity, we reconstruct A linearly from concatenated:
        # However, since we already have concatenated, we can directly pass concatenated as A with
        # A_stride_m = H and A_stride_k = 1. But concatenated is [B, S, H] with strides (S*H, H, 1).
        # We will use A_ptr pointing to concatenated as a flattened [B*S, H] by mapping m->b and m'.
        # In Triton, we can't easily index a 3D tensor like that, so instead we compute per-row using:
        # For each (b, m), m < T: row from enc[b, m, :], else from hid[b, m-T, :]. But we already
        # concatenated, so we can simply read from concatenated: A[m] = concatenated[b, m, :].
        # Therefore, A_ptr = concatenated, A_stride_m = concatenated.stride(1) * concatenated.stride(0) is not correct.
        # Better: We will treat concatenated as a 2D flattened A with rows m in [0, B*S):
        # But Triton kernels don't support dynamic 3D indexing like that. Instead, we can pass A as a flattened
        # [B*S, H] view using .view, but we still need strides.
        # To keep it simple and correct: recompute per-row by copying from concatenated in matmul kernel via pointer math.
        # We'll pass A_ptr = concatenated, A_stride_m = concatenated.stride(1) * 0 + concatenated.stride(2) doesn't help.
        # Simpler approach: since we already concatenated, for each (b, m), we can load concatenated[b, m, :]
        # into A_row via pointer math. Triton supports loading from 3D pointer if we pass appropriate strides.
        # So, we will set A_ptr = concatenated and in kernel, for each m, load row as A_row_ptr = concatenated + b*S*H + m*H + k.
        # This requires us to pass b and m, and we can compute pointer as follows:
        # We cannot directly pass b inside kernel; instead we compute in kernel using tl.program_id(0) and tl.program_id(1).
        # We need to map m to concatenated[b, m, :]. We can do that by passing B and S and using b=pid0, m=pid1.
        # Then A_row_ptr = concatenated + b * (S*H) + m * H. But concatenated shape is [B, S, H], so address is
        # concatenated + b*(S*H) + m*H + k, where k in [0, H).

        # Launch matmul row-wise kernel over (B, S)
        grid_matmul = (B, S)
        # Important: For A_ptr, we want each program to read row m of concatenated[b, m, :].
        # Since Triton will iterate m in [0, S), we can set A_row_ptr = concatenated + b * (S*H) + m * H
        # but concatenated is [B, S, H] so address is b*(S*H) + m*H + k. We can pass A_ptr = concatenated,
        # and in kernel use (b = tl.program_id(0), m = tl.program_id(1)), then:
        # A_row_ptr = concatenated + b * concatenated.stride(0) + m * concatenated.stride(1)
        # Here concatenated.stride(0)=S*H, concatenated.stride(1)=H, concatenated.stride(2)=1.
        matmul_row_kernel[grid_matmul](
            concatenated, Bw, C,
            B, S, H,
            concatenated.stride(0), concatenated.stride(2),     # A[m, k] strides: row stride (B*S*H), col stride (1)
            Bw.stride(0), Bw.stride(1),                         # B[k, n] strides
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split in Triton
        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
