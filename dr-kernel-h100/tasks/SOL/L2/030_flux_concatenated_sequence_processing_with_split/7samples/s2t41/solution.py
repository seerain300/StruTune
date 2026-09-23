import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,         # *float, [B, T, H]
    hid_ptr,         # *float, [B, I, H]
    concat_ptr,      # *float, [B, S, H], S = T + I
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    enc_stride0, enc_stride1, enc_stride2,
    hid_stride0, hid_stride1, hid_stride2,
    concat_stride0, concat_stride1, concat_stride2,
):
    # Each program copies one row from enc/hid into concat for a fixed batch
    b = tl.program_id(0)
    if b >= B:
        return
    # Copy encoder rows
    for t in range(0, T):
        src = enc_ptr + b * enc_stride0 + t * enc_stride1
        dst = concat_ptr + b * concat_stride0 + t * concat_stride1
        # H is known; we can load/store contiguous along H
        for j in range(0, H):
            val = tl.load(src + j * enc_stride2)
            tl.store(dst + j * concat_stride2, val)
    # Copy hidden rows
    for i in range(0, I):
        src = hid_ptr + b * hid_stride0 + i * hid_stride1
        dst = concat_ptr + b * concat_stride0 + (T + i) * concat_stride1
        for j in range(0, H):
            val = tl.load(src + j * hid_stride2)
            tl.store(dst + j * concat_stride2, val)


@triton.jit
def matmul_row_kernel(
    A_ptr,           # *float, [S, H], row-major contiguous
    B_ptr,           # *float, [H, H], row-major contiguous (process_weight.T)
    C_ptr,           # *float, [S, H], row-major contiguous
    S: tl.constexpr,  # total rows
    H: tl.constexpr,  # feature dim
    A_stride0, A_stride1,
    B_stride0, B_stride1,
    C_stride0, C_stride1,
    BLOCK_K: tl.constexpr = 64,
):
    pid_m = tl.program_id(0)  # row index in [0, S)
    if pid_m >= S:
        return
    # Initialize accumulator
    n = 0
    acc = tl.zeros((1,), dtype=tl.float32)
    # Loop over K dimension in chunks
    for k in range(0, H, BLOCK_K):
        # For a single row, we compute:
        # acc = sum_{kk=0..H-1} A[pid_m, kk] * B[kk, :]
        # But since A is [S, H] row, we load A_row and multiply with B_tile
        # However, Triton expects 2D matmul-like layout. Here we implement per-row loop.
        # Compute the row A[pid_m, :]
        a_row = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            kk_abs = k + kk
            if kk_abs < H:
                a_val = tl.load(A_ptr + pid_m * A_stride0 + kk_abs * A_stride1)
                # B_tile: B[kk_abs, :]
                b_tile = tl.load(B_ptr + kk_abs * B_stride0 + tl.arange(0, BLOCK_K) * B_stride1)
                # a_val is scalar; broadcast to vector
                acc += a_val * b_tile
    # Store acc to C[pid_m, :]
    for j in range(0, H):
        tl.store(C_ptr + pid_m * C_stride0 + j * C_stride1, acc[0])


@triton.jit
def split_seqs_kernel(
    C_ptr,           # *float, [B, S, H]
    out_encoder_ptr, # *float, [B, T, H]
    out_hidden_ptr,  # *float, [B, I, H]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    C_stride0, C_stride1, C_stride2,
    out_encoder_stride0, out_encoder_stride1, out_encoder_stride2,
    out_hidden_stride0, out_hidden_stride1, out_hidden_stride2,
):
    b = tl.program_id(0)
    if b >= B:
        return
    # Copy first T rows to encoder output
    for t in range(0, T):
        src = C_ptr + b * C_stride0 + t * C_stride1
        dst_e = out_encoder_ptr + b * out_encoder_stride0 + t * out_encoder_stride1
        for j in range(0, H):
            val = tl.load(src + j * C_stride2)
            tl.store(dst_e + j * out_encoder_stride2, val)
    # Copy remaining I rows to hidden output
    for i in range(0, I):
        src = C_ptr + b * C_stride0 + (T + i) * C_stride1
        dst_h = out_hidden_ptr + b * out_hidden_stride0 + i * out_hidden_stride1
        for j in range(0, H):
            val = tl.load(src + j * C_stride2)
            tl.store(dst_h + j * out_hidden_stride2, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension
        - Apply linear projection (matmul with process_weight.T)
        - Split back into separate encoder and image streams
        """
        # Ensure CUDA and float32
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32"

        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H

        # Allocate concatenated [B, S, H]
        S = T + I
        concatenated = torch.empty((B, S, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)

        # 1) Concatenate in Triton
        grid_concat = (B,)
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: C = concatenated @ process_weight.T
        # Use A as concatenated [S, H], B as process_weight.T [H, H], C as [S, H]
        # For robustness, we make A and B contiguous row-major
        A = concatenated  # already contiguous in last dim; A_stride1=1
        Bw_T = process_weight.transpose(0, 1).contiguous()  # [H, H]
        C = torch.empty((S, H), device=A.device, dtype=A.dtype)

        grid_matmul = (S,)
        matmul_row_kernel[grid_matmul](
            A, Bw_T, C,
            S, H,
            A.stride(0), A.stride(1),
            Bw_T.stride(0), Bw_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split C into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        processed_hidden = torch.empty((B, I, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
