import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(enc_ptr, hid_ptr, out_ptr,
                       B, T, I, H,
                       BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr):
    # grid = (B,)
    b = tl.program_id(0)

    # Rows for encoder part: [0, T)
    t_rows = tl.arange(0, BLOCK_T)
    mask_t = t_rows < T
    src_enc = enc_ptr + b * H + t_rows * H  # enc[b, t, :]
    dst_t = out_ptr + b * H + t_rows * H    # out[b, t, :]
    tl.store(dst_t, tl.load(src_enc, mask=mask_t, other=0.0))

    # Rows for hidden part: [T, T+I)
    i_rows = tl.arange(0, BLOCK_I)
    mask_i = i_rows < I
    src_hid = hid_ptr + b * H + i_rows * H  # hid[b, i, :]
    dst_i = out_ptr + (b * S) * H + (t_rows + T) * H
    tl.store(dst_i, tl.load(src_hid, mask=mask_i, other=0.0))


@triton.jit
def matmul_row_kernel(A_ptr, B_ptr, C_ptr,
                      S, H,
                      BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # grid = (S, 1) where 1 corresponds to a single N tile covering H
    m = tl.program_id(0)  # row index in [0, S)
    n_offsets = tl.arange(0, BLOCK_N)  # columns, BLOCK_N must equal H for this workload

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, H, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + m * H + k_offsets, mask=k_offsets < H, other=0.0)  # [BLOCK_K]
        b = tl.load(B_ptr + k_offsets[:, None] + n_offsets[None, :],  # [BLOCK_K, BLOCK_N]
                    mask=(k_offsets[:, None] < H) & (n_offsets[None, :] < H),
                    other=0.0)
        # acc += a[:, None] * b
        acc += tl.sum(a[:, None] * b, axis=0)

    # Store result
    tl.store(C_ptr + m * H + n_offsets, acc, mask=n_offsets < H)


@triton.jit
def split_seqs_kernel(C_ptr, enc_out_ptr, hid_out_ptr,
                      B, T, I, H, S,
                      BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr):
    # grid = (B,)
    b = tl.program_id(0)

    # First T rows: encoder part
    t_rows = tl.arange(0, BLOCK_T)
    mask_t = t_rows < T
    src = C_ptr + b * H + t_rows * H
    dst = enc_out_ptr + b * H + t_rows * H
    tl.store(dst, tl.load(src, mask=mask_t, other=0.0))

    # Remaining I rows: hidden part
    i_rows = tl.arange(0, BLOCK_I)
    mask_i = i_rows < I
    src2 = C_ptr + (b * S) * H + (t_rows + T) * H
    dst2 = hid_out_ptr + b * H + i_rows * H
    tl.store(dst2, tl.load(src2, mask=mask_i, other=0.0))


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes: encoder_hidden_states [B, T, H], hidden_states [B, I, H], process_weight [H, H]
        assert encoder_hidden_states.dim() == 3 and hidden_states.dim() == 3, "Inputs must be [B, dim, H]"
        assert process_weight.dim() == 2, "process_weight must be [H, H]"
        assert encoder_hidden_states.dtype == hidden_states.dtype == process_weight.dtype, "All tensors must have same dtype"
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"

        B, T, H = encoder_hidden_states.shape
        _, I, _ = hidden_states.shape
        S = T + I

        # Ensure contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        Bw_T = process_weight.transpose(0, 1).contiguous()  # [H, H]
        device = enc.device

        # 1) Concatenate into [B, S, H]
        concatenated = torch.empty((B, S, H), device=device, dtype=enc.dtype)
        grid_concat = (B,)
        concat_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H,
            BLOCK_T=128, BLOCK_I=128,
            num_warps=4, num_stages=2,
        )

        # 2) Compute matmul: C = concatenated @ process_weight.T, shape [S, H]
        # Treat concatenated as [S, H] per batch
        A = concatenated.view(S, H).contiguous()
        C = torch.empty((S, H), device=device, dtype=enc.dtype)

        grid_m = (S,)
        grid_n = (1,)  # since BLOCK_N = H
        matmul_row_kernel[grid_m * grid_n](
            A, Bw_T, C,
            S, H,
            BLOCK_K=64, BLOCK_N=1024,  # BLOCK_N must equal H for this workload
            num_warps=4, num_stages=2,
        )

        # 3) Split C into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=enc.dtype)

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            BLOCK_T=128, BLOCK_I=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
