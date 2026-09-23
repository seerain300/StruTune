import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,       # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,        # *hidden_states [B, L_img, D]
    out_ptr,       # *output [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    out_stride_b: tl.int32, out_stride_s: tl.int32, out_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # 3D grid: (B, tiles over sequence, tiles over D)
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    M = L_txt + L_img
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < M
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Base pointers for this batch
    ehs_base = ehs_ptr + pid_b * ehs_stride_b
    hs_base = hs_ptr + pid_b * hs_stride_b
    out_base = out_ptr + pid_b * out_stride_b

    # Masks for source tensors
    mask_e = mask_s[:, None] & (offs_s[:, None] < L_txt) & mask_d[None, :]
    mask_h = mask_s[:, None] & (offs_s[:, None] >= L_txt) & mask_d[None, :]

    # Load encoder part and store to output at positions s
    e_ptrs = ehs_base + offs_s[:, None] * ehs_stride_s + offs_d[None, :] * ehs_stride_d
    val_e = tl.load(e_ptrs, mask=mask_e, other=0.0)
    out_ptrs = out_base + offs_s[:, None] * out_stride_s + offs_d[None, :] * out_stride_d
    tl.store(out_ptrs, val_e, mask=mask_s[:, None] & mask_d[None, :])

    # Load hidden part and store to output at positions s + L_txt
    h_ptrs = hs_base + (offs_s[:, None] - L_txt) * hs_stride_s + offs_d[None, :] * hs_stride_d
    val_h = tl.load(h_ptrs, mask=mask_h, other=0.0)
    out_h_ptrs = out_base + (offs_s[:, None] + L_txt) * out_stride_s + offs_d[None, :] * out_stride_d
    tl.store(out_h_ptrs, val_h, mask=mask_s[:, None] & mask_d[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton implementation:
        - Concatenate [B, L_txt, D] and [B, L_img, D] along seq dim in Triton.
        - Apply linear projection using torch.matmul (GPU).
        - Return processed_encoder and processed_hidden.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA for Triton kernels."

        # Shapes
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguous
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        # process_weight: [D, D]; we need A @ process_weight.T, so we pass process_weight.T
        process_weight_T = process_weight.t().contiguous()  # [D, D]

        # Allocate concatenated [B, L_txt + L_img, D]
        M = L_txt + L_img
        A = torch.empty((B, M, D), device=ehs.device, dtype=ehs.dtype)

        # Launch Triton concat kernel
        BLOCK_S = 128
        BLOCK_D = 128
        grid = (B, triton.cdiv(M, BLOCK_S), triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid](
            ehs, hs, A,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # Linear projection via torch (robust and fast)
        # processed [B, M, D] = A @ process_weight_T
        processed = torch.matmul(A, process_weight_T)  # [B, M, D]

        # Split into two streams using torch slicing (reliable and fast)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
