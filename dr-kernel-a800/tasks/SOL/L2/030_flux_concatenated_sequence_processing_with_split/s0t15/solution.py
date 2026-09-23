import torch
import triton
import triton.language as tl


@triton.jit
def copy_d_dim_kernel(
    src_ptr,            # *source [B, S, D]
    dst_ptr,            # *destination [B, S, D]
    B: tl.int32,
    S: tl.int32,
    D: tl.int32,
    src_stride_b: tl.int32,
    src_stride_s: tl.int32,
    src_stride_d: tl.int32,
    dst_stride_b: tl.int32,
    dst_stride_s: tl.int32,
    dst_stride_d: tl.int32,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, S, ceil(D / BLOCK_D))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d_block = tl.program_id(2)

    d_offsets = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    src_b_ptr = src_ptr + pid_b * src_stride_b
    dst_b_ptr = dst_ptr + pid_b * dst_stride_b

    vals = tl.load(
        src_b_ptr + pid_s * src_stride_s + d_offsets * src_stride_d,
        mask=mask_d,
        other=0.0
    )
    tl.store(
        dst_b_ptr + pid_s * dst_stride_s + d_offsets * dst_stride_d,
        vals,
        mask=mask_d
    )


@triton.jit
def split_copy_kernel(
    src_ptr,            # *processed [B, L_txt + L_img, D]
    out1_ptr,           # *processed_encoder [B, L_txt, D]
    out2_ptr,           # *processed_hidden [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    src_stride_b: tl.int32,
    src_stride_s: tl.int32,
    src_stride_d: tl.int32,
    out1_stride_b: tl.int32, out1_stride_s: tl.int32, out1_stride_d: tl.int32,
    out2_stride_b: tl.int32, out2_stride_s: tl.int32, out2_stride_d: tl.int32,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, L_txt, ceil(D / BLOCK_D)) for out1 and (B, L_img, ceil(D / BLOCK_D)) for out2
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d_block = tl.program_id(2)

    d_offsets = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    src_b_ptr = src_ptr + pid_b * src_stride_b
    out1_b_ptr = out1_ptr + pid_b * out1_stride_b
    out2_b_ptr = out2_ptr + pid_b * out2_stride_b

    # Copy encoder part
    vals = tl.load(
        src_b_ptr + pid_s * src_stride_s + d_offsets * src_stride_d,
        mask=mask_d,
        other=0.0
    )
    tl.store(
        out1_b_ptr + pid_s * out1_stride_s + d_offsets * out1_stride_d,
        vals,
        mask=mask_d
    )
    # Copy hidden part
    vals2 = tl.load(
        src_b_ptr + (pid_s + L_txt) * src_stride_s + d_offsets * src_stride_d,
        mask=mask_d,
        other=0.0
    )
    tl.store(
        out2_b_ptr + pid_s * out2_stride_s + d_offsets * out2_stride_d,
        vals2,
        mask=mask_d
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate along sequence dim using Triton copy kernels (no torch.cat).
        - Matmul using torch.matmul for robustness and speed.
        - Split outputs back using Triton copy kernel.
        """
        # Shapes
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert D == encoder_hidden_states.shape[2], "hidden_dim must match across inputs"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # 1) Concatenate along sequence dim: [B, L_txt + L_img, D] without torch.cat
        S_total = L_txt + L_img
        dst = torch.empty((B, S_total, D), dtype=hidden_states.dtype, device=hidden_states.device)

        # Prepare strides
        ehs = encoder_hidden_states
        hs = hidden_states
        ehs_stride_b, ehs_stride_s, ehs_stride_d = ehs.stride(0), ehs.stride(1), ehs.stride(2)
        hs_stride_b, hs_stride_s, hs_stride_d = hs.stride(0), hs.stride(1), hs.stride(2)
        dst_stride_b, dst_stride_s, dst_stride_d = dst.stride(0), dst.stride(1), dst.stride(2)

        # Copy encoder_hidden_states into dst[:, :L_txt, :]
        BLOCK_D = 128
        grid_ehs = (B, L_txt, triton.cdiv(D, BLOCK_D))
        copy_d_dim_kernel[grid_ehs](
            ehs, dst, B, L_txt, D,
            ehs_stride_b, ehs_stride_s, ehs_stride_d,
            dst_stride_b, dst_stride_s, dst_stride_d,
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2
        )

        # Copy hidden_states into dst[:, L_txt:, :]
        grid_hs = (B, L_img, triton.cdiv(D, BLOCK_D))
        copy_d_dim_kernel[grid_hs](
            hs, dst, B, L_img, D,
            hs_stride_b, hs_stride_s, hs_stride_d,
            dst_stride_b, dst_stride_s * L_txt, dst_stride_d,
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2
        )

        # 2) Apply linear projection using torch for robustness and speed
        # process_weight is [D, D]; concatenated [B, S_total, D] @ process_weight.T -> [B, S_total, D]
        processed = torch.matmul(dst, process_weight.t())

        # 3) Split back using Triton kernel: out1 [B, L_txt, D], out2 [B, L_img, D]
        out1 = torch.empty((B, L_txt, D), dtype=processed.dtype, device=processed.device)
        out2 = torch.empty((B, L_img, D), dtype=processed.dtype, device=processed.device)

        split_copy_kernel[(B, L_txt, triton.cdiv(D, BLOCK_D))](
            processed, out1, out2, B, L_txt, L_img, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            out1.stride(0), out1.stride(1), out1.stride(2),
            out2.stride(0), out2.stride(1), out2.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2
        )

        return out1, out2


def run(*args):
    return ModelNew()(*args)
