import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    src1_ptr,  # *ptr to encoder_hidden_states: [B, L_txt, D]
    src2_ptr,  # *ptr to hidden_states: [B, L_img, D]
    dst_ptr,   # *ptr to dst_concat: [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    stride_src1_b: tl.int32, stride_src1_t: tl.int32, stride_src1_d: tl.int32,
    stride_src2_b: tl.int32, stride_src2_i: tl.int32, stride_src2_d: tl.int32,
    stride_dst_b: tl.int32, stride_dst_s: tl.int32, stride_dst_d: tl.int32,
):
    pid_b = tl.program_id(0)
    # Copy src1 -> dst[:, :L_txt, :]
    for t in range(0, L_txt):
        src1_off = pid_b * stride_src1_b + t * stride_src1_t
        dst_off_prefix = pid_b * stride_dst_b + t * stride_dst_s
        for d in range(0, D):
            val = tl.load(src1_ptr + src1_off + d * stride_src1_d)
            tl.store(dst_ptr + dst_off_prefix + d * stride_dst_d, val)

    # Copy src2 -> dst[:, L_txt:, :]
    for i in range(0, L_img):
        src2_off = pid_b * stride_src2_b + i * stride_src2_i
        dst_off_suffix = pid_b * stride_dst_b + (L_txt + i) * stride_dst_s
        for d in range(0, D):
            val = tl.load(src2_ptr + src2_off + d * stride_src2_d)
            tl.store(dst_ptr + dst_off_suffix + d * stride_dst_d, val)


@triton.jit
def copy_seq_prefix_kernel(
    src_ptr,   # *ptr to processed: [B, L_txt + L_img, D]
    dst_ptr,   # *ptr to processed_encoder: [B, L_txt, D]
    B: tl.int32,
    L_txt: tl.int32,
    D: tl.int32,
    stride_src_b: tl.int32, stride_src_s: tl.int32, stride_src_d: tl.int32,
    stride_dst_b: tl.int32, stride_dst_t: tl.int32, stride_dst_d: tl.int32,
):
    pid_b = tl.program_id(0)
    for t in range(0, L_txt):
        src_off = pid_b * stride_src_b + t * stride_src_s
        dst_off = pid_b * stride_dst_b + t * stride_dst_t
        for d in range(0, D):
            val = tl.load(src_ptr + src_off + d * stride_src_d)
            tl.store(dst_ptr + dst_off + d * stride_dst_d, val)


@triton.jit
def copy_seq_suffix_kernel(
    src_ptr,   # *ptr to processed: [B, L_txt + L_img, D]
    dst_ptr,   # *ptr to processed_hidden: [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    stride_src_b: tl.int32, stride_src_s: tl.int32, stride_src_d: tl.int32,
    stride_dst_b: tl.int32, stride_dst_i: tl.int32, stride_dst_d: tl.int32,
):
    pid_b = tl.program_id(0)
    for i in range(0, L_img):
        src_off = pid_b * stride_src_b + (L_txt + i) * stride_src_s
        dst_off = pid_b * stride_dst_b + i * stride_dst_i
        for d in range(0, D):
            val = tl.load(src_ptr + src_off + d * stride_src_d)
            tl.store(dst_ptr + dst_off + d * stride_dst_d, val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenation of sequences via Triton kernel (no torch.cat).
        - Linear projection via torch.matmul for correctness.
        - Splitting into encoder and hidden streams via Triton copy kernels (no torch slicing).
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D
        assert process_weight.shape == (D, D)

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Allocate concatenated tensor [B, L_txt + L_img, D]
        L_total = L_txt + L_img
        dst_concat = torch.empty((B, L_total, D), device=device, dtype=dtype)

        # Launch Triton concatenation kernel with correct grid
        grid = (B, 1, 1)
        concat_seqs_kernel[grid](
            encoder_hidden_states, hidden_states, dst_concat,
            B, L_txt, L_img, D,
            *encoder_hidden_states.stride(),
            *hidden_states.stride(),
            *dst_concat.stride(),
        )

        # Linear projection: processed = concatenated @ process_weight.T
        # Use PyTorch for correctness and performance
        processed = torch.matmul(dst_concat, process_weight.t())

        # Allocate outputs
        processed_encoder = torch.empty((B, L_txt, D), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, L_img, D), device=device, dtype=dtype)

        # Launch Triton copy kernels to split
        copy_seq_prefix_kernel[grid](
            processed, processed_encoder,
            B, L_txt, D,
            *processed.stride(),
            *processed_encoder.stride(),
        )

        copy_seq_suffix_kernel[grid](
            processed, processed_hidden,
            B, L_txt, L_img, D,
            *processed.stride(),
            *processed_hidden.stride(),
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
