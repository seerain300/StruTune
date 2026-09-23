import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,   # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,    # *hidden_states [B, L_img, D]
    dst_ptr,   # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)  # tile along sequence dimension

    S_total = L_txt + L_img
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S_total
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # For positions s < L_txt, copy from encoder_hidden_states
    e_base = ehs_ptr + pid_b * ehs_stride_b
    e_ptrs = e_base + (offs_s)[:, None] * ehs_stride_s + offs_d[None, :] * ehs_stride_d
    e_mask = (offs_s[:, None] < L_txt) & mask_d[None, :]
    e_val = tl.load(e_ptrs, mask=e_mask, other=0.0)

    dst_e_base = dst_ptr + pid_b * dst_stride_b
    dst_e_ptrs = dst_e_base + offs_s[:, None] * dst_stride_s + offs_d[None, :] * dst_stride_d
    tl.store(dst_e_ptrs, e_val, mask=e_mask)

    # For positions s >= L_txt, copy from hidden_states at offset s - L_txt
    h_base = hs_ptr + pid_b * hs_stride_b
    h_ptrs = h_base + (offs_s - L_txt)[:, None] * hs_stride_s + offs_d[None, :] * hs_stride_d
    h_mask = mask_s[:, None] & mask_d[None, :]
    h_val = tl.load(h_ptrs, mask=h_mask, other=0.0)

    dst_h_base = dst_ptr + pid_b * dst_stride_b
    dst_h_ptrs = dst_h_base + offs_s[:, None] * dst_stride_s + offs_d[None, :] * dst_stride_d
    tl.store(dst_h_ptrs, h_val, mask=h_mask)


@triton.jit
def split_outputs_kernel(
    C_ptr,        # *processed [B, M, D], M = L_txt + L_img
    out_e_ptr,    # *processed_encoder [B, L_txt, D]
    out_h_ptr,    # *processed_hidden [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_d: tl.int32,
    out_e_stride_b: tl.int32, out_e_stride_s: tl.int32, out_e_stride_d: tl.int32,
    out_h_stride_b: tl.int32, out_h_stride_s: tl.int32, out_h_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)  # tile along sequence dimension

    # Encoder part: s in [0, L_txt)
    S_e = L_txt
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S_e
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    c_base = C_ptr + pid_b * C_stride_b
    c_ptrs = c_base + offs_s[:, None] * C_stride_m + offs_d[None, :] * C_stride_d
    val_e = tl.load(c_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    out_e_base = out_e_ptr + pid_b * out_e_stride_b
    out_e_ptrs = out_e_base + offs_s[:, None] * out_e_stride_s + offs_d[None, :] * out_e_stride_d
    tl.store(out_e_ptrs, val_e, mask=mask_s[:, None] & mask_d[None, :])

    # Hidden part: s in [L_txt, L_txt + L_img)
    S_h = L_img
    hs_base = C_ptr + pid_b * C_stride_b
    hs_ptrs = hs_base + (offs_s + L_txt)[:, None] * C_stride_m + offs_d[None, :] * C_stride_d
    val_h = tl.load(hs_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

    out_h_base = out_h_ptr + pid_b * out_h_stride_b
    out_h_ptrs = out_h_base + offs_s[:, None] * out_h_stride_s + offs_d[None, :] * out_h_stride_d
    tl.store(out_h_ptrs, val_h, mask=mask_s[:, None] & mask_d[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton implementation:
        - Concatenate [B, L_txt, D] and [B, L_img, D] along seq dim in Triton.
        - Apply linear projection using torch.matmul (GPU) for robustness.
        - Split outputs in Triton, casting to original hidden_states dtype.
        """
        assert hidden


def run(*args):
    return ModelNew()(*args)
