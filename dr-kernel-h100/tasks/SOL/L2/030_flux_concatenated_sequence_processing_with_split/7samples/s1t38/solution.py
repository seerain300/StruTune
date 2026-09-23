import torch
import triton
import triton.language as tl


@triton.jit
def _fused_cat_matmul_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D]
    H_ptr,        # hidden_states: [B, I, D]
    W_ptr,        # process_weight.T: [D, D]
    Y_ptr,        # output processed: [B, T+I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    W_d0_stride, W_d1_stride,
    Y_b_stride, Y_t_stride, Y_d_stride,
):
    # program ids
    pid_b = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # row in [0, T+I)
    t_total = T + I
    # decide stream and source row
    stream = pid_m // t_total  # should be 0 for all pid_m < t_total
    pos = pid_m
    # if stream == 1, pos corresponds to hidden index: pos - T
    src_row = pos
    # Load x (input row) based on stream
    src_ptr = None
    if stream == 0:
        src_ptr = E_ptr + pid_b * E_b_stride + src_row * E_t_stride
    else:
        src_ptr = H_ptr + pid_b * H_b_stride + (src_row - T) * H_i_stride

    # Load w = W[src_row, :]
    w_ptr = W_ptr + src_row * W_d0_stride
    w_vec = tl.load(w_ptr + tl.arange(0, D) * W_d1_stride)  # length D vector

    # Accumulate y over D
    y = tl.zeros([D], dtype=tl.float32)
    # Iterate over D in tiles of BLOCK_D (but here we assume D known and loop fully)
    for d in range(0, D):
        x_val = tl.load(src_ptr + d * E_d_stride)  # load x[d]
        w_val = tl.load(w_ptr + d * W_d1_stride)   # w[d]
        y += x_val * w_val

    # Store y to output
    dst_ptr = Y_ptr + pid_b * Y_b_stride + pid_m * Y_t_stride
    tl.store(dst_ptr + tl.arange(0, D) * Y_d_stride, y)


@triton.jit
def _split_copy_kernel(
    Y_ptr,             # processed: [B, T+I, D]
    out0_ptr,          # processed_encoder: [B, T, D]
    out1_ptr,          # processed_hidden: [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_t_stride, Y_d_stride,
    out0_b_stride, out0_d_stride,
    out1_b_stride, out1_d_stride,
    BLOCK_D: tl.constexpr,
):
    # 3D grid: (batch, which stream, row in stream)
    pid_b = tl.program_id(0)
    stream = tl.program_id(1)  # 0 for encoder, 1 for hidden
    pid_row = tl.program_id(2)

    if stream == 0:
        # copy rows 0..T-1
        src_ptr = Y_ptr + pid_b * Y_b_stride + pid_row * Y_t_stride
        dst_ptr = out0_ptr + pid_b * out0_b_stride + pid_row * out0_d_stride
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        vals = tl.load(src_ptr + offs * Y_d_stride, mask=mask, other=0.0)
        tl.store(dst_ptr + offs * out0_d_stride, vals, mask=mask)
    else:
        # copy rows T..T+I-1
        src_ptr = Y_ptr + pid_b * Y_b_stride + (pid_row + T) * Y_t_stride
        dst_ptr = out1_ptr + pid_b * out1_b_stride + pid_row * out1_d_stride
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        vals = tl.load(src_ptr + offs * Y_d_stride, mask=mask, other=0.0)
        tl.store(dst_ptr + offs * out1_d_stride, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of:
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, D]
            processed = concatenated @ process_weight.T  # [B, T+I, D]
            processed_encoder = processed[:, :T, :]
            processed_hidden = processed[:, T:, :]
        All computation performed in Triton @triton.jit kernels.
        """
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D and process_weight.shape[1] == D and process_weight.shape[0] == D, "Mismatched hidden_dim"

        # Make contiguous for predictable strides
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()  # W is [D, D]; we'll use W.t() as input to kernel

        # Allocate output tensor [B, T+I, D]
        T_total = T + I
        Y = torch.empty((B, T_total, D), device=hidden_states.device, dtype=torch.float32)

        # Launch fused cat+matmul kernel: grid over (batch, T_total rows)
        grid = (B, T_total)
        _fused_cat_matmul_kernel[grid](
            E, H, W, Y,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            num_warps=4, num_stages=2,
        )

        # Split outputs using Triton copy kernel
        processed_encoder = torch.empty((B, T, D), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=hidden_states.device, dtype=torch.float32)

        grid_split = (B, 2, I)  # stream dimension is 2 (0 for encoder rows, 1 for hidden rows)
        _split_copy_kernel[grid_split](
            Y, processed_encoder, processed_hidden,
            B, T, I, D,
            Y.stride(0), Y.stride(1), Y.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_D=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
