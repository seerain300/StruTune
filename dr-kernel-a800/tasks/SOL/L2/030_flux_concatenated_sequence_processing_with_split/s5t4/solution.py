import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    e_ptr,            # *f32, encoder_hidden_states [B, Stext, H]
    h_ptr,            # *f32, hidden_states [B, Simg, H]
    out_ptr,          # *f32, concatenated [B, T, H], T = Stext + Simg
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    stride_e_b: tl.constexpr, stride_e_t: tl.constexpr, stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr, stride_h_t: tl.constexpr, stride_h_h: tl.constexpr,
    stride_out_b: tl.constexpr, stride_out_t: tl.constexpr, stride_out_h: tl.constexpr,
):
    # 2D grid over (B, T)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Determine source based on pid_t
    is_text = pid_t < Stext

    # Base pointers for this batch
    e_base = e_ptr + pid_b * stride_e_b
    h_base = h_ptr + pid_b * stride_h_b
    out_base = out_ptr + pid_b * stride_out_b

    # Copy one full row from encoder or hidden
    if is_text:
        src_base = e_base + pid_t * stride_e_t
        out_row_ptr = out_base + pid_t * stride_out_t
        for h in range(0, H):
            val = tl.load(src_base + h * stride_e_h)
            tl.store(out_row_ptr + h * stride_out_h, val)
    else:
        src_t = pid_t - Stext
        src_base = h_base + src_t * stride_h_t
        out_row_ptr = out_base + pid_t * stride_out_t
        for h in range(0, H):
            val = tl.load(src_base + h * stride_h_h)
            tl.store(out_row_ptr + h * stride_out_h, val)


@triton.jit
def batched_gemm_kernel(
    A_ptr,            # *f32, out_cat [B, T, H]
    B_ptr,            # *f32, process_weight_T [H, H]
    Out_ptr,          # *f32, output [B, T, H]
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    stride_A_b: tl.constexpr, stride_A_t: tl.constexpr, stride_A_h: tl.constexpr,
    stride_B_k: tl.constexpr, stride_B_h: tl.constexpr,
    stride_Out_b: tl.constexpr, stride_Out_t: tl.constexpr, stride_Out_h: tl.constexpr,
):
    # 2D grid over (B, T): each program computes full output vector for (b, t)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    out_row_ptr = Out_ptr + pid_b * stride_Out_b + pid_t * stride_Out_t

    # Accumulator for the entire output vector
    acc = tl.zeros((H,), dtype=tl.float32)

    # Loop over k = 0..H-1
    for k in range(0, H):
        # Load A[b, t, k]
        a_val = tl.load(A_ptr + pid_b * stride_A_b + pid_t * stride_A_t + k * stride_A_h)

        # Load B[k, :] and multiply: B is [H, H], we need column k
        b_vec = tl.load(B_ptr + k * stride_B_k + tl.arange(0, H) * stride_B_h)
        acc += a_val * b_vec

    # Store full acc
    for h in range(0, H):
        tl.store(out_row_ptr + h * stride_Out_h, acc[h])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension inside Triton.
        - Applies linear projection (process_weight.T) to the concatenated sequences inside Triton.
        - Returns split outputs: (processed_encoder, processed_hidden).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "Inputs must be CUDA tensors for Triton execution."

        # Shapes
        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # Ensure contiguity
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()

        # Concatenation output [B, T, H]
        out_cat = torch.empty((B, T, H), device=e.device, dtype=e.dtype)

        # Launch concat kernel: 2D grid (B, T)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            e, h, out_cat,
            B, Stext, Simg, H, T,
            e.stride(0), e.stride(1), e.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=4, num_stages=2,
        )

        # Prepare weight as [H, H], contiguous
        W_T = process_weight.transpose(0, 1).contiguous()

        # Output [B, T, H]
        out = torch.empty((B, T, H), device=e.device, dtype=e.dtype)

        # Launch GEMM kernel: 2D grid (B, T), each program computes full vector
        grid_gemm = (B, T)
        batched_gemm_kernel[grid_gemm](
            out_cat, W_T, out,
            B, T, H,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            W_T.stride(0), W_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=4, num_stages=2,
        )

        # Split outputs: return view slices (no compute)
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
