import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr, hs_ptr, out_ptr,
    B, L_txt, L_img, D,
    stride_eb, stride_es, stride_ed,
    stride_hb, stride_hs, stride_hd,
    stride_ob, stride_os, stride_od,
    BLOCK_N: tl.constexpr
):
    # program_id(0) iterates over batch, program_id(1) over segments: [0..L_txt) and [L_txt..L_txt+L_img)
    b = tl.program_id(0)
    seg = tl.program_id(1)
    # segment 0: encoder_hidden_states, segment 1: hidden_states
    if seg < L_txt:
        t = seg
        e_off = b * stride_eb + t * stride_es
        o_off = b * stride_ob + t * stride_os
        # copy vector of length D
        for d in range(0, D, BLOCK_N):
            cols = d + tl.arange(0, BLOCK_N)
            mask = cols < D
            vals = tl.load(ehs_ptr + e_off + cols * stride_ed, mask=mask, other=0.0)
            tl.store(out_ptr + o_off + cols * stride_od, vals, mask=mask)
    else:
        t = seg - L_txt
        h_off = b * stride_hb + t * stride_hs
        o_off = b * stride_ob + (t + L_txt) * stride_os
        for d in range(0, D, BLOCK_N):
            cols = d + tl.arange(0, BLOCK_N)
            mask = cols < D
            vals = tl.load(hs_ptr + h_off + cols * stride_hd, mask=mask, other=0.0)
            tl.store(out_ptr + o_off + cols * stride_od, vals, mask=mask)


@triton.jit
def matmul_kernel_single(
    A_ptr, W_ptr, C_ptr,
    B, M, D,
    stride_Ab, stride_Am, stride_Ad,
    stride_Wk, stride_Wn,  # W is [D, D], we address as [k, n]
    stride_Cb, stride_Cm, stride_Cn,
    BLOCK_K: tl.constexpr
):
    # Each program computes one output element C[b, m, n]
    pid = tl.program_id(0)
    # Map pid -> (b, m, n)
    # We assume grid size B * M * D; so b = pid // (M * D); m = (pid % (M * D)) // D; n = pid % D
    tmp = pid // D
    b = tmp // M
    m = tmp % M
    n = pid % D

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, D, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)  # vector of K indices
        mask_k = ks < D
        # Load A_row[b, m, ks] as vector
        a_ptrs = A_ptr + b * stride_Ab + m * stride_Am + ks * stride_Ad
        a_vals = tl.load(a_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]
        # Load W_row[ks, n] as vector
        w_ptrs = W_ptr + ks * stride_Wk + n * stride_Wn
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]
        # Accumulate dot product
        # Ensure a_vals and w_vals have same shape for tl.sum
        prod = a_vals * w_vals  # broadcast allowed: each element in a_vals multiplies scalar w_vals[k]
        acc += tl.sum(prod, axis=0)

    # Store result
    c_ptr = C_ptr + b * stride_Cb + m * stride_Cm + n * stride_Cn
    tl.store(c_ptr, acc)


@triton.jit
def split_seqs_kernel(
    in_ptr, out1_ptr, out2_ptr,
    B, L_txt, L_img, D,
    stride_in_b, stride_in_s, stride_in_d,
    stride_out1_b, stride_out1_s, stride_out1_d,
    stride_out2_b, stride_out2_s, stride_out2_d,
    BLOCK_N: tl.constexpr
):
    # program_id(0) iterates over batch, program_id(1) over sequence positions
    b = tl.program_id(0)
    t = tl.program_id(1)
    # Copy in[:, t, :] to out1 and out2
    in_off = b * stride_in_b + t * stride_in_s
    # processed[:, :L_txt, :] output index
    out1_off = b * stride_out1_b + t * stride_out1_s
    # processed[:, L_txt:, :] output index
    out2_off = b * stride_out2_b + (t + L_txt) * stride_out2_s
    for d in range(0, D, BLOCK_N):
        cols = d + tl.arange(0, BLOCK_N)
        mask = cols < D
        vals = tl.load(in_ptr + in_off + cols * stride_in_d, mask=mask, other=0.0)
        tl.store(out1_ptr + out1_off + cols * stride_out1_d, vals, mask=mask)
        tl.store(out2_ptr + out2_off + cols * stride_out2_d, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Performs:
          - concatenation in Triton
          - linear projection (matmul) in Triton
          - splitting in Triton
        Returns:
          - processed_encoder: [B, L_txt, D]
          - processed_hidden: [B, L_img, D]
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be 3D [B, length, D]"
        assert process_weight.ndim == 2 and process_weight.shape[0] == process_weight.shape[1], "process_weight must be square [D, D]"
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D
        assert process_weight.shape[1] == D

        device = hidden_states.device
        in_dtype = hidden_states.dtype  # preserve original dtype

        # 1) Concatenate sequences along sequence dimension using Triton
        M = L_txt + L_img
        A = torch.empty((B, M, D), dtype=torch.float32, device=device)  # compute in float32
        ehs = encoder_hidden_states.to(torch.float32)
        hs = hidden_states.to(torch.float32)

        grid_concat = (B, triton.cdiv(M, 1))
        concat_seqs_kernel[grid_concat](
            ehs, hs, A,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            BLOCK_N=64,
            num_warps=1, num_stages=1
        )

        # 2) Linear projection A @ process_weight.T using Triton GEMM (simple per-element kernel)
        # Note: This kernel computes one output element per program and loops over K.
        # It is designed for correctness (over simplicity); for speed, a tiled matmul would be better.
        W_T = process_weight.t().to(torch.float32)  # [D, D]
        C = torch.empty((B, M, D), dtype=torch.float32, device=device)

        grid_matmul = (B * M * D,)
        matmul_kernel_single[grid_matmul](
            A, W_T, C,
            B, M, D,
            A.stride(0), A.stride(1), A.stride(2),
            W_T.stride(0), W_T.stride(1),  # W_T is [D, D], addressing as [k, n]
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_K=128,
            num_warps=1, num_stages=1
        )

        # 3) Split back into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, L_txt, D), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, L_img, D), dtype=torch.float32, device=device)

        grid_split = (B, M)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_N=64,
            num_warps=1, num_stages=1
        )

        # Cast back to original dtype if needed
        if in_dtype != torch.float32:
            processed_encoder = processed_encoder.to(in_dtype)
            processed_hidden = processed_hidden.to(in_dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
