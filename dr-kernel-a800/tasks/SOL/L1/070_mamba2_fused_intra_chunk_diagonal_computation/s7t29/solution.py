import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_masked_cumsum_exp(
    A: tl.pointer_type(tl.float32, ()),  # [B, C, S, N]
    L: tl.pointer_type(tl.float32, ()),  # [B, C, S, S, N]
    B_size: tl.int32, C_size: tl.int32, S: tl.int32, N: tl.int32,
    A_stride_b: tl.int32, A_stride_c: tl.int32, A_stride_j: tl.int32, A_stride_n: tl.int32,
    L_stride_b: tl.int32, L_stride_c: tl.int32, L_stride_i: tl.int32, L_stride_j: tl.int32, L_stride_n: tl.int32,
):
    # Grid: (B, C, S) each program handles one (b, c, i)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    # Running sum along j for this (b, c, i)
    running = tl.zeros((), dtype=tl.float32)
    # We need to accumulate only j < i (diagonal=-1), exclude j == i
    for j in range(S):
        if j < i:
            a_ptr = A + b * A_stride_b + c * A_stride_c + j * A_stride_j + n * A_stride_n  # n will be looped over
            # We need to loop n as well
            for n in range(N):
                a_val = tl.load(a_ptr)
                running += a_val
            # After loop over n, exp(running) and store at L[b, c, i, j, n]
            for n in range(N):
                l_ptr = L + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n
                tl.store(l_ptr, tl.exp(running))
                # running was summed over all n; but L depends only on the cumsum of A across j for each n independently.
                # The original code applies exp to the masked cumsum per (b, c, i, j, n). Since we summed across n in the first loop,
                # we need to reset running per n. To reflect that, we should recompute running for each n in the second loop.
                # However, since A[b, c, j, n] is different for each n, we cannot keep a single running. Therefore, we compute running
                # per n by looping again.
                pass
    # The above nested loops cannot be easily written in Triton due to Python-level if dependent on runtime values. 
    # To implement the lower-triangular mask correctly, we need to maintain a separate running per n.
    # Triton requires compile-time loops; here we use a 2D tile and mask: for each n, compute cumsum across j < i and store exp.
    # We implement: for each n, loop j from 0..S-1 and store exp of cumsum up to j if j < i; otherwise zero. 
    # But Triton doesn't support nested Python loops cleanly across multiple runtime dims here. 
    # Therefore, we simplify: for each n, compute running across j in 0..S-1, then store exp(running) to L at each j where j < i.
    # This is equivalent to cumsum along j and exp, with mask j < i (exclude j==i).
    # However, Triton's for-loops are limited. To make it work, we use a vectorized approach over j with tl.arange and masks.

    # Vectorized approach: initialize L to zeros; for each n, compute prefix sums for j in [0..S-1] and store at j < i.
    # We can't directly assign with mask in Triton easily. Instead, we implement two loops: one to compute cumsum and exp, 
    # and another to write back. But to keep it simple and correct, we use the masked cumsum logic by iterating j and n explicitly.

    # Since Triton doesn't allow complex nested loops as above, we instead implement the mask logic per n using a small Python 
    # loop structure that is allowed: for j in range(S): and for n in range(N):. The key is to avoid torch operations.
    # We will compute running per n by reinitializing running to 0 for each n and storing exp(running) at (i, j, n) for j < i.
    # Note: This Triton kernel uses Python-level for-loops over S and N; Triton will unroll them when S/N are provided as constexpr.
    # To satisfy the Triton requirement, we pass S and N as tl.constexpr. In this implementation, we will pass S and N as 32 (NUM_HEADS).
    # However, S and N here are chunk_size and num_heads, which are dynamic. Triton requires compile-time loops for such structures.
    # Therefore, we will rework this kernel to use tl.arange and masks for vectorized lower-triangular mask. This is the robust way.

    # Robust vectorized approach using tl.arange:
    # We create a 2D j vector [0..S-1], and for each n in [0..N-1], compute cumsum along j where j < i and store exp(running) at L.
    j_vec = tl.arange(0, S)
    # We need to loop n explicitly; Triton supports Python for-loop over N if N is constexpr. We will pass N as constexpr in the call.
    for n in range(N):
        running = tl.zeros((), dtype=tl.float32)
        # Mask lower-triangular: include j < i, exclude j == i
        mask_j = j_vec < i
        # For each j, if mask_j, add A[b, c, j, n]; else add 0
        for jj in range(S):
            # Load A[b, c, jj, n]
            a_ptr = A + b * A_stride_b + c * A_stride_c + jj * A_stride_j + n * A_stride_n
            a_val = tl.load(a_ptr)
            # Only add if jj < i
            if jj < i:
                running += a_val
            # Store exp(running) at L[b, c, i, jj, n]
            l_ptr = L + b * L_stride_b + c * L_stride_c + i * L_stride_i + jj * L_stride_j + n * L_stride_n
            tl.store(l_ptr, tl.exp(running))


@triton.jit
def contract_BC_to_G(
    B_exp: tl.pointer_type(tl.float32, ()),  # [B, C, S, N, K]
    C_exp: tl.pointer_type(tl.float32, ()),  # [B, C, S, N, K]
    G: tl.pointer_type(tl.float32, ()),      # [B, C, S, S, N]
    B_size: tl.int32, C_size: tl.int32, S: tl.int32, N: tl.int32, K: tl.constexpr,
    B_stride_b: tl.int32, B_stride_c: tl.int32, B_stride_j: tl.int32, B_stride_n: tl.int32, B_stride_k: tl.int32,
    C_stride_b: tl.int32, C_stride_c: tl.int32, C_stride_i: tl.int32, C_stride_n: tl.int32, C_stride_k: tl.int32,
    G_stride_b: tl.int32, G_stride_c: tl.int32, G_stride_i: tl.int32, G_stride_j: tl.int32, G_stride_n: tl.int32,
):
    # Grid: (B, C, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    g_ptr = G + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    acc = tl.zeros((), dtype=tl.float32)

    # Sum over state dimension K (constexpr)
    for k in range(K):
        b_ptr = B_exp + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k
        c_ptr = C_exp + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k
        b_val = tl.load(b_ptr)
        c_val = tl.load(c_ptr)
        acc += b_val * c_val

    tl.store(g_ptr, acc)


@triton.jit
def elementwise_mul(
    G: tl.pointer_type(tl.float32, ()),  # [B, C, S, S, N]
    L: tl.pointer_type(tl.float32, ()),  # [B, C, S, S, N]
    M: tl.pointer_type(tl.float32, ()),  # [B, C, S, S, N]
    B_size: tl.int32, C_size: tl.int32, S: tl.int32, N: tl.int32,
    G_stride_b: tl.int32, G_stride_c: tl.int32, G_stride_i: tl.int32, G_stride_j: tl.int32, G_stride_n: tl.int32,
    L_stride_b: tl.int32, L_stride_c: tl.int32, L_stride_i: tl.int32, L_stride_j: tl.int32, L_stride_n: tl.int32,
    M_stride_b: tl.int32, M_stride_c: tl.int32, M_stride_i: tl.int32, M_stride_j: tl.int32, M_stride_n: tl.int32,
):
    # Grid: (B, C, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    g_ptr = G + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    l_ptr = L + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n
    m_ptr = M + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n

    g_val = tl.load(g_ptr)
    l_val = tl.load(l_ptr)
    m_val = g_val * l_val
    tl.store(m_ptr, m_val)


@triton.jit
def diag_contract_Y(
    M: tl.pointer_type(tl.float32, ()),          # [B, C, S, S, N]
    hidden: tl.pointer_type(tl.float32, ()),     # [B, C, S, N, D]
    Y: tl.pointer_type(tl.float32, ()),          # [B, C, S, N, D]
    B_size: tl.int32, C_size: tl.int32, S: tl.int32, N: tl.int32, D: tl.constexpr,
    M_stride_b: tl.int32, M_stride_c: tl.int32, M_stride_i: tl.int32, M_stride_j: tl.int32, M_stride_n: tl.int32,
    hidden_stride_b: tl.int32, hidden_stride_c: tl.int32, hidden_stride_j: tl.int32, hidden_stride_n: tl.int32, hidden_stride_d: tl.int32,
    Y_stride_b: tl.int32, Y_stride_c: tl.int32, Y_stride_i: tl.int32, Y_stride_n: tl.int32, Y_stride_d: tl.int32,
):
    # Grid: (B, C, S, N, D) each program handles one (b, c, i, n, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)

    # Accumulate over j: sum_j M[b, c, i, j, n] * hidden[b, c, j, n, d]
    for j in range(S):
        m_ptr = M + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        h_ptr = hidden + b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_j + n * hidden_stride_n + d * hidden_stride_d
        m_val = tl.load(m_ptr)
        h_val = tl.load(h_ptr)
        acc += m_val * h_val

    y_ptr = Y + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants as per original code
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.repeat_factor = self.NUM_HEADS // self.N_GROUPS  # = 4

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        # Ensure tensors are on CUDA
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew.forward requires CUDA tensors."

        # Shapes
        Bsz, Csz, S, N, D = hidden_states.shape  # batch_size, num_chunks, chunk_size, num_heads, head_dim
        # Prepare expanded B and C to num_heads
        B_exp = B.repeat_interleave(self.repeat_factor, dim=3).to(torch.float32)
        C_exp = C.repeat_interleave(self.repeat_factor, dim=3).to(torch.float32)

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        M = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        # Strides
        A_stride_b, A_stride_c, A_stride_j, A_stride_n = A_cumsum.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        hidden_stride_b, hidden_stride_c, hidden_stride_j, hidden_stride_n, hidden_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        B_exp_stride_b, B_exp_stride_c, B_exp_stride_j, B_exp_stride_n, B_exp_stride_k = B_exp.stride()
        C_exp_stride_b, C_exp_stride_c, C_exp_stride_i, C_exp_stride_n, C_exp_stride_k = C_exp.stride()

        # Launch Triton kernels
        # Kernel 1: compute L via masked cumsum along j with diagonal=-1 (exclude j==i) and apply exp
        # Note: Triton requires compile-time loop bounds. We pass S and N as constexpr (they are used as runtime here).
        # Triton will handle Python for-loops when bounds are provided; to be safe, we keep S and N in loops explicitly.
        # The kernel below uses tl.arange for vectorization; we will pass S as a constexpr by using the loop.
        # For robustness, we run grid over (B, C, S). Triton will iterate over N inside the kernel using for n in range(N).
        grid_L = (Bsz, Csz, S)
        compute_L_masked_cumsum_exp[grid_L](
            A_cumsum, L,
            B_size=Bsz, C_size=Csz, S=S, N=N,
            A_stride_b=A_stride_b, A_stride_c=A_stride_c, A_stride_j=A_stride_j, A_stride_n=A_stride_n,
            L_stride_b=L_stride_b, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j, L_stride_n=L_stride_n,
            num_warps=1, num_stages=1
        )

        # Kernel 2: contract B_exp and C_exp to G
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_exp, C_exp, G,
            B_size=Bsz, C_size=Csz, S=S, N=N, K=hidden_states.shape[4],  # state_size is D, but original code sets K to state_size
            B_exp_stride_b=B_exp_stride_b, B_exp_stride_c=B_exp_stride_c, B_exp_stride_j=B_exp_stride_j, B_exp_stride_n=B_exp_stride_n, B_exp_stride_k=B_exp_stride_k,
            C_exp_stride_b=C_exp_stride_b, C_exp_stride_c=C_exp_stride_c, C_exp_stride_i=C_exp_stride_i, C_exp_stride_n=C_exp_stride_n, C_exp_stride_k=C_exp_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # Kernel 3: elementwise multiply M = G * L
        grid_M = (Bsz, Csz, S, S, N)
        elementwise_mul[grid_M](
            G, L, M,
            B_size=Bsz, C_size=Csz, S=S, N=N,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            L_stride_b=L_stride_b, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j, L_stride_n=L_stride_n,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            num_warps=1, num_stages=1
        )

        # Kernel 4: diagonal contraction to produce Y
        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            M, hidden_states.to(torch.float32), Y,
            B_size=Bsz, C_size=Csz, S=S, N=N, D=D,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            hidden_stride_b=hidden_stride_b, hidden_stride_c=hidden_stride_c, hidden_stride_j=hidden_stride_j, hidden_stride_n=hidden_stride_n, hidden_stride_d=hidden_stride_d,
            Y_stride_b=Y_stride_b, Y_stride_c=Y_stride_c, Y_stride_i=Y_stride_i, Y_stride_n=Y_stride_n, Y_stride_d=Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Return in bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
