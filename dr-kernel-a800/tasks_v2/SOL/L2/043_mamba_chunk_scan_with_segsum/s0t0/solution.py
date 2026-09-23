import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: elementwise exp for a 1D flattened tensor
@triton.jit
def exp_kernel(x_ptr, y_ptr, n_elements: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(y_ptr + offsets, y, mask=mask)


# Triton kernel: elementwise add (y = y + add), works for any shape by flattening
@triton.jit
def add_kernel(y_ptr, add_ptr, n_elements: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    y = y + a
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_exp_inplace(x: torch.Tensor, out: torch.Tensor):
    """
    Compute out = exp(x) using Triton. x and out are float32 CUDA tensors of same shape.
    """
    assert TRITON_AVAILABLE and x.is_cuda and out.is_cuda
    n_elements = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    exp_kernel[grid](x, out, n_elements, BLOCK=BLOCK, num_warps=4)


def triton_add_inplace(y: torch.Tensor, add: torch.Tensor):
    """
    y: destination tensor (float32 CUDA)
    add: source tensor (float32 CUDA), can be broadcastable scalar/vector; here we assume same shape.
    y += add using Triton elementwise add.
    """
    assert TRITON_AVAILABLE and y.is_cuda and add.is_cuda
    n_elements = y.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    add_kernel[grid](y, add, n_elements, BLOCK=BLOCK, num_warps=4)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, initial_states: torch.Tensor):
        # Preserve original behavior and shapes
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = 256
        n_groups = 1
        chunk_size = 256

        # Compute padding size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_size

        # Convert to float32
        hidden_states_f = hidden_states.to(torch.float32)
        A_f = A.to(torch.float32)
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        D_f = D.to(torch.float32)
        initial_states_f = initial_states.to(torch.float32)

        # Expand B and C to num_heads
        B_expanded = B_f.expand(batch_size, seq_len, num_heads, state_size)
        C_expanded = C_f.expand(batch_size, seq_len, num_heads, state_size)

        # Pad hidden states
        hidden_states_padded = F.pad(hidden_states_f, (0, 0, 0, 0, 0, pad_size, 0, 0), mode='constant', value=0)

        # Reshape into chunks
        hidden_states_chunked = hidden_states_padded.reshape(
            batch_size, -1, chunk_size, num_heads, head_dim
        )

        # Transpose A for cumsum and reshape
        A_transposed = A_f.transpose(1, 2)  # [batch, seq_len, num_heads]
        A_chunked = A_transposed.reshape(batch_size, seq_len_padded, chunk_size, num_heads)

        # Compute A_cumsum (PyTorch) and exp of it using Triton
        A_chunked_perm = A_chunked.permute(0, 3, 1, 2)  # [batch, num_heads, num_chunks, chunk_size]
        A_cumsum = torch.cumsum(A_chunked_perm, dim=-1)  # [batch, num_heads, num_chunks, chunk_size]

        # Triton elementwise exp of A_cumsum
        A_cumsum_exp = torch.empty_like(A_cumsum)
        triton_exp_inplace(A_cumsum, A_cumsum_exp)

        # For simplicity, we avoid reimplementing segment_sum and other complex ops in Triton here.
        # The original code heavily relies on torch.tril + masked_fill + cumsum for segment_sum, which is


def run(*args):
    return ModelNew()(*args)
