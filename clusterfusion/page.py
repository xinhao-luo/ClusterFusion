import torch

def get_seq_lens(
    kv_indptr: torch.Tensor, kv_last_page_len: torch.Tensor, page_size: int
) -> torch.Tensor:
    r"""Convert KV indptr and last page length to sequence lengths.

    Parameters
    ----------
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    page_size : int
        The size of a page in the paged kv-cache.

    Returns
    -------
    seq_lens: torch.Tensor
        The sequence lengths of each request in the paged kv-cache, shape: ``[batch_size]``.
    """
    return (
        torch.clamp(kv_indptr[1:] - kv_indptr[:-1] - 1, min=0) * page_size
        + kv_last_page_len
    )