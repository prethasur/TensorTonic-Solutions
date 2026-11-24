import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Handle empty input list
    if not seqs:
        return np.empty((0, max_len or 0), dtype=int)

    # Determine target length L
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    padded_seqs = []
    for seq in seqs:
        # Copy the sequence to avoid mutating the caller's original list
        row = list(seq[:max_len])
        
        # Pad if shorter than max_len
        d = max_len - len(row)
        if d > 0:
            row.extend([pad_value] * d)
            
        padded_seqs.append(row)

    return np.array(padded_seqs)
       
    pass