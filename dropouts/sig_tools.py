def trim_range(d_frq: float, lcf: Optional[float] = None,
               ucf: Optional[float] = None) -> slice:
    """Return slice of trim indices regarding an array ``frqs`` of DFT
    frquencies, such that both boundaries are included.

    Args:
        d_frq:   Frequency spacing.
        lcf:     Lower cut-off frequency.
        ucf:     Upper cut-off frequency.

    Returns:
        Slice of trim indices.
    """
    try:
        lcf = int(lcf//d_frq)
    except TypeError:
        lcf = None
    try:
        ucf = int(ucf//d_frq)
    except TypeError:
        ucf = None
    return slice(lcf, ucf)
