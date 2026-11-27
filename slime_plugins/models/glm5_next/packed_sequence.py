"""Packed document offsets shared by the GLM-5.3 operator adapters."""


def packed_boundaries(offsets, token_count):
    """Return integer offsets covering the stream, including empty documents."""
    try:
        values = offsets.tolist() if hasattr(offsets, "tolist") else list(offsets)
    except TypeError as exc:
        raise ValueError("Expected a one-dimensional sequence of integer offsets") from exc
    if not isinstance(values, list) or len(values) < 2 or any(type(value) is not int for value in values):
        raise ValueError("Expected a one-dimensional sequence of at least two integer offsets")
    if values[0] != 0 or values[-1] != token_count:
        raise ValueError(f"Packed boundaries must cover {token_count} tokens")
    if any(end < start for start, end in zip(values, values[1:])):
        raise ValueError("Packed boundaries must be nondecreasing")
    return values
