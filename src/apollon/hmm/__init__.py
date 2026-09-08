"""
Hidden Markov Models

This subpackage requires the optional ``chainsaddiction`` dependency.
Install it with ``pip install apollon[hmm]``.
"""
try:
    import chainsaddiction  # pylint: disable = unused-import
except ImportError as err:
    raise ImportError(
        "The `apollon.hmm` subpackage requires the optional "
        "`chainsaddiction` package, which is not installed. Install it "
        "with `pip install apollon[hmm]`."
    ) from err
