# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Apollon is a Python + C framework for content-based music analysis: audio
feature extraction, onset detection, time-series segmentation, Hidden Markov
Models, and Self-Organizing Maps. Package sources live under `src/apollon`,
performance-critical routines are implemented as C extension modules.

## Commands

The project uses Poetry for dependency management and packaging.

```bash
poetry install --with=development   # install runtime + dev deps (pylint, mypy, ruff, hypothesis)
poetry run -- python -m unittest    # run the full test suite
poetry run -- mypy --strict src/apollon   # type-check
poetry run -- pylint --persistent n --jobs 0 src/apollon   # lint
```

Run a single test module or case with unittest's dotted path syntax, e.g.:

```bash
poetry run -- python -m unittest tests.signal.test_features
poetry run -- python -m unittest tests.signal.test_features.SomeTestCase.test_something
```

`tox.ini` targets py311–py314 and additionally pulls in `hypothesis` for
property-based tests (see `tests/signal/strategies.py` and `tests/hmm/test_hmm.py`).

The C extensions (`apollon.signal._features`, `apollon.som._distance`) are
built from `build.py` via the Poetry build script, so a plain `pip install -e .`
or `poetry install` is required after changing any `.c`/`.h` source before the
new symbols are importable.

CI (`.github/workflows/`) runs `mypy --strict`, `pylint`, and `python -m
unittest` on pushes for Python 3.11–3.13; `release.yml` builds and publishes
sdists/wheels to PyPI on GitHub release.

## Architecture

### Package layout
- `apollon/signal/` — DSP: `spectral.py` (STFT/DFT via `TransformResult`
  subclasses), `features.py`, `filter.py`, `tools.py`, `critical_bands.py`.
  Backed by the `apollon.signal._features` C extension (`cdim.c`,
  `correlogram.c`) for correlation dimension / correlogram computation.
- `apollon/segment/` — windowing/segmentation of arrays (`ArraySegmentation`
  → `Segments`) and of audio files read lazily from disk
  (`FileSegmentation`, wraps an `AudioFile`). Both expose segments as
  `Segment` model instances with frame bounds/center.
- `apollon/onsets/` — `OnsetDetector` abstract base class with concrete
  `EntropyOnsetDetector` (delay-embedding entropy, uses `apollon.fractal`)
  and `FluxOnsetDetector` (spectral flux via `signal.spectral.Stft`). Peak
  picking is delegated to `apollon.peak_picking.FilterPeakPicker`.
- `apollon/hmm/` — thin pydantic wrappers (`PoissonHmmParams`,
  `PoissonHmmQualityMeasures`) around the external `chainsaddiction.poishmm`
  package; this repo does not implement the HMM training algorithm itself.
  `chainsaddiction` is an optional dependency (the `hmm` extra —
  `pip install apollon[hmm]`); `apollon/hmm/__init__.py` raises an explicit
  `ImportError` if it's missing, so `import apollon.hmm` fails fast with a
  helpful message instead of a bare `ModuleNotFoundError` from `models.py`.
- `apollon/som/` — only the C distance-metric extension
  (`apollon.som._distance`, from `distance.c`) lives here. The actual
  Self-Organizing Map implementation is the separate `awesom` package
  (a project dependency); this module exists to feed it fast distance
  computations.
- `apollon/audio.py` — `AudioFile`, a thin wrapper around `soundfile` for
  lazy, offset-based reads (supports negative offsets for pre-padding) with
  optional mono downmix/normalization.
- `apollon/io/` — pickle/npy persistence helpers (`io.py`) and JSON
  helpers (`json.py`).
- `apollon/typing.py` — the project's shared type aliases (`Array`,
  `FloatArray`, `IntArray`, `ComplexArray`, `PathType`, `SomDims`, ...).
  Import this module qualified (e.g. `from .. typing import FloatArray`)
  rather than importing individual names star-style.

### Parameter/model pattern
Each subpackage that exposes configurable algorithms defines its parameters
as pydantic `BaseModel`s in a sibling `models.py` (e.g.
`apollon.signal.models.StftParams`, `apollon.segment.models.SegmentationParams`,
`apollon.onsets.models.FluxODParams`, top-level `apollon.models` for
cross-cutting params like `PeakPickingParams`). Classes that perform
computation store their resolved params on `self._params` and expose them
via a `params` property, so results stay reproducible/serializable
independent of the object that produced them. Follow this pattern
(model class in `models.py` + a `params` property) when adding new
transforms or detectors.

### C extensions
New C-level acceleration belongs in `src/apollon/<pkg>/*.c` with headers in
`include/`, registered as an `Extension` in `build.py`, and given a `.pyi`
stub next to the compiled module (see `apollon/som/_distance.pyi`) for mypy.
Extensions are referenced by their dotted import path (e.g.
`apollon.signal._features`, `apollon.som._distance`) and are allow-listed in
`pyproject.toml` under `tool.pylint.'MESSAGES CONTROL'.extension-pkg-allow-list`
since pylint cannot introspect compiled modules — add new extension modules
there too.

## Conventions
- Google-style docstrings; every public function/method/class should have one.
- Full static typing enforced with `mypy --strict`; add new shared aliases to
  `apollon/typing.py` rather than inlining `numpy.ndarray[...]` annotations.
- Name types after the thing they describe, not suffixed with `Type` (e.g.
  `Bike`, not `BicycleType`).
