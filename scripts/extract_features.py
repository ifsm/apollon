#!/usr/bin/env python3
"""
Timbre and MFCC feature extraction
==================================

Compute frame-wise timbre features and Mel-frequency cepstral coefficients of
a set of audio files in parallel, and store them as parquet files::

    <outpath>/timbre/files/<stem>.parquet   one column per timbre feature
    <outpath>/mfcc/files/<stem>.parquet     one column per coefficient
    <outpath>/{timbre,mfcc}/params.json     parameters of each feature set
    <outpath>/manifest.csv                  outcome for each input file

Rows are analysis frames, indexed by ``time``, the centre of each frame in
seconds. Only frames that lie completely inside the signal are kept. Both
feature sets are computed on a grid anchored at the first sample, and the
timbre hop is a multiple of the MFCC hop, so every timbre frame has an MFCC
frame at the same time.

The parameters suit short sustained vowels, recorded as 16-bit PCM at
44.1 kHz. Each file is freed from DC and scaled to a common RMS level before
the analysis, which makes the level-dependent features comparable across
recordings of unknown gain.

Usage::

    python scripts/extract_features.py INPUT [INPUT ...] -o OUTPATH [-j N]
        [--overwrite] [-v]

Each INPUT is a wav file or a directory, which is searched recursively for
wav files.
"""
import os

# Each worker process runs single-threaded. Pin the BLAS and OpenMP pools
# before numpy is imported, so that the workers do not oversubscribe the CPUs.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

# pylint: disable = wrong-import-position
import argparse
import logging
import logging.handlers
import multiprocessing as mp
import pathlib
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing.queues import Queue
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import (BaseModel, NonNegativeFloat, PositiveFloat,
                      model_validator)

from apollon._defaults import SPL_REF
from apollon.audio import AudioFile
from apollon.segment import ArraySegmentation
from apollon.signal import critical_bands as _cb
from apollon.signal import features, tools
from apollon.signal.cepstral import Mfcc
from apollon.signal.models import (CepstrumParams, CorrDimParams, MfccParams,
                                   StftParams, TriangFilterSpec)
from apollon.signal.spectral import Spectrogram, Stft
from apollon.typing import FloatArray, IntArray, floatarray


LOGGER = logging.getLogger("extract_features")
LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(processName)s] %(message)s"

FPS = 44100
"""Sample rate the frame lengths below are given at."""

CDIM_BADER_MIN = 2390
"""Samples ``cdim_bader`` consumes on top of its embedding span, i.e.,
``CDIM_BADER_N_SAMPLES - CDIM_BADER_BOUND`` of ``include/cdim.h``."""


def cdim_frame_length(params: CorrDimParams) -> int:
    """Compute the number of samples ``cdim_bader`` consumes per frame.

    Args:
        params:  Parameters of the correlation dimension

    Returns:
        Frame length in samples.
    """
    return CDIM_BADER_MIN + (params.m_dim-1) * params.delay


def hop_size(params: StftParams) -> int:
    """Return the distance between consecutive frames in samples."""
    return params.n_perseg - params.n_overlap


class LevelParams(BaseModel):
    """Level normalization applied to each file before the analysis.

    Each file is scaled to an RMS of ``target_dbfs`` relative to 1.0. For the
    psychoacoustic features, that level is taken to be ``ref_spl`` dB SPL.
    """
    target_dbfs: float = -20.0
    ref_spl: float = 70.0

    @property
    def target_rms(self) -> float:
        """RMS each file is scaled to, relative to full scale."""
        return float(10**(self.target_dbfs/20))

    @property
    def pa_per_fs(self) -> float:
        """Sound pressure in Pa that corresponds to 1.0 full scale."""
        return float(SPL_REF * 10**(self.ref_spl/20) / self.target_rms)


class BandParams(BaseModel):
    """Frequency band the spectral features are restricted to, in Hz."""
    low: NonNegativeFloat
    high: PositiveFloat

    @model_validator(mode="after")
    def _check_low_lt_high(self) -> Self:
        if self.low >= self.high:
            raise ValueError(f"low ({self.low} Hz) must be less than high "
                             f"({self.high} Hz).")
        return self


class TimbreTrackParams(BaseModel):
    """Parameters of the timbre features."""
    stft: StftParams
    band: BandParams
    level: LevelParams
    cdim: CorrDimParams
    loudness_resolution: PositiveFloat = 0.1

    @model_validator(mode="after")
    def _check_grid(self) -> Self:
        if not self.stft.extend:
            raise ValueError("stft.extend must be True. The STFT and the "
                             "correlation dimension share a frame grid "
                             "anchored at the first sample.")
        if self.band.high > self.stft.fps / 2:
            raise ValueError(f"band.high ({self.band.high} Hz) exceeds the "
                             f"Nyquist frequency ({self.stft.fps/2} Hz).")
        if cdim_frame_length(self.cdim) > self.stft.n_perseg:
            raise ValueError(f"cdim frames ({cdim_frame_length(self.cdim)} "
                             "samples) must not be longer than STFT frames "
                             f"({self.stft.n_perseg} samples).")
        return self


class MfccTrackParams(BaseModel):
    """Parameters of the Mel-frequency cepstral coefficients."""
    level: LevelParams
    mfcc: MfccParams

    @model_validator(mode="after")
    def _check_grid(self) -> Self:
        if not self.mfcc.stft.extend:
            raise ValueError("mfcc.stft.extend must be True. Frames are "
                             "selected on a grid anchored at the first "
                             "sample.")
        return self


LEVEL = LevelParams(target_dbfs=-20.0, ref_spl=70.0)

# Formants and aspiration noise lie within this band. Above it, microphone
# and preamp noise, which differ between recording sessions, dominate.
BAND = BandParams(low=50.0, high=8000.0)

TIMBRE = TimbreTrackParams(
    # 92.9 ms Hann window with 10.77 Hz bins, fine enough for
    # ``roughness_helmholtz`` to resolve partials 33 Hz apart. Hop 20 ms.
    stft=StftParams(fps=FPS, window="hann", n_fft=None, n_perseg=4096,
                    n_overlap=4096-882, extend=True, pad=True),
    band=BAND,
    level=LEVEL,
    # Delay of about a quarter period of F1. Embedding span 39*14 samples,
    # 12.4 ms; frames of 2936 samples, 66.6 ms.
    cdim=CorrDimParams(delay=14, m_dim=40, n_bins=1000, scaling_size=10),
    loudness_resolution=0.1)

MFCC = MfccTrackParams(
    level=LEVEL,
    mfcc=MfccParams(
        # 23.2 ms Hann window smooths the harmonics towards the spectral
        # envelope. Hop 10 ms, half the timbre hop.
        stft=StftParams(fps=FPS, window="hann", n_fft=2048, n_perseg=1024,
                        n_overlap=1024-441, extend=True, pad=True),
        fb=TriangFilterSpec(low=BAND.low, high=BAND.high, n_filters=40,
                            scale="mel"),
        # c0 is kept and no lifter applied; downstream may truncate, drop c0,
        # and standardize.
        cepstrum=CepstrumParams(n_coefs=20, dct_type=2, lifter_gain=0.0),
        preemphasis=0.97,
        # 50 dB below the normalized RMS level. Weak high bands otherwise
        # follow the noise floor of each recording, which differs with gain.
        floor_dbfs=-70.0))


def full_frames(n_samples: int, n_perseg: int, hop: int) -> IntArray:
    """Select the frames that lie completely inside a signal.

    Frame ``k`` covers the samples from ``k*hop - n_perseg//2`` up to, but
    excluding, ``k*hop - n_perseg//2 + n_perseg``. This is the segmentation
    of ``ArraySegmentation`` with ``extend=True``, which centres frame ``k``
    on sample ``k*hop``.

    Args:
        n_samples:  Length of the signal
        n_perseg:   Frame length in samples
        hop:        Distance between consecutive frames in samples

    Returns:
        Indices of the frames that hold no padding. Empty if the signal is
        too short for a single one.
    """
    half = n_perseg // 2
    first = -(-half // hop)
    last = (n_samples - n_perseg + half) // hop
    return np.arange(first, last+1)


def condition(sig: FloatArray, level: LevelParams) -> FloatArray:
    """Remove DC from ``sig`` and scale it to the level of ``level``.

    Args:
        sig:    Single-channel signal, shaped ``(n_samples, 1)``
        level:  Level normalization

    Returns:
        Conditioned copy of ``sig``.

    Raises:
        ValueError: If ``sig`` is silent.
    """
    out = sig - sig.mean(axis=0)
    rms = features.rms(out).item()
    if rms == 0:
        raise ValueError("Signal is silent.")
    return floatarray(out * (level.target_rms/rms))


class TimbreExtractor:
    """Compute the timbre features of a signal."""
    def __init__(self, params: TimbreTrackParams) -> None:
        """Set up the transforms.

        Args:
            params:  Parameters of the timbre features
        """
        self._params = params
        stp = params.stft
        self._stft = Stft(fps=stp.fps, n_perseg=stp.n_perseg,
                          n_overlap=stp.n_overlap, window=stp.window,
                          n_fft=stp.n_fft, norm=stp.norm,
                          single_sided=stp.single_sided, extend=stp.extend,
                          pad=stp.pad)
        n_cdim = cdim_frame_length(params.cdim)
        self._cdim_cutter = ArraySegmentation(n_cdim, n_cdim-hop_size(stp),
                                              extend=True, pad=True)

    @property
    def params(self) -> TimbreTrackParams:
        """Return parameters."""
        return self._params

    def transform(self, sig: FloatArray) -> pd.DataFrame:
        """Compute the timbre features of ``sig``.

        The correlation dimension is estimated on frames of its own length,
        centred on the same samples as the STFT frames.

        Args:
            sig:  Single-channel signal, shaped ``(n_samples, 1)``

        Returns:
            One column per feature and one row per frame, indexed by the
            frame centre in seconds.

        Raises:
            ValueError: If ``sig`` is silent, or too short for a single
                complete frame.
        """
        stp = self._params.stft
        hop = hop_size(stp)
        idx = full_frames(sig.shape[0], stp.n_perseg, hop)
        if idx.size == 0:
            raise ValueError(f"Signal of {sig.shape[0]} samples holds no "
                             f"complete {stp.n_perseg}-sample frame.")

        sig = condition(sig, self._params.level)
        sxx = self._stft.transform(sig)
        cols = self._spectral(sxx, idx)
        cols.update(self._psychoacoustic(sxx, idx))
        cols["cdim"] = self._cdim(sig, idx)
        return pd.DataFrame(cols, index=pd.Index(idx*hop/stp.fps, name="time"))

    def _spectral(self, sxx: Spectrogram, idx: IntArray) -> dict[str, FloatArray]:
        """Compute the features that act on the magnitude spectrum."""
        band = self._params.band
        frqs = sxx.frqs
        in_band = ((frqs >= band.low) & (frqs <= band.high)).ravel()
        mag = sxx.abs[:, idx]
        pwr = np.square(mag[in_band])

        centroid = features.spectral_centroid(frqs[in_band], pwr)
        spread = features.spectral_spread(frqs[in_band], pwr, centroid)

        # The first frame has no predecessor; spectral_flux reads 0 there.
        flux = features.spectral_flux(mag[in_band], delta=hop_size(sxx.params)/sxx.params.fps)
        flux[0, 0] = np.nan

        rough = features.roughness_helmholtz(sxx.d_frq,
                                             np.where(frqs >= band.low, mag, 0.0),
                                             frq_max=band.high)
        return {"spectral_centroid": centroid.ravel(),
                "spectral_spread": spread.ravel(),
                "spectral_flux": flux.ravel(),
                "roughness": rough.ravel()}

    def _psychoacoustic(self, sxx: Spectrogram, idx: IntArray
                        ) -> dict[str, FloatArray]:
        """Compute loudness and sharpness from one excitation pattern."""
        band = self._params.band
        frqs = sxx.frqs.ravel()
        power = sxx.ms_power[:, idx] * self._params.level.pa_per_fs**2
        power[(frqs < band.low) | (frqs > band.high)] = 0.0
        excitation = _cb.excitation_pattern(frqs, power,
                                            self._params.loudness_resolution)
        return {"loudness": _cb.total_loudness(excitation, spread_input=False),
                "sharpness": _cb.sharpness(excitation, spread_input=False)}

    def _cdim(self, sig: FloatArray, idx: IntArray) -> FloatArray:
        """Estimate the correlation dimension of the frames in ``idx``.

        ``features.cdim`` converts its input to int16, so the signal is
        normalized to its peak first, which uses the full range.
        """
        cdp = self._params.cdim
        segs = self._cdim_cutter.transform(tools.normalize(sig))
        return features.cdim(segs.data[:, idx], cdp.delay, cdp.m_dim,
                             cdp.n_bins, cdp.scaling_size).ravel()


class MfccExtractor:
    """Compute the Mel-frequency cepstral coefficients of a signal."""
    def __init__(self, params: MfccTrackParams) -> None:
        """Set up the transform.

        Args:
            params:  Parameters of the coefficients
        """
        mfp = params.mfcc
        self._mfcc = Mfcc(mfp.stft, mfp.fb, mfp.cepstrum,
                          preemphasis=mfp.preemphasis or 0.0,
                          floor_dbfs=mfp.floor_dbfs)
        self._params = MfccTrackParams(level=params.level,
                                       mfcc=self._mfcc.params)

    @property
    def params(self) -> MfccTrackParams:
        """Return parameters."""
        return self._params

    def transform(self, sig: FloatArray) -> pd.DataFrame:
        """Compute the Mel-frequency cepstral coefficients of ``sig``.

        Args:
            sig:  Single-channel signal, shaped ``(n_samples, 1)``

        Returns:
            One column per coefficient, ``c0`` to ``c<n-1>``, and one row per
            frame, indexed by the frame centre in seconds.

        Raises:
            ValueError: If ``sig`` is silent, or too short for a single
                complete frame.
        """
        stp = self._params.mfcc.stft
        hop = hop_size(stp)
        idx = full_frames(sig.shape[0], stp.n_perseg, hop)
        if idx.size == 0:
            raise ValueError(f"Signal of {sig.shape[0]} samples holds no "
                             f"complete {stp.n_perseg}-sample frame.")

        coefs = self._mfcc.transform(condition(sig, self._params.level)).coefs
        return pd.DataFrame(coefs[:, idx].T,
                            columns=[f"c{i}" for i in range(coefs.shape[0])],
                            index=pd.Index(idx*hop/stp.fps, name="time"))


@dataclass(frozen=True)
class FileStatus:
    """Outcome of processing one input file."""
    path: str
    stem: str
    status: Literal["ok", "skipped", "error"]
    duration_s: float = np.nan
    n_frames_timbre: int = 0
    n_frames_mfcc: int = 0
    error: str = ""


class _Worker:
    # pylint: disable = too-few-public-methods
    """State of a pool worker process, set up by ``_init_worker``."""
    timbre: TimbreExtractor
    mfcc: MfccExtractor


def _init_worker(queue: "Queue[logging.LogRecord]", level: int,
                 timbre: TimbreTrackParams, mfcc: MfccTrackParams) -> None:
    """Set up logging and the extractors of a pool worker process.

    Workers do not inherit the handlers of the main process. They hand their
    records to ``queue`` instead, which the main process drains.

    Args:
        queue:   Queue drained by the ``QueueListener`` of the main process
        level:   Logging level
        timbre:  Parameters of the timbre features
        mfcc:    Parameters of the Mel-frequency cepstral coefficients
    """
    proc = mp.current_process()
    proc.name = "worker-" + proc.name.rsplit("-", 1)[-1]
    LOGGER.handlers[:] = [logging.handlers.QueueHandler(queue)]
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    _Worker.timbre = TimbreExtractor(timbre)
    _Worker.mfcc = MfccExtractor(mfcc)


def output_paths(path: pathlib.Path, outpath: pathlib.Path
                 ) -> tuple[pathlib.Path, pathlib.Path]:
    """Return where the features of ``path`` are stored.

    Args:
        path:     Input audio file
        outpath:  Root of the output tree

    Returns:
        Paths of the timbre and the MFCC parquet file.
    """
    name = f"{path.stem}.parquet"
    return (outpath / "timbre" / "files" / name,
            outpath / "mfcc" / "files" / name)


def read_audio(path: pathlib.Path, fps: int) -> FloatArray:
    """Read an audio file, mixed down to a single channel.

    Args:
        path:  Audio file
        fps:   Expected sample rate

    Returns:
        Signal shaped ``(n_samples, 1)``.

    Raises:
        ValueError: If the file is not sampled at ``fps``.
    """
    snd = AudioFile(path)
    try:
        if snd.fps != fps:
            raise ValueError(f"Sample rate is {snd.fps} Hz, expected {fps} Hz.")
        return floatarray(snd.read(mono=True))
    finally:
        snd.close()


def write_parquet(frame: pd.DataFrame, path: pathlib.Path,
                  attrs: dict[str, Any]) -> None:
    """Write ``frame`` to ``path`` atomically.

    The frame is written to a temporary file first and then moved into
    place, so that an interrupted run never leaves a truncated file behind.

    Args:
        frame:  Data to write
        path:   Target file
        attrs:  Metadata, stored in the parquet file along with ``frame``
    """
    frame.attrs.update(attrs)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, engine="pyarrow")
    os.replace(tmp, path)


def process_file(path: pathlib.Path, outpath: pathlib.Path) -> FileStatus:
    """Compute and store both feature sets of ``path``.

    This is the task run by the pool workers. It never raises: any failure
    is logged and reported in the returned status.

    Args:
        path:     Input audio file
        outpath:  Root of the output tree

    Returns:
        Outcome of processing ``path``.
    """
    timbre, mfcc = _Worker.timbre, _Worker.mfcc
    LOGGER.debug("processing %s", path)
    start = time.perf_counter()
    try:
        sig = read_audio(path, timbre.params.stft.fps)
        tdf = timbre.transform(sig)
        mdf = mfcc.transform(sig)
        attrs = {"source": str(path), "n_samples": sig.shape[0],
                 "fps": timbre.params.stft.fps}
        timbre_path, mfcc_path = output_paths(path, outpath)
        write_parquet(tdf, timbre_path, attrs)
        write_parquet(mdf, mfcc_path, attrs)
    except Exception as err:        # pylint: disable = broad-exception-caught
        LOGGER.exception("failed %s", path)
        return FileStatus(str(path), path.stem, "error",
                          error=f"{type(err).__name__}: {err}")

    LOGGER.debug("finished %s: %d timbre / %d MFCC frames in %.2f s", path,
                 len(tdf), len(mdf), time.perf_counter()-start)
    return FileStatus(str(path), path.stem, "ok",
                      duration_s=sig.shape[0]/timbre.params.stft.fps,
                      n_frames_timbre=len(tdf), n_frames_mfcc=len(mdf))


def collect_files(inputs: Sequence[pathlib.Path]) -> list[pathlib.Path]:
    """Collect the wav files among ``inputs``.

    Args:
        inputs:  Files, or directories to search recursively

    Returns:
        Sorted list of files, each listed once.

    Raises:
        FileNotFoundError: If an input does not exist.
    """
    files: dict[pathlib.Path, pathlib.Path] = {}
    for inp in inputs:
        if inp.is_dir():
            found = (pth for pth in inp.rglob("*")
                     if pth.is_file() and pth.suffix.lower() == ".wav")
        elif inp.is_file():
            found = (pth for pth in (inp,))
        else:
            raise FileNotFoundError(f"No such file or directory: {inp}")
        for pth in found:
            files.setdefault(pth.resolve(), pth)
    return sorted(files.values())


def write_params(outpath: pathlib.Path, name: str, params: BaseModel,
                 overwrite: bool) -> None:
    """Store the parameters of a feature set in ``<outpath>/<name>``.

    Args:
        outpath:    Root of the output tree
        name:       Name of the feature set
        params:     Its parameters
        overwrite:  If ``True``, replace parameters that differ

    Raises:
        ValueError: If ``outpath`` holds features computed with other
            parameters, and ``overwrite`` is ``False``.
    """
    path = outpath / name / "params.json"
    new = params.model_dump_json(indent=2)
    if path.exists() and path.read_text(encoding="utf-8") != new and not overwrite:
        raise ValueError(f"{path} holds other parameters. Pass --overwrite "
                         "to recompute all files, or choose another output "
                         "path.")
    (outpath / name / "files").mkdir(parents=True, exist_ok=True)
    path.write_text(new, encoding="utf-8")


def run_pool(files: Sequence[pathlib.Path], outpath: pathlib.Path,
             n_workers: int, timbre: TimbreTrackParams,
             mfcc: MfccTrackParams) -> list[FileStatus]:
    """Process ``files`` on a pool of worker processes.

    Processes rather than threads: the correlation dimension, which takes
    most of the time, holds the GIL, and so do the per-frame Python loops of
    loudness, sharpness and roughness.

    Args:
        files:      Input audio files
        outpath:    Root of the output tree
        n_workers:  Number of worker processes
        timbre:     Parameters of the timbre features
        mfcc:       Parameters of the Mel-frequency cepstral coefficients

    Returns:
        Outcome of each file, in order of completion.
    """
    # Forking while the listener thread runs could copy a held lock into the
    # workers. Start them as fresh interpreters instead.
    ctx = mp.get_context("spawn")
    queue: Queue[logging.LogRecord] = ctx.Queue()
    handlers = LOGGER.handlers[:]
    listener = logging.handlers.QueueListener(queue, *handlers)
    listener.start()
    # Route the records of this process through the queue, too, so that the
    # listener emits them in the order they were logged.
    LOGGER.handlers[:] = [logging.handlers.QueueHandler(queue)]

    out: list[FileStatus] = []
    step = max(1, len(files) // 10)
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=(queue, LOGGER.level, timbre, mfcc)
                                 ) as pool:
            futures: dict[Future[FileStatus], pathlib.Path] = {
                pool.submit(process_file, pth, outpath): pth for pth in files}
            try:
                for fut in as_completed(futures):
                    out.append(_result(fut, futures[fut]))
                    if len(out) % step == 0 and len(out) < len(files):
                        LOGGER.info("progress: %d/%d files", len(out), len(files))
            except KeyboardInterrupt:
                LOGGER.error("interrupted; cancelling the remaining files")
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    finally:
        listener.stop()
        LOGGER.handlers[:] = handlers
    return out


def _result(fut: Future[FileStatus], path: pathlib.Path) -> FileStatus:
    """Return the outcome of ``fut``, also if its worker died."""
    try:
        return fut.result()
    except Exception as err:        # pylint: disable = broad-exception-caught
        LOGGER.error("failed %s: %s", path, err)
        return FileStatus(str(path), path.stem, "error",
                          error=f"{type(err).__name__}: {err}")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Extract timbre features and MFCCs from wav files.")
    parser.add_argument("inputs", type=pathlib.Path, nargs="+",
                        metavar="INPUT",
                        help="wav file, or directory to search recursively")
    parser.add_argument("-o", "--outpath", type=pathlib.Path, required=True,
                        help="root of the output tree")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1,
                        help="number of worker processes (default: number "
                             "of CPUs)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute files whose outputs exist")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="log each file as it is processed")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def setup_logging(verbose: bool) -> None:
    """Log to stderr, and each file as it is processed if ``verbose``."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    LOGGER.handlers[:] = [handler]
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    """Run the feature extraction.

    Args:
        argv:  Command line arguments, without the program name

    Returns:
        Exit status: 0 if all files succeeded, 1 if any failed, and 2 if the
        run was aborted before processing any.
    """
    args = parse_args(argv)
    setup_logging(args.verbose)
    outpath: pathlib.Path = args.outpath

    try:
        files = collect_files(args.inputs)
    except FileNotFoundError as err:
        LOGGER.error("%s", err)
        return 2
    if not files:
        LOGGER.error("no wav files found in %s",
                     ", ".join(str(inp) for inp in args.inputs))
        return 2

    dups = {stem: n for stem, n in Counter(pth.stem for pth in files).items()
            if n > 1}
    if dups:
        LOGGER.error("%d file names occur more than once, so their outputs "
                     "would collide: %s", len(dups), ", ".join(sorted(dups)))
        return 2

    timbre = TimbreExtractor(TIMBRE).params
    mfcc = MfccExtractor(MFCC).params
    try:
        write_params(outpath, "timbre", timbre, args.overwrite)
        write_params(outpath, "mfcc", mfcc, args.overwrite)
    except ValueError as err:
        LOGGER.error("%s", err)
        return 2

    todo: list[pathlib.Path] = []
    statuses: list[FileStatus] = []
    for pth in files:
        if args.overwrite or not all(out.exists() for out in output_paths(pth, outpath)):
            todo.append(pth)
        else:
            LOGGER.debug("skipped %s: outputs exist", pth)
            statuses.append(FileStatus(str(pth), pth.stem, "skipped"))

    n_workers = min(args.jobs, max(1, len(todo)))
    LOGGER.info("extracting features of %d files (%d skipped, outputs exist) "
                "into %s, worker processes: %d", len(todo), len(statuses),
                outpath, n_workers)
    start = time.perf_counter()
    try:
        if todo:
            statuses.extend(run_pool(todo, outpath, n_workers, timbre, mfcc))
    except KeyboardInterrupt:
        return 130

    manifest = outpath / "manifest.csv"
    pd.DataFrame([asdict(st) for st in statuses]).to_csv(manifest, index=False)
    counts = Counter(st.status for st in statuses)
    LOGGER.info("done in %.1f s: %d ok, %d skipped, %d failed; manifest: %s",
                time.perf_counter()-start, counts["ok"], counts["skipped"],
                counts["error"], manifest)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
