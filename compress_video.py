"""Prepare security clips for Perceptron: convert to MP4 and compress under the upload limit.

Requires ffmpeg and ffprobe on PATH (``brew install ffmpeg`` on macOS).

Examples::

    python compress_video.py ~/Downloads/example_vid.mov
    python compress_video.py clip.avi -o ready.mp4 --max-mb 15
    python compress_video.py long_clip.mp4 --trim 128
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Matches analyze.MAX_VIDEO_MB — keep in sync with the app's upload guard.
DEFAULT_MAX_MB = 15
DEFAULT_TRIM_SECONDS = 128  # Perceptron meaningfully samples ~2 minutes of footage.
ACCEPTED_SUFFIXES = {".mp4", ".webm"}


class VideoPreparationError(Exception):
    """Raised when a clip cannot be converted or compressed for upload."""


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise VideoPreparationError(
            f"{name} not found on PATH. Install ffmpeg (includes ffprobe), e.g.\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg"
        )
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def probe(path: Path) -> dict:
    _require_tool("ffprobe")
    result = _run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def _video_stream(probe_data: dict) -> dict | None:
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    return None


def duration_seconds(probe_data: dict) -> float:
    fmt = probe_data.get("format", {})
    if "duration" in fmt:
        return float(fmt["duration"])
    video = _video_stream(probe_data)
    if video and "duration" in video:
        return float(video["duration"])
    return 0.0


def is_mp4_h264(path: Path, probe_data: dict) -> bool:
    if path.suffix.lower() != ".mp4":
        return False
    video = _video_stream(probe_data)
    return bool(video and video.get("codec_name") == "h264")


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_compressed.mp4")


def _scale_filter(probe_data: dict, max_height: int | None) -> str | None:
    video = _video_stream(probe_data)
    if not video or max_height is None:
        return None
    height = int(video.get("height") or 0)
    if height <= max_height:
        return None
    return f"scale=-2:{max_height}"


def encode(
    input_path: Path,
    output_path: Path,
    *,
    probe_data: dict,
    crf: int,
    max_height: int | None,
    trim_seconds: float | None,
    audio_bitrate_k: int,
) -> None:
    _require_tool("ffmpeg")
    duration = duration_seconds(probe_data)
    if trim_seconds is not None and duration > trim_seconds:
        duration = trim_seconds

    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]

    if trim_seconds is not None:
        cmd.extend(["-t", str(trim_seconds)])

    vf = _scale_filter(probe_data, max_height)
    if vf:
        cmd.extend(["-vf", vf])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            f"{audio_bitrate_k}k",
            "-movflags",
            "+faststart",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            str(output_path),
        ]
    )
    _run(cmd)


def needs_preparation(path: Path, *, max_mb: float = DEFAULT_MAX_MB) -> bool:
    if not path.is_file():
        return False
    if size_mb(path) > max_mb:
        return True
    if path.suffix.lower() not in ACCEPTED_SUFFIXES:
        return True
    if path.suffix.lower() == ".mp4" and ffmpeg_available():
        try:
            return not is_mp4_h264(path, probe(path))
        except Exception:
            return True
    return False


def prepare_video_for_upload(
    input_path: Path,
    *,
    max_mb: float = DEFAULT_MAX_MB,
    trim_seconds: float | None = None,
) -> Path:
    """Return an API-ready MP4 path, converting/compressing the upload when needed."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    probe_data: dict | None = None
    duration = 0.0
    if ffmpeg_available():
        probe_data = probe(input_path)
        duration = duration_seconds(probe_data)

    if trim_seconds is None and duration > DEFAULT_TRIM_SECONDS and size_mb(input_path) > max_mb:
        trim_seconds = DEFAULT_TRIM_SECONDS

    needs_compress = needs_preparation(input_path, max_mb=max_mb)
    needs_trim = trim_seconds is not None and duration > trim_seconds
    if not needs_compress and not needs_trim:
        return input_path

    if not ffmpeg_available():
        if needs_compress:
            raise VideoPreparationError(
                f"Uploaded {input_path.suffix.lower() or 'video'} ({size_mb(input_path):.1f} MB) "
                f"must be converted to MP4 under {max_mb} MB before analysis. "
                "Install ffmpeg (`brew install ffmpeg`) or run:\n"
                f"  python compress_video.py {input_path}"
            )
        return input_path

    if probe_data is None:
        probe_data = probe(input_path)

    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", prefix="perceptron_prepared_")
    os.close(fd)
    output_path = Path(tmp_name)
    try:
        return compress_video(
            input_path,
            output_path,
            max_mb=max_mb,
            trim_seconds=trim_seconds,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


def compress_video(
    input_path: Path,
    output_path: Path,
    *,
    max_mb: float = DEFAULT_MAX_MB,
    trim_seconds: float | None = None,
) -> Path:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    probe_data = probe(input_path)
    if _video_stream(probe_data) is None:
        raise ValueError(f"No video stream found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        is_mp4_h264(input_path, probe_data)
        and size_mb(input_path) <= max_mb
        and trim_seconds is None
    ):
        if input_path.resolve() != output_path.resolve():
            shutil.copy2(input_path, output_path)
        return output_path

    # Try progressively stronger compression until under the size cap.
    attempts: list[tuple[int, int | None]] = [
        (28, None),
        (32, None),
        (34, 720),
        (36, 720),
        (38, 480),
    ]

    temp_output = output_path
    if output_path.exists():
        temp_output = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")

    try:
        for crf, max_height in attempts:
            encode(
                input_path,
                temp_output,
                probe_data=probe_data,
                crf=crf,
                max_height=max_height,
                trim_seconds=trim_seconds,
                audio_bitrate_k=96,
            )
            if size_mb(temp_output) <= max_mb:
                if temp_output != output_path:
                    temp_output.replace(output_path)
                return output_path

        final_mb = size_mb(temp_output)
        raise RuntimeError(
            f"Could not compress below {max_mb} MB (best result: {final_mb:.1f} MB). "
            "Try a shorter clip with --trim or lower resolution source footage."
        )
    finally:
        if temp_output != output_path and temp_output.exists():
            temp_output.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert security footage to MP4 (H.264/AAC) and compress for Perceptron uploads."
    )
    parser.add_argument("input", type=Path, help="Source video (MOV, AVI, MKV, WebM, etc.)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output MP4 path (default: <name>_compressed.mp4 next to input)",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=DEFAULT_MAX_MB,
        help=f"Target max file size in MB (default: {DEFAULT_MAX_MB})",
    )
    parser.add_argument(
        "--trim",
        type=float,
        metavar="SECONDS",
        help=(
            f"Keep only the first N seconds (default: no trim; "
            f"Perceptron samples ~{DEFAULT_TRIM_SECONDS}s)"
        ),
    )
    args = parser.parse_args()

    output = args.output or default_output_path(args.input)
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")

    try:
        result = compress_video(
            args.input,
            output,
            max_mb=args.max_mb,
            trim_seconds=args.trim,
        )
    except VideoPreparationError as exc:
        sys.exit(str(exc))
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
            sys.exit(f"ffmpeg failed: {detail or exc}")
        sys.exit(str(exc))

    print(f"Wrote {result} ({size_mb(result):.2f} MB)")


if __name__ == "__main__":
    main()
