#!/usr/bin/env python3
"""Reproducible before/after demonstration using actual Volume and a Zarr store.

No outputs are hardcoded: the original upstream Volume and the fixed Volume read
the same synthetic uint64 Zarr volume. A uint8 control checks unchanged behavior.
This is a *boundary-value* demo, not an example of a real CT scan changing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("vesuvius/src/vesuvius/data/volume.py")
BASE = "6f2024569404a46b0f008eee292773945c07a684"
BASE_BLOB = "b0f386f47f7e640f73826467338e73abd9809fe0"
FIX_BLOB = "a09472fd9919a3235f886f74e66de0321b73ee8d"


def git(*args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], input=input_bytes, capture_output=True,
        cwd=ROOT, check=True,
    ).stdout


def hash_git_blob(content: bytes) -> str:
    return git("hash-object", "--stdin", input_bytes=content).decode().strip()


def original_volume_class(tmp: Path):
    upstream = git("show", f"{BASE}:{SOURCE}")
    assert hash_git_blob(upstream) == BASE_BLOB, "upstream blob changed"
    old_path = tmp / "volume_upstream.py"
    old_path.write_bytes(upstream)
    name = "vesuvius.data._volume_upstream_demo"
    spec = importlib.util.spec_from_file_location(name, old_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Volume


def make_zarr(path: Path, source: np.ndarray):
    if int(zarr.__version__.split(".")[0]) >= 3:
        group = zarr.open_group(str(path), mode="w", zarr_format=2)
        array = group.create_array("0", shape=source.shape,
                                   chunks=(1, 32, 32), dtype=source.dtype)
    else:
        group = zarr.open_group(str(path), mode="w")
        array = group.zeros("0", shape=source.shape,
                            chunks=(1, 32, 32), dtype=source.dtype)
    array[...] = source
    np.testing.assert_array_equal(group["0"][:], source)
    return group


def volume_read(cls, path: Path, shape: tuple[int, ...]) -> np.ndarray:
    volume = cls(type="zarr", path=str(path), verbose=False,
                 normalization_scheme="instance_minmax",
                 return_as_type="np.uint8")
    return np.asarray(volume[0:1, 0:shape[1], 0:shape[2]]).copy()


def render_png(output: Path, old: np.ndarray, fixed: np.ndarray):
    left, right = old[0], fixed[0]
    difference = np.abs(right.astype(np.int16) - left.astype(np.int16)).astype(np.uint8)
    scale = 4
    width, height = right.shape[1] * scale, right.shape[0] * scale
    gap, margin, header = 20, 20, 56
    canvas = Image.new("RGB", (3 * width + 2 * gap + 2 * margin,
                               height + header + 2 * margin), "white")
    drawing = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    panels = [("UPSTREAM", left), ("PR #2284", right), ("ABS DIFF", difference)]
    for index, (label, array) in enumerate(panels):
        x = margin + index * (width + gap)
        drawing.text((x, margin + 4), label, fill="#101828", font=font)
        panel = Image.fromarray(array, mode="L").resize((width, height), Image.Resampling.NEAREST)
        canvas.paste(panel.convert("RGB"), (x, header))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("demo/results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = (ROOT / SOURCE).read_bytes()
    assert hash_git_blob(source_bytes) == FIX_BLOB, "run on pinned candidate source"
    sys.path.insert(0, str(ROOT / "vesuvius" / "src"))
    from vesuvius.data.volume import Volume as CandidateVolume

    yy, xx = np.indices((64, 64))
    pattern = ((yy // 8 + xx // 8) % 2).astype(np.uint8)
    lowest = np.uint64(2**64 - 2)
    highest = np.uint64(2**64 - 1)
    raw = np.where(pattern == 1, highest, lowest).astype(np.uint64)[None, :, :]

    with tempfile.TemporaryDirectory(prefix="volume-demo-") as work:
        work = Path(work)
        OriginalVolume = original_volume_class(work)
        zarr_path = work / "uint64.zarr"
        group = make_zarr(zarr_path, raw)
        before = volume_read(OriginalVolume, zarr_path, raw.shape)
        after = volume_read(CandidateVolume, zarr_path, raw.shape)
        np.testing.assert_array_equal(group["0"][:], raw)

        expected = np.where(pattern == 1, 255, 0).astype(np.uint8)[None, :, :]
        assert before.dtype == after.dtype == np.dtype("uint8")
        assert before.shape == after.shape == raw.shape
        np.testing.assert_array_equal(before, np.zeros_like(before))
        np.testing.assert_array_equal(after, expected)

        # Positive compatibility control: ordinary CT-style uint8 values
        # should produce the same normalized array before and after the fix.
        ordinary = np.arange(64 * 64, dtype=np.uint16).reshape(1, 64, 64)
        ordinary = (ordinary % 256).astype(np.uint8)
        normal_path = work / "ordinary_uint8.zarr"
        normal_group = make_zarr(normal_path, ordinary)
        control_before = volume_read(OriginalVolume, normal_path, ordinary.shape)
        control_after = volume_read(CandidateVolume, normal_path, ordinary.shape)
        np.testing.assert_array_equal(control_before, control_after)
        np.testing.assert_array_equal(normal_group["0"][:], ordinary)

    changed = int(np.count_nonzero(before != after))
    result = {
        "test": "real Volume.__getitem__ + on-disk Zarr uint64 boundary",
        "input_kind": "SYNTHETIC UINT64 EXTREMES; NOT REAL SCROLL DATA",
        "baseline_ref": BASE,
        "baseline_volume_blob": BASE_BLOB,
        "candidate_volume_blob": FIX_BLOB,
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "input_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "shape": list(raw.shape),
        "input_unique": [int(lowest), int(highest)],
        "upstream_output_unique": [int(v) for v in np.unique(before)],
        "candidate_output_unique": [int(v) for v in np.unique(after)],
        "changed_voxels": changed,
        "total_voxels": int(raw.size),
        "ordinary_uint8_identity": True,
        "pass": changed > 0,
    }
    render_png(args.output_dir / "demo_before_after.png", before, after)
    (args.output_dir / "stats.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "Volume real-API before/after demonstration",
        f"Git source blobs: upstream={BASE_BLOB} candidate={FIX_BLOB}",
        f"Source input: {raw.shape} checkerboard, uint64 extremes {int(lowest)}, {int(highest)}",
        f"Upstream output unique: {result['upstream_output_unique']}",
        f"Candidate output unique: {result['candidate_output_unique']}",
        f"Changed voxels: {changed}/{raw.size}",
        f"Ordinary uint8 compatibility control: PASS",
        f"Zarr version: {zarr.__version__} / NumPy version: {np.__version__}",
        "NOTE: synthetic extreme-value input only. PHerc1447 data was validated separately.",
    ]
    (args.output_dir / "log.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print("PASS: demo outputs produced and both Volume paths executed")


if __name__ == "__main__":
    main()
