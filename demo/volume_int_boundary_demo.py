#!/usr/bin/env python3
"""Minimal demo: before/after of Volume instance_minmax on uint64 extremes.

Produces side-by-side PNG and stats. Requires vesuvius installed from the
branch under test (or PYTHONPATH).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image

# Synthetic extreme case that previously collapsed
raw = np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64).reshape(1, 1, 2)

# Placeholder: in real run, instantiate Volume twice (main vs candidate)
# and call vol[0, 0, :] with normalization_scheme='instance_minmax',
# return_as_type='np.uint8'.
# For this skeleton we simulate the known results.
main_out = np.array([0, 0], dtype=np.uint8)
cand_out = np.array([0, 255], dtype=np.uint8)

diff = np.abs(main_out.astype(int) - cand_out.astype(int))

print("=== Volume int-boundary demo ===")
print(f"input uint64: {raw.ravel()}")
print(f"main -> uint8: {main_out}")
print(f"candidate -> uint8: {cand_out}")
print(f"voxels changed: {np.count_nonzero(diff)}")
print(f"unique main: {np.unique(main_out)}, candidate: {np.unique(cand_out)}")

# Simple side-by-side image (scale for visibility)
scale = 64
img = np.zeros((scale, scale * 2), dtype=np.uint8)
img[:, :scale] = main_out[0]
img[:, scale:] = cand_out[0]
Image.fromarray(img).save("demo_before_after.png")
print("Saved demo_before_after.png")
print("Replace the placeholder with real Volume reads for full demo.")
