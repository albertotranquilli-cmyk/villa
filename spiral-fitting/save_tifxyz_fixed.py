"""
Minimal compatible patch proposal for save_tifxyz.
- Fail-fast on non-finite values (aligned with save_combined_tifxyz)
- ANY -> ALL for partial sentinels
- json.dump(..., allow_nan=False) as final guardrail
"""
import numpy as np
import json
from pathlib import Path
from PIL import Image

def save_tifxyz_fixed(zyxs, path, uuid, step_size=1.0, voxel_size_um=1.0, source="test", *, layout_metadata=None):
    # 1. Fail-fast on non-finite
    if not np.isfinite(zyxs[zyxs != -1]).all():
        raise ValueError("Non-finite values in non-sentinel vertices")

    path = Path(path) / uuid
    path.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.asarray(zyxs[..., 2], dtype=np.float32)).save(path / "x.tif")
    Image.fromarray(np.asarray(zyxs[..., 1], dtype=np.float32)).save(path / "y.tif")
    Image.fromarray(np.asarray(zyxs[..., 0], dtype=np.float32)).save(path / "z.tif")

    # 2. ALL predicate for partial sentinels
    valid_vertex = np.all(zyxs != -1, axis=-1)
    valid_quad = (valid_vertex[:-1, :-1] & valid_vertex[1:, :-1] &
                  valid_vertex[:-1, 1:] & valid_vertex[1:, 1:])
    area_vx2 = int(valid_quad.sum()) * step_size ** 2
    valid_zyxs = zyxs[valid_vertex]
    bbox = (
        [valid_zyxs.min(axis=0)[::-1].tolist(), valid_zyxs.max(axis=0)[::-1].tolist()]
        if valid_vertex.any()
        else [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
    )
    metadata = {
        "scale": [1 / step_size, 1 / step_size],
        "bbox": bbox,
        "area_vx2": area_vx2,
        "source": source,
    }
    # 3. Strict JSON
    (path / "meta.json").write_text(json.dumps(metadata, allow_nan=False))
    return metadata

# Self-test
if __name__ == "__main__":
    import tempfile
    cases = {
        "nan": (np.array([[[1.,2.,3.],[4.,5.,np.nan]]], dtype=np.float32), True),
        "inf": (np.array([[[1.,2.,3.],[4.,5.,np.inf]]], dtype=np.float32), True),
        "partial": (np.array([[[1.,2.,3.],[-1.,10.,-1.]]], dtype=np.float32), False),
        "clean": (np.array([[[1.,2.,3.],[4.,5.,6.]]], dtype=np.float32), False),
    }
    for name, (zyxs, should_fail) in cases.items():
        try:
            with tempfile.TemporaryDirectory() as tmp:
                meta = save_tifxyz_fixed(zyxs, tmp, name)
                print(f"{name}: OK, bbox={meta['bbox']}")
                assert not should_fail
        except ValueError as e:
            print(f"{name}: REJECTED ({e})")
            assert should_fail
    print("All self-tests passed")
