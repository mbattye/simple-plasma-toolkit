"""Geometry / STL loading / direction / neighbour helpers."""

from __future__ import annotations

import numpy as np
import pytest

from heatstl.geometry import (
    SurfaceMesh,
    auto_hex_pitch,
    direction_from_angles,
    hex_neighbor_offsets,
    parse_direction,
)


def test_parse_direction_normalises():
    v = parse_direction("0,0,-2")
    assert v == pytest.approx([0.0, 0.0, -1.0])


def test_parse_direction_bad_shape():
    with pytest.raises(ValueError):
        parse_direction("0,0")


def test_parse_direction_zero():
    with pytest.raises(ValueError):
        parse_direction("0,0,0")


def test_direction_from_angles_zero_is_minus_z():
    v = direction_from_angles(0.0, 0.0)
    assert v == pytest.approx([0.0, 0.0, -1.0])


def test_direction_from_angles_45_along_x():
    v = direction_from_angles(45.0, 0.0)
    expected = np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5)])
    assert v == pytest.approx(expected)


def test_direction_from_angles_90_is_unit_in_xy():
    v = direction_from_angles(90.0, 90.0)
    assert v == pytest.approx([0.0, 1.0, 0.0])


def _square_tile(half: float = 0.075, thickness: float = 0.025) -> SurfaceMesh:
    import trimesh
    box = trimesh.creation.box(extents=(2 * half, 2 * half, thickness))
    return SurfaceMesh(
        vertices=np.asarray(box.vertices, dtype=float),
        faces=np.asarray(box.faces, dtype=np.int64),
        face_normals=np.asarray(box.face_normals, dtype=float),
        face_areas=np.asarray(box.area_faces, dtype=float),
    )


def test_auto_hex_pitch_for_axis_aligned_box():
    surf = _square_tile(half=0.075)
    pitch = auto_hex_pitch(surf, np.array([0.0, 0.0, -1.0]))
    assert pitch == pytest.approx(0.150)


def test_hex_neighbor_offsets_in_perpendicular_plane():
    surf = _square_tile(half=0.075)
    p_hat = np.array([0.0, 0.0, -1.0])
    offs = hex_neighbor_offsets(surf, p_hat, pitch=0.16, gap=0.0)
    assert offs.shape == (6, 3)
    # All offsets perpendicular to p_hat.
    assert np.allclose(offs @ p_hat, 0.0, atol=1e-10)
    # All same distance from origin.
    dists = np.linalg.norm(offs, axis=1)
    assert np.allclose(dists, 0.16, atol=1e-10)
    # Six distinct angles, evenly spaced.
    angles = np.sort(np.arctan2(offs[:, 1], offs[:, 0]))
    diffs = np.diff(angles)
    assert np.allclose(diffs, np.deg2rad(60.0), atol=1e-10)
