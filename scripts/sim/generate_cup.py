#!/usr/bin/env python3
"""Procedural generator for the hollow tapered ALOHA target cup OBJ meshes."""

import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from physical_ai.paths import ALOHA_ASSETS_DIR


def main():
    os.makedirs(ALOHA_ASSETS_DIR, exist_ok=True)
    n_sector = 64

    r_top_out = 0.070
    r_top_in = 0.066
    r_bot_out = 0.035
    r_bot_in = 0.031
    h = 0.045
    z_base = 0.002
    z_floor = 0.005

    # 1. Outer Blue Shell (cup_outer.obj)
    verts_outer = [(0.0, 0.0, z_base)]
    for i in range(n_sector):
        th = 2.0 * np.pi * i / n_sector
        verts_outer.append((r_bot_out * np.cos(th), r_bot_out * np.sin(th), z_base))
    for i in range(n_sector):
        th = 2.0 * np.pi * i / n_sector
        verts_outer.append((r_top_out * np.cos(th), r_top_out * np.sin(th), h))

    faces_outer = [(1, 2 + (i + 1) % n_sector, 2 + i) for i in range(n_sector)]
    for i in range(n_sector):
        b1, b2 = 2 + i, 2 + (i + 1) % n_sector
        t1, t2 = 2 + n_sector + i, 2 + n_sector + (i + 1) % n_sector
        faces_outer.append((b1, b2, t2))
        faces_outer.append((b1, t2, t1))

    with open(os.path.join(ALOHA_ASSETS_DIR, "cup_outer.obj"), "w", encoding="utf-8") as f:
        f.write("# Tapered Cup Outer Shell\n")
        for v in verts_outer:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fa in faces_outer:
            f.write(f"f {fa[0]} {fa[1]} {fa[2]}\n")

    # 2. Inner White Hollow Shell (cup_inner.obj)
    verts_inner = [(0.0, 0.0, z_floor)]
    for i in range(n_sector):
        th = 2.0 * np.pi * i / n_sector
        verts_inner.append((r_bot_in * np.cos(th), r_bot_in * np.sin(th), z_floor))
    for i in range(n_sector):
        th = 2.0 * np.pi * i / n_sector
        verts_inner.append((r_top_in * np.cos(th), r_top_in * np.sin(th), h))

    faces_inner = [(1, 2 + i, 2 + (i + 1) % n_sector) for i in range(n_sector)]
    for i in range(n_sector):
        b1, b2 = 2 + i, 2 + (i + 1) % n_sector
        t1, t2 = 2 + n_sector + i, 2 + n_sector + (i + 1) % n_sector
        faces_inner.append((b1, t2, b2))
        faces_inner.append((b1, t1, t2))

    with open(os.path.join(ALOHA_ASSETS_DIR, "cup_inner.obj"), "w", encoding="utf-8") as f:
        f.write("# Tapered Cup Inner Hollow Cavity\n")
        for v in verts_inner:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fa in faces_inner:
            f.write(f"f {fa[0]} {fa[1]} {fa[2]}\n")

    # 3. Curled Top Rim (cup_rim.obj)
    verts_rim = []
    n_rim_steps = 8
    r_rim_center = (r_top_out + r_top_in) / 2.0
    r_rim_tube = (r_top_out - r_top_in) / 2.0
    for j in range(n_rim_steps + 1):
        phi = np.pi * j / n_rim_steps
        r_j = r_rim_center + r_rim_tube * np.cos(phi)
        z_j = h + r_rim_tube * np.sin(phi)
        for i in range(n_sector):
            th = 2.0 * np.pi * i / n_sector
            verts_rim.append((r_j * np.cos(th), r_j * np.sin(th), z_j))

    faces_rim = []
    for j in range(n_rim_steps):
        for i in range(n_sector):
            v00 = 1 + j * n_sector + i
            v01 = 1 + j * n_sector + (i + 1) % n_sector
            v10 = 1 + (j + 1) * n_sector + i
            v11 = 1 + (j + 1) * n_sector + (i + 1) % n_sector
            faces_rim.append((v00, v01, v11))
            faces_rim.append((v00, v11, v10))

    with open(os.path.join(ALOHA_ASSETS_DIR, "cup_rim.obj"), "w", encoding="utf-8") as f:
        f.write("# Tapered Cup White Curled Rim\n")
        for v in verts_rim:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fa in faces_rim:
            f.write(f"f {fa[0]} {fa[1]} {fa[2]}\n")

    print(f"Generated cup OBJ meshes in {ALOHA_ASSETS_DIR}")


if __name__ == "__main__":
    main()
