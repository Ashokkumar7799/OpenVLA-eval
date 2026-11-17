import os
import numpy as np

assets_dir = r"E:\mediapipe\physical_ai_eval\mujoco_menagerie\aloha\assets"
os.makedirs(assets_dir, exist_ok=True)

N_SECTOR = 64  # High-resolution smooth circular profile

# ============================================================
# Geometry Specs — Wide Short Bowl-like Cup matching real video
# Real cup: very wide diameter, short height, heavily tapered
# ============================================================
R_top_out = 0.070       # 14.0 cm top outer diameter (much wider)
R_top_in = 0.066        # 13.2 cm top inner diameter
R_bot_out = 0.035       # 7.0 cm bottom outer diameter (narrower bottom = more taper)
R_bot_in = 0.031        # 6.2 cm bottom inner diameter
H = 0.045               # 4.5 cm height (shorter — bowl-like)
Z_base = 0.002
Z_floor = 0.005

# 1. Generate Outer Blue Shell (cup_outer.obj)
verts_outer = []
faces_outer = []

# Base center vertex
verts_outer.append((0.0, 0.0, Z_base))
# Bottom ring vertices (index 1 to N_SECTOR)
for i in range(N_SECTOR):
    theta = 2.0 * np.pi * i / N_SECTOR
    verts_outer.append((R_bot_out * np.cos(theta), R_bot_out * np.sin(theta), Z_base))

# Top ring vertices (index N_SECTOR+1 to 2*N_SECTOR)
for i in range(N_SECTOR):
    theta = 2.0 * np.pi * i / N_SECTOR
    verts_outer.append((R_top_out * np.cos(theta), R_top_out * np.sin(theta), H))

# Faces for base disk
for i in range(N_SECTOR):
    faces_outer.append((1, 2 + (i + 1) % N_SECTOR, 2 + i))

# Faces for outer conical wall
for i in range(N_SECTOR):
    b1 = 2 + i
    b2 = 2 + (i + 1) % N_SECTOR
    t1 = 2 + N_SECTOR + i
    t2 = 2 + N_SECTOR + (i + 1) % N_SECTOR
    faces_outer.append((b1, b2, t2))
    faces_outer.append((b1, t2, t1))

with open(os.path.join(assets_dir, "cup_outer.obj"), "w") as f:
    f.write("# Tapered Cup Outer Shell - Wide Short Bowl\n")
    for v in verts_outer:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for fa in faces_outer:
        f.write(f"f {fa[0]} {fa[1]} {fa[2]}\n")

print(f"Generated cup_outer.obj with {len(verts_outer)} vertices, {len(faces_outer)} faces")

# 2. Generate Inner White Hollow Shell (cup_inner.obj)
verts_inner = []
faces_inner = []

# Floor center vertex
verts_inner.append((0.0, 0.0, Z_floor))
# Bottom inner ring
for i in range(N_SECTOR):
    theta = 2.0 * np.pi * i / N_SECTOR
    verts_inner.append((R_bot_in * np.cos(theta), R_bot_in * np.sin(theta), Z_floor))

# Top inner ring
for i in range(N_SECTOR):
    theta = 2.0 * np.pi * i / N_SECTOR
    verts_inner.append((R_top_in * np.cos(theta), R_top_in * np.sin(theta), H))

# Floor disk faces (pointing up into cavity)
for i in range(N_SECTOR):
    faces_inner.append((1, 2 + i, 2 + (i + 1) % N_SECTOR))

# Inner conical wall faces (pointing inward into cavity)
for i in range(N_SECTOR):
    b1 = 2 + i
    b2 = 2 + (i + 1) % N_SECTOR
    t1 = 2 + N_SECTOR + i
    t2 = 2 + N_SECTOR + (i + 1) % N_SECTOR
    faces_inner.append((b1, t2, b2))
    faces_inner.append((b1, t1, t2))

with open(os.path.join(assets_dir, "cup_inner.obj"), "w") as f:
    f.write("# Tapered Cup Inner Hollow Cavity - Wide Short Bowl\n")
    for v in verts_inner:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for fa in faces_inner:
        f.write(f"f {fa[0]} {fa[1]} {fa[2]}\n")

print(f"Generated cup_inner.obj with {len(verts_inner)} vertices, {len(faces_inner)} faces")

# 3. Generate White Curled Top Rim (cup_rim.obj)
verts_rim = []
faces_rim = []
N_RIM_STEPS = 8
R_rim_center = (R_top_out + R_top_in) / 2.0
r_rim_tube = (R_top_out - R_top_in) / 2.0  # 2 mm tube radius

for j in range(N_RIM_STEPS + 1):
    phi = np.pi * j / N_RIM_STEPS  # from 0 (outer) to pi (inner)
    r_j = R_rim_center + r_rim_tube * np.cos(phi)
    z_j = H + r_rim_tube * np.sin(phi)
    for i in range(N_SECTOR):
        theta = 2.0 * np.pi * i / N_SECTOR
        verts_rim.append((r_j * np.cos(theta), r_j * np.sin(theta), z_j))

for j in range(N_RIM_STEPS):
    for i in range(N_SECTOR):
        v00 = 1 + j * N_SECTOR + i
        v01 = 1 + j * N_SECTOR + (i + 1) % N_SECTOR
        v10 = 1 + (j + 1) * N_SECTOR + i
        v11 = 1 + (j + 1) * N_SECTOR + (i + 1) % N_SECTOR
        faces_rim.append((v00, v01, v11))
        faces_rim.append((v00, v11, v10))

with open(os.path.join(assets_dir, "cup_rim.obj"), "w") as f:
    f.write("# Tapered Cup White Curled Rim - Wide Short Bowl\n")
    for v in verts_rim:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for fa in faces_rim:
        f.write(f"f {fa[0]} {fa[1]} {fa[2]}\n")

print(f"Generated cup_rim.obj with {len(verts_rim)} vertices, {len(faces_rim)} faces")
print(f"\nCup Dimensions:")
print(f"  Height: {H*100:.1f} cm")
print(f"  Top Diameter (outer): {R_top_out*200:.1f} cm")
print(f"  Bottom Diameter (outer): {R_bot_out*200:.1f} cm")
print(f"  Wall Thickness: {(R_top_out - R_top_in)*1000:.1f} mm")
