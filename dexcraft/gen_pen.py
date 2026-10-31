"""Procedural tools: a ballpoint-style pen, a calligraphy brush (bamboo
handle, metal ferrule, flexible tuft) and a chopstick. Visual meshes are surfaces of
revolution; writes assets/pen_*.stl, assets/brush_*.stl and assets/chopstick.stl.

Dimensions (m): length 0.140, body r 0.0045, rubber grip r 0.0042 over the
front 0.030, 0.012 metal cone down to a 0.0006 ball tip, domed cap, clip.
z runs from the tip (z = -PEN_HALF) to the cap end (z = +PEN_HALF).

    python -m dexcraft.gen_pen
"""
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

PEN_LEN = 0.140
PEN_HALF = PEN_LEN / 2
BODY_R = 0.0045
GRIP_R = 0.0042
GRIP_LEN = 0.030
CONE_LEN = 0.012
TIP_R = 0.0006
TIP_Z = -PEN_HALF
CONE_Z = TIP_Z + CONE_LEN              # cone base
GRIP_Z = CONE_Z + GRIP_LEN             # grip / body junction
CAP_Z = PEN_HALF - 0.006               # where the dome starts

# (z, r) profiles, nib -> cap
PROFILES = {
    "pen_cone": [(TIP_Z, 0.0), (TIP_Z + 0.0004, TIP_R), (TIP_Z + 0.0015, TIP_R * 1.2), (CONE_Z - 0.002, 0.0032),
                 (CONE_Z, 0.0040), (CONE_Z + 0.0005, 0.0040)],
    "pen_grip": [(CONE_Z + 0.0005, 0.0040), (CONE_Z + 0.004, GRIP_R), (GRIP_Z - 0.012, GRIP_R),
                 (GRIP_Z - 0.010, 0.0040), (GRIP_Z - 0.008, GRIP_R), (GRIP_Z - 0.006, 0.0040),
                 (GRIP_Z - 0.004, GRIP_R), (GRIP_Z, BODY_R)],
    "pen_body": [(GRIP_Z, BODY_R), (CAP_Z, BODY_R), (CAP_Z + 0.003, 0.0040), (CAP_Z + 0.005, 0.0025),
                 (PEN_HALF, 0.0)],
}


# ---- brush: same body frame convention as the pen (grip point 0.042 below the
# origin along -z, tip direction -z) so the cached grip works for both tools
BRUSH_HANDLE_R = 0.0045
BRUSH_GRIP_R = 0.0042
BRUSH_TOP_Z = 0.110
BRUSH_FERRULE_Z = -0.072           # handle ends / ferrule starts
BRUSH_TUFT_Z = -0.090              # ferrule ends / tuft root
BRUSH_TUFT1_LEN = 0.018
BRUSH_TUFT2_LEN = 0.017
BRUSH_TIP_Z = BRUSH_TUFT_Z - BRUSH_TUFT1_LEN - BRUSH_TUFT2_LEN       # -0.125
BRUSH_PROFILES = {
    "brush_handle": [(BRUSH_FERRULE_Z, 0.0044), (BRUSH_FERRULE_Z + 0.006, BRUSH_GRIP_R), (-0.012, BRUSH_GRIP_R),
                     (-0.006, BRUSH_HANDLE_R), (BRUSH_TOP_Z - 0.004, BRUSH_HANDLE_R), (BRUSH_TOP_Z, 0.0)],
    "brush_ferrule": [(BRUSH_TUFT_Z, 0.0040), (BRUSH_TUFT_Z + 0.004, 0.0046), (BRUSH_FERRULE_Z - 0.002, 0.0046),
                      (BRUSH_FERRULE_Z, 0.0044)],
    # tuft pieces are in their own body frames (root at z = 0, pointing -z)
    "brush_tuft1": [(-BRUSH_TUFT1_LEN, 0.0030), (-BRUSH_TUFT1_LEN * 0.5, 0.0038), (0.0, 0.0040)],
    "brush_tuft2": [(-BRUSH_TUFT2_LEN, 0.0), (-BRUSH_TUFT2_LEN + 0.001, 0.0005), (-BRUSH_TUFT2_LEN * 0.5, 0.0018),
                    (0.0, 0.0030)],
}


# ---- chopstick: same frame convention again (grip point 0.042 below the origin,
# tip along -z). Held 150 mm above the tip, 230 mm long, round section tapering
# from 9 mm at the top to 2.4 mm at the tip.
CHOP_LEN = 0.230
CHOP_GRIP_FROM_TIP = 0.150
CHOP_TIP_Z = -0.042 - CHOP_GRIP_FROM_TIP          # -0.192
CHOP_TOP_Z = CHOP_TIP_Z + CHOP_LEN                # +0.038
CHOP_TIP_R = 0.0012
CHOP_PROFILES = {
    "chopstick": [(CHOP_TIP_Z, 0.0), (CHOP_TIP_Z + 0.002, CHOP_TIP_R), (CHOP_TIP_Z + 0.060, 0.0024),
                  (-0.062, 0.0038), (-0.030, GRIP_R), (-0.020, BODY_R), (CHOP_TOP_Z - 0.003, BODY_R),
                  (CHOP_TOP_Z, 0.0)],
}


def revolve(profile, n=40):
    z = np.array([p[0] for p in profile])
    r = np.array([p[1] for p in profile])
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.array([[ri * np.cos(t), ri * np.sin(t), zi] for zi, ri in zip(z, r) for t in th])
    F = []
    m = len(profile)
    for i in range(m - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = a + n, b + n
            F += [(a, b, c), (b, d, c)]
    # close the ends with fans when the end radius is > 0
    for i, flip in ((0, True), (m - 1, False)):
        if r[i] > 1e-6:
            cidx = len(V)
            V = np.vstack([V, [[0, 0, z[i]]]])
            for j in range(n):
                a, b = i * n + j, i * n + (j + 1) % n
                F.append((cidx, a, b) if flip else (cidx, b, a))
    return V, np.array(F, dtype=np.int32)


def write_stl(path, V, F):
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(F)))
        for a, b, c in F:
            n = np.cross(V[b] - V[a], V[c] - V[a])
            n = n / (np.linalg.norm(n) + 1e-12)
            fh.write(struct.pack("<3f", *n))
            for v in (V[a], V[b], V[c]):
                fh.write(struct.pack("<3f", *v))
            fh.write(b"\0\0")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    for name, prof in {**PROFILES, **BRUSH_PROFILES, **CHOP_PROFILES}.items():
        V, F = revolve(prof)
        write_stl(os.path.join(ASSETS, name + ".stl"), V, F)
        print(f"{name}: {len(V)} verts, {len(F)} tris")


if __name__ == "__main__":
    main()
