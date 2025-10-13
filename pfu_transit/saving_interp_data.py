import os, re, io, glob
import numpy as np

def _natural_cubic_spline_build(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 2 or x.ndim != 1 or y.ndim != 1 or n != len(y):
        raise ValueError("x and y must be same-length 1D arrays with n >= 2")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")

    h = np.diff(x)
    m = np.diff(y) / h

    lower = np.zeros(n-1)
    diag  = np.zeros(n)
    upper = np.zeros(n-1)
    rhs   = np.zeros(n)

    diag[0] = 2.0
    upper[0] = 1.0
    rhs[0] = 3.0 * m[0]

    for i in range(1, n-1):
        lower[i-1] = h[i]
        diag[i]    = 2.0 * (h[i-1] + h[i])
        upper[i-1] = h[i-1]
        rhs[i]     = 3.0 * (h[i] * m[i-1] + h[i-1] * m[i])

    lower[n-2] = 1.0
    diag[n-1]  = 2.0
    rhs[n-1]   = 3.0 * m[-1]

    # Thomas solve
    for i in range(1, n):
        w = lower[i-1] / diag[i-1]
        diag[i] -= w * upper[i-1]
        rhs[i]  -= w * rhs[i-1]

    s = np.empty(n)
    s[-1] = rhs[-1] / diag[-1]
    for i in range(n-2, -1, -1):
        s[i] = (rhs[i] - (upper[i] * s[i+1] if i < n-1 else 0.0)) / diag[i]

    a = y[:-1]
    b = s[:-1]
    c = (3*m - 2*s[:-1] - s[1:]) / h
    d = (s[:-1] + s[1:] - 2*m) / (h*h)
    return x, a, b, c, d

def cubic_spline_interpolator(x, y, clamp=True):
    xk, a, b, c, d = _natural_cubic_spline_build(x, y)
    def f(xq):
        xq = np.asarray(xq, float)
        if clamp:
            xq = np.clip(xq, xk[0], xk[-1])
        idx = np.searchsorted(xk, xq, side='right') - 1
        idx = np.clip(idx, 0, len(xk)-2)
        dx = xq - xk[idx]
        return ((d[idx]*dx + c[idx])*dx + b[idx])*dx + a[idx]
    return f

# ---------- helpers ----------
def read_xy_relaxed(path):
    """Read two numeric columns; ignore non-numeric header lines, spaces/commas ok."""
    numeric = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)) >= 2:
                numeric.append(line)
    if not numeric:
        raise ValueError(f"No numeric data found in {path}")
    # split on whitespace or commas
    rows = []
    for ln in numeric:
        parts = re.split(r"[\s,]+", ln.strip())
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    arr = np.array(rows, dtype=float)
    x, y = arr[:,0], arr[:,1]
    # sort & dedupe x
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = np.r_[True, np.diff(x) > 0]
    return x[keep], y[keep]

# ---------- main: read, interpolate, save ----------
SRC_DIR = "../assets/exoplanet_data"
OUT_DIR = "../assets/interp_data"
N_GRID  = 1000  # points per file; change if you want denser/sparser

os.makedirs(OUT_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC_DIR, "*.txt")))
if not files:
    raise SystemExit(f"No .txt files found in {SRC_DIR}")

# compute overlapping common grid (optional)
mins, maxs, xy_map = [], [], {}
for p in files:
    x, y = read_xy_relaxed(p)
    xy_map[p] = (x, y)
    mins.append(x.min()); maxs.append(x.max())

x_lo, x_hi = max(mins), min(maxs)
x_common = np.linspace(x_lo, x_hi, N_GRID) if x_hi > x_lo else None

for p in files:
    x, y = xy_map[p]
    spline = cubic_spline_interpolator(x, y, clamp=True)
    grid = x_common if x_common is not None else np.linspace(x.min(), x.max(), N_GRID)
    y_new = spline(grid)

    planet = os.path.splitext(os.path.basename(p))[0]
    out_csv = os.path.join(OUT_DIR, f"{planet}_interpolated.txt")

    # save CSV without pandas
    header = "x,y_interpolated"
    data_to_save = np.column_stack([grid, y_new])
    np.savetxt(out_csv, data_to_save, delimiter=",", header=header, comments="", fmt="%.10g")
    print(f"Saved {out_csv}")

print(f"✅ All interpolated files saved to: {os.path.abspath(OUT_DIR)}")
