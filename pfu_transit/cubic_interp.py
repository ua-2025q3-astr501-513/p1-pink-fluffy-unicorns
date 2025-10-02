import numpy as np


def _natural_cubic_spline_build(x, y):
    """
    Computes cubic spline coefficients for 1D data using natural boundary conditions.

    This function solves for the per-interval cubic coefficients (a, b, c, d) that 
    define a smooth, twice-differentiable spline passing through all input points.
    The "natural" boundary condition is enforced, meaning the second derivative 
    is zero at both endpoints.

    Parameters
    ----------
    x : array-like
        1D array of strictly increasing values (the independent variable).
    y : array-like
        1D array of dependent variable values corresponding to `x`. Must be the 
        same length as `x`.

    Returns
    -------
    x : ndarray
        Knot positions (copy of the input `x` array).
    a : ndarray
        Constant term coefficients for each interval [x[i], x[i+1]].
    b : ndarray
        Linear term coefficients for each interval.
    c : ndarray
        Quadratic term coefficients for each interval.
    d : ndarray
        Cubic term coefficients for each interval.

    Notes
    -----
    - The spline is piecewise cubic: on interval [x[i], x[i+1]], the polynomial 
    is given by:

        S_i(dx) = a[i] + b[i]*dx + c[i]*dx^2 + d[i]*dx^3,   where dx = xq - x[i]

    - Coefficients are constructed so that the spline is continuous in value, 
    first derivative, and second derivative across all knots.
    - Natural boundary condition is applied: S''(x[0]) = S''(x[-1]) = 0.
    - Intended for internal use; end users should call `cubic_spline_interpolator`.
    """


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
        lower[i-1] = h[i]                  # for s[i+1]
        diag[i]    = 2.0 * (h[i-1] + h[i]) # for s[i]
        upper[i-1] = h[i-1]                # for s[i-1]
        rhs[i]     = 3.0 * (h[i] * m[i-1] + h[i-1] * m[i])


    lower[n-2] = 1.0
    diag[n-1]  = 2.0
    rhs[n-1]   = 3.0 * m[-1]

    for i in range(1, n):
        w = lower[i-1] / diag[i-1]
        diag[i] -= w * upper[i-1]
        rhs[i]  -= w * rhs[i-1]
    s = np.empty(n)
    s[-1] = rhs[-1] / diag[-1]
    for i in range(n-2, -1, -1):
        s[i] = (rhs[i] - (upper[i] * s[i+1] if i < n-1 else 0.0)) / diag[i]

    # S_i(dx) = a[i] + b[i]*dx + c[i]*dx^2 + d[i]*dx^3, dx = t - x[i]
    a = y[:-1]
    b = s[:-1]
    c = (3*m - 2*s[:-1] - s[1:]) / h
    d = (s[:-1] + s[1:] - 2*m) / (h*h)

    return x, a, b, c, d

def cubic_spline_interpolator(x, y, clamp=True):
    
    """
    Constructs a 1D natural cubic spline interpolator for smooth curve fitting.

    The spline passes exactly through all input points (x, y) and enforces 
    "natural" boundary conditions, meaning the second derivative of the spline 
    is zero at both endpoints.

    Parameters
    ----------
    x : array-like
        1D array of strictly increasing values (the independent variable).
    y : array-like
        1D array of dependent variable values corresponding to `x`. 
        Must have the same length as `x`.
    extrapolate : bool, optional
        If True (default), queries outside the input domain [x[0], x[-1]] 
        are clamped to the endpoint values. If False, queries outside 
        will return NaN.

    Returns
-------
    spline : callable
        A function `f(xq)` that evaluates the cubic spline at arbitrary query 
        points `xq`. Supports scalar or array inputs.

    Notes
    -----
    - The spline is piecewise cubic: each interval [x[i], x[i+1]] has its own 
    cubic polynomial.
    - The coefficients are chosen so that:
        * The spline passes through all input points.
        * The first derivative is continuous across intervals.
        * The second derivative is continuous across intervals.
        * Natural boundary condition: S''(x[0]) = S''(x[-1]) = 0.
    - Internally, a tridiagonal linear system is solved to compute the first 
    derivatives at the knots, which are then used to build cubic coefficients.
    - Evaluation time is O(log n) per query due to interval search.

    """
    
    xk, a, b, c, d = _natural_cubic_spline_build(x, y)

    def f(xq):
        xq = np.asarray(xq, float)
        if clamp:
            xq = np.clip(xq, xk[0], xk[-1])
        idx = np.searchsorted(xk, xq, side='right') - 1
        idx = np.clip(idx, 0, len(xk) - 2)
        dx = xq - xk[idx]
        return ((d[idx]*dx + c[idx])*dx + b[idx])*dx + a[idx]

    return f
