import torch
from typing import Sequence, Literal
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.colors as mcolors
# Define color map
colors = ["#2989ff","#ffffff", "#ff424f"]
my_cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", colors)

colors = ["#8c52ff","#ffffff", "#ff66c4"]
my_cmap_2 = mcolors.LinearSegmentedColormap.from_list("my_cmap", colors)
from matplotlib.ticker import AutoLocator, ScalarFormatter

def plot_2D_comparison_with_coverage_error_compare(
    XY_test,
    error,
    pred_set_uncal,
    pred_set_cp,
    pred_set_local_cp,
    true_solution,
    *,
    grid_size=200,
    show_pts=False,
    X_pts=None,
    vlim=None,                  # one shared vlim for ALL panels; auto if None
    cmap=my_cmap,               # one unique cmap for ALL panels
    title_fontsize=16,
    tick_labelsize=22,
    wspace=0.15,                # spacing between the 4 panels
    # --- NEW: contour options ---
    noisy_mask=None,            # bool/0-1 mask for noisy region (len N, len G, or [GxG])
    sigma_field=None,           # numeric field to threshold into a noisy region (alt to mask)
    sigma_thresh=None,          # explicit threshold for sigma_field (if None, use quantile)
    sigma_quantile=0.80,        # used if sigma_thresh is None (e.g., top 20% as noisy)
    contour_kwargs=None,        # dict passed to ax.contour (colors, linewidths, linestyles, alpha)
    x_tick_step=None,            # e.g., 0.1 for major x ticks
    y_tick_step=None,            # e.g., 0.1 for major y ticks
    x_minor_step=None,           # e.g., 0.05 for minor x ticks
    y_minor_step=None,           # e.g., 0.05 for minor y ticks
):
    """
    4×1 panels (left→right): |ŷ - y|, width Before CP, width After CP, width After Local CP
    - Single shared colormap + single colorbar placed to the right of the entire row.
    - `vlim`: (vmin, vmax) shared across all panels; auto-computed from all fields if None.
    - `wspace`: horizontal spacing between panels.
    - NEW: If `noisy_mask` (bool) or `sigma_field` (numeric) is provided, draw a contour
      enclosing the local noisy region on ALL panels. For `sigma_field`, region is
      {x : sigma(x) >= sigma_thresh}, with `sigma_thresh` inferred from `sigma_quantile`
      if not given.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def _scatter_to_grid(vals_scatter, XY_scatter, Xg, Yg):
        """Nearest-neighbor map from scatter (N,) onto (grid_size, grid_size)."""
        XYs = to_numpy(XY_scatter).reshape(-1, 2)
        v   = to_numpy(vals_scatter).reshape(-1)
        P   = np.stack([Xg.ravel(), Yg.ravel()], axis=1)  # (G,2)
        d2  = ((P[:, None, :] - XYs[None, :, :]) ** 2).sum(axis=2)
        idx = d2.argmin(axis=1)
        return v[idx].reshape(grid_size, grid_size)

    # ---- Build grid over XY_test bounds ----
    xy_np = to_numpy(XY_test)
    x_lin = np.linspace(xy_np[:,0].min(), xy_np[:,0].max(), grid_size)
    y_lin = np.linspace(xy_np[:,1].min(), xy_np[:,1].max(), grid_size)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    XY_grid = torch.tensor(
        np.stack([Xg.ravel(), Yg.ravel()], axis=1),
        dtype=(XY_test.dtype if isinstance(XY_test, torch.Tensor) else torch.float32),
    )
    extent = [x_lin.min(), x_lin.max(), y_lin.min(), y_lin.max()]
    G = grid_size * grid_size

    # ---- True field on the grid ----
    true_grid = to_numpy(true_solution(XY_grid)).reshape(grid_size, grid_size)

    # ---- Intervals -> mean + width (reshaped to grid) ----
    def prep_interval(pred_set):
        lower = to_numpy(pred_set[0]).reshape(-1)
        upper = to_numpy(pred_set[1]).reshape(-1)
        mean  = (lower + upper) / 2.0
        width = upper - lower
        return mean.reshape(grid_size, grid_size), width.reshape(grid_size, grid_size)

    mean_uncal,    width_uncal    = prep_interval(pred_set_uncal)
    mean_cp,       width_cp       = prep_interval(pred_set_cp)
    mean_lcp,      width_lcp      = prep_interval(pred_set_local_cp)

    # ---- Error grid ----
    if error is None:
        err_grid = np.abs(mean_uncal - true_grid)
    else:
        err_np = to_numpy(error).reshape(-1)
        if err_np.size == G:
            err_grid = err_np.reshape(grid_size, grid_size)
        elif err_np.size == xy_np.shape[0]:
            err_grid = _scatter_to_grid(err_np, xy_np, Xg, Yg)
        else:
            raise ValueError(
                f"`error` must have length {G} or len(XY_test)={xy_np.shape[0]}, got {err_np.size}."
            )

    # ---- Shared vlim across ALL panels ----
    if vlim is None:
        vmin = float(
            np.nanmin([err_grid.min(), width_uncal.min(), width_cp.min(), width_lcp.min()])
        )
        vmax = float(
            np.nanmax([err_grid.max(), width_uncal.max(), width_cp.max(), width_lcp.max()])
        )
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        vlim = (vmin, vmax)

    norm = mcolors.Normalize(vmin=vlim[0], vmax=vlim[1])

    # ---- Figure & axes (4 by 1) ----
    fig, axs = plt.subplots(1, 4, figsize=(22, 5), gridspec_kw={"wspace": wspace})
    ax_err, ax_wb, ax_wc, ax_wlcp = axs

    # ---- Draw images with one shared cmap/norm ----
    im_err = ax_err.imshow(err_grid, extent=extent, origin='lower', aspect='auto',
                           interpolation='bilinear', cmap=cmap, norm=norm)
    ax_err.set_ylabel("y", fontsize=tick_labelsize)
    ax_err.set_xlabel("x", fontsize=tick_labelsize)

    im_wb  = ax_wb.imshow(width_uncal, extent=extent, origin='lower', aspect='auto',
                          interpolation='bilinear', cmap=cmap, norm=norm)
    ax_wb.set_yticks([])
    ax_wb.set_xlabel("x", fontsize=tick_labelsize)

    im_wc  = ax_wc.imshow(width_cp, extent=extent, origin='lower', aspect='auto',
                          interpolation='bilinear', cmap=cmap, norm=norm)
    ax_wc.set_yticks([])
    ax_wc.set_xlabel("x", fontsize=tick_labelsize)

    im_wl  = ax_wlcp.imshow(width_lcp, extent=extent, origin='lower', aspect='auto',
                            interpolation='bilinear', cmap=cmap, norm=norm)
    ax_wlcp.set_yticks([])
    ax_wlcp.set_xlabel("x", fontsize=tick_labelsize)

    # ---- Optional: resolve "local noisy region" (mask or thresholded sigma) to a grid ----
    def _to_grid_like(arr):
        arr = to_numpy(arr)
        if arr.ndim == 2 and arr.shape == (grid_size, grid_size):
            return arr
        arr = arr.reshape(-1)
        if arr.size == G:
            return arr.reshape(grid_size, grid_size)
        elif arr.size == xy_np.shape[0]:
            return _scatter_to_grid(arr, xy_np, Xg, Yg)
        raise ValueError(
            f"`noisy_mask`/`sigma_field` must be length {G}, len(XY_test)={xy_np.shape[0]}, "
            f"or shape ({grid_size},{grid_size}). Got {arr.shape}."
        )

    noisy_grid = None
    if noisy_mask is not None:
        ng = _to_grid_like(noisy_mask)
        if ng.dtype != bool:
            ng = ng > 0.5
        noisy_grid = ng.astype(float)
    elif sigma_field is not None:
        sg = _to_grid_like(sigma_field).astype(float)
        if sigma_thresh is None:
            finite = np.isfinite(sg)
            if finite.any():
                thr = np.quantile(sg[finite], sigma_quantile)
            else:
                thr = np.nan
        else:
            thr = float(sigma_thresh)
        if np.isfinite(thr):
            noisy_grid = (sg >= thr).astype(float)

    # ---- Draw contour outlining the noisy region on ALL panels ----
    if noisy_grid is not None:
        ckw = {'colors': 'k', 'linewidths': 1.6, 'linestyles': '--', 'alpha': 0.9}
        if contour_kwargs:
            ckw.update(contour_kwargs)
        for ax in axs:
            try:
                ax.contour(Xg, Yg, noisy_grid, levels=[0.5], **ckw)
            except Exception:
                # If no valid contour (e.g., mask all-True or all-False), just skip gracefully
                pass

    # ---- Optional scatter points ----
    if show_pts and X_pts is not None:
        pts = to_numpy(X_pts)
        for ax in axs:
            ax.scatter(pts[:,0], pts[:,1], s=10, c="k", alpha=0.8)
    from matplotlib.ticker import MultipleLocator
    x0, x1, y0, y1 = extent

    def _ticks(lo, hi, step):
        # start on a clean multiple of `step` at or below `lo`
        start = np.floor(lo / step) * step
        # ensure inclusive coverage to hi (small epsilon)
        return np.arange(start, hi + 1e-12, step)

    if x_tick_step is not None and x_tick_step > 0:
        x_ticks = _ticks(x0, x1, x_tick_step)
        for ax in axs:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(ax.get_xticks(), fontsize=14)
    if y_tick_step is not None and y_tick_step > 0:
        y_ticks = _ticks(y0, y1, y_tick_step)
        for ax in axs:
            ax.set_yticks(y_ticks if ax is ax_err else [])  # keep only left-most y ticks
            ax.set_yticklabels(ax.get_yticks(), fontsize=14)
            # If you want y ticks on all panels, remove the conditional and use:
            # ax.set_yticks(y_ticks)

    # Minor ticks via MultipleLocator (doesn't need explicit arrays)
    if x_minor_step is not None and x_minor_step > 0:
        for ax in axs:
            ax.xaxis.set_minor_locator(MultipleLocator(x_minor_step))
            ax.tick_params(which='minor', length=3)
    if y_minor_step is not None and y_minor_step > 0:
        for ax in axs:
            ax.yaxis.set_minor_locator(MultipleLocator(y_minor_step))
            ax.tick_params(which='minor', length=3)
            
    
    # ---- Single colorbar on the far right (shared across all axes) ----
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, location="right", fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=16)
    return fig, axs


def estimate_true_error_local_max(
    xy: torch.Tensor,          # (N, 2)
    u: torch.Tensor,           # (N, 1)  <-- use your observed y if you want data noise; don't pass model mean
    nghd_size,                 # float -> radius mode; int>=1 -> kNN mode
    *,
    include_self: bool = True,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    For each point i, compute a local noise proxy:
        noise(i) = max_{j,k in N(i)} |u[j] - u[k]|
                  = (max_{j in N(i)} u[j]) - (min_{k in N(i)} u[k])

    Neighborhood N(i):
        - Radius mode (nghd_size: float): points with dist <= radius.
        - kNN mode    (nghd_size: int>=1): k nearest neighbors.

    Returns:
        (N,1) tensor of local ranges per point.

    Notes:
      • If you want "intrinsic data noise", pass your *observed* targets with noise as `u`.
      • Using ground-truth noiseless u will measure local function variability instead.
    """
    assert xy.ndim == 2 and xy.shape[1] == 2, "xy must be (N, 2)"
    assert u.ndim == 2 and u.shape[1] == 1, "u must be (N, 1)"
    N = xy.shape[0]
    device = xy.device
    u1 = u.squeeze(1)  # (N,)

    # ---------- kNN mode ----------
    if isinstance(nghd_size, int):
        assert nghd_size >= 1, "For kNN mode, nghd_size must be int>=1"
        k = min(nghd_size, N)
        row_block = max(1, chunk_size // 4)

        out = torch.empty((N,), device=device)
        for r0 in range(0, N, row_block):
            r1 = min(N, r0 + row_block)
            dists = torch.cdist(xy[r0:r1], xy)  # (r, N)

            if not include_self:
                for i in range(r0, r1):
                    dists[i - r0, i] = inf

            _, idx = torch.topk(dists, k=k, largest=False, dim=1)   # (r, k)
            neigh_u = u1[idx]                                       # (r, k)
            local_max = neigh_u.max(dim=1).values
            local_min = neigh_u.min(dim=1).values
            out[r0:r1] = (local_max - local_min)

        return out.unsqueeze(1)

    # ---------- radius mode ----------
    elif isinstance(nghd_size, float):
        radius = nghd_size
        assert radius > 0.0, "For radius mode, nghd_size must be a positive float"

        # Running extrema per row
        run_max = torch.full((N,), -inf, device=device)
        run_min = torch.full((N,),  inf, device=device)

        for c0 in range(0, N, chunk_size):
            c1 = min(N, c0 + chunk_size)
            dblock = torch.cdist(xy, xy[c0:c1])                     # (N, B)

            if not include_self:
                idx_rows = torch.arange(c0, c1, device=device)
                dblock[idx_rows, idx_rows - c0] = inf               # remove self

            mask = dblock <= radius                                 # (N, B)
            u_block = u1[c0:c1].unsqueeze(0).expand_as(dblock)      # (N, B)

            # masked maxima/minima in this block
            blk_max = torch.where(mask, u_block, torch.full_like(u_block, -inf)).max(dim=1).values
            blk_min = torch.where(mask, u_block, torch.full_like(u_block,  inf)).min(dim=1).values

            # update running extrema
            run_max = torch.maximum(run_max, blk_max)
            run_min = torch.minimum(run_min, blk_min)

        # Rows with no neighbors found (possible only if include_self=False and radius tiny)
        no_neigh = torch.isinf(run_max) | torch.isinf(run_min)
        if no_neigh.any():
            # define their range as 0 (or use self value to avoid NaNs)
            run_max[no_neigh] = u1[no_neigh]
            run_min[no_neigh] = u1[no_neigh]

        return (run_max - run_min).unsqueeze(1)

    else:
        raise ValueError("nghd_size must be int (kNN) or float (radius).")
   

import torch
import matplotlib.pyplot as plt

def visualize_selected_points(
    X: torch.Tensor,
    idx: torch.Tensor,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
    figsize=(6,6),
    selected_color="red",
    background_color="lightgray",
    marker_size=30,
    title: str | None = "Selected points"
):
    """
    Visualize selected points (idx) on 2D domain [-1,1]^2.

    Parameters
    ----------
    X : torch.Tensor
        (N, 2) full set of points in domain.
    idx : torch.Tensor
        1D LongTensor of selected indices.
    domain : tuple, default (-1,1)
        Plot limits for both x and y.
    figsize : tuple
        Matplotlib figure size.
    selected_color : str
        Color for selected points.
    background_color : str
        Color for non-selected points.
    marker_size : int
        Scatter marker size.
    title : str or None
        Plot title.
    """
    X = X.detach().cpu()
    idx = idx.detach().cpu()

    mask = torch.zeros(X.shape[0], dtype=bool)
    mask[idx] = True

    plt.figure(figsize=figsize)
    plt.scatter(X[~mask,0], X[~mask,1], c=background_color, s=marker_size, alpha=0.5, label="Unselected")
    plt.scatter(X[mask,0], X[mask,1], c=selected_color, s=marker_size, label="Selected")

    plt.xlim(domain)
    plt.ylim(domain)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()




import torch
from typing import Sequence, Literal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_metrics_table(
    X_test: torch.Tensor,
    cp_uncal_predset,
    cp_cal_predset,
    true_solution,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str="Uncalibrated",
    df2_name: str="Calibrated",
    title: str = "",
    main_title: str | None = None,
    X_vis=None, Y_vis=None,
    alpha_level: float = 0.05,
    figsize: tuple = (9, 2.5),
    max_digits_display = lambda x: f"{x:.4g}"
):
    """
    Display a side-by-side metrics comparison (table) for the uncalibrated and
    calibrated models at a single alpha level.
    """
    # Compute the coverage deviation using mae
    def prepare_coverage_data(df):
        expected = 1 - df["alpha"]
        empirical = df["coverage"]
        exp_full = pd.concat([pd.Series([0.0]), expected, pd.Series([1.0])], ignore_index=True)
        emp_full = pd.concat([pd.Series([0.0]), empirical, pd.Series([1.0])], ignore_index=True)
        sort_idx = exp_full.argsort()
        exp_sorted, emp_sorted = exp_full[sort_idx], emp_full[sort_idx]
        return exp_sorted.to_numpy(), emp_sorted.to_numpy()

    def coverage_deviation(exp, emp, how="mae"):
        diff = np.abs(emp - exp)
        if   how == "mae":  return diff.mean()
        elif how == "rmse": return np.sqrt((diff**2).mean())
        elif how == "max":  return diff.max()
        else:
            raise ValueError("metric must be 'mae', 'rmse', or 'max'")

    exp1, emp1 = prepare_coverage_data(df1)
    exp2, emp2 = prepare_coverage_data(df2)
    dev1 = coverage_deviation(exp1, emp1)  # Using the default metrics
    dev2 = coverage_deviation(exp2, emp2)  # Using the default metrics
    print(f"Uncal dev:{dev1}")
    print(f"Cal dev:{dev2}")
    alpha_level_upper = alpha_level + 1e-3
    alpha_level_lower = alpha_level - 1e-3
    
    # ────────────────────── 1. Slice the two rows ──────────────────────
    row_uncal = df1.loc[(df1["alpha"] <= alpha_level_upper) & 
                           (df1["alpha"] >= alpha_level_lower)].copy()
    row_uncal["model"] = df1_name
    row_uncal["expected coverage"] = (1 - row_uncal["alpha"])
    row_uncal["mean coverage deviation"] = "{:.4f}".format(dev1)
    row_uncal["coverage"] = (row_uncal["coverage"]).map("{:.2f}".format)

    row_cal = df2.loc[(df2["alpha"] <= alpha_level_upper) & 
                        (df2["alpha"] >= alpha_level_lower)].copy()
    row_cal["model"] = df2_name
    row_cal["expected coverage"] = (1- row_cal["alpha"])
    row_cal["mean coverage deviation"] = "{:.4f}".format(dev2)
    row_cal["coverage"] = (row_cal["coverage"]).map("{:.2f}".format)

    if row_uncal.empty or row_cal.empty:
        raise ValueError(f"alpha={alpha_level} not found in both data frames.")

    # ───────────────────── 2. Stack & tidy up ──────────────────────────
    rows = pd.concat([row_uncal, row_cal], axis=0).reset_index(drop=True)
    rows = rows.rename(columns={"coverage": "actual coverage"})
    # Get all columns except 'model' for the selection
    other_cols = [c for c in rows.columns if c != "model"]
    rows = rows.loc[:, ["model"] + other_cols]

    
    # nice ordering: model | expected alpha | true alpha | <metrics…>
    metric_cols = [c for c in rows.columns if c not in ("model", "expected coverage", "actual coverage", "mean coverage deviation", "sharpness")]
    rows = rows[["model", "expected coverage", "actual coverage", "mean coverage deviation", "sharpness"]]
    

    # ──────────────── 2.5. Format numeric values ───────────────────────
    # Format all numeric columns to 4 decimal places (excluding 'model' column)
    for col in rows.columns:
        if pd.api.types.is_numeric_dtype(rows[col]):
            rows[col] = rows[col].apply(max_digits_display)  # .4g gives up to 4 significant 

    # ───────────────────── 3. Plot as table ────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=rows.values,
        colLabels=rows.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    if main_title is not None:
        plt.title(main_title, pad=20, fontsize=12)

    plt.tight_layout()
