import torch


import torch, numpy as np, matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from typing import Sequence, Tuple

def theoretical_noise_uniform(
    n: int,
    domain: Tuple[float, float],
    *,
    centers: Sequence[float] = (1.0, 2.0, 3.0),
    widths: Sequence[float] | float = (0.2, 0.2, 0.2),
    bump: float = 0.3,
    baseline: float = 0.05,
    device=None,
    dtype=None,
    return_points: bool = False,   # if True, also return X grid (n,1)
    # --- NEW ---
    X: torch.Tensor | None = None,     # if provided, use these points instead of a uniform grid
    homoscedastic: float | None = None # if provided, ignore islands and return constant σ=homoscedastic
):
    """
    Build the theoretical heteroscedastic noise level σ(x).

    If `homoscedastic` is provided, returns a constant σ(x) = homoscedastic,
    aligned to X (if given) or to a generated uniform grid.

    Otherwise (original behavior):
    σ(x) = baseline + sum_c bump * exp(-0.5 * ((x-c)/w)^2) * 1{|x-c| ≤ w}
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.get_default_dtype()

    # Choose points X
    if X is not None:
        assert X.ndim == 2 and X.shape[1] == 1, "X must be (N,1)"
        X_grid = X.to(device=device, dtype=dtype)
        n_eff = X_grid.shape[0]
    else:
        t0, t1 = float(domain[0]), float(domain[1])
        X_grid = torch.linspace(t0, t1, steps=n, device=device, dtype=dtype).unsqueeze(1)  # (n,1)
        n_eff = n

    # If asked to match data_generation's scalar noise, return constant σ
    if homoscedastic is not None:
        sigma = torch.full((n_eff, 1), float(homoscedastic), device=device, dtype=dtype)
        return (sigma, X_grid) if return_points else sigma

    # Otherwise, build the islands model (original behavior)
    if isinstance(widths, (int, float)):
        widths = [float(widths)] * len(centers)
    assert len(widths) == len(centers), "`widths` must match length of `centers`"

    x = X_grid
    sigma = torch.full((n_eff, 1), fill_value=baseline, device=device, dtype=dtype)
    eps = 1e-12

    for c, w in zip(centers, widths):
        c = float(c); w = float(w)
        phi = torch.exp(-0.5 * ((x - c) / (w + eps))**2)       # (n,1)
        box = ((x >= c - w) & (x <= c + w)).to(dtype=dtype)    # (n,1)
        sigma = sigma + bump * (phi * box)

    return (sigma, X_grid) if return_points else sigma



@torch.no_grad()
def local_noise_max_from_xy(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int = 15,
    metric: str = "mae",   # 'mae' -> |Yi-Yj|, 'mse' -> (Yi-Yj)^2
):
    """
    Estimate local noise at each X_i as the maximum pairwise target difference
    within its k-NN neighborhood (in X-space), excluding self.

    Args
    ----
    X : (n,d) or (n,) torch tensor
    Y : (n,)  or (n,1) torch tensor
    k : int   number of neighbors (incl. self in search)
    metric : {'mae','mse'}

    Returns
    -------
    local_noise_max : (n,) tensor
        max_j |Yi - Yj| (or squared) over the k-NN of X_i, j != i.
        If k <= 1, returns zeros.
    nn_idx : (n,k) LongTensor
        Indices of the k nearest neighbors for each point (includes self).
    argmax_idx : (n,) LongTensor
        Index (into original dataset) of the neighbor achieving the max for each i.
        For k <= 1, filled with i.
    """
    # normalize shapes
    if X.ndim == 1:
        X = X.unsqueeze(-1)
    Y = Y.reshape(-1)

    n = X.shape[0]
    k = int(max(1, min(k, n)))

    # kNN (Euclidean). nn_idx includes self (distance 0).
    dists = torch.cdist(X, X, p=2)                              # (n, n)
    nn_idx = torch.topk(dists, k=k, dim=1, largest=False).indices  # (n, k)

    # gather neighbor targets and build pairwise diffs
    Y_nei = Y[nn_idx]                                           # (n, k)
    Yi = Y.unsqueeze(1)                                         # (n, 1)
    if metric.lower() == "mse":
        diffs = (Y_nei - Yi).pow(2)
    else:  # 'mae'
        diffs = (Y_nei - Yi).abs()

    # exclude self from the max
    rows = torch.arange(n, device=nn_idx.device).unsqueeze(1)   # (n,1)
    self_mask = nn_idx.eq(rows)                                 # (n,k)
    if k == 1:
        # no neighbors besides self
        local_noise_max = torch.zeros(n, dtype=diffs.dtype, device=diffs.device)
        argmax_idx = rows.squeeze(1)
        return local_noise_max, nn_idx, argmax_idx

    diffs_masked = diffs.masked_fill(self_mask, float("-inf"))
    # max over neighbors
    max_vals, which = torch.max(diffs_masked, dim=1)            # (n,), (n,)
    # map which (position in k) -> original dataset index
    argmax_idx = nn_idx[torch.arange(n, device=nn_idx.device), which]

    # if a row was all -inf (shouldn't happen unless k==1), clamp to 0
    max_vals = torch.where(torch.isinf(max_vals), torch.zeros_like(max_vals), max_vals)

    return max_vals, nn_idx, argmax_idx


def plot_local_error_grid_2x2(
    *,
    X_test: torch.Tensor,
    true_err,
    heu_width, 
    cp_width,
    adap_width,
    k = (20, 20, 20, 20),
    nbar = (24, 24, 24, 24),
    ylims = (None, (0, 1.5), None, (0, 2.5)),
    colors = (
        ("#dfdfdf", "#dfdfdf"),
        ("#dfdfdf", "#6ea4f1"),
        ("#dfdfdf", "#f0b7b2"),
        ("#dfdfdf", "#f5e3b3"),
    ),
    labels = ("True error", "After CP", "After Local CP", "Before CP"),
    tick_count: int = 6,
    bar_width: float = 0.7,
    overlay_frac: float = 1.0,
    figsize=(12, 10),
    suptitle: str | None = None,
    sharey: bool = False,
):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharey=sharey)
    axs = axs.ravel()

    # (1) True error only (show both axes)
    plot_local_error_bar(
        X_test, true_err,
        k=k[0], nbar=nbar[0],
        colors=colors[0], label=labels[0],
        bar_width=bar_width, overlay_frac=overlay_frac,
        tick_count=tick_count, ylim=ylims[0],
        ax=axs[0],
        display_x=False, display_y=True, plot_true_error=True
    )

    # (2) Heuristic vs True (hide both)
    plot_local_error_bar(
        X_test, true_err,
        heuristics=heu_width,
        k=k[3], nbar=nbar[3],
        colors=colors[3], label=labels[3],
        bar_width=bar_width, overlay_frac=overlay_frac,
        tick_count=tick_count, ylim=ylims[3],
        ax=axs[1],
        display_x=False, display_y=False
    )

    # (3) CP vs True (hide both)
    plot_local_error_bar(
        X_test, true_err,
        heuristics=cp_width,
        k=k[1], nbar=nbar[1],
        colors=colors[1], label=labels[1],
        bar_width=bar_width, overlay_frac=overlay_frac,
        tick_count=tick_count, ylim=ylims[1],
        ax=axs[2],
        display_x=True, display_y=True
    )

    # (4) Local CP vs True (show x only)
    plot_local_error_bar(
        X_test, true_err,
        heuristics=adap_width,
        k=k[2], nbar=nbar[2],
        colors=colors[2], label=labels[2],
        bar_width=bar_width, overlay_frac=overlay_frac,
        tick_count=tick_count, ylim=ylims[2],
        ax=axs[3],
        display_x=True, display_y=False
    )

    # if suptitle:
    #     fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if suptitle else None)
    return fig, axs



@torch.no_grad()
def plot_local_error_bar(
    X: torch.Tensor,
    local_err: torch.Tensor,
    heuristics: torch.Tensor | None = None,
    *,
    k: int = 15,
    nbar: int | None = None,
    colors=("#d9534f", "#4287f5"),
    label="Est.",
    bar_width: float = 0.7,
    overlay_frac: float = 1.0,
    figsize=(5, 5),
    title="Local error across grid",
    tick_count: int = 6,
    top_pad: float = 0.12,
    ylim: tuple[float, float] | None = (0, 1.3),
    est_alpha: float = 1.0,
    edge_lw: float = 0.4,
    e_alpha: float = 0.7,
    ax: plt.Axes | None = None,
    display_x: bool = False,
    display_y: bool = False,
    plot_true_error: bool = False
):
    # ---- data & kNN (unchanged) ----
    Xc = X.detach().cpu()
    if Xc.ndim == 1: Xc = Xc.unsqueeze(-1)
    n = Xc.shape[0]
    k = int(max(1, min(k, n)))

    true_err = local_err.detach().cpu().reshape(-1)
    dists = torch.cdist(Xc, Xc, p=2)
    nn_idx = torch.topk(dists, k=k, dim=1, largest=False).indices

    true_knn = true_err[nn_idx].mean(dim=1).numpy()
    heur_knn = None
    if heuristics is not None:
        heur = heuristics.detach().cpu().reshape(-1)
        heur_knn = heur[nn_idx].mean(dim=1).numpy()

    x1d = X.detach().cpu().reshape(-1).numpy() if (X.dim()==1 or Xc.shape[1]==1) else np.arange(n, dtype=float)
    order = np.argsort(x1d)
    x_sorted, true_sorted = x1d[order], true_knn[order]
    heur_sorted = heur_knn[order] if heur_knn is not None else None

    if nbar is not None and nbar < len(true_sorted):
        idx = np.linspace(0, len(true_sorted)-1, nbar).astype(int)
        x_sorted, true_sorted = x_sorted[idx], true_sorted[idx]
        if heur_sorted is not None: heur_sorted = heur_sorted[idx]

    m = len(true_sorted); pos = np.arange(m)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # ---- bars ----
    base_label = "True" if plot_true_error else None
    ax.bar(pos, true_sorted, width=bar_width, color=colors[0],
           edgecolor="black", linewidth=edge_lw, zorder=5, alpha=e_alpha,
           label=base_label)

    if heur_sorted is not None:
        narrow_w = max(1e-3, min(bar_width, bar_width * overlay_frac))
        ax.bar(pos, heur_sorted, width=narrow_w, color=colors[1],
               edgecolor="black", linewidth=edge_lw, alpha=est_alpha,
               label=label, zorder=4)

    # ---- limits ----
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        data_max = float(np.nanmax([true_sorted.max(), (heur_sorted.max() if heur_sorted is not None else 0.0)]))
        ax.set_ylim(0.0, (1.0 + top_pad) * (data_max if data_max > 0 else 1.0))

    # ---- tick/label visibility (share-aware) ----
    # detect sharing
    sharey_group = ax.get_shared_y_axes().get_siblings(ax)
    is_sharey = len(sharey_group) > 1
    sharex_group = ax.get_shared_x_axes().get_siblings(ax)
    is_sharex = len(sharex_group) > 1

    # X axis
    if display_x:
        xt_n = int(max(2, min(tick_count, m))) if m > 1 else 1
        xticks = np.linspace(0, m-1, xt_n).astype(int) if m > 1 else np.array([0])
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x_sorted[t]:.2f}" for t in xticks])
        ax.set_xlabel(r"$x$", fontsize=17)
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, length=3)
        if not is_sharex:
            ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks()))
    else:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        # only kill locator/formatter if not shared
        if not is_sharex:
            ax.xaxis.set_major_locator(mticker.NullLocator())
            ax.xaxis.set_major_formatter(mticker.NullFormatter())

    # Y axis
    if display_y:
        y_min_curr, y_max_curr = ax.get_ylim()
        yt_n = int(max(2, tick_count))
        y_ticks = np.linspace(y_min_curr, y_max_curr, yt_n)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{yt:.2f}" for yt in y_ticks])
        ax.set_ylabel("Error", fontsize=17)
        ax.tick_params(axis="y", which="both", left=True, labelleft=True, length=3)
        if not is_sharey:
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
    else:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        if not is_sharey:
            ax.yaxis.set_major_locator(mticker.NullLocator())
            ax.yaxis.set_major_formatter(mticker.NullFormatter())

    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)

    # legend only if something labeled
    handles, labels_ = ax.get_legend_handles_labels()
    if labels_:
        ax.legend(frameon=False, loc="upper left", fontsize=14.5)

    if created_fig:
        fig.tight_layout()

    return fig, ax
