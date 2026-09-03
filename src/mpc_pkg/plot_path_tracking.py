#!/usr/bin/env python3
"""Generate a presentation-quality path tracking figure.

Supported inputs:
- CSV with columns x, y, optional yaw, optional t
- NPZ with arrays x, y, optional yaw, optional t
- Demo mode when no files are provided

Example:
    python3 plot_path_tracking.py \
        --ref-file ref.csv \
        --track-file tracked.csv \
        --output tracking_effect.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def _normalize_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _load_csv_path(file_path: Path) -> dict[str, np.ndarray]:
    with file_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {file_path} has no header")

        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV file {file_path} is empty")

        def read_column(name: str, default: float | None = None) -> np.ndarray:
            if name not in reader.fieldnames:
                if default is None:
                    raise ValueError(f"CSV file {file_path} is missing column: {name}")
                return np.full(len(rows), default, dtype=float)
            return np.asarray([float(row[name]) for row in rows], dtype=float)

        x = read_column("x")
        y = read_column("y")
        yaw = read_column("yaw", default=np.nan)
        t = read_column("t", default=np.nan)

    data: dict[str, np.ndarray] = {"x": x, "y": y}
    if not np.all(np.isnan(yaw)):
        data["yaw"] = yaw
    if not np.all(np.isnan(t)):
        data["t"] = t
    return data


def _load_npz_path(file_path: Path, role: str) -> dict[str, np.ndarray]:
    with np.load(file_path, allow_pickle=False) as data:
        keys = set(data.files)

        def pick(*candidates: str) -> np.ndarray:
            for candidate in candidates:
                if candidate in keys:
                    return np.asarray(data[candidate], dtype=float)
            raise ValueError(f"NPZ file {file_path} is missing one of: {candidates}")

        if role not in {"ref", "track"}:
            raise ValueError(f"Unknown role: {role}")

        prefix = "ref_" if role == "ref" else "track_"
        payload: dict[str, np.ndarray] = {
            "x": pick(f"{prefix}x", "ref_x", "track_x", "x"),
            "y": pick(f"{prefix}y", "ref_y", "track_y", "y"),
        }

        yaw_candidates = [f"{prefix}yaw", "ref_yaw", "track_yaw", "yaw"]
        if any(k in keys for k in yaw_candidates):
            payload["yaw"] = pick(*yaw_candidates)

        time_candidates = [f"{prefix}t", "ref_t", "track_t", "t"]
        if any(k in keys for k in time_candidates):
            payload["t"] = pick(*time_candidates)

        for extra_key in ["cmd_speed", "obs_speed", "track_error", "cmd_vx", "cmd_vy", "cmd_vw", "obs_vx", "obs_vy", "obs_vw"]:
            if extra_key in keys:
                payload[extra_key] = np.asarray(data[extra_key], dtype=float)

        return payload


def load_path_file(file_path: str | Path | None, role: str) -> dict[str, np.ndarray] | None:
    if file_path is None:
        return None
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_path(path)
    if suffix == ".npz":
        return _load_npz_path(path, role=role)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def demo_paths() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from linear import SplinePlanner

    planner = SplinePlanner()
    x_ref, y_ref, yaw_ref = planner.generate_path(
        x_pts=np.array([0.0, 1.6, 3.1, 4.6, 6.6, 8.0]),
        y_pts=np.array([0.0, 1.0, 0.2, 2.0, 1.6, 4.0]),
        step_cm=6.0,
    )

    s = np.linspace(0.0, 1.0, len(x_ref))
    yaw_unwrapped = np.unwrap(yaw_ref)
    yaw_rate = np.abs(np.gradient(yaw_unwrapped, s, edge_order=1))
    turn_factor = yaw_rate / (np.max(yaw_rate) + 1e-9)

    tangent_x = np.cos(yaw_ref)
    tangent_y = np.sin(yaw_ref)
    normal_x = -np.sin(yaw_ref)
    normal_y = np.cos(yaw_ref)

    # 直线段误差收敛到 0.5 cm 左右，弯道最大误差约 7 cm，并且偏向沿前方切弯
    end_window = np.clip((s - 0.88) / 0.12, 0.0, 1.0)
    end_relax = 1.0 - end_window * end_window * (3.0 - 2.0 * end_window)
    error_mag = 0.005 + 0.087 * turn_factor * end_relax

    bend_dir_x = 0.58 * tangent_x + 0.82 * normal_x
    bend_dir_y = 0.58 * tangent_y + 0.82 * normal_y
    bend_norm = np.sqrt(bend_dir_x * bend_dir_x + bend_dir_y * bend_dir_y) + 1e-9
    bend_dir_x /= bend_norm
    bend_dir_y /= bend_norm

    side_wobble = 0.004 * np.sin(4.0 * np.pi * s) * (0.40 + 0.60 * (1.0 - turn_factor))

    x_track = x_ref + error_mag * bend_dir_x + side_wobble * normal_x
    y_track = y_ref + error_mag * bend_dir_y + side_wobble * normal_y
    yaw_track = _normalize_angle(yaw_ref + 0.045 * np.sin(5.0 * np.pi * s) - 0.018)

    ref_segment = np.sqrt(np.diff(x_ref) ** 2 + np.diff(y_ref) ** 2)
    ref_t = np.concatenate(([0.0], np.cumsum(ref_segment / 2.0)))
    t_ref = ref_t
    t_track = ref_t.copy()
    ref_speed = np.full_like(t_ref, 2.0, dtype=float)
    time_norm = t_track / max(float(t_track[-1]), 1e-9)
    quick_rise = 1.0 - np.exp(-14.0 * time_norm)
    bend_slowdown = 1.0 - 0.12 * turn_factor
    slow_tail = np.clip((time_norm - 0.90) / 0.10, 0.0, 1.0)
    smooth_tail = slow_tail * slow_tail * (3.0 - 2.0 * slow_tail)
    cmd_speed = 2.0 * quick_rise * bend_slowdown * (1.0 - smooth_tail)
    return (
        {"x": x_ref, "y": y_ref, "yaw": yaw_ref, "t": t_ref, "ref_speed": ref_speed},
        {"x": x_track, "y": y_track, "yaw": yaw_track, "t": t_track, "cmd_speed": cmd_speed},
    )


def _path_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    points = np.column_stack([x, y])
    return np.stack([points[:-1], points[1:]], axis=1)


def _draw_gradient_path(ax: plt.Axes, x: np.ndarray, y: np.ndarray, cmap: str, linewidth: float, alpha: float = 1.0) -> None:
    if len(x) < 2:
        return
    segments = _path_segments(x, y)
    colors = np.linspace(0.0, 1.0, len(segments))
    collection = LineCollection(segments, cmap=cmap, linewidths=linewidth, alpha=alpha)
    collection.set_array(colors)
    collection.set_zorder(3)
    ax.add_collection(collection)


def _draw_heading_arrows(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray | None,
    color: str,
    stride: int,
    scale: float,
    alpha: float,
) -> None:
    if yaw is None or len(x) == 0:
        return
    stride = max(1, stride)
    indices = np.arange(0, len(x), stride)
    u = np.cos(yaw[indices]) * scale
    v = np.sin(yaw[indices]) * scale
    ax.quiver(
        x[indices],
        y[indices],
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        headwidth=4.5,
        headlength=6.0,
        headaxislength=5.0,
        color=color,
        alpha=alpha,
        zorder=4,
    )


def _nearest_error(reference: dict[str, np.ndarray], tracked: dict[str, np.ndarray]) -> np.ndarray:
    ref_points = np.column_stack([reference["x"], reference["y"]])
    track_points = np.column_stack([tracked["x"], tracked["y"]])
    diff = track_points[:, None, :] - ref_points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(dist, axis=1)


def _infer_time(data: dict[str, np.ndarray], fallback_dt: float = 0.1) -> np.ndarray:
    if "t" in data:
        t = np.asarray(data["t"], dtype=float)
        if len(t) >= 2 and np.all(np.isfinite(t)):
            return t
    n = len(np.asarray(data["x"], dtype=float))
    return np.arange(n, dtype=float) * fallback_dt


def _compute_speed(data: dict[str, np.ndarray], fallback_dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    t = _infer_time(data, fallback_dt=fallback_dt)

    if len(x) < 2:
        return t, np.zeros_like(t)

    dt = np.diff(t)
    dt = np.where(dt <= 1e-9, fallback_dt, dt)
    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    speed = np.concatenate(([0.0], dist / dt))
    return t, speed


def build_figure(reference: dict[str, np.ndarray], tracked: dict[str, np.ndarray], title: str) -> plt.Figure:
    ref_x = np.asarray(reference["x"], dtype=float)
    ref_y = np.asarray(reference["y"], dtype=float)
    ref_yaw = np.asarray(reference.get("yaw")) if "yaw" in reference else None

    track_x = np.asarray(tracked["x"], dtype=float)
    track_y = np.asarray(tracked["y"], dtype=float)
    track_yaw = np.asarray(tracked.get("yaw")) if "yaw" in tracked else None
    ref_t = _infer_time(reference)
    ref_speed = np.asarray(reference["ref_speed"], dtype=float) if "ref_speed" in reference else _compute_speed(reference)[1]
    track_t, computed_track_speed = _compute_speed(tracked)

    track_speed = np.asarray(tracked["cmd_speed"], dtype=float) if "cmd_speed" in tracked else computed_track_speed
    obs_speed = np.asarray(tracked["obs_speed"], dtype=float) if "obs_speed" in tracked else None
    track_error = np.asarray(tracked["track_error"], dtype=float) if "track_error" in tracked else _nearest_error(reference, tracked)
    mean_error = float(np.mean(track_error))
    max_error = float(np.max(track_error))
    end_error = float(track_error[-1])

    fig = plt.figure(figsize=(17, 10), dpi=220)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.6, 1.25],
        height_ratios=[1.0, 1.0],
        wspace=0.22,
        hspace=0.16,
    )
    ax = fig.add_subplot(grid[:, 0])
    ax_speed = fig.add_subplot(grid[0, 1])
    ax_error = fig.add_subplot(grid[1, 1])

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax_speed.set_facecolor("#f8fafc")
    ax_error.set_facecolor("#f8fafc")

    _draw_gradient_path(ax, track_x, track_y, cmap="viridis", linewidth=4.0, alpha=0.98)
    ax.plot(
        ref_x,
        ref_y,
        linestyle=(0, (7, 5)),
        linewidth=3.0,
        color="#0f172a",
        alpha=0.85,
        label="Reference path",
        zorder=2,
    )

    if len(track_x) > 0:
        ax.scatter(
            track_x[0],
            track_y[0],
            s=120,
            marker="o",
            facecolor="#ffffff",
            edgecolor="#ef4444",
            linewidth=2.0,
            zorder=6,
            label="Track start",
        )
        ax.scatter(
            track_x[-1],
            track_y[-1],
            s=180,
            marker="*",
            facecolor="#ef4444",
            edgecolor="#7f1d1d",
            linewidth=1.0,
            zorder=7,
            label="Track end",
        )

    if len(ref_x) > 0:
        ax.scatter(
            ref_x[0],
            ref_y[0],
            s=120,
            marker="s",
            facecolor="#ffffff",
            edgecolor="#1d4ed8",
            linewidth=2.0,
            zorder=6,
            label="Reference start",
        )
        ax.scatter(
            ref_x[-1],
            ref_y[-1],
            s=180,
            marker="D",
            facecolor="#1d4ed8",
            edgecolor="#1e3a8a",
            linewidth=1.0,
            zorder=7,
            label="Reference end",
        )

    _draw_heading_arrows(ax, ref_x, ref_y, ref_yaw, color="#334155", stride=max(1, len(ref_x) // 18), scale=0.18, alpha=0.55)
    _draw_heading_arrows(ax, track_x, track_y, track_yaw, color="#f97316", stride=max(1, len(track_x) // 16), scale=0.22, alpha=0.75)

    if len(track_x) > 0 and len(ref_x) > 0:
        sample_idx = np.linspace(0, len(track_x) - 1, num=min(18, len(track_x)), dtype=int)
        ref_points = np.column_stack([ref_x, ref_y])
        for idx in sample_idx:
            point = np.array([track_x[idx], track_y[idx]])
            nearest = ref_points[np.argmin(np.sum((ref_points - point) ** 2, axis=1))]
            ax.plot([point[0], nearest[0]], [point[1], nearest[1]], color="#94a3b8", linewidth=0.8, alpha=0.45, zorder=1)

    ax.text(
        0.02,
        0.98,
        f"mean error = {mean_error:.3f} m\nmax error = {max_error:.3f} m\nfinal error = {end_error:.3f} m",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
        zorder=10,
    )

    ax.set_title(title, fontsize=22, fontweight="bold", pad=18, color="#0f172a")
    ax.set_xlabel("X / m", fontsize=15)
    ax.set_ylabel("Y / m", fontsize=15)
    ax.grid(True, which="major", color="#cbd5e1", linewidth=0.9, alpha=0.7)
    ax.grid(True, which="minor", color="#e2e8f0", linewidth=0.5, alpha=0.5)
    ax.minorticks_on()
    ax.set_aspect("equal", adjustable="box")

    all_x = np.concatenate([ref_x, track_x]) if len(ref_x) and len(track_x) else (ref_x if len(ref_x) else track_x)
    all_y = np.concatenate([ref_y, track_y]) if len(ref_y) and len(track_y) else (ref_y if len(ref_y) else track_y)
    if len(all_x) > 0:
        x_span = float(np.max(all_x) - np.min(all_x))
        y_span = float(np.max(all_y) - np.min(all_y))
        margin = 0.12 * max(x_span, y_span, 1.0)
        ax.set_xlim(float(np.min(all_x) - margin), float(np.max(all_x) + margin))
        ax.set_ylim(float(np.min(all_y) - margin), float(np.max(all_y) + margin))

    legend = ax.legend(loc="lower right", frameon=True, fontsize=12)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cbd5e1")
    legend.get_frame().set_alpha(0.96)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#94a3b8")

    # 速度小图：折线图
    ax_speed.plot(ref_t, ref_speed, color="#64748b", linewidth=2.0, linestyle="--", label="Ref speed")
    ax_speed.plot(track_t, track_speed, color="#2563eb", linewidth=2.4, label="Cmd speed")
    ax_speed.fill_between(track_t, track_speed, color="#93c5fd", alpha=0.22)
    if obs_speed is not None:
        ax_speed.plot(track_t, obs_speed, color="#f97316", linewidth=1.8, linestyle=":", label="Observed speed")
    ax_speed.set_title("Speed", fontsize=15, fontweight="bold", color="#0f172a")
    ax_speed.set_ylabel("m/s", fontsize=12)
    ax_speed.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.7)
    ax_speed.legend(loc="upper right", fontsize=10, frameon=True)
    for spine in ax_speed.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#94a3b8")

    # 追踪误差小图：柱状图
    error_bar_color = np.where(track_error <= mean_error, "#22c55e", "#ef4444")
    ax_error.bar(track_t, track_error, width=max(0.06, float(np.ptp(track_t)) / max(len(track_t), 30)), color=error_bar_color, alpha=0.8)
    ax_error.axhline(mean_error, color="#0f172a", linewidth=1.6, linestyle="--", label=f"Mean {mean_error:.3f} m")
    ax_error.set_title("Tracking Error", fontsize=15, fontweight="bold", color="#0f172a")
    ax_error.set_xlabel("Time / s", fontsize=12)
    ax_error.set_ylabel("m", fontsize=12)
    ax_error.grid(True, axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.7)
    ax_error.legend(loc="upper right", fontsize=10, frameon=True)
    for spine in ax_error.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#94a3b8")

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot path tracking result as a single PPT-friendly figure.")
    parser.add_argument("--ref-file", type=str, default=None, help="Reference path file (.csv or .npz)")
    parser.add_argument("--track-file", type=str, default=None, help="Tracked path file (.csv or .npz)")
    parser.add_argument("--output", type=str, default="tracking_effect.png", help="Output image path")
    parser.add_argument("--title", type=str, default="Path Tracking Result", help="Figure title")
    parser.add_argument("--dpi", type=int, default=320, help="Output DPI")
    parser.add_argument("--pdf", action="store_true", help="Also save a PDF copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ref_file is None and args.track_file is None:
        reference, tracked = demo_paths()
        title = args.title + " (demo)"
    else:
        if args.ref_file is None or args.track_file is None:
            raise SystemExit("Please provide both --ref-file and --track-file, or omit both for demo mode.")
        reference = load_path_file(args.ref_file, role="ref")
        tracked = load_path_file(args.track_file, role="track")
        if reference is None or tracked is None:
            raise SystemExit("Failed to load path data.")
        title = args.title

    fig = build_figure(reference, tracked, title=title)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    if args.pdf:
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()