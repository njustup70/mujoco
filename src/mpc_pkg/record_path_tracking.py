#!/usr/bin/env python3
"""Record live MPC control-node data and export a PPT-ready tracking figure.

This node subscribes to:
- /mpc/reference_path
- /mpc/tracked_path
- /state/cmd_vel
- /state/observe_vel

It records a short window of real data, saves it to NPZ, and then calls the
plotting script to produce a single summary image.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node


@dataclass
class PathSnapshot:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


class TrackingRecorder(Node):
    def __init__(self, output_npz: Path, sample_period: float, duration: float, plot_script: Path, output_png: Path, title: str) -> None:
        super().__init__("tracking_recorder")
        self.output_npz = output_npz
        self.sample_period = float(sample_period)
        self.duration = float(duration)
        self.plot_script = plot_script
        self.output_png = output_png
        self.title = title

        self.reference_path: Optional[PathSnapshot] = None
        self.tracked_path: Optional[PathSnapshot] = None
        self.last_cmd_vel = np.zeros(3, dtype=float)
        self.last_obs_vel = np.zeros(3, dtype=float)
        self.samples: list[dict[str, float]] = []
        self.start_time = self.get_clock().now().nanoseconds * 1e-9
        self.finished = False

        self.create_subscription(PathMsg, "/mpc/reference_path", self._reference_path_callback, 10)
        self.create_subscription(PathMsg, "/mpc/tracked_path", self._tracked_path_callback, 10)
        self.create_subscription(Twist, "/state/cmd_vel", self._cmd_vel_callback, 10)
        self.create_subscription(Vector3Stamped, "/state/observe_vel", self._observe_vel_callback, 10)

        self.timer = self.create_timer(self.sample_period, self._sample_once)
        self.stop_timer = self.create_timer(0.25, self._check_finish)

    @staticmethod
    def _path_to_snapshot(msg: PathMsg) -> Optional[PathSnapshot]:
        if not msg.poses:
            return None
        xs = []
        ys = []
        yaws = []
        for pose_stamped in msg.poses:
            pose = pose_stamped.pose
            xs.append(float(pose.position.x))
            ys.append(float(pose.position.y))
            q = pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaws.append(float(np.arctan2(siny_cosp, cosy_cosp)))
        return PathSnapshot(x=np.asarray(xs, dtype=float), y=np.asarray(ys, dtype=float), yaw=np.asarray(yaws, dtype=float))

    @staticmethod
    def _nearest_error(reference: PathSnapshot, tracked_point: np.ndarray) -> float:
        ref_points = np.column_stack([reference.x, reference.y])
        diff = ref_points - tracked_point[None, :]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        return float(np.min(dist))

    def _reference_path_callback(self, msg: PathMsg) -> None:
        snapshot = self._path_to_snapshot(msg)
        if snapshot is not None:
            self.reference_path = snapshot

    def _tracked_path_callback(self, msg: PathMsg) -> None:
        snapshot = self._path_to_snapshot(msg)
        if snapshot is not None:
            self.tracked_path = snapshot

    def _cmd_vel_callback(self, msg: Twist) -> None:
        self.last_cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=float)

    def _observe_vel_callback(self, msg: Vector3Stamped) -> None:
        self.last_obs_vel = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)

    def _sample_once(self) -> None:
        if self.reference_path is None or self.tracked_path is None:
            return

        if len(self.tracked_path.x) == 0:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self.start_time
        track_x = float(self.tracked_path.x[-1])
        track_y = float(self.tracked_path.y[-1])
        track_yaw = float(self.tracked_path.yaw[-1])
        error = self._nearest_error(self.reference_path, np.array([track_x, track_y], dtype=float))
        cmd_speed = float(np.hypot(self.last_cmd_vel[0], self.last_cmd_vel[1]))
        obs_speed = float(np.hypot(self.last_obs_vel[0], self.last_obs_vel[1]))

        self.samples.append(
            {
                "t": elapsed,
                "x": track_x,
                "y": track_y,
                "yaw": track_yaw,
                "error": error,
                "cmd_vx": float(self.last_cmd_vel[0]),
                "cmd_vy": float(self.last_cmd_vel[1]),
                "cmd_vw": float(self.last_cmd_vel[2]),
                "cmd_speed": cmd_speed,
                "obs_vx": float(self.last_obs_vel[0]),
                "obs_vy": float(self.last_obs_vel[1]),
                "obs_vw": float(self.last_obs_vel[2]),
                "obs_speed": obs_speed,
            }
        )

    def _check_finish(self) -> None:
        if self.finished:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.start_time >= self.duration:
            self.finished = True
            self._finalize_and_exit()

    def _finalize_and_exit(self) -> None:
        if self.reference_path is None or self.tracked_path is None:
            self.get_logger().error("No path data received; nothing to save.")
            rclpy.shutdown()
            return

        samples = self.samples
        if not samples:
            self.get_logger().error("No time samples recorded; nothing to save.")
            rclpy.shutdown()
            return

        sample_t = np.asarray([row["t"] for row in samples], dtype=float)
        sample_x = np.asarray([row["x"] for row in samples], dtype=float)
        sample_y = np.asarray([row["y"] for row in samples], dtype=float)
        sample_yaw = np.asarray([row["yaw"] for row in samples], dtype=float)
        sample_error = np.asarray([row["error"] for row in samples], dtype=float)
        cmd_vx = np.asarray([row["cmd_vx"] for row in samples], dtype=float)
        cmd_vy = np.asarray([row["cmd_vy"] for row in samples], dtype=float)
        cmd_vw = np.asarray([row["cmd_vw"] for row in samples], dtype=float)
        cmd_speed = np.asarray([row["cmd_speed"] for row in samples], dtype=float)
        obs_vx = np.asarray([row["obs_vx"] for row in samples], dtype=float)
        obs_vy = np.asarray([row["obs_vy"] for row in samples], dtype=float)
        obs_vw = np.asarray([row["obs_vw"] for row in samples], dtype=float)
        obs_speed = np.asarray([row["obs_speed"] for row in samples], dtype=float)

        np.savez(
            self.output_npz,
            ref_x=self.reference_path.x,
            ref_y=self.reference_path.y,
            ref_yaw=self.reference_path.yaw,
            ref_t=np.arange(len(self.reference_path.x), dtype=float) * self.sample_period,
            track_x=sample_x,
            track_y=sample_y,
            track_yaw=sample_yaw,
            t=sample_t,
            track_error=sample_error,
            cmd_vx=cmd_vx,
            cmd_vy=cmd_vy,
            cmd_vw=cmd_vw,
            cmd_speed=cmd_speed,
            obs_vx=obs_vx,
            obs_vy=obs_vy,
            obs_vw=obs_vw,
            obs_speed=obs_speed,
        )
        self.get_logger().info(f"Saved NPZ to {self.output_npz}")

        cmd = [
            "/usr/bin/python",
            str(self.plot_script),
            "--ref-file",
            str(self.output_npz),
            "--track-file",
            str(self.output_npz),
            "--output",
            str(self.output_png),
            "--title",
            self.title,
        ]
        subprocess.run(cmd, check=True)
        self.get_logger().info(f"Saved figure to {self.output_png}")
        rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record MPC control-node data and generate a summary figure.")
    parser.add_argument("--duration", type=float, default=15.0, help="Recording duration in seconds")
    parser.add_argument("--sample-period", type=float, default=0.1, help="Sampling period in seconds")
    parser.add_argument("--output-npz", type=str, default="tracking_record.npz", help="Output NPZ path")
    parser.add_argument("--output-png", type=str, default="tracking_record.png", help="Output PNG path")
    parser.add_argument("--title", type=str, default="Path Tracking Result", help="Figure title")
    parser.add_argument(
        "--plot-script",
        type=str,
        default="src/mpc_pkg/plot_path_tracking.py",
        help="Path to the plotting script",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()

    node = TrackingRecorder(
        output_npz=Path(args.output_npz).resolve(),
        sample_period=args.sample_period,
        duration=args.duration,
        plot_script=Path(args.plot_script).resolve(),
        output_png=Path(args.output_png).resolve(),
        title=args.title,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
