#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RC 2027 field — minimal MuJoCo loader / inspector.

用法 / Usage:
    pip install mujoco
    python example_load.py                # 加载两版模型 + 静置 1 s + 打印可动件
    python example_load.py --steps 1500   # 静置 3 s

输出 / Output:
    · 每版模型的 geom/mesh/body/site 计数（可用来确认文件没被改坏）
    · field_with_objects.xml 里 53 个可动件的初始位姿表（按名字索引，可直接进观测）
    · 静置测试：各可动件的竖向/水平位移（应为毫米级 = 只是落到接触面上）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import mujoco
except ImportError:                                     # pragma: no cover
    sys.exit("需要 mujoco：pip install mujoco（本包用 3.14.0 验证）")

try:
    import numpy as np
except ImportError:                                     # pragma: no cover
    sys.exit("需要 numpy：pip install numpy")

HERE = Path(__file__).resolve().parent
EXPECT = {"field.xml": (39, {39}, 0, 10), "field_with_objects.xml": (190, {62, 65}, 53, 10)}
MOVABLE = ([f"skyblock_{i:02d}" for i in range(1, 13)] + ["mustika"]
           + [f"earthblock_{s}_{i:02d}" for s in ("red", "blue") for i in range(1, 21)])


def load(name: str):
    """用 bytes assets 加载，避免 MuJoCo C++ 层处理非 ASCII 目录。"""
    xml = (HERE / name).read_text(encoding="utf-8")
    mesh_dir = HERE / "meshes"
    assets = {str(p.relative_to(mesh_dir)): p.read_bytes()
              for p in sorted(mesh_dir.rglob("*.obj"))}
    return mujoco.MjModel.from_xml_string(xml, assets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="field_with_objects.xml",
                    choices=["field.xml", "field_with_objects.xml"])
    ap.add_argument("--steps", type=int, default=500, help="静置步数（2 ms/步，500 = 1 s）")
    args = ap.parse_args()

    for name in ("field.xml", "field_with_objects.xml"):
        m = load(name)
        exp = EXPECT[name]
        got = (m.ngeom, m.nmesh, m.nbody - 1, m.nsite)
        flag = "✓" if got[0] == exp[0] and got[1] in exp[1] and got[2:] == exp[2:] else "✗"
        print(f"{name:<20} geom={got[0]:<4} mesh={got[1]:<3} body={got[2]:<3} site={got[3]:<3} "
              f"nq={m.nq:<4} {flag}")

    print(f"\n加载 {args.model} 并静置 {args.steps} 步 …")
    m = load(args.model)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in MOVABLE}
    missing = [n for n, i in ids.items() if i < 0]
    if missing:
        print(f"（本模型没有这些可动件：{len(missing)} 个，例 {missing[:3]}）")
    ids = {n: i for n, i in ids.items() if i >= 0}
    p0 = {n: d.xpos[i].copy() for n, i in ids.items()}

    for _ in range(args.steps):
        mujoco.mj_step(m, d)

    print(f"\n可动件 {len(ids)} 个（质量与尺寸见 README §4）：")
    print(f"  {'body':<22}{'x':>9}{'y':>9}{'z':>9}   {'Δz (mm)':>9}{'Δxy (mm)':>10}")
    worst = 0.0
    for n, i in ids.items():
        p = d.xpos[i]
        dz = float(p[2] - p0[n][2]) * 1000
        dxy = float(np.linalg.norm(p[:2] - p0[n][:2])) * 1000
        worst = max(worst, abs(dz), dxy)
        print(f"  {n:<22}{p[0]:>9.3f}{p[1]:>9.3f}{p[2]:>9.3f}   {dz:>9.1f}{dxy:>10.1f}")
    print(f"\n最大位移 {worst:.1f} mm → "
          + ("静置稳定 ✓（毫米级 = 只是落到接触面上）" if worst < 50 else "请检查（>50 mm）"))

    print("\n提示：给可动件加速度即可测试抓取/推挤，例如")
    print("  jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'mustika_joint')")
    print("  d.qvel[m.jnt_dofadr[jid]] = 2.0        # 向东 2 m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
