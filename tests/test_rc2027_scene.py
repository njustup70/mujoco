"""Check the integrated scene against the unmodified movable field model."""

from pathlib import Path
import unittest

import mujoco
import numpy as np


FIELD = Path(__file__).resolve().parents[1] / "src/mujoco_ros2_bridge/model/rc2027"
OBJECTS = (
    [f"skyblock_{i:02d}" for i in range(1, 13)]
    + ["mustika"]
    + [f"earthblock_{team}_{i:02d}" for team in ("red", "blue") for i in range(1, 21)]
)


class Rc2027SceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = mujoco.MjModel.from_xml_path(str(FIELD / "field_with_objects.xml"))
        cls.scene = mujoco.MjModel.from_xml_path(str(FIELD / "scene.xml"))

    def test_game_objects_keep_world_poses_and_free_joints(self):
        source = mujoco.MjData(self.field)
        combined = mujoco.MjData(self.scene)
        mujoco.mj_forward(self.field, source)
        mujoco.mj_forward(self.scene, combined)
        self.assertEqual(len(OBJECTS), 53)
        # The main-branch chassis has four steering pivots and four wheel bodies.
        self.assertEqual(self.scene.nbody - self.field.nbody, 9)
        for name in OBJECTS:
            np.testing.assert_allclose(combined.body(name).xpos, source.body(name).xpos, atol=1e-9)
            np.testing.assert_allclose(combined.body(name).xquat, source.body(name).xquat, atol=1e-9)
            self.assertEqual(self.scene.joint(f"{name}_joint").type, mujoco.mjtJoint.mjJNT_FREE)

    def test_chassis_starts_in_blue_zone_facing_north(self):
        data = mujoco.MjData(self.scene)
        mujoco.mj_forward(self.scene, data)
        np.testing.assert_allclose(data.body("chassis").xpos, [4.75, -5.10, 0.235], atol=1e-9)
        np.testing.assert_allclose(data.body("chassis").xmat.reshape(3, 3)[:, 0], [0, 1, 0], atol=1e-9)
        # Rear wheels clear the south fence, even though the robot is wider than the start zone.
        for i in range(4):
            wheel = data.body(f"wheel{i}").xpos
            self.assertGreater(wheel[0], 4.0)
            self.assertLess(wheel[0], 5.5)
            self.assertGreater(wheel[1] - 0.058, -5.5)
        self.assertEqual(self.scene.nu, 8)

    def test_settling_and_forward_velocity(self):
        data = mujoco.MjData(self.scene)
        mujoco.mj_forward(self.scene, data)
        initial = {name: data.body(name).xpos.copy() for name in OBJECTS}
        for _ in range(1000):
            mujoco.mj_step(self.scene, data)
        max_drift = max(np.linalg.norm(data.body(name).xpos - initial[name]) for name in OBJECTS)
        self.assertLess(max_drift, 0.01)
        np.testing.assert_allclose(data.body("chassis").xpos[:2], [4.75, -5.10], atol=0.01)

        # Positive wheel drive at +90° yaw must move the main-branch robot north.
        data = mujoco.MjData(self.scene)
        mujoco.mj_forward(self.scene, data)
        for i in range(4):
            data.actuator(f"drive{i}").ctrl[0] = 3.0
        for _ in range(500):
            mujoco.mj_step(self.scene, data)
        self.assertGreater(data.body("chassis").xpos[1], -5.05)
        self.assertAlmostEqual(data.body("chassis").xpos[0], 4.75, delta=0.01)


if __name__ == "__main__":
    unittest.main()
