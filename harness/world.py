"""Named world state read from a LIBERO env: object poses, fixture joints, EEF and sim state."""

from dataclasses import asdict, dataclass

import numpy as np

from harness.schema import WorldState

_JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


@dataclass(frozen=True)
class FixtureJoint:
    name: str
    type: str
    range: tuple[float, float]
    qpos_addr: int


@dataclass(frozen=True)
class SceneLayout:
    object_names: tuple[str, ...]
    fixture_joints: tuple[FixtureJoint, ...]
    # Position + xyzw quaternion at layout time; fixtures never move, only their joints do.
    static_fixture_poses: dict[str, tuple[float, ...]]

    def metadata(self) -> dict:
        return {
            "object_names": list(self.object_names),
            "fixture_joints": [asdict(j) for j in self.fixture_joints],
            "static_fixture_poses": {k: list(v) for k, v in self.static_fixture_poses.items()},
        }


def scene_layout(env) -> SceneLayout:
    inner = env.env
    model = env.sim.model
    joints = []
    for fixture_name in inner.fixtures_dict:
        for joint_name in inner.get_object(fixture_name).joints:
            joint_id = model.joint_name2id(joint_name)
            joints.append(
                FixtureJoint(
                    name=joint_name,
                    type=_JOINT_TYPES[int(model.jnt_type[joint_id])],
                    range=tuple(float(x) for x in model.jnt_range[joint_id]),
                    qpos_addr=int(model.get_joint_qpos_addr(joint_name)),
                )
            )
    poses = {}
    for fixture_name in inner.fixtures_dict:
        body_id = inner.obj_body_id[fixture_name]
        quat_wxyz = env.sim.data.body_xquat[body_id]
        poses[fixture_name] = (
            *map(float, env.sim.data.body_xpos[body_id]),
            *map(float, quat_wxyz[[1, 2, 3, 0]]),
        )
    return SceneLayout(tuple(inner.objects_dict), tuple(joints), poses)


def read_world_state(env, obs, layout: SceneLayout) -> WorldState:
    qpos = env.sim.data.qpos
    object_poses = [
        np.concatenate([obs[f"{name}_pos"], obs[f"{name}_quat"]]) for name in layout.object_names
    ]
    return WorldState(
        sim_state=env.get_sim_state(),
        object_poses=np.array(object_poses, dtype=np.float32).reshape(-1, 7),
        fixture_joint_qpos=np.array(
            [qpos[j.qpos_addr] for j in layout.fixture_joints], dtype=np.float32
        ),
        eef_pos=np.array(obs["robot0_eef_pos"]),
        eef_quat_raw=np.array(obs["robot0_eef_quat"]),
        gripper_qpos=np.array(obs["robot0_gripper_qpos"]),
    )
