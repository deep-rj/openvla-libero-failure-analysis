import pytest

pytest.importorskip("libero.libero")

from envs import build_env

from harness import world, world_extract

pytestmark = pytest.mark.env


@pytest.fixture(scope="module")
def subgoals():
    cache = {}

    def get(task_id):
        if task_id not in cache:
            env, _, _, _ = build_env(task_id)
            try:
                names = world.scene_layout(env).object_names
                cache[task_id] = world_extract.build_subgoals(env.env, names)
            finally:
                env.close()
        return cache[task_id]

    return get


def _by_verb(subs):
    return {s.predicate[0]: s for s in subs}


@pytest.mark.parametrize("task_id", [3, 9])
def test_closing_a_container_depends_on_placing_into_it(subgoals, task_id):
    subs = subgoals(task_id)
    close, place = _by_verb(subs)["close"], _by_verb(subs)["in"]
    assert close.depends_on == (subs.index(place),)
    assert place.depends_on == ()


@pytest.mark.parametrize("task_id", [0, 2, 4, 6, 8])
def test_unordered_tasks_have_no_dependencies(subgoals, task_id):
    assert all(s.depends_on == () for s in subgoals(task_id))


def test_fixture_thresholds_match_libero(subgoals):
    assert _by_verb(subgoals(2))["turnon"].threshold == 0.5
    assert _by_verb(subgoals(3))["close"].threshold == 0.0
    assert _by_verb(subgoals(9))["close"].threshold == -0.005


def test_caddy_compartments_are_wrong_target_candidates(subgoals):
    (book,) = subgoals(5)
    assert "desk_caddy_1_front_contain_region" in book.candidates
    assert "desk_caddy_1_back_contain_region" not in book.candidates
    assert not any("study_table" in c for c in book.candidates)


def test_other_plate_is_a_wrong_target_candidate(subgoals):
    white_mug = next(s for s in subgoals(4) if s.predicate[1] == "porcelain_mug_1")
    assert "plate_2" in white_mug.candidates
    assert "plate_1" not in white_mug.candidates


def test_fixtures_are_moved_to_the_logged_poses():
    env, _, _, _ = build_env(5)
    try:
        logged = world.scene_layout(env).static_fixture_poses
        env.reset()
        assert world.scene_layout(env).static_fixture_poses != logged
        world_extract.place_fixtures(env, logged)
        placed = world.scene_layout(env).static_fixture_poses
        for name, pose in logged.items():
            assert placed[name] == pytest.approx(pose, abs=1e-6)
    finally:
        env.close()


def test_arm_joints_are_the_seven_hinges_with_their_ranges():
    env, _, _, _ = build_env(0)
    try:
        joints = world_extract.arm_joints(env.env)
    finally:
        env.close()
    assert [j["name"] for j in joints] == [f"robot0_joint{i}" for i in range(1, 8)]
    joint4 = joints[3]
    assert joint4["range"] == pytest.approx([-3.0718, -0.0698], abs=1e-3)
