import numpy as np

from harness import attempts

T, COMMAND = 200, 50


def _episode(settle: float = 0.03, reopen_at: int | None = None):
    """One goal object at 5 cm; the fingers close from 80 mm over 10 steps from COMMAND."""
    cmd = np.full(T, -1.0)
    cmd[COMMAND:] = 1.0
    width = np.full(T, 0.08)
    width[COMMAND : COMMAND + 10] = np.linspace(0.08, settle, 10)
    width[COMMAND + 10 :] = settle
    if reopen_at is not None:
        cmd[reopen_at:] = -1.0
        width[reopen_at:] = 0.08
    dist = np.full((T, 1), 0.05)
    return cmd, width, dist


def _label(cmd, width, dist, holds=None):
    holds = [[]] * dist.shape[1] if holds is None else holds
    return attempts.label_attempts(cmd, width, dist, holds, start=10)


def test_a_grasp_starting_before_the_fingers_reopen_is_a_successful_attempt():
    cmd, width, dist = _episode(reopen_at=COMMAND + 120)
    [a] = _label(cmd, width, dist, [[(COMMAND + 100, COMMAND + 120, False)]])
    assert (a.command, a.outcome) == (COMMAND, "success")


def test_a_grasp_starting_after_the_fingers_reopen_leaves_the_attempt_failed():
    cmd, width, dist = _episode(reopen_at=COMMAND + 40)
    [a] = _label(cmd, width, dist, [[(COMMAND + 45, COMMAND + 60, False)]])
    assert a.outcome == "failure"


def test_a_grasp_that_ends_in_a_slip_is_a_slip():
    cmd, width, dist = _episode()
    [a] = _label(cmd, width, dist, [[(COMMAND + 15, COMMAND + 30, True)]])
    assert a.outcome == "slip"


def test_no_grasp_during_the_attempt_is_a_failure():
    cmd, width, dist = _episode()
    [a] = _label(cmd, width, dist)
    assert a.outcome == "failure"


def test_fingers_reopening_before_they_pass_the_open_width_abort_the_attempt():
    cmd, width, dist = _episode(reopen_at=COMMAND + 2)
    [a] = _label(cmd, width, dist)
    assert a.outcome == "aborted"


def test_fingers_reopening_after_closing_past_the_open_width_leave_a_failure():
    cmd, width, dist = _episode(reopen_at=COMMAND + 5)
    [a] = _label(cmd, width, dist)
    assert a.outcome == "failure"


def test_a_close_while_a_goal_object_is_held_is_not_an_attempt():
    cmd, width, dist = _episode()
    assert _label(cmd, width, dist, [[(20, 120, False)]]) == []


def test_a_close_far_from_every_goal_object_is_not_an_attempt():
    cmd, width, dist = _episode()
    dist[:] = 0.20
    assert _label(cmd, width, dist) == []


def test_the_attempt_is_on_the_nearest_goal_object():
    cmd, width, _ = _episode()
    dist = np.tile([0.10, 0.04], (T, 1))
    [a] = _label(cmd, width, dist)
    assert a.obj == 1
