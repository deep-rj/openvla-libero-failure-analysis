import numpy as np

from harness import review

P = review.DEFAULT_STALL


def _width(*segments):
    return np.concatenate([np.full(n, w) for n, w in segments])


def test_settled_counts_only_still_steps_between_empty_and_open():
    width = _width((5, 0.08), (5, 0.022), (5, 0.001))
    expected = [False] * 6 + [True] * 3 + [False] * 6  # segment edges are moving steps
    assert review.settled_mask(width, P).tolist() == expected


def test_settled_excludes_steps_where_the_fingers_are_still_moving():
    width = np.array([0.060, 0.050, 0.040, 0.030, 0.030, 0.030])
    assert review.settled_mask(width, P).tolist() == [False, False, False, False, True, True]


def test_stall_is_a_long_settled_close_on_an_object_that_never_rises():
    assert review.is_stall("failure", settled_steps=203, settled_width_m=0.022, rise_m=0.007)


def test_stall_requires_a_failed_attempt():
    assert not review.is_stall("slip", settled_steps=203, settled_width_m=0.022, rise_m=0.007)


def test_stall_requires_enough_settled_steps():
    assert not review.is_stall("failure", settled_steps=29, settled_width_m=0.022, rise_m=0.0)


def test_stall_excludes_a_lifted_object():
    assert not review.is_stall("failure", settled_steps=168, settled_width_m=0.029, rise_m=0.288)


def test_stall_excludes_fingers_near_open():
    assert not review.is_stall("failure", settled_steps=105, settled_width_m=0.0698, rise_m=0.0)


def test_window_drivers_lists_steps_above_the_floor_by_size_with_offsets():
    exp_h = np.ones(20)
    exp_h[[14, 17, 18]] = [1.3, 1.02, 1.9]
    drivers = review.window_drivers(exp_h, np.arange(10, 20), command=19, floor=0.05)
    assert [(round(e, 3), o) for e, o in drivers] == [(0.9, -1), (0.3, -5)]
