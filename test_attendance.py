"""Unit tests for apply_punch / plan_punches — sequences from the attendance plan."""

from datetime import datetime, time, timedelta

from attendance import MIN_BREAK_SEC, TZ, apply_punch, plan_punches


def _at(hour, minute=0, second=0):
    return datetime(2026, 11, 6, hour, minute, second, tzinfo=TZ)


def test_first_in_early_arrival():
    p = apply_punch("absent", None, None, None, "gate_in", "in", _at(9, 0))
    assert p.action == "clock_in"
    assert p.state == "present"


def test_debounce_same_in():
    t0 = _at(9, 0)
    p = apply_punch("present", "gate_in", t0, None, "gate_in", "in", t0 + timedelta(seconds=10))
    assert p.action == "ignored"
    assert p.state == "present"
    assert p.reason == "debounce_same_camera"


def test_break_start_and_end():
    p1 = apply_punch("present", "gate_in", _at(9, 0), None, "gate_out", "out", _at(11, 0))
    assert p1.action == "break_start"
    assert p1.state == "on_break"
    p2 = apply_punch(
        "on_break",
        "gate_out",
        _at(11, 0),
        _at(11, 0),
        "gate_in",
        "in",
        _at(11, 20),
    )
    assert p2.action == "break_end"
    assert p2.state == "present"


def test_short_bounce_cancels_break():
    start = _at(11, 0)
    p = apply_punch(
        "on_break",
        "gate_out",
        start,
        start,
        "gate_in",
        "in",
        start + timedelta(seconds=30),
    )
    assert p.action == "ignored"
    assert p.state == "present"
    assert p.reason == "short_bounce_cancel_break"
    assert 30 < MIN_BREAK_SEC


def test_out_after_shift_clock_out():
    p = apply_punch(
        "present",
        "gate_in",
        _at(9, 0),
        None,
        "gate_out",
        "out",
        _at(18, 35),
        shift_end=time(18, 30),
    )
    assert p.action == "clock_out"
    assert p.state == "left"


def test_first_event_out_infers_clock_in_then_break():
    planned = plan_punches(
        "absent",
        None,
        None,
        None,
        "gate_out",
        "out",
        _at(10, 0),
        shift_end=time(18, 30),
    )
    assert len(planned) == 2
    assert planned[0].action == "clock_in"
    assert planned[0].state == "present"
    assert planned[0].reason == "inferred_in_from_out"
    assert planned[0].camera_id == "gate_in"
    assert planned[0].role == "in"
    assert planned[1].action == "break_start"
    assert planned[1].state == "on_break"
    assert planned[1].camera_id == "gate_out"
    assert planned[1].role == "out"


def test_first_event_out_after_shift_infers_then_clock_out():
    planned = plan_punches(
        "absent",
        None,
        None,
        None,
        "gate_out",
        "out",
        _at(18, 45),
        shift_end=time(18, 30),
    )
    assert len(planned) == 2
    assert planned[0].action == "clock_in"
    assert planned[0].reason == "inferred_in_from_out"
    assert planned[1].action == "clock_out"
    assert planned[1].state == "left"


def test_after_left_in_ignored():
    p = apply_punch("left", "gate_out", _at(18, 35), None, "gate_in", "in", _at(19, 0))
    assert p.action == "ignored"
    assert p.state == "left"


def test_out_while_on_break_ignored():
    p = apply_punch(
        "on_break",
        "gate_out",
        _at(11, 0),
        _at(11, 0),
        "gate_out",
        "out",
        _at(11, 10),
    )
    assert p.action == "ignored"
    assert p.state == "on_break"


def test_custom_shift_end_affects_break_vs_out():
    p = apply_punch(
        "present",
        "gate_in",
        _at(9, 0),
        None,
        "gate_out",
        "out",
        _at(17, 0),
        shift_end=time(16, 30),
    )
    assert p.action == "clock_out"
    assert p.state == "left"
