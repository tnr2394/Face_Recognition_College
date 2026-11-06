"""Shift-aware IN/OUT punch rules (pure function — no DB / Flask)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Optional
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Kolkata")
SHIFT_END_DEFAULT = time(18, 30)
SHIFT_START_DEFAULT = time(9, 30)
DEBOUNCE_SEC = 45
MIN_BREAK_SEC = 60

# Back-compat alias used by older tests/imports
SHIFT_END = SHIFT_END_DEFAULT

ROLE_BY_CAMERA = {
    "gate_in": "in",
    "gate_out": "out",
    "cam1": "in",
}

INFERRED_IN_CAMERA = "gate_in"


@dataclass
class Punch:
    action: str  # clock_in | break_start | break_end | clock_out | ignored
    state: str  # absent | present | on_break | left
    reason: str


@dataclass
class PlannedPunch:
    action: str
    state: str
    reason: str
    camera_id: str
    role: str


def camera_role(camera_id: Optional[str]) -> Optional[str]:
    if not camera_id:
        return None
    return ROLE_BY_CAMERA.get(str(camera_id).strip())


def _parse_hhmm(value, fallback: time) -> time:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    parts = text.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return fallback
        return time(hour, minute)
    except (TypeError, ValueError, IndexError):
        return fallback


def get_office_hours(path=None):
    """Read shift_start / shift_end from cameras.json office section."""
    office = {}
    try:
        from camera_config import load_cameras

        loaded = load_cameras(path).get("office") or {}
        if isinstance(loaded, dict):
            office = loaded
    except Exception:
        office = {}
    return {
        "shift_start": _parse_hhmm(office.get("shift_start"), SHIFT_START_DEFAULT),
        "shift_end": _parse_hhmm(office.get("shift_end"), SHIFT_END_DEFAULT),
    }


def get_shift_end(path=None) -> time:
    return get_office_hours(path)["shift_end"]


def apply_punch(
    state: str,
    last_camera: Optional[str],
    last_at: Optional[datetime],
    break_started_at: Optional[datetime],
    camera_id: str,
    role: str,
    at: datetime,
    shift_end: Optional[time] = None,
) -> Punch:
    """
    Apply one camera recognition to attendance state.
    `at` must be timezone-aware (prefer Asia/Kolkata).
    """
    if at.tzinfo is None:
        at = at.replace(tzinfo=TZ)
    else:
        at = at.astimezone(TZ)

    state = state or "absent"
    role = (role or "").strip().lower()
    camera_id = str(camera_id or "").strip()
    end = shift_end if shift_end is not None else get_shift_end()

    if last_at is not None:
        last = last_at if last_at.tzinfo else last_at.replace(tzinfo=TZ)
        last = last.astimezone(TZ)
        if last_camera == camera_id and (at - last).total_seconds() < DEBOUNCE_SEC:
            return Punch("ignored", state, "debounce_same_camera")

    if state == "absent":
        if role == "in":
            return Punch("clock_in", "present", "first_in")
        return Punch("ignored", "absent", "out_without_entry")

    if state == "present":
        if role == "in":
            return Punch("ignored", "present", "already_present")
        if role == "out":
            if at.timetz().replace(tzinfo=None) < end:
                return Punch("break_start", "on_break", "out_during_shift")
            return Punch("clock_out", "left", "out_after_shift")
        return Punch("ignored", state, "unknown_role")

    if state == "on_break":
        if role == "out":
            return Punch("ignored", "on_break", "already_on_break")
        if role == "in":
            started = break_started_at
            if started is not None:
                started = started if started.tzinfo else started.replace(tzinfo=TZ)
                started = started.astimezone(TZ)
                if (at - started).total_seconds() < MIN_BREAK_SEC:
                    return Punch("ignored", "present", "short_bounce_cancel_break")
            return Punch("break_end", "present", "return_from_break")
        return Punch("ignored", state, "unknown_role")

    if state == "left":
        return Punch("ignored", "left", "already_left")

    return Punch("ignored", state, "unknown_state")


def plan_punches(
    state: str,
    last_camera: Optional[str],
    last_at: Optional[datetime],
    break_started_at: Optional[datetime],
    camera_id: str,
    role: str,
    at: datetime,
    shift_end: Optional[time] = None,
) -> List[PlannedPunch]:
    """
    Plan one or more punches for a recognition.
    Absent + OUT -> inferred clock_in (gate_in) then normal OUT punch.
    """
    camera_id = str(camera_id or "").strip()
    role = (role or "").strip().lower()
    state = state or "absent"
    end = shift_end if shift_end is not None else get_shift_end()

    if state == "absent" and role == "out":
        inferred = PlannedPunch(
            action="clock_in",
            state="present",
            reason="inferred_in_from_out",
            camera_id=INFERRED_IN_CAMERA,
            role="in",
        )
        out_punch = apply_punch(
            "present",
            INFERRED_IN_CAMERA,
            at,
            None,
            camera_id,
            "out",
            at,
            shift_end=end,
        )
        return [
            inferred,
            PlannedPunch(
                action=out_punch.action,
                state=out_punch.state,
                reason=out_punch.reason,
                camera_id=camera_id,
                role="out",
            ),
        ]

    punch = apply_punch(
        state,
        last_camera,
        last_at,
        break_started_at,
        camera_id,
        role,
        at,
        shift_end=end,
    )
    return [
        PlannedPunch(
            action=punch.action,
            state=punch.state,
            reason=punch.reason,
            camera_id=camera_id,
            role=role,
        )
    ]
