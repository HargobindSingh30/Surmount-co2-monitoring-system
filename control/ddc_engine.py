"""
DDC Control Engine - faithful to SBI GITC Building BuildTrack spec.

Thresholds are ABSOLUTE ppm values:
  Lower:    490 ppm  (differential = 90)
  Upper:    520 ppm  (differential = 120)
  IGBC max: 530 ppm  (differential = 130)

Damper variant: +/-20% steps, init at 20%, min 0%, max 100%
Fan variant:    ON >= upper, OFF < lower
Log trigger:    every 25 ppm differential change OR event type change
"""

from dataclasses import dataclass, field

ATMOSPHERIC_BASELINE = 400.0

# Absolute CO2 thresholds (ppm)
UPPER_PPM = 520.0
LOWER_PPM = 490.0
IGBC_MAX  = 530.0

# As differentials (what the engine actually compares)
THRESHOLD_UPPER = UPPER_PPM - ATMOSPHERIC_BASELINE  # 120 ppm differential
THRESHOLD_LOWER = LOWER_PPM - ATMOSPHERIC_BASELINE  #  90 ppm differential

DAMPER_STEP = 20
DAMPER_MIN  = 0
DAMPER_MAX  = 100
LOG_EVERY_PPM = 25


@dataclass
class DDCState:
    sensor_id:        str
    is_fan_variant:   bool = False
    damper_position:  int  = 20
    fan_on:           bool = False
    last_logged_diff: float = field(default=None, init=False)
    last_event:       str   = field(default="INIT", init=False)


def _determine_event(diff, state):
    if state.is_fan_variant:
        if diff >= THRESHOLD_UPPER:
            return "FAN_ON",  state.damper_position, True
        elif diff < THRESHOLD_LOWER:
            return "FAN_OFF", state.damper_position, False
        else:
            return "HOLD",    state.damper_position, state.fan_on
    else:
        if diff >= THRESHOLD_UPPER:
            new_pos = min(DAMPER_MAX, state.damper_position + DAMPER_STEP)
            return "OPEN",  new_pos, False
        elif diff < THRESHOLD_LOWER:
            new_pos = max(DAMPER_MIN, state.damper_position - DAMPER_STEP)
            return "CLOSE", new_pos, False
        else:
            return "HOLD",  state.damper_position, False


def _should_log(diff, state, event):
    if event != state.last_event:
        return True
    if state.last_logged_diff is None:
        return True
    if abs(diff - state.last_logged_diff) >= LOG_EVERY_PPM:
        return True
    return False


def process_reading(reading, state):
    diff = reading["differential"]
    event, new_damper, new_fan = _determine_event(diff, state)

    should_log = _should_log(diff, state, event)

    state.damper_position = new_damper
    state.fan_on          = new_fan
    state.last_event      = event
    if should_log:
        state.last_logged_diff = diff

    if not should_log:
        return None, state

    log_record = {
        "sensor_id":        reading["sensor_id"],
        "floor":            reading["floor"],
        "wing":             reading["wing"],
        "timestamp":        reading["timestamp"],
        "co2_level":        reading["co2_level"],
        "differential":     diff,
        "damper_position":  new_damper,
        "fan_on":           new_fan,
        "event_type":       event,
        "occupancy":        reading.get("occupancy"),
        "predicted_breach": None,
    }

    reading["damper_position"] = new_damper
    reading["fan_on"]          = new_fan

    return log_record, state


def run_pipeline(readings):
    """Process pre-generated readings. No DDC->simulator feedback."""
    states = {}
    log_records = []
    for reading in readings:
        sid = reading["sensor_id"]
        if sid not in states:
            is_fan = reading.get("is_fan_variant", sid.endswith("AHU02"))
            states[sid] = DDCState(sensor_id=sid, is_fan_variant=is_fan)
        log_rec, states[sid] = process_reading(reading, states[sid])
        if log_rec is not None:
            log_records.append(log_rec)
    return log_records
