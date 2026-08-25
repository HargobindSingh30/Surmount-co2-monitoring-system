"""CO2 Data Simulator - SBI GITC Building, Floor 1, Wing A."""

import random
from datetime import datetime, timedelta
from dataclasses import dataclass

ATMOSPHERIC_BASELINE   = 400.0
CO2_PER_PERSON_PER_MIN = 0.30   # ppm/person/min (~500 m3 office)
NATURAL_DECAY_RATE     = 0.035  # decay fraction of differential per minute

OCCUPANCY_SCHEDULE = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
    6: 2, 7: 8, 8: 25, 9: 40, 10: 45,
    11: 45, 12: 30, 13: 35, 14: 42,
    15: 45, 16: 38, 17: 20, 18: 10,
    19: 5, 20: 2, 21: 0, 22: 0, 23: 0,
}


@dataclass
class SensorConfig:
    sensor_id:      str
    floor:          int
    wing:           str
    is_fan_variant: bool = False
    max_occupancy:  int  = 50


def _get_occupancy(hour, minute, config):
    base      = OCCUPANCY_SCHEDULE.get(hour, 0)
    next_base = OCCUPANCY_SCHEDULE.get((hour + 1) % 24, 0)
    frac      = minute / 60.0
    interp    = base + (next_base - base) * frac
    noisy     = interp * random.uniform(0.90, 1.10)
    return max(0, min(config.max_occupancy, round(noisy)))


def _maybe_inject_spike(co2, hour):
    if 8 <= hour <= 18:
        if random.random() < 0.002:
            return co2 + random.uniform(10, 35)
    return co2


class CO2Simulator:
    def __init__(self, config, start_time, initial_co2=420.0):
        self.config          = config
        self.current_time    = start_time
        self.co2             = initial_co2
        self.damper_position = 20
        self.fan_on          = False

    @property
    def differential(self):
        return round(self.co2 - ATMOSPHERIC_BASELINE, 1)

    def next_tick(self, dt_minutes=1.0):
        hour      = self.current_time.hour
        minute    = self.current_time.minute
        occupancy = _get_occupancy(hour, minute, self.config)

        generation = CO2_PER_PERSON_PER_MIN * occupancy * dt_minutes
        if self.config.is_fan_variant:
            ventilation = (18.0 if self.fan_on else 1.5) * dt_minutes
        else:
            ventilation = (self.damper_position / 100.0) * 18.0 * dt_minutes
        natural_decay = (
            NATURAL_DECAY_RATE
            * max(0, self.co2 - ATMOSPHERIC_BASELINE)
            * dt_minutes
        )

        self.co2 = max(ATMOSPHERIC_BASELINE, self.co2 + generation - ventilation - natural_decay)
        self.co2 = min(self.co2, 1500.0)
        self.co2 = _maybe_inject_spike(self.co2, hour)

        reading = {
            "sensor_id":       self.config.sensor_id,
            "floor":           self.config.floor,
            "wing":            self.config.wing,
            "timestamp":       self.current_time,
            "co2_level":       round(self.co2, 1),
            "differential":    self.differential,
            "occupancy":       occupancy,
            "damper_position": self.damper_position,
            "fan_on":          self.fan_on,
        }
        self.current_time += timedelta(minutes=dt_minutes)
        return reading


def generate_historical_data(days=30, start_time=None, dt_minutes=1.0):
    """Batch generation without DDC feedback. Use runner.py for coupled sim."""
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(days=days)
    sensors = [
        SensorConfig("F1-WA-AHU01", floor=1, wing="A", is_fan_variant=False),
        SensorConfig("F1-WA-AHU02", floor=1, wing="A", is_fan_variant=True),
    ]
    sims = [CO2Simulator(cfg, start_time) for cfg in sensors]
    out  = []
    for _ in range(int(days * 24 * 60 / dt_minutes)):
        for sim in sims:
            out.append(sim.next_tick(dt_minutes))
    return out
