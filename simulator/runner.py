"""
Coupled simulation runner.
Advances simulator one tick at a time, feeds DDC output (damper %) back
into the simulator before the next tick — closing the control loop properly.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from simulator.simulator import CO2Simulator, SensorConfig
from control.ddc_engine import DDCState, process_reading

SENSORS = [
    SensorConfig("F1-WA-AHU01", floor=1, wing="A", is_fan_variant=False),
    SensorConfig("F1-WA-AHU02", floor=1, wing="A", is_fan_variant=True),
]


def generate_coupled_data(
    days: int = 30,
    start_time: datetime | None = None,
    dt_minutes: float = 1.0,
    initial_co2: float = 420.0,
) -> tuple[list[dict], list[dict]]:
    """
    Run simulator + DDC engine in a coupled loop.

    Returns:
        raw_readings  — every tick (for ML feature engineering)
        log_records   — only rows that pass the DDC log filter (for DB / display)
    """
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(days=days)

    sims   = {cfg.sensor_id: CO2Simulator(cfg, start_time, initial_co2) for cfg in SENSORS}
    states = {cfg.sensor_id: DDCState(cfg.sensor_id, is_fan_variant=cfg.is_fan_variant)
              for cfg in SENSORS}

    raw_readings = []
    log_records  = []
    total_ticks  = int(days * 24 * 60 / dt_minutes)

    for _ in range(total_ticks):
        for sid, sim in sims.items():
            # 1. Advance simulator (uses current damper/fan state)
            reading = sim.next_tick(dt_minutes)
            raw_readings.append(reading)

            # 2. DDC processes reading
            log_rec, states[sid] = process_reading(reading, states[sid])

            # 3. Feed DDC output back into simulator for next tick
            sim.damper_position = states[sid].damper_position
            sim.fan_on          = states[sid].fan_on

            if log_rec is not None:
                log_records.append(log_rec)

    return raw_readings, log_records


if __name__ == "__main__":
    import pandas as pd
    from collections import Counter

    print("Running 1-day coupled simulation...")
    raw, logs = generate_coupled_data(days=1, start_time=datetime(2025, 9, 1, 0, 0))

    df_raw = pd.DataFrame(raw)
    df_log = pd.DataFrame(logs)

    print(f"Raw ticks:       {len(raw)}")
    print(f"Log entries:     {len(logs)}")
    print(f"Events:          {dict(Counter(r['event_type'] for r in logs))}")
    print(f"CO2 range:       {df_raw.co2_level.min():.1f} – {df_raw.co2_level.max():.1f} ppm")
    print(f"Diff range:      {df_log.differential.min():.1f} – {df_log.differential.max():.1f} ppm")
    print(f"Damper range:    {df_log.damper_position.min()}% – {df_log.damper_position.max()}%")
    print(f"Breach events:   {(df_log.differential >= 520).sum()}")
    print(f"IGBC violations: {(df_log.differential >= 530).sum()}")

    # Print a sample of the day's activity
    print("\nSample log (first 10 entries):")
    for r in logs[:10]:
        print(f"  {r['timestamp'].strftime('%H:%M')} | {r['sensor_id']} | "
              f"CO2={r['co2_level']:.1f} diff={r['differential']:.1f} "
              f"damper={r['damper_position']}% event={r['event_type']}")
