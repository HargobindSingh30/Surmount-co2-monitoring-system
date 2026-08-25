-- Smart CO2 Monitoring System — PostgreSQL Schema
-- SBI GITC Building, CBD Belapur, Navi Mumbai
-- One floor / one wing prototype

CREATE TABLE IF NOT EXISTS activity_log (
    id              BIGSERIAL PRIMARY KEY,
    sensor_id       VARCHAR(20)     NOT NULL,   -- e.g. "F1-W1-AHU01"
    floor           SMALLINT        NOT NULL,
    wing            CHAR(1)         NOT NULL,   -- A/B/C/D
    timestamp       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    co2_level       NUMERIC(6,1)    NOT NULL,   -- raw ppm from sensor
    differential    NUMERIC(6,1)    NOT NULL,   -- co2_level - 400
    damper_position SMALLINT        NOT NULL,   -- 0-100 %
    fan_on          BOOLEAN         NOT NULL DEFAULT FALSE,  -- only for fan variant
    event_type      VARCHAR(30)     NOT NULL,   -- OPEN / HOLD / CLOSE / FAN_ON / FAN_OFF / INIT
    occupancy       SMALLINT,                   -- estimated occupants at time of log
    predicted_breach BOOLEAN                    -- ML layer output (nullable until model runs)
);

CREATE INDEX IF NOT EXISTS idx_activity_log_timestamp  ON activity_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_activity_log_sensor_id  ON activity_log (sensor_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_activity_log_floor_wing ON activity_log (floor, wing, timestamp DESC);

-- View: latest state per sensor (useful for dashboard)
CREATE OR REPLACE VIEW sensor_latest AS
SELECT DISTINCT ON (sensor_id)
    sensor_id, floor, wing, timestamp,
    co2_level, differential, damper_position, fan_on, event_type, predicted_breach
FROM activity_log
ORDER BY sensor_id, timestamp DESC;

-- View: KPI 1 — average CO2 differential per zone (last 24 h)
CREATE OR REPLACE VIEW kpi_avg_differential AS
SELECT
    sensor_id, floor, wing,
    ROUND(AVG(differential), 1) AS avg_differential_ppm,
    COUNT(*)                    AS readings_24h
FROM activity_log
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY sensor_id, floor, wing
ORDER BY avg_differential_ppm DESC;

-- View: KPI 2 — breach frequency (differential >= 520) per zone (last 24 h)
CREATE OR REPLACE VIEW kpi_breach_frequency AS
SELECT
    sensor_id, floor, wing,
    COUNT(*) FILTER (WHERE differential >= 520) AS breach_count,
    COUNT(*)                                    AS total_readings,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE differential >= 520) / NULLIF(COUNT(*), 0),
        1
    )                                           AS breach_pct
FROM activity_log
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY sensor_id, floor, wing
ORDER BY breach_count DESC;

-- View: KPI 3 — average recovery time (breach end to differential < 490)
-- Approximated as avg minutes between OPEN and next CLOSE event per sensor
CREATE OR REPLACE VIEW kpi_recovery_time AS
SELECT
    sensor_id, floor, wing,
    ROUND(AVG(EXTRACT(EPOCH FROM (close_ts - open_ts)) / 60.0), 1) AS avg_recovery_min
FROM (
    SELECT
        sensor_id, floor, wing,
        timestamp                                                          AS open_ts,
        LEAD(timestamp) OVER (PARTITION BY sensor_id ORDER BY timestamp)   AS close_ts,
        event_type,
        LEAD(event_type) OVER (PARTITION BY sensor_id ORDER BY timestamp)  AS next_event
    FROM activity_log
) t
WHERE event_type = 'OPEN' AND next_event = 'CLOSE'
GROUP BY sensor_id, floor, wing;
