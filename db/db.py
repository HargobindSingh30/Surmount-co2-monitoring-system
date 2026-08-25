"""
Database connection + helpers.
Reads DATABASE_URL from environment (set in .env or docker-compose).
"""
import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/co2monitor")


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_log(record: dict) -> None:
    """
    Insert a single activity log row.
    record keys: sensor_id, floor, wing, timestamp, co2_level, differential,
                 damper_position, fan_on, event_type, occupancy, predicted_breach
    """
    sql = """
        INSERT INTO activity_log
            (sensor_id, floor, wing, timestamp, co2_level, differential,
             damper_position, fan_on, event_type, occupancy, predicted_breach)
        VALUES
            (%(sensor_id)s, %(floor)s, %(wing)s, %(timestamp)s, %(co2_level)s,
             %(differential)s, %(damper_position)s, %(fan_on)s, %(event_type)s,
             %(occupancy)s, %(predicted_breach)s)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, record)


def bulk_insert_logs(records: list[dict]) -> None:
    """Bulk insert for simulator runs."""
    sql = """
        INSERT INTO activity_log
            (sensor_id, floor, wing, timestamp, co2_level, differential,
             damper_position, fan_on, event_type, occupancy, predicted_breach)
        VALUES
            (%(sensor_id)s, %(floor)s, %(wing)s, %(timestamp)s, %(co2_level)s,
             %(differential)s, %(damper_position)s, %(fan_on)s, %(event_type)s,
             %(occupancy)s, %(predicted_breach)s)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)


def fetch_recent_logs(sensor_id: str, limit: int = 500) -> list[dict]:
    sql = """
        SELECT * FROM activity_log
        WHERE sensor_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (sensor_id, limit))
            return [dict(r) for r in cur.fetchall()]


def fetch_all_recent(hours: int = 24) -> list[dict]:
    sql = """
        SELECT * FROM activity_log
        WHERE timestamp >= NOW() - INTERVAL '%s hours'
        ORDER BY timestamp ASC
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (hours,))
            return [dict(r) for r in cur.fetchall()]


def fetch_kpis() -> dict:
    """Return all three KPIs as dicts."""
    kpis = {}
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for view, key in [
                ("kpi_avg_differential",  "avg_differential"),
                ("kpi_breach_frequency",  "breach_frequency"),
                ("kpi_recovery_time",     "recovery_time"),
            ]:
                cur.execute(f"SELECT * FROM {view}")
                kpis[key] = [dict(r) for r in cur.fetchall()]
    return kpis
