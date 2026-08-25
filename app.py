"""
Smart CO2 Monitoring & Control System - SBI GITC Building
BMS operations console. Run: streamlit run app.py
"""
import time, sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from simulator.simulator import CO2Simulator, SensorConfig
from simulator.runner import generate_coupled_data
from control.ddc_engine import DDCState, process_reading

st.set_page_config(
    page_title="CO2 Monitor | SBI GITC",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
UPPER_PPM  = 520.0   # breach threshold (ppm absolute)
LOWER_PPM  = 490.0   # hold threshold
IGBC_PPM   = 530.0   # IGBC certification limit
BASE_PPM   = 400.0   # atmospheric baseline
SENSORS    = ["F1-WA-AHU01", "F1-WA-AHU02"]
TICK_SEC   = 2
SEED_DAYS  = 3
CHART_HRS  = 6

# ── Palette (IBM Carbon inspired, light console) ──────────────────────────────
C = {
    "bg":        "#F4F4F4",
    "surface":   "#FFFFFF",
    "surface2":  "#F4F4F4",
    "border":    "#E0E0E0",
    "border2":   "#C6C6C6",
    "text":      "#161616",
    "text2":     "#525252",
    "text3":     "#6F6F6F",
    "accent":    "#0F62FE",
    "safe":      "#198038",
    "safe_bg":   "#DEFBE6",
    "safe_bd":   "#A7F0BA",
    "warn":      "#8E6A00",
    "warn_bg":   "#FCF4D6",
    "warn_bd":   "#E8C84A",
    "crit":      "#DA1E28",
    "crit_bg":   "#FFF1F1",
    "crit_bd":   "#FFB3B8",
    "ahu01":     "#0F62FE",
    "ahu02":     "#8A3FFC",
    "header":    "#161616",
}

FONT_SANS = "IBM Plex Sans, sans-serif"
FONT_MONO = "IBM Plex Mono, monospace"


# ── Global CSS ────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
  font-family: {FONT_SANS};
  background: {C['bg']};
  color: {C['text']};
}}
header[data-testid="stHeader"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1320px; }}

/* ── Console header bar ── */
.bms-header {{
  background: {C['header']};
  border-bottom: 3px solid {C['accent']};
  padding: 10px 20px;
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 0;
}}
.bms-left  {{ display: flex; align-items: center; gap: 14px; }}
.bms-logo  {{
  font-family: {FONT_MONO}; font-size: 12px; font-weight: 500;
  color: #FFFFFF; background: {C['accent']}; padding: 4px 8px;
}}
.bms-title {{ font-size: 13px; font-weight: 600; color: #F4F4F4; }}
.bms-sub   {{ font-size: 11px; color: #8D8D8D; }}
.bms-right {{ display: flex; align-items: center; gap: 18px; }}
.bms-live  {{
  display: flex; align-items: center; gap: 7px;
  border: 1px solid #3D3D3D; padding: 3px 10px;
  font-size: 10px; font-weight: 600; color: #42BE65;
  letter-spacing: 0.1em; text-transform: uppercase;
}}
.bms-live-dot {{
  width: 7px; height: 7px; background: #42BE65;
  animation: bmspulse 2s ease-in-out infinite;
}}
@keyframes bmspulse {{ 0%,100% {{opacity:1}} 50% {{opacity:.25}} }}
.bms-clock {{
  font-family: {FONT_MONO}; font-size: 12px; color: #8D8D8D;
  font-variant-numeric: tabular-nums;
}}

/* ── Nav (radio styled as tabs) ── */
div[data-testid="stRadioGroup"] {{
  gap: 0 !important; background: {C['surface']};
  border: 1px solid {C['border']}; border-top: none;
  padding: 0 8px; flex-direction: row; flex-wrap: nowrap;
}}
label[data-testid="stRadioOption"] {{
  padding: 10px 18px !important; margin: 0 !important;
  border-bottom: 2px solid transparent; border-radius: 0;
  cursor: pointer; background: transparent !important;
}}
label[data-testid="stRadioOption"]:hover {{ background: {C['surface2']} !important; }}
label[data-testid="stRadioOption"] > div > div > div:first-child {{ display: none; }}
label[data-testid="stRadioOption"][data-selected="true"] {{
  border-bottom: 2px solid {C['accent']};
}}
label[data-testid="stRadioOption"][data-selected="true"] p {{
  font-weight: 600; color: {C['text']} !important;
}}
label[data-testid="stRadioOption"] p {{
  font-family: {FONT_SANS}; font-size: 13px; color: {C['text2']};
}}

/* ── Section label ── */
.sec-label {{
  font-size: 11px; font-weight: 600; color: {C['text3']};
  letter-spacing: 0.1em; text-transform: uppercase;
  margin: 18px 0 10px;
}}

/* ── KPI row ── */
.kpi-row {{
  display: grid; grid-template-columns: repeat(5, 1fr);
  gap: 1px; background: {C['border']};
  border: 1px solid {C['border']}; margin-bottom: 8px;
}}
.kpi-card {{
  background: {C['surface']}; border-top: 3px solid {C['border2']};
  padding: 13px 15px; display: flex; flex-direction: column; gap: 5px;
}}
.kpi-card.safe {{ border-top-color: {C['safe']}; }}
.kpi-card.warn {{ border-top-color: {C['warn']}; }}
.kpi-card.crit {{ border-top-color: {C['crit']}; }}
.kpi-label {{
  font-size: 10px; font-weight: 600; color: {C['text3']};
  letter-spacing: 0.08em; text-transform: uppercase;
}}
.kpi-value {{
  font-family: {FONT_MONO}; font-size: 25px; color: {C['text']};
  line-height: 1.1; font-variant-numeric: tabular-nums;
}}
.kpi-value small {{ font-size: 13px; color: {C['text3']}; }}
.kpi-sub {{ font-size: 11px; color: {C['text3']}; }}
.kpi-tag {{
  display: inline-block; font-size: 10px; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 2px 8px; align-self: flex-start; margin-top: 2px;
}}
.kpi-tag.safe {{ background:{C['safe_bg']}; color:{C['safe']}; border:1px solid {C['safe_bd']}; }}
.kpi-tag.warn {{ background:{C['warn_bg']}; color:{C['warn']}; border:1px solid {C['warn_bd']}; }}
.kpi-tag.crit {{ background:{C['crit_bg']}; color:{C['crit']}; border:1px solid {C['crit_bd']}; }}
.kpi-alert {{
  margin-top: 7px; padding: 7px 10px; font-size: 11px; line-height: 1.45;
  color: {C['text2']}; background: {C['surface2']};
  border-left: 2px solid {C['border2']};
}}
.kpi-alert.warn {{ border-left-color: {C['warn']}; }}
.kpi-alert.crit {{ border-left-color: {C['crit']}; }}
.kpi-alert b {{
  display: block; font-size: 9px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 2px;
}}
.kpi-alert.warn b {{ color: {C['warn']}; }}
.kpi-alert.crit b {{ color: {C['crit']}; }}

/* ── Card shells (Streamlit bordered containers) ── */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border: 1px solid {C['border']} !important;
  border-radius: 2px !important;
  background: {C['surface']} !important;
}}
.card-title   {{ font-size: 13px; font-weight: 600; color: {C['text']}; }}
.card-caption {{
  font-size: 11px; color: {C['text3']}; margin-top: 2px;
  padding-bottom: 8px; border-bottom: 1px solid {C['border']};
}}
.rec-strip {{
  padding: 8px 12px; background: {C['surface2']};
  border-left: 2px solid {C['warn']};
  font-size: 11px; color: {C['text2']}; line-height: 1.5;
}}
.rec-strip.crit {{ border-left-color: {C['crit']}; }}
.rec-strip b {{
  display: block; font-size: 9px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: {C['warn']}; margin-bottom: 2px;
}}
.rec-strip.crit b {{ color: {C['crit']}; }}

/* tighten vertical rhythm inside cards */
[data-testid="stVerticalBlock"] {{ gap: 0.65rem; }}

/* ── Sensor cards ── */
.sensor-row {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
}}
.sensor-card {{
  background: {C['surface']}; border: 1px solid {C['border']};
  padding: 15px 16px; display: flex; flex-direction: column; gap: 11px;
}}
.sensor-head {{ display: flex; align-items: flex-start; justify-content: space-between; }}
.sensor-id   {{ font-family: {FONT_MONO}; font-size: 12px; font-weight: 500; color: {C['text']}; }}
.sensor-type {{ font-size: 11px; color: {C['text3']}; margin-top: 2px; }}
.sensor-ppm  {{
  font-family: {FONT_MONO}; font-size: 33px;
  font-variant-numeric: tabular-nums; line-height: 1;
}}
.sensor-ppm span {{ font-size: 12px; color: {C['text3']}; font-family: {FONT_SANS}; }}
.ppm-safe {{ color: {C['safe']}; }}
.ppm-warn {{ color: {C['warn']}; }}
.ppm-crit {{ color: {C['crit']}; }}
.damper-label {{
  display: flex; justify-content: space-between;
  font-size: 11px; color: {C['text2']};
}}
.damper-val {{ font-family: {FONT_MONO}; font-weight: 500; }}
.progress-track {{
  height: 5px; background: {C['surface2']};
  border: 1px solid {C['border']}; overflow: hidden;
}}
.progress-fill {{ height: 100%; background: {C['ahu01']}; }}
.fan-line {{ display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 500; }}
.fan-dot  {{ width: 8px; height: 8px; }}
.fan-on   {{ color: {C['safe']}; }}
.fan-on .fan-dot  {{ background: {C['safe']}; }}
.fan-off  {{ color: {C['text3']}; }}
.fan-off .fan-dot {{ background: {C['border2']}; }}

/* ── Toggle & dataframe font ── */
[data-testid="stToggle"] p {{ font-size: 12px; color: {C['text2']}; }}
[data-testid="stDataFrame"] {{ font-family: {FONT_SANS}; }}
</style>""", unsafe_allow_html=True)


# ── Session init (no ML: seed 3 days of coupled simulation) ──────────────────
def init_session():
    if st.session_state.get("initialized"):
        return
    with st.spinner(f"Starting up: generating {SEED_DAYS} days of simulated history..."):
        seed_start = datetime.utcnow() - timedelta(days=SEED_DAYS)
        raw, logs  = generate_coupled_data(days=SEED_DAYS, start_time=seed_start)
        df_raw = pd.DataFrame(raw)
        df_log = pd.DataFrame(logs)

        last = df_raw.groupby("sensor_id")["co2_level"].last()
        now  = datetime.utcnow()
        cfgs = {
            "F1-WA-AHU01": SensorConfig("F1-WA-AHU01", 1, "A", is_fan_variant=False),
            "F1-WA-AHU02": SensorConfig("F1-WA-AHU02", 1, "A", is_fan_variant=True),
        }
        st.session_state["df_log"] = df_log
        st.session_state["simulators"] = {
            s: CO2Simulator(cfgs[s], now, float(last.get(s, 420))) for s in SENSORS
        }
        st.session_state["ddc_states"] = {
            s: DDCState(s, is_fan_variant=("AHU02" in s)) for s in SENSORS
        }
        st.session_state["last_tick"]   = time.time()
        st.session_state["initialized"] = True


def advance_tick():
    if time.time() - st.session_state.get("last_tick", 0) < TICK_SEC:
        return
    new_rows = []
    for sid in SENSORS:
        sim   = st.session_state["simulators"][sid]
        state = st.session_state["ddc_states"][sid]
        sim.damper_position = state.damper_position
        sim.fan_on          = state.fan_on
        reading = sim.next_tick(1.0)
        log_rec, st.session_state["ddc_states"][sid] = process_reading(reading, state)
        if log_rec is not None:
            new_rows.append(log_rec)
    if new_rows:
        st.session_state["df_log"] = pd.concat(
            [st.session_state["df_log"], pd.DataFrame(new_rows)], ignore_index=True
        )
    st.session_state["last_tick"] = time.time()


# ── Helpers ───────────────────────────────────────────────────────────────────
def naive_now():
    return pd.Timestamp.utcnow().replace(tzinfo=None)


def to_naive(series):
    s = pd.to_datetime(series)
    try:
        return s.dt.tz_localize(None)
    except TypeError:
        return s


def ppm_zone(co2):
    if co2 >= UPPER_PPM: return "crit"
    if co2 >= LOWER_PPM: return "warn"
    return "safe"


# ── KPI calculations (pure functions of the activity log) ─────────────────────
def calc_kpis(df):
    """Returns a dict with all 5 KPIs, each carrying value / status / alert."""
    out = {}
    if df.empty:
        return None

    d = df.copy()
    d["ts"] = to_naive(d["timestamp"])
    cutoff  = d["ts"].max() - pd.Timedelta(hours=24)
    w       = d[d["ts"] >= cutoff]
    if w.empty:
        w = d.tail(500)
    hours_span = max((w["ts"].max() - w["ts"].min()).total_seconds() / 3600, 0.5)

    # KPI 1: IGBC compliance (% of logged readings under 530 ppm)
    comp_pct = 100.0 * (w["co2_level"] < IGBC_PPM).mean()
    if comp_pct >= 98:   s1, a1 = "safe", None
    elif comp_pct >= 95: s1, a1 = "warn", "Compliance slipping. Review ventilation capacity before the next IGBC audit window."
    else:                s1, a1 = "crit", "Compliance below 95%. Certification at risk; escalate to facilities engineering."
    out["compliance"] = dict(value=f"{comp_pct:.1f}%", sub="of readings below 530 ppm",
                             status=s1, tag={"safe": "Compliant", "warn": "Watch", "crit": "At risk"}[s1], alert=a1)

    # KPI 2: breach frequency (readings at/above 520 ppm)
    breach_mask = w["co2_level"] >= UPPER_PPM
    breach_n    = int(breach_mask.sum())
    breach_pct  = 100.0 * breach_n / len(w)
    worst = w[breach_mask]["sensor_id"].mode()
    worst = worst.iloc[0] if not worst.empty else None
    if breach_n == 0:  s2, a2 = "safe", None
    elif breach_n <= 5:
        s2 = "warn"
        a2 = f"{worst or 'Zone'} has {breach_n} exceedance(s). Check ventilation sizing for this zone."
    else:
        s2 = "crit"
        a2 = f"{breach_n} exceedances in 24 h ({worst or 'both zones'}). Ventilation capacity is undersized for peak occupancy."
    out["breach"] = dict(value=str(breach_n), sub=f"above 520 ppm · {breach_pct:.1f}% of readings",
                         status=s2, tag={"safe": "Clear", "warn": "Review", "crit": "Frequent"}[s2], alert=a2)

    # KPI 3: avg recovery time (OPEN/FAN_ON to next CLOSE/FAN_OFF, per sensor)
    pairs = []
    for sid, grp in w.sort_values("ts").groupby("sensor_id"):
        open_ts  = grp[grp["event_type"].isin(["OPEN", "FAN_ON"])]["ts"].values
        close_ts = grp[grp["event_type"].isin(["CLOSE", "FAN_OFF"])]["ts"].values
        for ot in open_ts:
            later = close_ts[close_ts > ot]
            if len(later):
                pairs.append((later[0] - ot) / np.timedelta64(1, "m"))
    rec_min = round(float(np.mean(pairs)), 1) if pairs else None
    if rec_min is None:      s3, a3, val3 = "safe", None, "n/a"
    elif rec_min < 15:       s3, a3, val3 = "safe", None, f"{rec_min:.1f}"
    elif rec_min <= 20:
        s3, a3, val3 = "warn", "Recovery above the 15 min target. Review damper step size and fan staging.", f"{rec_min:.1f}"
    else:
        s3, a3, val3 = "crit", "Recovery well above target. Consider variable-speed control on AHU01.", f"{rec_min:.1f}"
    out["recovery"] = dict(value=val3, unit=" min" if rec_min is not None else "",
                           sub="target: below 15 min", status=s3,
                           tag={"safe": "On target", "warn": "Slow", "crit": "Above target"}[s3], alert=a3)

    # KPI 4: cycling / hunting frequency (control transitions per hour, worst zone)
    rates = {}
    for sid, grp in w.sort_values("ts").groupby("sensor_id"):
        ev = grp[grp["event_type"].isin(["OPEN", "CLOSE", "FAN_ON", "FAN_OFF"])]
        rates[sid] = len(ev) / hours_span
    worst_sid  = max(rates, key=rates.get) if rates else None
    worst_rate = rates.get(worst_sid, 0.0)
    other = {k: v for k, v in rates.items() if k != worst_sid}
    other_txt = " · ".join(f"{k[-5:]}: {v:.0f}/hr" for k, v in other.items()) or ""
    if worst_rate <= 4:  s4, a4 = "safe", None
    elif worst_rate <= 8:
        s4, a4 = "warn", f"{worst_sid[-5:]} cycling {worst_rate:.0f} times/hr. Monitor for hunting."
    else:
        s4, a4 = "crit", f"{worst_sid[-5:]} hunting at {worst_rate:.0f} transitions/hr. Widen the deadband (490-520 ppm) by 5-10 ppm."
    out["cycling"] = dict(value=f"{worst_rate:.0f}", unit="/hr",
                          sub=f"{worst_sid[-5:] if worst_sid else 'n/a'} worst zone · {other_txt}",
                          status=s4, tag={"safe": "Stable", "warn": "Elevated", "crit": "Hunting"}[s4], alert=a4)

    # KPI 5: daily breach peak hour (pattern over the full log)
    bre = d[d["co2_level"] >= UPPER_PPM]
    if bre.empty:
        out["peak"] = dict(value="n/a", sub="no breaches logged", status="safe", tag="No pattern", alert=None)
    else:
        hrs   = bre["ts"].dt.hour
        peak  = int(hrs.mode().iloc[0])
        share = float((hrs == peak).mean())
        n_days = max(bre["ts"].dt.date.nunique(), 1)
        if share >= 0.30 and len(bre) >= 4:
            s5 = "warn"
            a5 = (f"Recurring peak near {peak:02d}:00. Pre-raise the ventilation baseline "
                  f"at {peak-1:02d}:45 instead of waiting for the reactive OPEN event.")
        else:
            s5, a5 = "safe", None
        out["peak"] = dict(value=f"{peak:02d}:00", sub=f"{share:.0%} of breaches · {n_days} day(s) observed",
                           status=s5, tag={"safe": "Diffuse", "warn": "Pattern"}[s5], alert=a5)
    return out


def kpi_card_html(label, k):
    unit  = k.get("unit", "")
    alert = ""
    if k.get("alert"):
        alert = f'<div class="kpi-alert {k["status"]}"><b>Action</b>{k["alert"]}</div>'
    return (f'<div class="kpi-card {k["status"]}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{k["value"]}<small>{unit}</small></div>'
            f'<div class="kpi-sub">{k["sub"]}</div>'
            f'<span class="kpi-tag {k["status"]}">{k["tag"]}</span>'
            f'{alert}</div>')


def render_kpi_row(kpis):
    cards = (
        kpi_card_html("IGBC Compliance",   kpis["compliance"]) +
        kpi_card_html("Breach Events",     kpis["breach"]) +
        kpi_card_html("Avg Recovery Time", kpis["recovery"]) +
        kpi_card_html("Cycling Frequency", kpis["cycling"]) +
        kpi_card_html("Daily Breach Peak", kpis["peak"])
    )
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


# ── Plotly styling ────────────────────────────────────────────────────────────
def style_chart(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=16, b=10),
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(family=FONT_SANS, size=11, color=C["text2"]),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, family=FONT_MONO),
                     linecolor=C["border"], zeroline=False)
    fig.update_yaxes(gridcolor=C["border"], gridwidth=0.5,
                     tickfont=dict(size=10, family=FONT_MONO), zeroline=False)
    return fig


def live_chart(df):
    fig = go.Figure()
    fig.add_hrect(y0=IGBC_PPM,  y1=650,       fillcolor=C["crit"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=UPPER_PPM, y1=IGBC_PPM,  fillcolor=C["crit"], opacity=0.04, line_width=0)
    fig.add_hrect(y0=LOWER_PPM, y1=UPPER_PPM, fillcolor=C["warn"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=350,       y1=LOWER_PPM, fillcolor=C["safe"], opacity=0.03, line_width=0)

    for level, col, name in [
        (IGBC_PPM,  C["crit"], "IGBC limit 530"),
        (UPPER_PPM, C["crit"], "Breach 520"),
        (LOWER_PPM, C["warn"], "Hold 490"),
    ]:
        fig.add_hline(y=level, line_dash="dot", line_color=col, line_width=1,
                      annotation_text=name, annotation_position="right",
                      annotation_font=dict(size=9, color=col, family=FONT_SANS))

    colours = {"F1-WA-AHU01": C["ahu01"], "F1-WA-AHU02": C["ahu02"]}
    names   = {"F1-WA-AHU01": "AHU01 · Damper zone", "F1-WA-AHU02": "AHU02 · Fan zone"}
    for sid in SENSORS:
        df_s = df[df["sensor_id"] == sid]
        if df_s.empty:
            continue
        ts     = to_naive(df_s["timestamp"])
        recent = df_s[ts >= ts.max() - pd.Timedelta(hours=CHART_HRS)]
        if recent.empty:
            recent = df_s.tail(60)
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["co2_level"], mode="lines",
            name=names[sid], line=dict(color=colours[sid], width=1.8),
        ))

    fig = style_chart(fig, height=360)
    fig.update_yaxes(title="CO2 (ppm)", range=[370, 560],
                     title_font=dict(size=11, family=FONT_SANS))
    fig.update_layout(margin=dict(l=10, r=80, t=16, b=10))
    return fig


def tod_breach_chart(df):
    d = df.copy()
    d["ts"] = to_naive(d["timestamp"])
    bre = d[d["co2_level"] >= UPPER_PPM]
    counts = bre["ts"].dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
    mx = counts.max() if counts.max() > 0 else 1
    peak_hour = int(counts.idxmax()) if mx > 0 else None

    def hour_label(h):
        if h == 0:   return "12am"
        if h < 12:   return f"{h}am"
        if h == 12:  return "12pm"
        return f"{h - 12}pm"

    # Only show 6am–9pm (hours 6–21) — the meaningful occupancy window
    display_hours = list(range(6, 22))
    x_labels = [hour_label(h) for h in display_hours]
    y_vals   = [int(counts[h]) for h in display_hours]
    disp_mx  = max(y_vals) if any(y_vals) else 1

    seg_colors = [
        C["crit"] if v / disp_mx > 0.70
        else C["warn"] if v / disp_mx > 0.30
        else C["safe"]
        for v in y_vals
    ]

    # Invisible traces for legend entries
    fig = go.Figure()
    for label, col in [("High (>70% of peak)", C["crit"]),
                        ("Moderate (30–70%)",   C["warn"]),
                        ("Low (<30%)",          C["safe"])]:
        fig.add_trace(go.Bar(
            x=[None], y=[None], name=label,
            marker_color=col, showlegend=True,
        ))

    fig.add_trace(go.Bar(
        x=x_labels, y=y_vals,
        marker_color=seg_colors, marker_line_width=0,
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>%{y} breach readings<extra></extra>",
    ))

    # Annotate the peak bar
    if peak_hour is not None and peak_hour in display_hours and disp_mx > 0:
        fig.add_annotation(
            x=hour_label(peak_hour), y=counts[peak_hour],
            text=f"Peak: {hour_label(peak_hour)}",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor=C["crit"], ax=0, ay=-28,
            font=dict(size=9, color=C["crit"], family=FONT_SANS),
            bgcolor=C["crit_bg"], bordercolor=C["crit_bd"],
            borderwidth=1, borderpad=3,
        )

    fig = style_chart(fig, height=300)
    fig.update_layout(
        bargap=0.2,
        legend=dict(
            orientation="h", y=-0.22, x=0,
            font=dict(size=10), traceorder="normal",
        ),
    )
    fig.update_xaxes(
        type="category",
        title="Hour of Day (6am to 9pm shown)",
        title_font=dict(size=10, family=FONT_SANS),
        tickangle=0,
        tickfont=dict(size=10, family=FONT_MONO),
    )
    fig.update_yaxes(
        title="Breach readings",
        title_font=dict(size=10),
    )
    return fig


def cycling_chart(df):
    d = df.copy()
    d["ts"]   = to_naive(d["timestamp"])
    d["date"] = d["ts"].dt.date
    fig = go.Figure()
    for sid, col in [("F1-WA-AHU01", C["ahu01"]), ("F1-WA-AHU02", C["ahu02"])]:
        grp = d[(d["sensor_id"] == sid) &
                (d["event_type"].isin(["OPEN", "CLOSE", "FAN_ON", "FAN_OFF"]))]
        per_day = grp.groupby("date").size()
        hours   = d.groupby("date")["ts"].agg(lambda s: max((s.max()-s.min()).total_seconds()/3600, 1))
        rate    = (per_day / hours).dropna()
        fig.add_trace(go.Bar(
            x=[f"{x:%d %b}" for x in rate.index], y=rate.round(1).values,
            name=sid[-5:], marker_color=col, marker_line_width=0,
        ))
    fig = style_chart(fig, height=260)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title="transitions / hr", title_font=dict(size=10))
    return fig


def efficiency_chart(df):
    d = df.copy()
    d["ts"]   = to_naive(d["timestamp"])
    d["date"] = d["ts"].dt.date
    d = d[(d.get("occupancy", pd.Series(dtype=float)).fillna(0) >= 5)]
    fig = go.Figure()
    for sid, col in [("F1-WA-AHU01", C["ahu01"]), ("F1-WA-AHU02", C["ahu02"])]:
        grp = d[d["sensor_id"] == sid]
        if grp.empty:
            continue
        daily = (grp["differential"] / grp["occupancy"]).groupby(grp["date"]).mean()
        fig.add_trace(go.Scatter(
            x=[f"{x:%d %b}" for x in daily.index], y=daily.round(2).values,
            mode="lines+markers", name=sid[-5:],
            line=dict(color=col, width=1.8), marker=dict(size=5),
        ))
    fig = style_chart(fig, height=260)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title="ppm differential / occupant", title_font=dict(size=10))
    return fig


def card_head(title, caption):
    st.markdown(
        f'<div class="card-title">{title}</div>'
        f'<div class="card-caption">{caption}</div>',
        unsafe_allow_html=True,
    )


def rec_strip_html(text, crit=False):
    cls = "rec-strip crit" if crit else "rec-strip"
    return f'<div class="{cls}"><b>Recommendation</b>{text}</div>'


# ── Panels ────────────────────────────────────────────────────────────────────
def render_overview(df, kpis):
    st.markdown('<div class="sec-label">System KPIs · Last 24 Hours</div>', unsafe_allow_html=True)
    if kpis:
        render_kpi_row(kpis)
    else:
        st.info("KPIs will populate as the simulation runs.")

    with st.container(border=True):
        card_head("Live CO2 Trend, Both Zones (Last 6 Simulated Hours)",
                  f"Refreshes every {TICK_SEC} seconds · 1 simulated minute = {TICK_SEC} real seconds")
        if not df.empty:
            st.plotly_chart(live_chart(df), use_container_width=True, key="live_chart",
                            config={"displayModeBar": False})

    st.markdown('<div class="sec-label">Current Sensor Status</div>', unsafe_allow_html=True)
    cards = []
    for sid in SENSORS:
        df_s   = df[df["sensor_id"] == sid]
        latest = df_s.iloc[-1] if not df_s.empty else None
        co2    = float(latest["co2_level"])     if latest is not None else 400.0
        diff   = float(latest["differential"])  if latest is not None else 0.0
        damper = int(latest["damper_position"]) if latest is not None else 20
        fan_on = bool(latest["fan_on"])         if latest is not None else False
        zone   = ppm_zone(co2)
        tag    = {"safe": "Normal", "warn": "Hold", "crit": "Breach"}[zone]
        is_fan = "AHU02" in sid

        if is_fan:
            state_line = (
                f'<div class="fan-line {"fan-on" if fan_on else "fan-off"}">'
                f'<span class="fan-dot"></span>'
                f'{"Fan ON (CO2 above breach threshold)" if fan_on else "Fan OFF (CO2 below hold threshold)"}'
                f'</div>'
            )
            type_txt = "Inline fresh-air fan · ON / OFF control"
        else:
            state_line = (
                f'<div class="damper-label"><span>Damper position</span>'
                f'<span class="damper-val">{damper}%</span></div>'
                f'<div class="progress-track"><div class="progress-fill" style="width:{damper}%"></div></div>'
            )
            type_txt = "Motorised damper · 20% steps · 0 to 100%"

        cards.append(
            f'<div class="sensor-card">'
            f'<div class="sensor-head"><div><div class="sensor-id">{sid}</div>'
            f'<div class="sensor-type">{type_txt}</div></div>'
            f'<span class="kpi-tag {zone}">{tag}</span></div>'
            f'<div class="sensor-ppm ppm-{zone}">{co2:.0f} <span>ppm · {diff:.0f} above outdoor</span></div>'
            f'{state_line}</div>'
        )
    st.markdown(f'<div class="sensor-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_analytics(df, kpis):
    if df.empty:
        st.info("Analytics will populate as the simulation runs.")
        return

    peak = kpis["peak"] if kpis else None
    with st.container(border=True):
        card_head("KPI 5: Time-of-Day Breach Pattern",
                  "Breach count aggregated by hour · full simulated history · both zones combined")
        st.plotly_chart(tod_breach_chart(df), use_container_width=True, key="tod_chart",
                        config={"displayModeBar": False})
        if peak and peak.get("alert"):
            st.markdown(rec_strip_html(peak["alert"]), unsafe_allow_html=True)
        elif peak and peak["value"] != "n/a":
            st.markdown(rec_strip_html(
                f'Breach activity is spread across hours (peak {peak["value"]}). '
                'No pre-emptive schedule change needed.'), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            card_head("KPI 4: Cycling / Hunting Frequency",
                      "OPEN/CLOSE and FAN transitions per hour · per day")
            st.plotly_chart(cycling_chart(df), use_container_width=True, key="cyc_chart",
                            config={"displayModeBar": False})
            cyc = kpis["cycling"] if kpis else None
            rec = rec_strip_html(cyc["alert"], crit=(cyc["status"] == "crit")) if cyc and cyc.get("alert") \
                  else rec_strip_html("Cycling within normal range. Deadband width is adequate.")
            st.markdown(rec, unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            card_head("Supplementary: Occupancy-to-CO2 Efficiency",
                      "Average ppm differential per occupant · daily · occupied hours only")
            st.plotly_chart(efficiency_chart(df), use_container_width=True, key="eff_chart",
                            config={"displayModeBar": False})
            st.markdown(rec_strip_html(
                "A rising ratio means CO2 climbs faster than occupancy explains: "
                "check filters, duct obstructions, or sensor drift."), unsafe_allow_html=True)


def render_log(df):
    with st.container(border=True):
        card_head("Activity Log",
                  "Every DDC control decision (last 50 events, newest first). "
                  "OPEN/CLOSE = damper step · FAN_ON/OFF = fan toggle · HOLD = no change")
        if df.empty:
            st.info("Log will populate as the simulation runs.")
            return
        show = ["timestamp", "sensor_id", "co2_level", "differential",
                "event_type", "damper_position", "fan_on", "occupancy"]
        show = [c for c in show if c in df.columns]
        disp = (df[show].tail(200)
                .sort_values("timestamp", ascending=False)
                .head(50).copy().reset_index(drop=True))
        disp["timestamp"] = disp["timestamp"].astype(str).str[:19]
        disp["co2_level"] = disp["co2_level"].round(1)

        st.dataframe(
            disp, use_container_width=True, height=430,
            column_config={
                "timestamp":       st.column_config.TextColumn("Time (UTC)"),
                "sensor_id":       st.column_config.TextColumn("Sensor"),
                "co2_level":       st.column_config.NumberColumn("CO2 (ppm)", format="%.1f"),
                "differential":    st.column_config.NumberColumn("Differential", format="%.1f"),
                "event_type":      st.column_config.TextColumn("Decision"),
                "damper_position": st.column_config.NumberColumn("Damper %", format="%d%%"),
                "fan_on":          st.column_config.CheckboxColumn("Fan on"),
                "occupancy":       st.column_config.NumberColumn("Occupancy"),
            },
        )


# ── Main ──────────────────────────────────────────────────────────────────────
@st.fragment(run_every=TICK_SEC)
def console():
    """Whole console re-renders every TICK_SEC seconds without a full-page rerun."""
    df = st.session_state.get("df_log", pd.DataFrame())

    # Console header
    now_str = datetime.utcnow().strftime("%H:%M:%S")
    st.markdown(f"""
<div class="bms-header">
  <div class="bms-left">
    <div class="bms-logo">CO₂</div>
    <div>
      <div class="bms-title">Smart CO₂ Monitoring &amp; Control · BMS Console</div>
      <div class="bms-sub">SBI GITC Building · Floor 1, Wing A · CBD Belapur, Navi Mumbai</div>
    </div>
  </div>
  <div class="bms-right">
    <div class="bms-live"><div class="bms-live-dot"></div>Live</div>
    <div class="bms-clock">{now_str} UTC · {len(df):,} events</div>
  </div>
</div>""", unsafe_allow_html=True)

    nav_col, pause_col = st.columns([5, 1])
    with nav_col:
        panel = st.radio("nav", ["Overview", "Analytics", "Activity Log"],
                         horizontal=True, label_visibility="collapsed", key="nav")
    with pause_col:
        paused = st.toggle("Pause", key="paused")

    if not paused:
        advance_tick()
        df = st.session_state.get("df_log", pd.DataFrame())

    kpis = calc_kpis(df)

    if panel == "Overview":
        render_overview(df, kpis)
    elif panel == "Analytics":
        render_analytics(df, kpis)
    else:
        render_log(df)


def main():
    inject_css()
    init_session()
    console()


if __name__ == "__main__":
    main()
