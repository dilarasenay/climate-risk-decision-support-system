import base64
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from db import load_data, insert_row

st.set_page_config(
    page_title="İklim Risk Gösterge Paneli",
    page_icon="🌍",
    layout="wide",
)

ASSETS_DIR = BASE_DIR / "assets"


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def safe_col(df: pd.DataFrame, candidates: list[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Eksik kolon. Beklenen kolonlardan biri yok: {candidates}")
    return None


def fmt_num(value, digits=2):
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def risk_etiketi(score: float) -> str:
    if score < 25:
        return "Düşük"
    if score < 50:
        return "Orta"
    if score < 75:
        return "Yüksek"
    return "Kritik"


def risk_rengi(score: float) -> str:
    if score < 25:
        return "linear-gradient(135deg,#16a34a,#22c55e)"
    if score < 50:
        return "linear-gradient(135deg,#facc15,#f97316)"
    if score < 75:
        return "linear-gradient(135deg,#f97316,#ef4444)"
    return "linear-gradient(135deg,#ef4444,#7f1d1d)"


def section_header(title: str, subtitle: str = ""):
    subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-mark"></div>
            <div>
                <div class="section-title">{title}</div>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_title(title: str, subtitle: str = ""):
    subtitle_html = f'<div class="chart-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="chart-head">
            <div class="chart-title">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_kpi(icon: str, title: str, value: str, unit: str, subtitle: str, accent: str, bg_base64: str = ""):
    bg_style = f"background-image:url('data:image/png;base64,{bg_base64}');" if bg_base64 else ""
    st.markdown(
        f"""
        <div class="kpi-card" style="{bg_style}">
            <div class="kpi-dark"></div>
            <div class="kpi-rainbow"></div>
            <div class="kpi-orb" style="background:{accent};"></div>
            <div class="kpi-icon" style="background:{accent};">{icon}</div>
            <div class="kpi-text">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}<span>{unit}</span></div>
                <div class="kpi-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_panel(title: str, big: str, lines: list[str], badge_text: str, badge_color: str):
    lines_html = "".join([f'<div class="info-line">{line}</div>' for line in lines])
    st.markdown(
        f"""
        <div class="info-panel">
            <div class="info-eyebrow">{title}</div>
            <div class="info-big">{big}</div>
            {lines_html}
            <div class="info-badge" style="background:{badge_color};">{badge_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_item(icon: str, title: str, text: str, accent: str):
    st.markdown(
        f"""
        <div class="insight-item">
            <div class="insight-icon" style="background:{accent};">{icon}</div>
            <div>
                <div class="insight-title">{title}</div>
                <div class="insight-text">{text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_style(fig, height=320, showlegend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0f1d3a", family="Inter, Segoe UI, sans-serif"),
        showlegend=showlegend,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)",
            bordercolor="rgba(15,23,42,0.10)",
            font=dict(color="#0f1d3a", size=13),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.055)",
        zeroline=False,
        linecolor="rgba(15,23,42,0.06)",
        tickfont=dict(color="#64748b", size=11),
        title_font=dict(color="#64748b", size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.055)",
        zeroline=False,
        linecolor="rgba(15,23,42,0.06)",
        tickfont=dict(color="#64748b", size=11),
        title_font=dict(color="#64748b", size=12),
    )
    return fig


def minmax_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


# =========================================================
# GÖRSELLER — SENİN RESİM YOLLARIN KORUNDU
# =========================================================
BG_IMG = img_to_base64(ASSETS_DIR / "background" / "background.png")
HERO_IMG = img_to_base64(BASE_DIR / "dashboard" / "earth_placeholder.png")

KPI_TEMP = img_to_base64(ASSETS_DIR / "kpi" / "temp.png") or BG_IMG
KPI_CO2 = img_to_base64(ASSETS_DIR / "kpi" / "co2.png") or BG_IMG
KPI_SEA = img_to_base64(ASSETS_DIR / "kpi" / "sea.png") or BG_IMG
KPI_WEATHER = img_to_base64(ASSETS_DIR / "kpi" / "weather.png") or BG_IMG
KPI_RISK = img_to_base64(ASSETS_DIR / "kpi" / "risk.png") or BG_IMG
KPI_BIO = img_to_base64(ASSETS_DIR / "kpi" / "bio.png") or BG_IMG


# =========================================================
# VERİ
# =========================================================
try:
    df = load_data().copy()
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

if df.empty:
    st.error("Veri bulunamadı.")
    st.stop()

try:
    COL_COUNTRY = safe_col(df, ["country", "Country"])
    COL_YEAR = safe_col(df, ["year", "Year"])
    COL_TEMP = safe_col(df, ["avg_temperature_°c", "avg_temperature_c", "avg_temperature", "temperature"])
    COL_CO2 = safe_col(df, ["co2_emissions_tons_capita", "co2_emissions", "co2"])
    COL_SEA = safe_col(df, ["sea_level_rise_mm", "sea_level_rise", "sea_level"])
    COL_EVENTS = safe_col(df, ["extreme_weather_events", "weather_events", "extreme_events"])
    COL_RISK = safe_col(df, ["climate_risk_score", "risk_score"])
except Exception as e:
    st.error(str(e))
    st.stop()

for col in [COL_YEAR, COL_TEMP, COL_CO2, COL_SEA, COL_EVENTS, COL_RISK]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[COL_COUNTRY, COL_YEAR, COL_CO2, COL_RISK, COL_TEMP, COL_SEA, COL_EVENTS]).copy()

if df.empty:
    st.error("Temizleme sonrası veri kalmadı.")
    st.stop()


# =========================================================
# NORMALİZASYON VE SKOR
# =========================================================
df["temp_norm"] = minmax_normalize(df[COL_TEMP])
df["co2_norm"] = minmax_normalize(df[COL_CO2])
df["sea_norm"] = minmax_normalize(df[COL_SEA])
df["events_norm"] = minmax_normalize(df[COL_EVENTS])

df["climate_pressure_score"] = (
    df["temp_norm"] * 0.25 +
    df["co2_norm"] * 0.35 +
    df["sea_norm"] * 0.20 +
    df["events_norm"] * 0.20
) * 100

df["pressure_alert"] = df["climate_pressure_score"].apply(
    lambda x: "Kritik İklim Baskısı" if x > 70 else
              "Yüksek Baskı" if x > 50 else
              "Normal"
)


# =========================================================
# CSS — ŞIKIR ŞIKIR PREMIUM FINAL
# =========================================================
st.markdown(
    f"""
<style>
:root {{
    --navy: #0b1635;
    --ink: #0f1d3a;
    --muted: #64748b;
    --soft-line: rgba(148, 163, 184, 0.25);
    --blue: #2563eb;
    --sky: #00d4ff;
    --cyan: #06b6d4;
    --green: #22c55e;
    --lime: #84cc16;
    --yellow: #facc15;
    --orange: #fb923c;
    --red: #ef4444;
    --pink: #ec4899;
    --purple: #8b5cf6;
}}

html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at 8% 10%, rgba(0,212,255,0.25), transparent 26%),
        radial-gradient(circle at 88% 9%, rgba(34,197,94,0.20), transparent 30%),
        radial-gradient(circle at 80% 78%, rgba(255,122,0,0.15), transparent 28%),
        radial-gradient(circle at 34% 86%, rgba(139,92,246,0.18), transparent 34%),
        linear-gradient(135deg, #e8f8ff 0%, #f7fff8 42%, #fff7ed 100%);
    color: var(--ink);
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(rgba(255,255,255,0.25), rgba(255,255,255,0.05)),
        radial-gradient(circle at 20% 20%, rgba(255,255,255,0.48), transparent 22%);
    z-index: 0;
}}

.stApp::after {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: .16;
    background-image:
        linear-gradient(rgba(15,23,42,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(15,23,42,0.035) 1px, transparent 1px);
    background-size: 46px 46px;
    z-index: 0;
}}

.block-container {{
    max-width: 1880px !important;
    padding: 1.35rem 1.35rem 2rem 1.35rem !important;
    position: relative;
    z-index: 1;
}}

header[data-testid="stHeader"], footer, #MainMenu {{
    visibility: hidden;
}}

section[data-testid="stSidebar"] {{
    display: none !important;
}}

.left-rail {{
    position: sticky;
    top: 1rem;
}}

.control-panel {{
    background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(231,247,255,0.64));
    backdrop-filter: blur(26px);
    -webkit-backdrop-filter: blur(26px);
    border: 1px solid rgba(255,255,255,0.85);
    border-radius: 30px;
    padding: 1.18rem 1.05rem;
    box-shadow:
        0 24px 70px rgba(37,99,235,0.12),
        0 12px 32px rgba(20,184,166,0.08),
        inset 0 1px 0 rgba(255,255,255,0.95);
}}

.brand-card {{
    display: flex;
    align-items: center;
    gap: .72rem;
    margin-bottom: 1.2rem;
}}

.brand-icon {{
    width: 48px;
    height: 48px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.55rem;
    background: linear-gradient(135deg, #00d4ff, #2563eb, #14b8a6);
    box-shadow: 0 16px 32px rgba(37,99,235,0.26);
}}

.brand-title {{
    font-size: 1.22rem;
    line-height: 1.08;
    font-weight: 950;
    color: #0b1635;
    letter-spacing: -.5px;
}}

.brand-sub {{
    font-size: .82rem;
    font-weight: 750;
    color: #64748b;
    margin-top: .14rem;
}}

.control-topline {{
    width: 138px;
    height: 7px;
    border-radius: 999px;
    background: linear-gradient(90deg, #00d4ff, #2563eb, #22c55e, #facc15, #fb7185, #8b5cf6);
    box-shadow: 0 12px 26px rgba(14,165,233,0.24);
    margin-bottom: .95rem;
}}

.control-title {{
    font-size: 1.12rem;
    font-weight: 950;
    color: #0f1d3a;
    margin-bottom: .28rem;
}}

.control-text {{
    font-size: .88rem;
    line-height: 1.58;
    color: #64748b;
    font-weight: 650;
    margin-bottom: 1rem;
}}

.control-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.18), transparent);
    margin: .95rem 0 1rem 0;
}}

.hero {{
    position: relative;
    overflow: hidden;
    min-height: 312px;
    border-radius: 36px;
    background:
        linear-gradient(105deg, rgba(255,255,255,0.86) 0%, rgba(255,255,255,0.48) 44%, rgba(255,255,255,0.06) 100%),
        url("data:image/png;base64,{HERO_IMG}");
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(255,255,255,0.82);
    box-shadow:
        0 34px 90px rgba(37,99,235,0.18),
        0 14px 44px rgba(20,184,166,0.10),
        inset 0 1px 0 rgba(255,255,255,0.95);
    padding: 2rem 2.35rem;
    margin-bottom: 1.02rem;
}}

.hero::after {{
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 76% 32%, rgba(255,255,255,0.32), transparent 26%),
        radial-gradient(circle at 68% 12%, rgba(0,212,255,0.16), transparent 22%),
        linear-gradient(180deg, transparent 38%, rgba(255,255,255,0.17));
    pointer-events: none;
}}

.hero-badge {{
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: .48rem;
    padding: .62rem 1rem;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(224,247,255,0.78));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.92);
    color: #0f1d3a;
    font-size: .82rem;
    font-weight: 950;
    margin-bottom: .92rem;
    box-shadow: 0 14px 30px rgba(37,99,235,0.10);
}}

.hero-title {{
    position: relative;
    z-index: 1;
    font-size: 3.25rem;
    line-height: .94;
    font-weight: 950;
    letter-spacing: -2px;
    color: #0b1635;
    margin-bottom: .66rem;
    max-width: 780px;
    text-shadow: 0 8px 28px rgba(255,255,255,0.38);
}}

.hero-text {{
    position: relative;
    z-index: 1;
    font-size: 1.02rem;
    line-height: 1.75;
    color: #3f536d;
    font-weight: 720;
    max-width: 760px;
}}

.section-header {{
    display: flex;
    align-items: flex-start;
    gap: .75rem;
    margin: .25rem 0 .82rem 0;
}}

.section-mark {{
    width: 8px;
    min-width: 8px;
    height: 44px;
    border-radius: 999px;
    background: linear-gradient(180deg, #00d4ff, #2563eb, #22c55e, #facc15, #fb7185);
    box-shadow: 0 12px 28px rgba(0,212,255,0.24);
}}

.section-title {{
    font-size: 1.22rem;
    line-height: 1.12;
    font-weight: 950;
    color: #0f1d3a;
    letter-spacing: -.35px;
}}

.section-subtitle {{
    margin-top: .15rem;
    font-size: .88rem;
    line-height: 1.5;
    color: #64748b;
    font-weight: 680;
}}

.kpi-card {{
    position: relative;
    min-height: 128px;
    border-radius: 24px;
    overflow: hidden;
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(255,255,255,0.82);
    box-shadow:
        0 18px 44px rgba(37,99,235,0.10),
        0 8px 26px rgba(20,184,166,0.08),
        inset 0 1px 0 rgba(255,255,255,0.95);
    transition: transform .22s ease, box-shadow .22s ease;
}}

.kpi-card:hover {{
    transform: translateY(-5px) scale(1.012);
    box-shadow:
        0 28px 70px rgba(37,99,235,0.16),
        0 14px 38px rgba(20,184,166,0.11),
        inset 0 1px 0 rgba(255,255,255,0.98);
}}

.kpi-card::after {{
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.42) 42%, transparent 62%);
    transform: translateX(-120%);
    transition: transform .60s ease;
}}

.kpi-card:hover::after {{
    transform: translateX(120%);
}}

.kpi-glass {{
    position: absolute;
    inset: 0;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.84), rgba(255,255,255,0.54)),
        radial-gradient(circle at 18% 18%, rgba(255,255,255,0.70), transparent 28%);
    backdrop-filter: blur(7px);
}}

.kpi-orb {{
    position: absolute;
    right: -34px;
    top: -34px;
    width: 128px;
    height: 128px;
    opacity: .24;
    filter: blur(12px);
    border-radius: 999px;
}}

.kpi-icon {{
    position: absolute;
    left: 1rem;
    top: 1rem;
    z-index: 2;
    width: 46px;
    height: 46px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.35rem;
    box-shadow: 0 14px 26px rgba(15,23,42,0.12), inset 0 1px 0 rgba(255,255,255,0.3);
}}

.kpi-text {{
    position: relative;
    z-index: 2;
    padding: 1rem 1rem 1rem 4.6rem;
}}

.kpi-title {{
    font-size: .82rem;
    font-weight: 950;
    line-height: 1.25;
    color: #18243f;
    margin-bottom: .42rem;
}}

.kpi-value {{
    font-size: 1.78rem;
    font-weight: 950;
    line-height: 1;
    letter-spacing: -1px;
    color: #0b1635;
}}

.kpi-value span {{
    font-size: .85rem;
    margin-left: .24rem;
    color: #334155;
    font-weight: 900;
}}

.kpi-subtitle {{
    font-size: .78rem;
    line-height: 1.38;
    color: #64748b;
    font-weight: 750;
    margin-top: .38rem;
}}

.info-panel {{
    background: linear-gradient(145deg, rgba(255,255,255,0.82), rgba(232,247,255,0.58));
    backdrop-filter: blur(22px);
    border: 1px solid rgba(255,255,255,0.85);
    border-radius: 26px;
    padding: 1rem;
    box-shadow:
        0 22px 62px rgba(37,99,235,0.10),
        inset 0 1px 0 rgba(255,255,255,0.95);
    min-height: 150px;
}}

.info-eyebrow {{
    font-size: .75rem;
    font-weight: 950;
    letter-spacing: .52px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: .42rem;
}}

.info-big {{
    font-size: 1.82rem;
    line-height: 1.05;
    font-weight: 950;
    color: #0f1d3a;
    letter-spacing: -.6px;
    margin-bottom: .38rem;
}}

.info-line {{
    font-size: .88rem;
    line-height: 1.56;
    color: #4f6178;
    font-weight: 700;
}}

.info-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: .42rem .86rem;
    border-radius: 999px;
    color: white;
    font-size: .76rem;
    font-weight: 950;
    margin-top: .74rem;
    box-shadow: 0 12px 24px rgba(15,23,42,0.12);
}}

.insight-item {{
    display: flex;
    gap: .75rem;
    align-items: flex-start;
    background: rgba(255,255,255,0.68);
    border: 1px solid rgba(255,255,255,0.76);
    border-radius: 20px;
    padding: .82rem;
    margin-bottom: .58rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
}}

.insight-icon {{
    width: 38px;
    height: 38px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.12rem;
    box-shadow: 0 12px 22px rgba(15,23,42,0.12);
}}

.insight-title {{
    font-size: .82rem;
    font-weight: 950;
    color: #0f1d3a;
    margin-bottom: .15rem;
}}

.insight-text {{
    font-size: .78rem;
    line-height: 1.42;
    color: #64748b;
    font-weight: 700;
}}

.chart-head {{
    margin: .12rem 0 .38rem 0;
}}

.chart-title {{
    font-size: .98rem;
    font-weight: 950;
    color: #0f1d3a;
    line-height: 1.16;
    letter-spacing: -.2px;
}}

.chart-subtitle {{
    font-size: .80rem;
    line-height: 1.45;
    color: #64748b;
    font-weight: 680;
    margin-top: .14rem;
}}

[data-testid="stPlotlyChart"] {{
    background: linear-gradient(145deg, rgba(255,255,255,0.66), rgba(228,248,255,0.42)) !important;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.86) !important;
    border-radius: 28px !important;
    box-shadow:
        0 24px 70px rgba(37,99,235,0.10),
        0 12px 34px rgba(20,184,166,0.08),
        inset 0 1px 0 rgba(255,255,255,0.92) !important;
    padding: .62rem !important;
    transition: transform .2s ease, box-shadow .2s ease;
    margin-bottom: .3rem !important;
}}

[data-testid="stPlotlyChart"] > div {{
    background: transparent !important;
}}

[data-testid="stPlotlyChart"]:hover {{
    transform: translateY(-4px);
    box-shadow:
        0 34px 90px rgba(37,99,235,0.15),
        0 18px 48px rgba(20,184,166,0.12),
        inset 0 1px 0 rgba(255,255,255,0.96) !important;
}}

.risk-gauge-shell {{
    background: linear-gradient(145deg, rgba(255,255,255,0.80), rgba(237,247,255,0.58));
    border: 1px solid rgba(255,255,255,0.90);
    border-radius: 30px;
    padding: .85rem;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.45),
        0 24px 70px rgba(37,99,235,0.12),
        0 18px 48px rgba(236,72,153,0.12),
        inset 0 1px 0 rgba(255,255,255,0.95);
}}

.alert-floating {{
    position: fixed;
    top: 14px;
    right: 14px;
    z-index: 9999;
    width: 292px;
    border-radius: 24px;
    padding: .82rem .9rem;
    background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(255,242,232,0.78));
    backdrop-filter: blur(22px);
    border: 1px solid rgba(255,255,255,0.80);
    box-shadow:
        0 26px 70px rgba(249,115,22,0.18),
        inset 0 1px 0 rgba(255,255,255,0.92);
}}

.alert-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .8rem;
    margin-bottom: .35rem;
}}

.alert-title {{
    font-size: .90rem;
    font-weight: 950;
    color: #b42318;
}}

.alert-pill {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: .26rem .54rem;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(239,68,68,0.16), rgba(249,115,22,0.12));
    color: #b42318;
    font-size: .68rem;
    font-weight: 950;
}}

.alert-text {{
    font-size: .78rem;
    line-height: 1.52;
    color: #8b2b2b;
    font-weight: 700;
}}

[data-testid="stDataFrame"] {{
    background: rgba(255,255,255,0.72) !important;
    backdrop-filter: blur(18px);
    border-radius: 24px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.78) !important;
    box-shadow:
        0 22px 58px rgba(37,99,235,0.08),
        inset 0 1px 0 rgba(255,255,255,0.85) !important;
}}

.stSelectbox label,
.stSlider label,
.stTextInput label,
.stNumberInput label,
.stCheckbox label {{
    color: #0f1d3a !important;
    font-weight: 850 !important;
    font-size: .84rem !important;
}}

.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {{
    background: rgba(255,255,255,0.68) !important;
    border: 1px solid rgba(255,255,255,0.82) !important;
    border-radius: 15px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.80);
}}

.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background-color: #2563eb !important;
    border-color: white !important;
    box-shadow: 0 0 0 5px rgba(37,99,235,0.18) !important;
}}

.stDownloadButton button,
.stFormSubmitButton button,
.stButton button {{
    background: linear-gradient(135deg, #2563eb, #00d4ff, #14b8a6) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 900 !important;
    box-shadow:
        0 14px 32px rgba(37,99,235,0.26),
        inset 0 1px 0 rgba(255,255,255,0.28) !important;
    transition: transform .18s ease, box-shadow .18s ease;
}}

.stDownloadButton button:hover,
.stFormSubmitButton button:hover,
.stButton button:hover {{
    transform: translateY(-2px);
    box-shadow:
        0 20px 44px rgba(37,99,235,0.34),
        inset 0 1px 0 rgba(255,255,255,0.32) !important;
}}

.element-container {{
    margin-bottom: .28rem !important;
}}

.stMarkdown h3 {{
    color: #0f1d3a !important;
    font-weight: 950 !important;
    letter-spacing: -.4px !important;
}}

/* =========================================================
   FINAL OVERRIDE — RENGARENK + GRAFİK/BACKGROUND BÜTÜN + KPI RESİMLERİ KORUNDU
   ========================================================= */
.stApp {
    background:
        radial-gradient(circle at 7% 8%, rgba(0,212,255,0.34), transparent 28%),
        radial-gradient(circle at 88% 10%, rgba(34,197,94,0.28), transparent 30%),
        radial-gradient(circle at 78% 72%, rgba(255,122,0,0.22), transparent 30%),
        radial-gradient(circle at 28% 88%, rgba(168,85,247,0.24), transparent 36%),
        radial-gradient(circle at 48% 44%, rgba(236,72,153,0.10), transparent 32%),
        linear-gradient(135deg, #ddf7ff 0%, #f0fff4 38%, #fff1df 72%, #f7edff 100%) !important;
}

.kpi-glass { display: none !important; }

.kpi-dark {
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(5,10,22,0.74), rgba(5,10,22,0.30)),
        linear-gradient(180deg, rgba(255,255,255,0.05), rgba(0,0,0,0.32));
}

.kpi-rainbow {
    position: absolute;
    inset: 0;
    opacity: .28;
    background:
        radial-gradient(circle at 16% 12%, rgba(0,212,255,0.55), transparent 28%),
        radial-gradient(circle at 82% 20%, rgba(34,197,94,0.48), transparent 30%),
        radial-gradient(circle at 76% 88%, rgba(249,115,22,0.46), transparent 32%),
        radial-gradient(circle at 20% 88%, rgba(168,85,247,0.44), transparent 32%);
}

.kpi-title { color: rgba(255,255,255,0.96) !important; }
.kpi-value { color: #fff !important; text-shadow: 0 10px 22px rgba(0,0,0,0.24) !important; }
.kpi-value span { color: rgba(255,255,255,0.88) !important; }
.kpi-subtitle { color: rgba(255,255,255,0.86) !important; }

[data-testid="stPlotlyChart"] {
    background:
        radial-gradient(circle at 10% 12%, rgba(0,212,255,0.20), transparent 34%),
        radial-gradient(circle at 88% 18%, rgba(34,197,94,0.18), transparent 34%),
        radial-gradient(circle at 50% 100%, rgba(249,115,22,0.14), transparent 38%),
        linear-gradient(145deg, rgba(255,255,255,0.40), rgba(255,255,255,0.14)) !important;
    backdrop-filter: blur(30px) saturate(1.38) !important;
    -webkit-backdrop-filter: blur(30px) saturate(1.38) !important;
    border: 1px solid rgba(255,255,255,0.48) !important;
    border-radius: 30px !important;
    box-shadow:
        0 22px 64px rgba(0,212,255,0.12),
        0 18px 50px rgba(34,197,94,0.09),
        0 12px 34px rgba(249,115,22,0.07),
        inset 0 1px 0 rgba(255,255,255,0.58) !important;
    padding: .55rem !important;
    margin-bottom: .18rem !important;
}

[data-testid="stPlotlyChart"] svg,
[data-testid="stPlotlyChart"] > div {
    background: transparent !important;
}

.control-panel,
.info-panel,
.risk-gauge-shell,
[data-testid="stDataFrame"] {
    background:
        radial-gradient(circle at 12% 10%, rgba(0,212,255,0.14), transparent 32%),
        radial-gradient(circle at 86% 16%, rgba(34,197,94,0.13), transparent 32%),
        linear-gradient(145deg, rgba(255,255,255,0.50), rgba(255,255,255,0.22)) !important;
    backdrop-filter: blur(28px) saturate(1.25) !important;
    -webkit-backdrop-filter: blur(28px) saturate(1.25) !important;
    border: 1px solid rgba(255,255,255,0.52) !important;
}

.kpi-card {
    box-shadow:
        0 18px 46px rgba(0,212,255,0.14),
        0 12px 32px rgba(34,197,94,0.10),
        0 8px 24px rgba(249,115,22,0.08),
        inset 0 1px 0 rgba(255,255,255,0.65) !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# ANA LAYOUT
# =========================================================
left_col, right_col = st.columns([0.86, 4.25], gap="large")

with left_col:
    st.markdown('<div class="left-rail"><div class="control-panel">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="brand-card">
            <div class="brand-icon">🌍</div>
            <div>
                <div class="brand-title">İKLİM RİSK</div>
                <div class="brand-sub">Gösterge Paneli</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="control-topline"></div>', unsafe_allow_html=True)
    st.markdown('<div class="control-title">🎛 Filtreler</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="control-text">Risk seviyesi, yıl aralığı ve çevresel eşikleri bu panelden yönet.</div>',
        unsafe_allow_html=True,
    )

    country_list = ["Tümü"] + sorted(df[COL_COUNTRY].dropna().unique().tolist())
    selected_country = st.selectbox("Ülke Seç", country_list, key="country_select")

    min_year = int(df[COL_YEAR].min())
    max_year = int(df[COL_YEAR].max())
    selected_year = st.slider("Yıl Aralığı", min_year, max_year, (min_year, max_year), key="year_slider")

    threshold = st.slider(
        "CO₂ Eşik Değeri",
        min_value=float(df[COL_CO2].min()),
        max_value=float(df[COL_CO2].max()),
        value=float(round(df[COL_CO2].quantile(0.75), 2)),
        step=0.01,
        key="threshold_slider",
    )

    st.markdown('<div class="control-divider"></div>', unsafe_allow_html=True)

    risk_level_options = ["Tümü", "Düşük", "Orta", "Yüksek", "Kritik"]
    selected_risk_level = st.selectbox("Risk Seviyesi", risk_level_options, key="risk_level_select")

    only_alerts = st.checkbox("Sadece yüksek emisyon uyarıları", value=False, key="only_alerts_check")

    sort_metric_map = {
        "Risk Skoru": COL_RISK,
        "CO₂ Emisyonu": COL_CO2,
        "Ortalama Sıcaklık": COL_TEMP,
        "Deniz Seviyesi Artışı": COL_SEA,
        "Aşırı Hava Olayı": COL_EVENTS,
        "İklim Baskı Skoru": "climate_pressure_score",
    }
    selected_sort_label = st.selectbox("Ülke Sıralama Ölçütü", list(sort_metric_map.keys()), key="sort_metric_select")
    selected_sort_col = sort_metric_map[selected_sort_label]

    top_n = st.slider("Grafiklerde Gösterilecek Ülke Sayısı", 5, 20, 10, key="top_n_slider")

    min_risk_filter = st.slider(
        "Minimum Risk Skoru",
        min_value=float(df[COL_RISK].min()),
        max_value=float(df[COL_RISK].max()),
        value=float(df[COL_RISK].min()),
        step=1.0,
        key="min_risk_slider",
    )

    temp_min = float(df[COL_TEMP].min())
    temp_max = float(df[COL_TEMP].max())
    selected_temp_range = st.slider(
        "Sıcaklık Aralığı",
        min_value=temp_min,
        max_value=temp_max,
        value=(temp_min, temp_max),
        step=0.01,
        key="temp_range_slider",
    )

    events_min = float(df[COL_EVENTS].min())
    events_max = float(df[COL_EVENTS].max())
    selected_events_range = st.slider(
        "Aşırı Hava Olayı Aralığı",
        min_value=events_min,
        max_value=events_max,
        value=(events_min, events_max),
        step=1.0,
        key="events_range_slider",
    )

    min_pressure_filter = st.slider(
        "Minimum İklim Baskı Skoru",
        min_value=float(df["climate_pressure_score"].min()),
        max_value=float(df["climate_pressure_score"].max()),
        value=float(df["climate_pressure_score"].min()),
        step=1.0,
        key="min_pressure_slider",
    )

    auto_focus_top_country = st.checkbox(
        "Detay panelinde otomatik en riskli ülkeyi göster",
        value=True,
        key="auto_focus_top_country_check"
    )

    st.markdown('<div class="control-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📌 Aktif Filtre Özeti")
    st.caption(f"🌐 Ülke: {selected_country}")
    st.caption(f"📅 Yıl: {selected_year[0]} - {selected_year[1]}")
    st.caption(f"🛡 Risk seviyesi: {selected_risk_level}")
    st.caption(f"⭐ Top N: {top_n}")
    st.caption(f"🚨 Sadece uyarılar: {'Evet' if only_alerts else 'Hayır'}")

    st.markdown('<div class="control-divider"></div>', unsafe_allow_html=True)
    st.markdown("### ➕ Yeni Kayıt")

    with st.form("new_record_form"):
        new_country = st.text_input("Ülke", placeholder="Örn: Mexico")
        new_year = st.number_input("Yıl", min_value=1900, max_value=2100, value=2025, step=1)
        new_temp = st.number_input("Ortalama Sıcaklık", value=0.50, step=0.01, format="%.2f")
        new_co2 = st.number_input("CO₂ Emisyonu", value=0.50, step=0.01, format="%.2f")
        new_sea = st.number_input("Deniz Seviyesi Artışı", value=0.50, step=0.01, format="%.2f")
        new_events = st.number_input("Aşırı Hava Olayı", value=50.0, step=1.0, format="%.0f")
        new_risk = st.number_input("Risk Skoru", value=50.0, step=1.0, format="%.1f")
        submitted = st.form_submit_button("Kaydı Veritabanına Ekle")

    if submitted:
        if new_country.strip() == "":
            st.error("Ülke adı boş olamaz.")
        else:
            try:
                insert_row(
                    {
                        COL_COUNTRY: new_country.strip(),
                        COL_YEAR: int(new_year),
                        COL_TEMP: float(new_temp),
                        COL_CO2: float(new_co2),
                        COL_SEA: float(new_sea),
                        COL_EVENTS: float(new_events),
                        COL_RISK: float(new_risk),
                    }
                )
                st.success("Yeni kayıt başarıyla eklendi. Sayfayı yenile.")
            except Exception as e:
                st.error(f"Kayıt eklenemedi: {e}")

    st.markdown("</div></div>", unsafe_allow_html=True)


# =========================================================
# FİLTRELEME
# =========================================================
df["dynamic_alert"] = df[COL_CO2].apply(
    lambda x: "Yüksek Karbon Emisyonu Uyarısı" if x > threshold else "Normal Durum"
)

filtered_df = df[
    (df[COL_YEAR] >= selected_year[0]) &
    (df[COL_YEAR] <= selected_year[1]) &
    (df[COL_RISK] >= min_risk_filter) &
    (df["climate_pressure_score"] >= min_pressure_filter) &
    (df[COL_TEMP] >= selected_temp_range[0]) &
    (df[COL_TEMP] <= selected_temp_range[1]) &
    (df[COL_EVENTS] >= selected_events_range[0]) &
    (df[COL_EVENTS] <= selected_events_range[1])
].copy()

if selected_country != "Tümü":
    filtered_df = filtered_df[filtered_df[COL_COUNTRY] == selected_country].copy()

if selected_risk_level != "Tümü":
    if selected_risk_level == "Düşük":
        filtered_df = filtered_df[filtered_df[COL_RISK] < 25]
    elif selected_risk_level == "Orta":
        filtered_df = filtered_df[(filtered_df[COL_RISK] >= 25) & (filtered_df[COL_RISK] < 50)]
    elif selected_risk_level == "Yüksek":
        filtered_df = filtered_df[(filtered_df[COL_RISK] >= 50) & (filtered_df[COL_RISK] < 75)]
    elif selected_risk_level == "Kritik":
        filtered_df = filtered_df[filtered_df[COL_RISK] >= 75]

if only_alerts:
    filtered_df = filtered_df[filtered_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı"].copy()

if filtered_df.empty:
    with right_col:
        st.warning("Seçilen filtrelere ait veri bulunamadı.")
    st.stop()


# =========================================================
# HESAPLAMALAR
# =========================================================
avg_temp = round(filtered_df[COL_TEMP].mean(), 2)
avg_co2 = round(filtered_df[COL_CO2].mean(), 3)
avg_sea = round(filtered_df[COL_SEA].mean(), 2)
avg_events = int(round(filtered_df[COL_EVENTS].mean(), 0))
avg_risk = round(filtered_df[COL_RISK].mean(), 1)
avg_pressure = round(filtered_df["climate_pressure_score"].mean(), 1)

alert_count = int((filtered_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı").sum())
pressure_alert_count = int((filtered_df["pressure_alert"] != "Normal").sum())

country_risk_map = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)
    .agg({
        COL_RISK: "mean",
        COL_CO2: "mean",
        COL_TEMP: "mean",
        COL_EVENTS: "mean",
        "climate_pressure_score": "mean",
    })
)

country_risk_bar = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)
    .agg({
        COL_RISK: "mean",
        COL_CO2: "mean",
        COL_TEMP: "mean",
        COL_SEA: "mean",
        COL_EVENTS: "mean",
        "climate_pressure_score": "mean",
    })
    .sort_values(selected_sort_col, ascending=False)
)

alerts = (
    filtered_df[filtered_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı"]
    .sort_values(COL_CO2, ascending=False)
)

top_country = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)[COL_RISK]
    .mean()
    .sort_values(COL_RISK, ascending=False)
)

if auto_focus_top_country:
    detail_country = top_country.iloc[0][COL_COUNTRY] if len(top_country) > 0 else "-"
else:
    detail_country = selected_country if selected_country != "Tümü" else (
        top_country.iloc[0][COL_COUNTRY] if len(top_country) > 0 else "-"
    )

detail_df = filtered_df[filtered_df[COL_COUNTRY] == detail_country].copy()
if detail_df.empty:
    detail_df = filtered_df.copy()

detail_avg_temp = round(detail_df[COL_TEMP].mean(), 2)
detail_avg_co2 = round(detail_df[COL_CO2].mean(), 3)
detail_avg_risk = round(detail_df[COL_RISK].mean(), 1)
detail_avg_pressure = round(detail_df["climate_pressure_score"].mean(), 1)
detail_alerts = int((detail_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı").sum())

detail_yearly = (
    detail_df.groupby(COL_YEAR, as_index=False)
    .agg({
        COL_RISK: "mean",
        COL_CO2: "mean",
        COL_TEMP: "mean",
        COL_EVENTS: "mean",
        "climate_pressure_score": "mean"
    })
    .sort_values(COL_YEAR)
)

most_risky_country = top_country.iloc[0][COL_COUNTRY] if len(top_country) > 0 else "-"
most_risky_score = round(top_country.iloc[0][COL_RISK], 1) if len(top_country) > 0 else 0

bubble_df = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)
    .agg({
        COL_CO2: "mean",
        COL_RISK: "mean",
        COL_EVENTS: "mean",
        "climate_pressure_score": "mean"
    })
    .sort_values(COL_RISK, ascending=False)
)


# =========================================================
# ALERT
# =========================================================
filter_signature = (
    selected_country,
    selected_year[0],
    selected_year[1],
    round(threshold, 2),
    selected_risk_level,
    only_alerts,
    selected_sort_label,
    top_n,
    min_risk_filter,
    min_pressure_filter,
    round(selected_temp_range[0], 2),
    round(selected_temp_range[1], 2),
    round(selected_events_range[0], 2),
    round(selected_events_range[1], 2),
    auto_focus_top_country
)

if "last_filter_signature" not in st.session_state:
    st.session_state.last_filter_signature = None

show_alert = st.session_state.last_filter_signature != filter_signature
st.session_state.last_filter_signature = filter_signature

if pressure_alert_count > 0 and show_alert:
    st.markdown(
        f"""
        <div class="alert-floating">
            <div class="alert-top">
                <div class="alert-title">🚨 Kritik İklim Uyarısı</div>
                <div class="alert-pill">Yüksek Öncelik</div>
            </div>
            <div class="alert-text">
                Filtrelenen sonuçlarda <strong>{pressure_alert_count}</strong> çevresel baskı sinyali ve
                <strong>{alert_count}</strong> yüksek CO₂ uyarısı bulundu.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SAĞ PANEL
# =========================================================
with right_col:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">🌍 İklim Zekâsı • Gerçek Zamanlı Sinyaller</div>
            <div class="hero-title">İKLİM RİSK<br>GÖSTERGE PANELİ</div>
            <div class="hero-text">
                İklim riski, karbon baskısı, okyanus stresi ve çevresel sinyalleri
                renkli, premium ve karar destek odaklı tek bir ekranda izleyin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
    with k1:
        metric_kpi("🌡", "Ortalama Sıcaklık", fmt_num(avg_temp, 2), "°C", "İklim ısı eğilimi", "linear-gradient(135deg,#ff7a00,#ff0054)", KPI_TEMP)
    with k2:
        metric_kpi("🏭", "Ortalama CO₂", fmt_num(avg_co2, 3), "", "Karbon baskısı", "linear-gradient(135deg,#00d4ff,#2563eb)", KPI_CO2)
    with k3:
        metric_kpi("🌊", "Deniz Seviyesi", fmt_num(avg_sea, 2), "mm", "Okyanus yükselişi", "linear-gradient(135deg,#06b6d4,#14b8a6)", KPI_SEA)
    with k4:
        metric_kpi("☔", "Aşırı Olay Sayısı", str(avg_events), "", "İklim olay sıklığı", "linear-gradient(135deg,#8b5cf6,#6366f1)", KPI_WEATHER)
    with k5:
        metric_kpi("🛡", "İklim Risk Skoru", fmt_num(avg_risk, 1), "", "Küresel risk endeksi", "linear-gradient(135deg,#ff4d6d,#ef4444)", KPI_RISK)
    with k6:
        metric_kpi("🌱", "İklim Baskı Skoru", fmt_num(avg_pressure, 1), "", "Birleşik skor", "linear-gradient(135deg,#22c55e,#84cc16)", KPI_BIO)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    left_main, right_side = st.columns([2.15, 1.55], gap="large")

    with left_main:
        chart_title("Küresel İklim Baskı Haritası", "Normalize edilmiş çevresel baskı skoruna göre dünya görünümü")

        map_df = country_risk_map.copy().sort_values("climate_pressure_score", ascending=False)
        top_map_points = map_df.head(5).copy()

        fig_map = px.choropleth(
            map_df,
            locations=COL_COUNTRY,
            locationmode="country names",
            color="climate_pressure_score",
            hover_name=COL_COUNTRY,
            hover_data={
                "climate_pressure_score": ':.1f',
                COL_RISK: ':.1f',
                COL_CO2: ':.3f',
                COL_TEMP: ':.2f',
                COL_EVENTS: ':.0f'
            },
            color_continuous_scale=[
                [0.00, "#20c997"],
                [0.25, "#a7f3d0"],
                [0.45, "#facc15"],
                [0.65, "#fb923c"],
                [1.00, "#ef4444"],
            ],
        )

        fig_map.update_traces(
            marker_line_color="rgba(255,255,255,0.90)",
            marker_line_width=1.1,
            hovertemplate="""
            <b>%{hovertext}</b><br>
            İklim Baskı Skoru: %{z:.1f}<br>
            Risk Skoru: %{customdata[1]:.1f}<br>
            CO₂: %{customdata[2]:.3f}<br>
            Sıcaklık: %{customdata[3]:.2f}°C<br>
            Aşırı Olay: %{customdata[4]:.0f}<br>
            <extra></extra>
            """
        )

        fig_map.add_trace(
            go.Scattergeo(
                locations=top_map_points[COL_COUNTRY],
                locationmode="country names",
                text=top_map_points[COL_COUNTRY],
                mode="markers+text",
                textposition="top center",
                marker=dict(
                    size=11,
                    color="#2563eb",
                    line=dict(width=3, color="white"),
                    opacity=0.98,
                ),
                textfont=dict(size=10, color="#0f1d3a", family="Inter, Segoe UI, sans-serif"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig_map.update_layout(
            height=390,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f1d3a", family="Inter, Segoe UI, sans-serif"),
            coloraxis_colorbar=dict(
                title=dict(text="Baskı Skoru", font=dict(color="#0f1d3a", size=12)),
                thickness=16,
                len=0.72,
                outlinewidth=0,
                tickfont=dict(color="#475569", size=11),
            ),
            geo=dict(
                projection_type="natural earth",
                showframe=False,
                showcoastlines=True,
                coastlinecolor="rgba(100,116,139,0.32)",
                coastlinewidth=0.8,
                showcountries=True,
                countrycolor="rgba(148,163,184,0.38)",
                countrywidth=0.6,
                showland=True,
                landcolor="rgba(239,248,242,0.78)",
                showocean=True,
                oceancolor="rgba(226,242,253,0.54)",
                showlakes=True,
                lakecolor="rgba(226,242,253,0.54)",
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        top5 = country_risk_bar.head(5).copy()
        if len(top5) > 0:
            top5_cols = st.columns(5, gap="small")
            medal_colors = ["#2563eb", "#f97316", "#22c55e", "#06b6d4", "#8b5cf6"]
            for i, (_, row) in enumerate(top5.iterrows()):
                with top5_cols[i]:
                    st.markdown(
                        f"""
                        <div class="insight-item" style="padding:.62rem;align-items:center;">
                            <div class="insight-icon" style="width:32px;height:32px;border-radius:999px;background:{medal_colors[i]};">{i+1}</div>
                            <div>
                                <div class="insight-title">{row[COL_COUNTRY]}</div>
                                <div class="insight-text">{row[selected_sort_col]:.1f}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with right_side:
        alert_card = st.container()
        with alert_card:
            st.markdown(
                f"""
                <div class="info-panel" style="background:linear-gradient(145deg,rgba(255,255,255,.88),rgba(255,236,236,.72));">
                    <div class="info-eyebrow" style="color:#b42318;">🚨 Kritik İklim Uyarısı</div>
                    <div class="info-line" style="color:#8b2b2b;font-size:.95rem;">
                        Filtrelenen sonuçlarda <b>{pressure_alert_count}</b> çevresel baskı sinyali ve
                        <b>{alert_count}</b> yüksek CO₂ uyarısı bulundu.
                    </div>
                    <div class="info-badge" style="background:linear-gradient(135deg,#ef4444,#f97316);">Yüksek Öncelik</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        s1, s2 = st.columns(2, gap="small")
        with s1:
            info_panel(
                "Yönetici Özeti",
                most_risky_country,
                [
                    f"Geçerli filtrelerde {most_risky_country} öne çıkıyor.",
                    f"Ortalama risk skoru {most_risky_score}.",
                ],
                risk_etiketi(most_risky_score),
                risk_rengi(most_risky_score),
            )
        with s2:
            info_panel(
                "Seçili Ülke",
                detail_country,
                [
                    f"Sıcaklık: {detail_avg_temp}°C",
                    f"CO₂: {detail_avg_co2}",
                    f"Risk: {detail_avg_risk}",
                    f"Baskı: {detail_avg_pressure}",
                    f"Uyarı: {detail_alerts}",
                ],
                risk_etiketi(detail_avg_risk),
                risk_rengi(detail_avg_risk),
            )

        st.markdown('<div class="risk-gauge-shell">', unsafe_allow_html=True)
        chart_title("İklim Baskı Ölçeri", "Seçili ülkenin birleşik çevresel baskı seviyesi")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=detail_avg_pressure,
                number={"font": {"size": 42, "color": "#0f766e"}, "suffix": ""},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                    "bar": {"color": "#2563eb", "thickness": 0.22},
                    "bgcolor": "rgba(255,255,255,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 25], "color": "#22c55e"},
                        {"range": [25, 50], "color": "#facc15"},
                        {"range": [50, 75], "color": "#fb923c"},
                        {"range": [75, 100], "color": "#ef4444"},
                    ],
                    "threshold": {
                        "line": {"color": "#0f1d3a", "width": 4},
                        "thickness": 0.82,
                        "value": detail_avg_pressure,
                    },
                },
            )
        )
        gauge.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#0f1d3a"},
        )
        st.plotly_chart(gauge, use_container_width=True)
        insight_item("📈", "Yön", "Çevresel baskı trendi seçili aralıkta izleniyor.", "linear-gradient(135deg,#22c55e,#14b8a6)")
        insight_item("⚡", "Durum", f"Ortalama baskı skoru {detail_avg_pressure} seviyesinde.", "linear-gradient(135deg,#f97316,#ef4444)")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        chart_title("Risk Skoru Trend Grafiği", "Seçili ülkenin yıllara göre değişimi")
        fig_risk_trend = px.line(detail_yearly, x=COL_YEAR, y=COL_RISK, markers=True)
        fig_risk_trend.update_traces(line=dict(width=3.6, color="#ff006e"), marker=dict(size=8, color="#ef4444"))
        apply_chart_style(fig_risk_trend, height=230)
        fig_risk_trend.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_risk_trend, use_container_width=True)

    with c2:
        chart_title("CO₂ Trend Grafiği", "Karbon emisyon yoğunluğu")
        fig_co2_trend = px.area(detail_yearly, x=COL_YEAR, y=COL_CO2)
        fig_co2_trend.update_traces(line=dict(width=3.2, color="#00b4d8"), fillcolor="rgba(0,180,216,0.34)")
        apply_chart_style(fig_co2_trend, height=230)
        fig_co2_trend.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_co2_trend, use_container_width=True)

    with c3:
        chart_title("Sıcaklık Trend Grafiği", "Ortalama sıcaklık değişimi")
        fig_temp = px.line(detail_yearly, x=COL_YEAR, y=COL_TEMP, markers=True)
        fig_temp.update_traces(line=dict(width=3.4, color="#ff0054"), marker=dict(size=8, color="#ff7a00"))
        apply_chart_style(fig_temp, height=230)
        fig_temp.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_temp, use_container_width=True)

    with c4:
        chart_title("Aşırı Olay Trend Grafiği", "Yıllara göre olay sıklığı")
        fig_events = px.bar(
            detail_yearly,
            x=COL_YEAR,
            y=COL_EVENTS,
            color=COL_EVENTS,
            color_continuous_scale=["#38bdf8", "#8b5cf6", "#f97316", "#ef4444"],
        )
        apply_chart_style(fig_events, height=230)
        fig_events.update_layout(xaxis_title="", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_events, use_container_width=True)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    row2_col1, row2_col2, row2_col3 = st.columns([1.05, 1.05, 1.0], gap="large")

    with row2_col1:
        chart_title(f"Ülke Karşılaştırması (İlk {top_n})", f"Sıralama ölçütü: {selected_sort_label}")
        topn_df = country_risk_bar.head(top_n)
        fig_bar = px.bar(
            topn_df,
            x=selected_sort_col,
            y=COL_COUNTRY,
            orientation="h",
            color=selected_sort_col,
            color_continuous_scale=["#14b8a6", "#22c55e", "#facc15", "#fb923c", "#ef4444"],
        )
        fig_bar.update_yaxes(categoryorder="total ascending")
        apply_chart_style(fig_bar, height=300)
        fig_bar.update_layout(xaxis_title="", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with row2_col2:
        chart_title("İklim Baskı Skoru Trendi", "Birleşik skorun yıllık hareketi")
        fig_pressure_trend = px.area(detail_yearly, x=COL_YEAR, y="climate_pressure_score")
        fig_pressure_trend.update_traces(line=dict(width=3.5, color="#00a896"), fillcolor="rgba(34,197,94,0.28)")
        apply_chart_style(fig_pressure_trend, height=300)
        fig_pressure_trend.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_pressure_trend, use_container_width=True)

    with row2_col3:
        chart_title("CO₂ ve Risk İlişkisi", "Emisyon, risk ve olay yoğunluğu")
        fig_scatter = px.scatter(
            bubble_df,
            x=COL_CO2,
            y=COL_RISK,
            size=COL_EVENTS,
            color="climate_pressure_score",
            hover_name=COL_COUNTRY,
            color_continuous_scale=["#14b8a6", "#22c55e", "#facc15", "#fb923c", "#ef4444"],
            size_max=42,
        )
        apply_chart_style(fig_scatter, height=300)
        fig_scatter.update_layout(xaxis_title="Ortalama CO₂", yaxis_title="Ortalama Risk")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    section_header("Karbon Emisyonu Uyarıları", "Eşik değeri aşan kritik kayıtların özet görünümü")

    if len(alerts) > 0:
        alert_table = alerts[
            [COL_COUNTRY, COL_YEAR, COL_CO2, COL_RISK, "climate_pressure_score", "dynamic_alert", "pressure_alert"]
        ].head(12).rename(
            columns={
                COL_COUNTRY: "Ülke",
                COL_YEAR: "Yıl",
                COL_CO2: "CO₂ Emisyonu",
                COL_RISK: "Risk Skoru",
                "climate_pressure_score": "İklim Baskı",
                "dynamic_alert": "CO₂ Uyarısı",
                "pressure_alert": "Baskı Durumu",
            }
        )
        st.dataframe(alert_table, use_container_width=True, hide_index=True)

        csv = alert_table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 CSV İndir",
            data=csv,
            file_name="climate_alerts.csv",
            mime="text/csv",
        )
    else:
        st.success("Kritik karbon emisyonu uyarısı bulunmuyor.")
