import base64
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from db import load_data, insert_row

st.set_page_config(
    page_title="İklim Risk Gösterge Paneli",
    page_icon="🌍",
    layout="wide",
)

ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = BASE_DIR / "src" / "trained_climate_risk_model.joblib"


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


def train_or_load_ml_model(data: pd.DataFrame, feature_cols: list[str], target_col: str):
    """Dashboard içinde API kullanmadan ML risk tahmini için modeli hazırlar."""
    if not ML_AVAILABLE:
        return None, {"error": "scikit-learn veya joblib kurulu değil."}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        try:
            model_data = joblib.load(MODEL_PATH)
            if model_data.get("features") == feature_cols:
                return model_data, model_data.get("metrics", {})
        except Exception:
            pass

    model_df = data[feature_cols + [target_col]].copy()
    for col in feature_cols + [target_col]:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.dropna(subset=feature_cols + [target_col])

    if len(model_df) < 5:
        return None, {"error": "ML modeli eğitmek için en az 5 temiz gözlem gerekli."}

    X = model_df[feature_cols]
    y = model_df[target_col]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        random_state=42,
        min_samples_leaf=2,
    )

    metrics = {"train_rows": int(len(model_df))}

    if len(model_df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics.update(
            {
                "mae": round(float(mean_absolute_error(y_test, preds)), 3),
                "r2": round(float(r2_score(y_test, preds)), 3),
                "test_rows": int(len(X_test)),
            }
        )
    else:
        model.fit(X, y)
        metrics.update({"mae": None, "r2": None, "test_rows": 0})

    model_data = {"model": model, "features": feature_cols, "metrics": metrics}
    joblib.dump(model_data, MODEL_PATH)
    return model_data, metrics


def predict_ml_risk(model_data: dict, temp: float, co2: float, sea: float, events: float) -> float:
    X_new = pd.DataFrame(
        [
            {
                "avg_temperature": temp,
                "co2_emissions": co2,
                "sea_level_rise": sea,
                "extreme_weather_events": events,
            }
        ]
    )

    X_new = X_new[model_data["features"]]
    pred = float(model_data["model"].predict(X_new)[0])
    return round(max(0, min(100, pred)), 2)


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

# ML tarafında kolon adları standartlaştırılıyor.
ml_df = df.rename(
    columns={
        COL_TEMP: "avg_temperature",
        COL_CO2: "co2_emissions",
        COL_SEA: "sea_level_rise",
        COL_EVENTS: "extreme_weather_events",
        COL_RISK: "climate_risk_score",
    }
).copy()

ML_FEATURES = [
    "avg_temperature",
    "co2_emissions",
    "sea_level_rise",
    "extreme_weather_events",
]
ML_TARGET = "climate_risk_score"


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

.popup-backdrop {{
    position: fixed;
    inset: 0;
    z-index: 9998;
    background:
        radial-gradient(circle at 12% 18%, rgba(0,212,255,0.18), transparent 28%),
        radial-gradient(circle at 86% 18%, rgba(249,115,22,0.16), transparent 30%),
        radial-gradient(circle at 68% 88%, rgba(139,92,246,0.16), transparent 34%),
        rgba(15, 23, 42, 0.22);
    backdrop-filter: blur(12px) saturate(1.25);
    -webkit-backdrop-filter: blur(12px) saturate(1.25);
}}

.climate-popup {{
    position: fixed;
    top: 92px;
    right: 32px;
    width: 390px;
    z-index: 9999;
    border-radius: 32px;
    padding: 1.35rem 1.35rem 1.2rem 1.35rem;
    overflow: hidden;
    background:
        radial-gradient(circle at 8% 10%, rgba(255,255,255,0.92), transparent 28%),
        radial-gradient(circle at 14% 18%, rgba(239,68,68,0.20), transparent 32%),
        radial-gradient(circle at 92% 18%, rgba(249,115,22,0.20), transparent 34%),
        radial-gradient(circle at 50% 100%, rgba(255,214,165,0.24), transparent 38%),
        linear-gradient(145deg, rgba(255,255,255,0.78), rgba(255,244,232,0.54));
    border: 1px solid rgba(255,255,255,0.78);
    box-shadow:
        0 34px 95px rgba(239,68,68,0.24),
        0 18px 54px rgba(15,23,42,0.18),
        inset 0 1px 0 rgba(255,255,255,0.92),
        inset 0 -1px 0 rgba(255,255,255,0.36);
    backdrop-filter: blur(30px) saturate(1.45);
    -webkit-backdrop-filter: blur(30px) saturate(1.45);
    animation: popupSlide .34s cubic-bezier(.2,.85,.25,1);
}}

.climate-popup::before {{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(120deg, rgba(255,255,255,0.42), transparent 28%, transparent 64%, rgba(255,255,255,0.18)),
        radial-gradient(circle at 100% 0%, rgba(255,255,255,0.52), transparent 22%);
}}

.climate-popup::after {{
    content: "";
    position: absolute;
    left: 1.35rem;
    right: 1.35rem;
    top: 0;
    height: 5px;
    border-radius: 0 0 999px 999px;
    background: linear-gradient(90deg, #ef4444, #f97316, #facc15, #22c55e, #06b6d4, #8b5cf6);
    box-shadow: 0 12px 28px rgba(249,115,22,0.26);
}}

@keyframes popupSlide {{
    from {{
        opacity: 0;
        transform: translateY(-18px) scale(.96);
        filter: blur(4px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0) scale(1);
        filter: blur(0);
    }}
}}

.popup-close {{
    position: absolute;
    top: .82rem;
    right: .86rem;
    z-index: 3;
    width: 36px;
    height: 36px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.78);
    background: rgba(255,255,255,0.72);
    color: #b42318 !important;
    font-size: 1.42rem;
    font-weight: 950;
    cursor: pointer;
    line-height: 34px;
    text-align: center;
    text-decoration: none !important;
    box-shadow: 0 10px 24px rgba(239,68,68,0.12);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: transform .16s ease, background .16s ease, box-shadow .16s ease;
}}

.popup-close:hover {{
    transform: scale(1.06) rotate(4deg);
    background: rgba(254,226,226,0.88);
    box-shadow: 0 14px 30px rgba(239,68,68,0.20);
}}

.popup-icon {{
    position: relative;
    z-index: 2;
    width: 56px;
    height: 56px;
    border-radius: 22px;
    background:
        radial-gradient(circle at 28% 22%, rgba(255,255,255,0.48), transparent 30%),
        linear-gradient(135deg,#ef4444,#f97316);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.55rem;
    margin-bottom: .85rem;
    box-shadow:
        0 18px 38px rgba(239,68,68,0.30),
        inset 0 1px 0 rgba(255,255,255,0.34);
}}

.popup-title {{
    position: relative;
    z-index: 2;
    font-size: 1.28rem;
    font-weight: 950;
    color: #991b1b;
    letter-spacing: -.25px;
    margin-bottom: .48rem;
}}

.popup-text {{
    position: relative;
    z-index: 2;
    font-size: .94rem;
    line-height: 1.62;
    color: #7f1d1d;
    font-weight: 780;
}}

.popup-filter-box {{
    position: relative;
    z-index: 2;
    margin-top: .9rem;
    padding: .78rem .85rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.54);
    border: 1px solid rgba(255,255,255,0.64);
    color: #334155;
    font-size: .83rem;
    line-height: 1.7;
    font-weight: 760;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.72);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}}

.popup-badge {{
    position: relative;
    z-index: 2;
    display: inline-flex;
    margin-top: .95rem;
    padding: .52rem .94rem;
    border-radius: 999px;
    background:
        radial-gradient(circle at 22% 20%, rgba(255,255,255,0.42), transparent 32%),
        linear-gradient(135deg,#ef4444,#f97316);
    color: white;
    font-size: .77rem;
    font-weight: 950;
    box-shadow: 0 14px 30px rgba(239,68,68,0.26);
}}


.ml-premium-card {{
    position: relative;
    overflow: hidden;
    border-radius: 30px;
    padding: 1.25rem;
    background:
        radial-gradient(circle at 10% 10%, rgba(0,212,255,0.24), transparent 32%),
        radial-gradient(circle at 88% 18%, rgba(139,92,246,0.22), transparent 34%),
        radial-gradient(circle at 50% 100%, rgba(34,197,94,0.18), transparent 36%),
        linear-gradient(135deg, rgba(255,255,255,0.56), rgba(255,255,255,0.24));
    border: 1px solid rgba(255,255,255,0.62);
    box-shadow:
        0 24px 70px rgba(37,99,235,0.14),
        0 14px 40px rgba(139,92,246,0.12),
        inset 0 1px 0 rgba(255,255,255,0.72);
    backdrop-filter: blur(26px) saturate(1.35);
}}

.ml-premium-title {{
    font-size: 1.18rem;
    font-weight: 950;
    color: #0f1d3a;
    letter-spacing: -.25px;
    margin-bottom: .25rem;
}}

.ml-premium-subtitle {{
    font-size: .86rem;
    color: #64748b;
    font-weight: 720;
    margin-bottom: .9rem;
}}

.ml-result-card {{
    margin-top: .9rem;
    border-radius: 22px;
    padding: 1rem;
    background:
        radial-gradient(circle at 12% 12%, rgba(34,197,94,0.22), transparent 34%),
        linear-gradient(135deg, rgba(255,255,255,0.70), rgba(240,253,244,0.52));
    border: 1px solid rgba(255,255,255,0.76);
    box-shadow: 0 16px 36px rgba(34,197,94,0.12);
}}


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
    auto_focus_top_country,
)

# İlk açılışta popup gösterme; sadece filtre değişince göster.
if "alert_initialized" not in st.session_state:
    st.session_state.alert_initialized = True
    st.session_state.alert_open = False
    st.session_state.last_filter_signature = filter_signature

show_alert = st.session_state.last_filter_signature != filter_signature

# Filtre değişince ve uyarı varsa popup açılır.
if show_alert and pressure_alert_count > 0:
    st.session_state.alert_open = True

st.session_state.last_filter_signature = filter_signature

# Popup'ı st.markdown yerine parent DOM'a JavaScript ile ekliyoruz.
# Böylece HTML kod gibi görünmez, X'e basınca da sayfa/query param değişmeden kapanır.
if st.session_state.alert_open and pressure_alert_count > 0:
    popup_html = f"""
<div class="popup-backdrop" id="climate-popup-backdrop">
    <div class="climate-popup">
        <button
            class="popup-close"
            type="button"
            onclick="document.getElementById('climate-popup-root')?.remove();"
            aria-label="Uyarıyı kapat"
        >×</button>

        <div class="popup-icon">🚨</div>
        <div class="popup-title">Kritik İklim Uyarısı</div>

        <div class="popup-text">
            Seçtiğin filtrelere göre <b>{pressure_alert_count}</b> çevresel baskı sinyali ve
            <b>{alert_count}</b> yüksek CO₂ uyarısı bulundu.
        </div>

        <div class="popup-filter-box">
            🌍 Ülke: <b>{selected_country}</b><br>
            📅 Yıl: <b>{selected_year[0]} - {selected_year[1]}</b><br>
            🛡 Risk seviyesi: <b>{selected_risk_level}</b><br>
            🏭 CO₂ eşik değeri: <b>{threshold:.2f}</b>
        </div>

        <div class="popup-badge">Yüksek Öncelik</div>
    </div>
</div>
"""

    components.html(
        f"""
<script>
(function() {{
    const popupHtml = {json.dumps(popup_html)};
    const rootId = "climate-popup-root";

    const oldPopup = parent.document.getElementById(rootId);
    if (oldPopup) {{
        oldPopup.remove();
    }}

    const root = parent.document.createElement("div");
    root.id = rootId;
    root.innerHTML = popupHtml;
    parent.document.body.appendChild(root);
}})();
</script>
        """,
        height=0,
        width=0,
    )

    # Aynı filtrede sonraki rerun'da popup tekrar zorla açılmasın.
    # Yeni filtre seçilirse show_alert tekrar True olur ve popup yeniden açılır.
    st.session_state.alert_open = False


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

        # Boşluğu dolduran premium ek grafik: Risk yoğunluk matrisi
        st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
        chart_title("Risk Yoğunluk Matrisi", "Ülke ve yıllara göre risk skorunun renkli dağılımı")

        heatmap_source = (
            filtered_df.groupby([COL_COUNTRY, COL_YEAR], as_index=False)[COL_RISK]
            .mean()
        )

        heatmap_top_countries = (
            filtered_df.groupby(COL_COUNTRY)[COL_RISK]
            .mean()
            .sort_values(ascending=False)
            .head(min(top_n, 12))
            .index
        )

        heatmap_source = heatmap_source[heatmap_source[COL_COUNTRY].isin(heatmap_top_countries)]
        heatmap_pivot = heatmap_source.pivot(index=COL_COUNTRY, columns=COL_YEAR, values=COL_RISK)

        if not heatmap_pivot.empty:
            fig_heatmap = px.imshow(
                heatmap_pivot,
                aspect="auto",
                color_continuous_scale=[
                    [0.00, "#14b8a6"],
                    [0.35, "#22c55e"],
                    [0.55, "#facc15"],
                    [0.75, "#fb923c"],
                    [1.00, "#ef4444"],
                ],
                labels=dict(x="Yıl", y="Ülke", color="Risk"),
            )
            apply_chart_style(fig_heatmap, height=270)
            fig_heatmap.update_layout(
                margin=dict(l=8, r=8, t=8, b=8),
                coloraxis_colorbar=dict(
                    title=dict(text="Risk", font=dict(size=11)),
                    thickness=12,
                    len=0.72,
                    outlinewidth=0,
                    tickfont=dict(size=10),
                ),
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

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

    section_header("ML Tabanlı Risk Tahmini", "Girilen çevresel değerlere göre makine öğrenmesi ile tahmini iklim risk skoru üretir")

    model_data, model_metrics = train_or_load_ml_model(ml_df, ML_FEATURES, ML_TARGET)

    if model_data is None:
        st.warning(f"ML modeli hazırlanamadı: {model_metrics.get('error', 'Bilinmeyen hata')}")
        st.caption("Gerekirse terminalde şunu çalıştır: pip install scikit-learn joblib")
    else:
        ml_col1, ml_col2 = st.columns([1.25, 1.0], gap="large")

        with ml_col1:
            with st.form("ml_prediction_form"):
                p1, p2 = st.columns(2)
                with p1:
                    ml_temp = st.number_input("Ortalama Sıcaklık", value=float(avg_temp), step=0.01, format="%.2f")
                    ml_sea = st.number_input("Deniz Seviyesi Artışı", value=float(avg_sea), step=0.01, format="%.2f")
                with p2:
                    ml_co2 = st.number_input("CO₂ Emisyonu", value=float(avg_co2), step=0.01, format="%.3f")
                    ml_events = st.number_input("Aşırı Hava Olayı", value=float(avg_events), step=1.0, format="%.0f")

                predict_submit = st.form_submit_button("🤖 ML ile Risk Tahmini Yap")

        with ml_col2:
            metric_lines = [f"Eğitim gözlemi: {model_metrics.get('train_rows', '-')}"]
            if model_metrics.get("mae") is not None:
                metric_lines.append(f"MAE: {model_metrics.get('mae')}")
                metric_lines.append(f"R²: {model_metrics.get('r2')}")
            else:
                metric_lines.append("Test metriği için veri sayısı sınırlı.")

            info_panel(
                "Model Durumu",
                "Aktif",
                metric_lines,
                "Random Forest",
                "linear-gradient(135deg,#2563eb,#00d4ff,#14b8a6)",
            )

        if predict_submit:
            predicted_score = predict_ml_risk(model_data, ml_temp, ml_co2, ml_sea, ml_events)
            predicted_label = risk_etiketi(predicted_score)

            st.markdown(
                f"""
                <div class="info-panel" style="margin-top:.7rem;">
                    <div class="info-eyebrow">🤖 Makine Öğrenmesi Tahmini</div>
                    <div class="info-big">{predicted_score}</div>
                    <div class="info-line">Girilen değerlere göre tahmini risk seviyesi: <b>{predicted_label}</b></div>
                    <div class="info-badge" style="background:{risk_rengi(predicted_score)};">{predicted_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
