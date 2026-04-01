import base64
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from db import load_data, insert_row

st.set_page_config(
    page_title="Global Climate Risk Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

ASSETS_DIR = BASE_DIR / "assets"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
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


def risk_label(score: float) -> str:
    if score < 25:
        return "Düşük"
    if score < 50:
        return "Orta"
    if score < 75:
        return "Yüksek"
    return "Kritik"


def risk_color(score: float) -> str:
    if score < 25:
        return "#22c55e"
    if score < 50:
        return "#facc15"
    if score < 75:
        return "#fb923c"
    return "#ef4444"


def kpi_card(title: str, value, subtitle: str, bg_base64: str):
    st.markdown(
        f"""
        <div class="kpi-wrap">
            <div class="kpi-bg" style="background-image:url('data:image/png;base64,{bg_base64}');"></div>
            <div class="kpi-overlay"></div>
            <div class="kpi-content">
                <div class="kpi-label">{title}</div>
                <div class="kpi-number">{value}</div>
                <div class="kpi-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# ASSETS
# --------------------------------------------------
bg_img = img_to_base64(ASSETS_DIR / "background" / "background.png")
temp_img = img_to_base64(ASSETS_DIR / "kpi" / "temp.png")
co2_img = img_to_base64(ASSETS_DIR / "kpi" / "co2.png")
sea_img = img_to_base64(ASSETS_DIR / "kpi" / "sea.png")
weather_img = img_to_base64(ASSETS_DIR / "kpi" / "weather.png")
risk_img = img_to_base64(ASSETS_DIR / "kpi" / "risk.png")
bio_img = img_to_base64(ASSETS_DIR / "kpi" / "bio.png")

# Fallback varsa boş kalmasın
if not temp_img:
    temp_img = bg_img
if not co2_img:
    co2_img = bg_img
if not sea_img:
    sea_img = bg_img
if not weather_img:
    weather_img = bg_img
if not risk_img:
    risk_img = bg_img
if not bio_img:
    bio_img = bg_img


# --------------------------------------------------
# DATA
# --------------------------------------------------
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

df = df.dropna(subset=[COL_COUNTRY, COL_YEAR, COL_CO2, COL_RISK]).copy()

if df.empty:
    st.error("Temizleme sonrası veri kalmadı.")
    st.stop()


# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown(
    f"""
<style>
/* Genel */
.stApp {{
    background:
        linear-gradient(rgba(245,250,248,0.90), rgba(240,248,246,0.92)),
        url("data:image/png;base64,{bg_img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #0f172a;
}}

.block-container {{
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}}

header[data-testid="stHeader"] {{
    background: transparent !important;
}}

button[kind="header"],
[data-testid="collapsedControl"] {{
    display: none !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(239,248,244,0.97), rgba(228,241,237,0.97));
    border-right: 1px solid rgba(15,23,42,0.08);
    box-shadow: 6px 0 30px rgba(15,23,42,0.05);
}}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div {{
    color: #0f172a;
}}

/* Floating alert */
.floating-alert {{
    position: fixed;
    top: 22px;
    right: 22px;
    z-index: 99999;
    min-width: 360px;
    max-width: 480px;
    background: linear-gradient(135deg, rgba(255,245,245,0.97), rgba(255,230,230,0.97));
    border: 1px solid rgba(239,68,68,0.22);
    box-shadow: 0 18px 50px rgba(239,68,68,0.18);
    border-radius: 22px;
    padding: 18px 20px;
    backdrop-filter: blur(18px);
    animation: fadeInSlide 0.45s ease;
}}

.floating-alert-title {{
    font-size: 16px;
    font-weight: 900;
    color: #991b1b;
    margin-bottom: 6px;
}}

.floating-alert-text {{
    font-size: 14px;
    color: #7f1d1d;
    line-height: 1.5;
}}

@keyframes fadeInSlide {{
    from {{
        opacity: 0;
        transform: translateY(-14px) translateX(10px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0) translateX(0);
    }}
}}

/* Hero */
.hero-shell {{
    background: linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,255,255,0.50));
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.45);
    border-radius: 34px;
    padding: 38px 34px;
    box-shadow: 0 18px 55px rgba(15,23,42,0.10);
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}}

.hero-shell::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(34,197,94,0.08), rgba(59,130,246,0.08), rgba(251,191,36,0.08));
    z-index: 0;
}}

.hero-content {{
    position: relative;
    z-index: 2;
}}

.hero-badge {{
    display: inline-block;
    padding: 10px 18px;
    border-radius: 999px;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(15,23,42,0.08);
    font-size: 14px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.05);
}}

.hero-title {{
    font-size: 64px;
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -2px;
    color: #0f172a;
    margin-bottom: 14px;
}}

.hero-subtitle {{
    font-size: 20px;
    color: #475569;
    font-weight: 500;
    max-width: 780px;
    line-height: 1.7;
}}

.status-card {{
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(255,255,255,0.56);
    border-radius: 24px;
    padding: 18px 20px;
    box-shadow: 0 10px 24px rgba(15,23,42,0.05);
    backdrop-filter: blur(12px);
    max-width: 760px;
}}

.status-label {{
    font-size: 15px;
    color: #334155;
    font-weight: 700;
    margin-bottom: 12px;
}}

.status-track {{
    width: 100%;
    height: 16px;
    border-radius: 999px;
    background: rgba(15,23,42,0.10);
    overflow: hidden;
    margin-bottom: 10px;
}}

.status-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e 0%, #facc15 35%, #fb923c 68%, #ff5f6d 100%);
}}

.status-score {{
    font-size: 14px;
    color: #475569;
    font-weight: 700;
}}

/* KPI */
.kpi-wrap {{
    border-radius: 28px;
    overflow: hidden;
    position: relative;
    min-height: 180px;
    box-shadow: 0 16px 45px rgba(15,23,42,0.10);
    border: 1px solid rgba(255,255,255,0.35);
    margin-bottom: 12px;
}}

.kpi-bg {{
    position: absolute;
    inset: 0;
    background-size: cover;
    background-position: center;
    filter: saturate(1.08) contrast(1.04);
    transform: scale(1.02);
}}

.kpi-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(10,15,25,0.10), rgba(10,15,25,0.52));
}}

.kpi-content {{
    position: relative;
    z-index: 2;
    padding: 20px 20px 18px 20px;
    color: white;
}}

.kpi-label {{
    font-size: 14px;
    font-weight: 700;
    opacity: 0.96;
    margin-bottom: 12px;
}}

.kpi-number {{
    font-size: 42px;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 8px;
}}

.kpi-sub {{
    font-size: 13px;
    opacity: 0.92;
    font-weight: 500;
}}

/* Section cards */
.section-shell {{
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(255,255,255,0.52);
    border-radius: 28px;
    padding: 20px;
    box-shadow: 0 12px 28px rgba(15,23,42,0.05);
    backdrop-filter: blur(12px);
}}

.section-title {{
    font-size: 22px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 14px;
}}

.insight-card {{
    background: rgba(255,255,255,0.84);
    border: 1px solid rgba(255,255,255,0.52);
    border-radius: 28px;
    padding: 20px;
    box-shadow: 0 12px 28px rgba(15,23,42,0.05);
}}

.tag-pill {{
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    color: white;
}}

/* Charts and tables */
[data-testid="stPlotlyChart"] {{
    background: rgba(255,255,255,0.62);
    border-radius: 24px;
    padding: 10px;
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0 10px 24px rgba(15,23,42,0.04);
}}

[data-testid="stDataFrame"] {{
    background: rgba(255,255,255,0.68);
    border-radius: 24px;
    padding: 8px;
    border: 1px solid rgba(255,255,255,0.55);
}}

hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(15,23,42,0.08), transparent);
    margin: 26px 0;
}}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("## 🌍 Climate Control")

country_list = ["Tümü"] + sorted(df[COL_COUNTRY].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Ülke Seç", country_list)

min_year = int(df[COL_YEAR].min())
max_year = int(df[COL_YEAR].max())
selected_year = st.sidebar.slider("Yıl Aralığı", min_year, max_year, (min_year, max_year))

threshold = st.sidebar.slider(
    "CO₂ Eşik Değeri",
    min_value=float(df[COL_CO2].min()),
    max_value=float(df[COL_CO2].max()),
    value=float(round(df[COL_CO2].quantile(0.75), 2)),
    step=0.01,
)

df["dynamic_alert"] = df[COL_CO2].apply(
    lambda x: "Yüksek Karbon Emisyonu Uyarısı" if x > threshold else "Normal Durum"
)

filtered_df = df[(df[COL_YEAR] >= selected_year[0]) & (df[COL_YEAR] <= selected_year[1])].copy()

if selected_country != "Tümü":
    filtered_df = filtered_df[filtered_df[COL_COUNTRY] == selected_country].copy()

if filtered_df.empty:
    st.warning("Seçilen filtrelere ait veri bulunamadı.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("## ➕ Yeni Kayıt")

with st.sidebar.form("new_record_form"):
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
        st.sidebar.error("Ülke adı boş olamaz.")
    else:
        try:
            new_row = {
                COL_COUNTRY: new_country.strip(),
                COL_YEAR: int(new_year),
                COL_TEMP: float(new_temp),
                COL_CO2: float(new_co2),
                COL_SEA: float(new_sea),
                COL_EVENTS: float(new_events),
                COL_RISK: float(new_risk),
            }
            insert_row(new_row)
            st.sidebar.success("Yeni kayıt başarıyla eklendi. Sayfayı yenile.")
        except Exception as e:
            st.sidebar.error(f"Kayıt eklenemedi: {e}")

# --------------------------------------------------
# METRICS / AGGREGATES
# --------------------------------------------------
avg_temp = round(filtered_df[COL_TEMP].mean(), 2)
avg_co2 = round(filtered_df[COL_CO2].mean(), 3)
avg_sea = round(filtered_df[COL_SEA].mean(), 2)
avg_events = int(round(filtered_df[COL_EVENTS].mean(), 0))
avg_risk = round(filtered_df[COL_RISK].mean(), 1)
alert_count = int((filtered_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı").sum())

risk_text = risk_label(avg_risk)
risk_fill = max(3, min(int(avg_risk), 100))

yearly = (
    filtered_df.groupby(COL_YEAR, as_index=False)
    .agg({COL_TEMP: "mean", COL_CO2: "mean", COL_RISK: "mean"})
    .sort_values(COL_YEAR)
)

country_risk_map = (
    df.groupby(COL_COUNTRY, as_index=False)
    .agg({COL_RISK: "mean", COL_CO2: "mean", COL_TEMP: "mean"})
)

country_risk_bar = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)[COL_RISK]
    .mean()
    .sort_values(COL_RISK, ascending=False)
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

top_hot = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)[COL_TEMP]
    .mean()
    .sort_values(COL_TEMP, ascending=False)
)

top_co2 = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)[COL_CO2]
    .mean()
    .sort_values(COL_CO2, ascending=False)
)

most_risky_country = top_country.iloc[0][COL_COUNTRY] if len(top_country) > 0 else "-"
most_risky_score = round(top_country.iloc[0][COL_RISK], 1) if len(top_country) > 0 else 0

detail_country = selected_country if selected_country != "Tümü" else most_risky_country
detail_df = filtered_df[filtered_df[COL_COUNTRY] == detail_country].copy()
if detail_df.empty:
    detail_df = filtered_df.copy()

detail_avg_temp = round(detail_df[COL_TEMP].mean(), 2)
detail_avg_co2 = round(detail_df[COL_CO2].mean(), 3)
detail_avg_risk = round(detail_df[COL_RISK].mean(), 1)
detail_alerts = int((detail_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı").sum())

# --------------------------------------------------
# FLOATING ALERT
# --------------------------------------------------
filter_signature = (selected_country, selected_year[0], selected_year[1], round(threshold, 2))
if "last_filter_signature" not in st.session_state:
    st.session_state.last_filter_signature = None

show_floating_alert = st.session_state.last_filter_signature != filter_signature
st.session_state.last_filter_signature = filter_signature

if alert_count > 0 and show_floating_alert:
    st.markdown(
        f"""
        <div class="floating-alert">
            <div class="floating-alert-title">🚨 Critical Climate Alert</div>
            <div class="floating-alert-text">
                Filtered results contain <b>{alert_count}</b> high carbon emission records.
                Highest observed CO₂ value is <b>{round(filtered_df[COL_CO2].max(), 3)}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-content">
            <div class="hero-badge">🌍 Climate Intelligence • Real-Time Environmental Signals</div>
            <div class="hero-title">GLOBAL CLIMATE<br>RISK DASHBOARD</div>
            <div class="hero-subtitle">
                Climate risk, biodiversity pressure, ocean stress and carbon dynamics
                brought together in a premium environmental decision support interface.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.25, 1], gap="large")

with hero_left:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">STATUS: <span style="color:{risk_color(avg_risk)}; font-weight:900;">{risk_text.upper()}</span></div>
            <div class="status-track">
                <div class="status-fill" style="width:{risk_fill}%;"></div>
            </div>
            <div class="status-score">Global risk score: {avg_risk}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        f"""
        <div class="insight-card">
            <div style="font-size:15px; color:#475569; font-weight:800;">Executive Summary</div>
            <div style="font-size:30px; font-weight:900; color:#0f172a; margin:10px 0 6px 0;">{most_risky_country}</div>
            <div style="color:#64748b; line-height:1.7;">
                Current filters highlight <b>{most_risky_country}</b> as the most climate-vulnerable country.
                Average risk score is <b>{most_risky_score}</b> with <b>{alert_count}</b> active high-emission signals.
            </div>
            <div style="margin-top:14px;">
                <span class="tag-pill" style="background:{risk_color(most_risky_score)};">{risk_label(most_risky_score)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# KPIS
# --------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="medium")

with k1:
    kpi_card("🌡 Ortalama Sıcaklık", avg_temp, "Climate heat trend", temp_img)
with k2:
    kpi_card("🏭 Ortalama CO₂", avg_co2, "Carbon emissions pressure", co2_img)
with k3:
    kpi_card("🌊 Deniz Seviyesi", avg_sea, "Ocean rise signal", sea_img)
with k4:
    kpi_card("⛈ Aşırı Hava Olayı", avg_events, "Extreme climate frequency", weather_img)
with k5:
    kpi_card("📊 Risk Skoru", avg_risk, "Global climate risk index", risk_img)
with k6:
    kpi_card("🌿 Biyo Baskı", alert_count, "Biodiversity & alert pressure", bio_img)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# MAP + DETAIL
# --------------------------------------------------
left_map, right_detail = st.columns([1.25, 1], gap="large")

with left_map:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗺 Global Climate Risk Map</div>', unsafe_allow_html=True)

    fig_map = px.choropleth(
        country_risk_map,
        locations=COL_COUNTRY,
        locationmode="country names",
        color=COL_RISK,
        hover_name=COL_COUNTRY,
        color_continuous_scale="YlOrRd",
    )
    fig_map.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#94a3b8",
            projection_type="natural earth",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#0f172a",
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_detail:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌍 Ülke Bazlı Detay Kartı</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="insight-card">
            <div style="font-size:15px; color:#475569; font-weight:700;">Seçili Ülke</div>
            <div style="font-size:28px; font-weight:900; color:#0f172a; margin:8px 0 14px 0;">{detail_country}</div>
            <div style="font-size:15px; color:#64748b; margin-bottom:8px;">Ortalama sıcaklık: {detail_avg_temp}°C</div>
            <div style="font-size:15px; color:#64748b; margin-bottom:8px;">CO₂ emisyonu: {detail_avg_co2}</div>
            <div style="font-size:15px; color:#64748b; margin-bottom:8px;">Risk skoru: {detail_avg_risk}</div>
            <div style="font-size:15px; color:#64748b; margin-bottom:14px;">Uyarı sayısı: {detail_alerts}</div>
            <span class="tag-pill" style="background:{risk_color(detail_avg_risk)};">{risk_label(detail_avg_risk)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    detail_yearly = (
        detail_df.groupby(COL_YEAR, as_index=False)[COL_RISK]
        .mean()
        .sort_values(COL_YEAR)
    )

    fig_detail = px.line(detail_yearly, x=COL_YEAR, y=COL_RISK, markers=True, title="Risk Score Trend")
    fig_detail.update_traces(line=dict(width=4, color="#f97316"))
    fig_detail.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#0f172a",
    )
    st.plotly_chart(fig_detail, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# TOP RISK + INSIGHTS
# --------------------------------------------------
left_risk, right_insight = st.columns([1.15, 1], gap="large")

with left_risk:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔥 En Riskli Ülkeler</div>', unsafe_allow_html=True)

    top10 = country_risk_bar.head(10)
    fig_bar = px.bar(
        top10,
        x=COL_RISK,
        y=COL_COUNTRY,
        orientation="h",
        color=COL_RISK,
        color_continuous_scale="Plasma",
    )
    fig_bar.update_yaxes(categoryorder="total ascending")
    fig_bar.update_layout(
        height=430,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Risk",
        yaxis_title="Ülke",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#0f172a",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_insight:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Top Insight</div>', unsafe_allow_html=True)

    hot_country = top_hot.iloc[0][COL_COUNTRY] if len(top_hot) > 0 else "-"
    hot_temp = round(top_hot.iloc[0][COL_TEMP], 2) if len(top_hot) > 0 else 0

    co2_country = top_co2.iloc[0][COL_COUNTRY] if len(top_co2) > 0 else "-"
    co2_value = round(top_co2.iloc[0][COL_CO2], 3) if len(top_co2) > 0 else 0

    st.markdown(
        f"""
        <div class="insight-card">
            <div style="font-size:15px; color:#475569; font-weight:800;">🌡 En Sıcak Bölge</div>
            <div style="font-size:22px; font-weight:900; margin:12px 0 6px 0;">{hot_country}</div>
            <div style="color:#64748b;">Ortalama sıcaklık: {hot_temp}°C</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="insight-card">
            <div style="font-size:15px; color:#475569; font-weight:800;">🏭 En Yüksek CO₂</div>
            <div style="font-size:22px; font-weight:900; margin:12px 0 6px 0;">{co2_country}</div>
            <div style="color:#64748b;">CO₂ ortalaması: {co2_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="insight-card">
            <div style="font-size:15px; color:#991b1b; font-weight:900;">🚨 Kritik Gözlem</div>
            <div style="margin-top:12px; color:#475569; line-height:1.7;">
                {most_risky_country}, mevcut filtrelerde en yüksek ortalama iklim risk skoruna sahip ülke olarak öne çıkıyor.
                Bu ülke için daha sıkı iklim önlemleri ve izleme mekanizmaları önerilir.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# ALERT TABLE + CSV
# --------------------------------------------------
st.markdown('<div class="section-shell">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🚨 Karbon Emisyonu Uyarıları</div>', unsafe_allow_html=True)

if len(alerts) > 0:
    alert_table = alerts[[COL_COUNTRY, COL_YEAR, COL_CO2, COL_RISK, "dynamic_alert"]].head(15).rename(
        columns={
            COL_COUNTRY: "Ülke",
            COL_YEAR: "Yıl",
            COL_CO2: "CO₂ Emisyonu",
            COL_RISK: "Risk Skoru",
            "dynamic_alert": "Uyarı",
        }
    )
    st.dataframe(alert_table, use_container_width=True, hide_index=True)

    csv = alert_table.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Uyarıları CSV Olarak İndir",
        data=csv,
        file_name="climate_alerts.csv",
        mime="text/csv",
    )
else:
    st.success("Kritik karbon emisyonu uyarısı bulunmuyor.")

st.markdown("</div>", unsafe_allow_html=True)