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
        return "#22c55e"
    if score < 50:
        return "#eab308"
    if score < 75:
        return "#f97316"
    return "#ef4444"


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


def metric_kpi(title: str, value: str, subtitle: str, bg_base64: str):
    st.markdown(
        f"""
        <div class="kpi-card" style="background-image:url('data:image/png;base64,{bg_base64}');">
            <div class="kpi-overlay"></div>
            <div class="kpi-content">
                <div class="kpi-chip"></div>
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
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


def apply_chart_style(fig, height=360, showlegend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fbfd",
        font=dict(color="#10203a", family="Inter, Segoe UI, sans-serif"),
        showlegend=showlegend,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.07)",
        zeroline=False,
        linecolor="rgba(15,23,42,0.08)",
        tickfont=dict(color="#64748b"),
        title_font=dict(color="#334155"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.07)",
        zeroline=False,
        linecolor="rgba(15,23,42,0.08)",
        tickfont=dict(color="#64748b"),
        title_font=dict(color="#334155"),
    )
    return fig


# =========================================================
# GÖRSELLER
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

df = df.dropna(subset=[COL_COUNTRY, COL_YEAR, COL_CO2, COL_RISK]).copy()

if df.empty:
    st.error("Temizleme sonrası veri kalmadı.")
    st.stop()

# =========================================================
# CSS
# =========================================================
st.markdown(
    f"""
<style>
:root {{
    --bg-main: #eef4f8;
    --bg-soft: #f8fbfd;
    --panel: rgba(255,255,255,0.96);
    --panel-soft: rgba(255,255,255,0.90);
    --ink: #10203a;
    --ink-soft: #64748b;
    --line: rgba(15,23,42,0.08);
    --shadow-lg: 0 18px 40px rgba(15,23,42,0.08);
    --shadow-md: 0 12px 24px rgba(15,23,42,0.05);
}}

html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.08), transparent 22%),
        linear-gradient(180deg, #edf4f8 0%, #f8fbfd 100%);
    color: var(--ink);
}}

.block-container {{
    max-width: 1800px !important;
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
}}

header[data-testid="stHeader"],
footer,
#MainMenu {{
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
    background:
        linear-gradient(180deg, rgba(248,251,253,0.98), rgba(241,247,250,0.96));
    border: 1px solid rgba(255,255,255,0.92);
    border-radius: 28px;
    padding: 1.1rem 1rem 1rem 1rem;
    box-shadow: var(--shadow-lg);
}}

.control-topline {{
    width: 72px;
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #06b6d4, #22c55e);
    margin-bottom: .9rem;
}}

.control-title {{
    font-size: 1.2rem;
    font-weight: 950;
    color: #10203a;
    margin-bottom: .35rem;
}}

.control-text {{
    font-size: .92rem;
    line-height: 1.65;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 1rem;
}}

.control-divider {{
    height: 1px;
    background: rgba(15,23,42,0.08);
    margin: .9rem 0 1rem 0;
}}

.hero {{
    position: relative;
    overflow: hidden;
    min-height: 280px;
    border-radius: 28px;
    background:
        linear-gradient(95deg, rgba(247,251,252,0.84) 0%, rgba(244,249,252,0.60) 42%, rgba(244,249,252,0.10) 100%),
        url("data:image/png;base64,{HERO_IMG}");
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(255,255,255,0.86);
    box-shadow: var(--shadow-lg);
    padding: 1.7rem 1.8rem;
    margin-bottom: 1rem;
}}

.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .58rem .92rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(255,255,255,0.92);
    color: #10203a;
    font-size: .86rem;
    font-weight: 850;
    margin-bottom: .85rem;
}}

.hero-title {{
    font-size: 3rem;
    line-height: .95;
    font-weight: 950;
    letter-spacing: -1.6px;
    color: #0c1631;
    margin-bottom: .58rem;
    max-width: 700px;
}}

.hero-text {{
    font-size: 1.02rem;
    line-height: 1.78;
    color: #435972;
    font-weight: 650;
    max-width: 740px;
}}

.section-header {{
    display: flex;
    align-items: flex-start;
    gap: .75rem;
    margin: .2rem 0 .8rem 0;
}}

.section-mark {{
    width: 8px;
    min-width: 8px;
    height: 38px;
    border-radius: 999px;
    background: linear-gradient(180deg, #06b6d4, #22c55e);
    box-shadow: 0 8px 16px rgba(6,182,212,0.18);
}}

.section-title {{
    font-size: 1.12rem;
    line-height: 1.12;
    font-weight: 950;
    color: #10203a;
}}

.section-subtitle {{
    margin-top: .14rem;
    font-size: .9rem;
    line-height: 1.5;
    color: #64748b;
    font-weight: 600;
}}

.kpi-card {{
    position: relative;
    overflow: hidden;
    min-height: 150px;
    border-radius: 22px;
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(255,255,255,0.20);
    box-shadow: 0 14px 28px rgba(15,23,42,0.14);
}}

.kpi-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(6,10,16,0.14), rgba(6,10,16,0.68));
}}

.kpi-content {{
    position: relative;
    z-index: 1;
    min-height: 150px;
    padding: .95rem;
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}}

.kpi-chip {{
    width: 42px;
    height: 4px;
    border-radius: 999px;
    background: rgba(255,255,255,0.92);
    margin-bottom: auto;
}}

.kpi-title {{
    font-size: .9rem;
    font-weight: 850;
    line-height: 1.28;
    margin-top: .7rem;
}}

.kpi-value {{
    font-size: 2.1rem;
    font-weight: 950;
    line-height: 1;
    letter-spacing: -1px;
    margin: .28rem 0;
}}

.kpi-subtitle {{
    font-size: .83rem;
    line-height: 1.42;
    color: rgba(255,255,255,0.96);
}}

.info-panel {{
    background: var(--panel);
    border: 1px solid rgba(255,255,255,0.98);
    border-radius: 22px;
    padding: .95rem;
    box-shadow: var(--shadow-md);
    min-height: 150px;
}}

.info-eyebrow {{
    font-size: .8rem;
    font-weight: 900;
    letter-spacing: .35px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: .42rem;
}}

.info-big {{
    font-size: 1.82rem;
    line-height: 1.05;
    font-weight: 950;
    color: #10203a;
    letter-spacing: -.5px;
    margin-bottom: .38rem;
}}

.info-line {{
    font-size: .9rem;
    line-height: 1.58;
    color: #55667c;
    font-weight: 600;
}}

.info-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: .38rem .78rem;
    border-radius: 999px;
    color: white;
    font-size: .78rem;
    font-weight: 900;
    margin-top: .72rem;
}}

.chart-head {{
    margin: .1rem 0 .35rem 0;
}}

.chart-title {{
    font-size: 1rem;
    font-weight: 950;
    color: #10203a;
    line-height: 1.16;
}}

.chart-subtitle {{
    font-size: .85rem;
    line-height: 1.45;
    color: #64748b;
    font-weight: 600;
    margin-top: .14rem;
}}

.alert-floating {{
    position: fixed;
    top: 14px;
    right: 16px;
    z-index: 9999;
    width: 290px;
    border-radius: 18px;
    padding: .82rem .88rem;
    background: rgba(255,242,242,0.98);
    border: 1px solid rgba(239,68,68,0.14);
    box-shadow: 0 14px 28px rgba(120,30,30,0.10);
}}

.alert-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .8rem;
    margin-bottom: .35rem;
}}

.alert-title {{
    font-size: .94rem;
    font-weight: 950;
    color: #b42318;
}}

.alert-pill {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: .28rem .58rem;
    border-radius: 999px;
    background: rgba(239,68,68,0.10);
    color: #b42318;
    font-size: .72rem;
    font-weight: 900;
}}

.alert-text {{
    font-size: .83rem;
    line-height: 1.56;
    color: #8b2b2b;
    font-weight: 600;
}}

[data-testid="stPlotlyChart"] {{
    background: rgba(255,255,255,0.92) !important;
    border: 1px solid rgba(255,255,255,0.98) !important;
    border-radius: 22px !important;
    box-shadow: 0 12px 24px rgba(15,23,42,0.05) !important;
    padding: .35rem !important;
}}

[data-testid="stDataFrame"] {{
    background: rgba(255,255,255,0.98) !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.98) !important;
}}

.stSelectbox label,
.stSlider label,
.stTextInput label,
.stNumberInput label,
.stCheckbox label {{
    color: #10203a !important;
    font-weight: 800 !important;
    font-size: .88rem !important;
}}

.stDownloadButton button,
.stFormSubmitButton button,
.stButton button {{
    border-radius: 14px !important;
    background: rgba(255,255,255,0.98) !important;
    color: #10203a !important;
    border: 1px solid rgba(15,23,42,0.10) !important;
    font-weight: 850 !important;
}}

.element-container {{
    margin-bottom: .34rem !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# ANA LAYOUT
# =========================================================
left_col, right_col = st.columns([0.92, 4.15], gap="large")

with left_col:
    st.markdown('<div class="left-rail"><div class="control-panel">', unsafe_allow_html=True)
    st.markdown('<div class="control-topline"></div>', unsafe_allow_html=True)
    st.markdown('<div class="control-title">İklim Kontrol Paneli</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="control-text">Filtreler, eşik değerleri ve yeni kayıt ekleme işlemleri bu sabit panelden yönetilir.</div>',
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
    st.markdown("###  Analiz Filtreleri")

    risk_level_options = ["Tümü", "Düşük", "Orta", "Yüksek", "Kritik"]
    selected_risk_level = st.selectbox("Risk Seviyesi", risk_level_options, key="risk_level_select")

    only_alerts = st.checkbox("Sadece yüksek emisyon uyarıları", value=False, key="only_alerts_check")

    sort_metric_map = {
        "Risk Skoru": COL_RISK,
        "CO₂ Emisyonu": COL_CO2,
        "Ortalama Sıcaklık": COL_TEMP,
        "Deniz Seviyesi Artışı": COL_SEA,
        "Aşırı Hava Olayı": COL_EVENTS,
    }
    selected_sort_label = st.selectbox(
        "Ülke Sıralama Ölçütü",
        list(sort_metric_map.keys()),
        key="sort_metric_select"
    )
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

    auto_focus_top_country = st.checkbox(
        "Detay panelinde otomatik en riskli ülkeyi göster",
        value=False,
        key="auto_focus_top_country_check"
    )

    st.markdown('<div class="control-divider"></div>', unsafe_allow_html=True)
    st.markdown("###  Aktif Filtre Özeti")
    st.caption(f"Ülke: {selected_country}")
    st.caption(f"Yıl: {selected_year[0]} - {selected_year[1]}")
    st.caption(f"Risk seviyesi: {selected_risk_level}")
    st.caption(f"Top N: {top_n}")
    st.caption(f"Sadece uyarılar: {'Evet' if only_alerts else 'Hayır'}")

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
alert_count = int((filtered_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı").sum())

country_risk_map = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)
    .agg({COL_RISK: "mean", COL_CO2: "mean", COL_TEMP: "mean", COL_EVENTS: "mean"})
)

country_risk_bar = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)
    .agg({
        COL_RISK: "mean",
        COL_CO2: "mean",
        COL_TEMP: "mean",
        COL_SEA: "mean",
        COL_EVENTS: "mean",
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
detail_alerts = int((detail_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı").sum())

detail_yearly = (
    detail_df.groupby(COL_YEAR, as_index=False)
    .agg({
        COL_RISK: "mean",
        COL_CO2: "mean",
        COL_TEMP: "mean",
        COL_EVENTS: "mean"
    })
    .sort_values(COL_YEAR)
)

most_risky_country = top_country.iloc[0][COL_COUNTRY] if len(top_country) > 0 else "-"
most_risky_score = round(top_country.iloc[0][COL_RISK], 1) if len(top_country) > 0 else 0
max_co2_value = round(filtered_df[COL_CO2].max(), 3) if not filtered_df.empty else 0

bubble_df = (
    filtered_df.groupby(COL_COUNTRY, as_index=False)
    .agg({
        COL_CO2: "mean",
        COL_RISK: "mean",
        COL_EVENTS: "mean"
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

if alert_count > 0 and show_alert:
    st.markdown(
        f"""
        <div class="alert-floating">
            <div class="alert-top">
                <div class="alert-title">Kritik İklim Uyarısı</div>
                <div class="alert-pill">Yüksek Öncelik</div>
            </div>
            <div class="alert-text">
                Filtrelenen sonuçlarda <strong>{alert_count}</strong> yüksek karbon emisyonu kaydı bulundu.
                En yüksek CO₂ değeri <strong>{max_co2_value}</strong>.
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
            <div class="hero-badge">🌍 İklim Zekâsı • Gerçek Zamanlı Çevresel Sinyaller</div>
            <div class="hero-title">İKLİM RİSK<br>GÖSTERGE PANELİ</div>
            <div class="hero-text">
                İklim riski, biyoçeşitlilik baskısı, okyanus stresi ve karbon dinamiklerini
                tek ekranda izleyebileceğiniz karar destek arayüzü.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Canlı İklim Göstergeleri", "Seçili filtrelere göre güncellenen temel çevresel sinyaller")
    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
    with k1:
        metric_kpi("🌡 Ortalama Sıcaklık", f"{avg_temp}", "İklim ısı eğilimi", KPI_TEMP)
    with k2:
        metric_kpi("🏭 Ortalama CO₂", f"{avg_co2}", "Karbon baskısı", KPI_CO2)
    with k3:
        metric_kpi("🌊 Deniz Seviyesi", f"{avg_sea}", "Okyanus yükselişi", KPI_SEA)
    with k4:
        metric_kpi("🌩 Aşırı Hava Olayı", f"{avg_events}", "İklim olay sıklığı", KPI_WEATHER)
    with k5:
        metric_kpi("📊 Risk Skoru", f"{avg_risk}", "Küresel risk endeksi", KPI_RISK)
    with k6:
        metric_kpi("🌿 Alarm Baskısı", f"{alert_count}", "Yüksek emisyon uyarıları", KPI_BIO)

    st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)

    section_header("Küresel Risk Görünümü", "Harita, yönetici özeti ve seçili ülke bazlı risk değerlendirmesi")
    map_col, info_col = st.columns([1.8, 1.0], gap="large")

    with map_col:
        chart_title("Küresel İklim Risk Haritası", "Ülkelerin ortalama risk skorları ve çevresel baskı göstergeleri")

        map_df = country_risk_map.copy().sort_values(COL_RISK, ascending=False)
        top_map_points = map_df.head(5).copy()

        fig_map = px.choropleth(
            map_df,
            locations=COL_COUNTRY,
            locationmode="country names",
            color=COL_RISK,
            hover_name=COL_COUNTRY,
            hover_data={
                COL_RISK: ':.1f',
                COL_CO2: ':.3f',
                COL_TEMP: ':.2f',
                COL_EVENTS: ':.0f'
            },
            color_continuous_scale=[
                [0.00, "#d8f3dc"],
                [0.20, "#95d5b2"],
                [0.40, "#52b788"],
                [0.60, "#ffd166"],
                [0.78, "#f8961e"],
                [1.00, "#d62828"],
            ],
        )

        fig_map.update_traces(
            marker_line_color="rgba(255,255,255,0.88)",
            marker_line_width=1.2,
            hovertemplate="""
            <b>%{hovertext}</b><br>
            Risk Skoru: %{z:.1f}<br>
            CO₂: %{customdata[1]:.3f}<br>
            Sıcaklık: %{customdata[2]:.2f}°C<br>
            Aşırı Olay: %{customdata[3]:.0f}<br>
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
                    size=10,
                    color="#10203a",
                    line=dict(width=2, color="white"),
                    opacity=0.95,
                ),
                textfont=dict(
                    size=10,
                    color="#10203a",
                    family="Inter, Segoe UI, sans-serif"
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig_map.update_layout(
            height=540,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#10203a", family="Inter, Segoe UI, sans-serif"),
            coloraxis_colorbar=dict(
                title=dict(text="Risk", font=dict(color="#10203a")),
                thickness=16,
                len=0.72,
                outlinewidth=0,
                tickfont=dict(color="#475569"),
            ),
            geo=dict(
                projection_type="natural earth",
                showframe=False,
                showcoastlines=True,
                coastlinecolor="rgba(100,116,139,0.55)",
                coastlinewidth=1.0,
                showcountries=True,
                countrycolor="rgba(148,163,184,0.55)",
                countrywidth=0.7,
                showland=True,
                landcolor="rgba(250,252,253,0.96)",
                showocean=True,
                oceancolor="rgba(226,238,248,0.88)",
                showlakes=True,
                lakecolor="rgba(226,238,248,0.88)",
                bgcolor="rgba(0,0,0,0)",
            ),
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="rgba(15,23,42,0.08)",
                font=dict(color="#10203a", size=13),
            ),
        )

        st.plotly_chart(fig_map, use_container_width=True)

    with info_col:
        info_panel(
            "Yönetici Özeti",
            most_risky_country,
            [
                f"Geçerli filtrelerde {most_risky_country} en dikkat çeken ülke olarak öne çıkıyor.",
                f"Ortalama risk skoru {most_risky_score} ve aktif yüksek emisyon sinyali {alert_count}."
            ],
            risk_etiketi(most_risky_score),
            risk_rengi(most_risky_score),
        )

        st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)

        info_panel(
            "Seçili Ülke",
            detail_country,
            [
                f"Ortalama sıcaklık: {detail_avg_temp}°C",
                f"CO₂ emisyonu: {detail_avg_co2}",
                f"Risk skoru: {detail_avg_risk}",
                f"Uyarı sayısı: {detail_alerts}",
            ],
            risk_etiketi(detail_avg_risk),
            risk_rengi(detail_avg_risk),
        )

        st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=detail_avg_risk,
                number={"font": {"size": 34}},
                title={"text": "İklim Risk Ölçeri", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#ef4444", "thickness": 0.28},
                    "steps": [
                        {"range": [0, 25], "color": "#22c55e"},
                        {"range": [25, 50], "color": "#eab308"},
                        {"range": [50, 75], "color": "#fb923c"},
                        {"range": [75, 100], "color": "#ef4444"},
                    ],
                },
            )
        )
        gauge.update_layout(
            height=270,
            margin=dict(l=10, r=10, t=28, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#10203a"},
        )
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)

    section_header("Trend Analitiği", "Zamana bağlı risk, karbon ve ülke karşılaştırmaları")
    row1_col1, row1_col2 = st.columns(2, gap="large")

    with row1_col1:
        chart_title("Risk Skoru Trend Grafiği", "Seçili ülkenin yıllara göre ortalama risk değişimi")
        fig_risk_trend = px.line(detail_yearly, x=COL_YEAR, y=COL_RISK, markers=True)
        fig_risk_trend.update_traces(line=dict(width=3.5, color="#f97316"), marker=dict(size=7))
        apply_chart_style(fig_risk_trend, height=340)
        fig_risk_trend.update_layout(xaxis_title="Yıl", yaxis_title="Risk")
        st.plotly_chart(fig_risk_trend, use_container_width=True)

    with row1_col2:
        chart_title("CO₂ Trend Grafiği", "Karbon emisyon yoğunluğunun yıllık hareketi")
        fig_co2_trend = px.area(detail_yearly, x=COL_YEAR, y=COL_CO2)
        fig_co2_trend.update_traces(line=dict(width=3, color="#0ea5e9"))
        apply_chart_style(fig_co2_trend, height=340)
        fig_co2_trend.update_layout(xaxis_title="Yıl", yaxis_title="CO₂")
        st.plotly_chart(fig_co2_trend, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2, gap="large")

    with row2_col1:
        chart_title(f"En Riskli Ülkeler (İlk {top_n})", f"Sıralama ölçütü: {selected_sort_label}")
        topn_df = country_risk_bar.head(top_n)
        fig_bar = px.bar(
            topn_df,
            x=COL_RISK,
            y=COL_COUNTRY,
            orientation="h",
            color=COL_RISK,
            color_continuous_scale="Plasma",
        )
        fig_bar.update_yaxes(categoryorder="total ascending")
        apply_chart_style(fig_bar, height=350)
        fig_bar.update_layout(xaxis_title="Risk", yaxis_title="Ülke")
        st.plotly_chart(fig_bar, use_container_width=True)

    with row2_col2:
        chart_title("CO₂ ve Risk İlişkisi", "Emisyon, risk ve olay yoğunluğu arasındaki dağılım ilişkisi")
        fig_scatter = px.scatter(
            bubble_df,
            x=COL_CO2,
            y=COL_RISK,
            size=COL_EVENTS,
            color=COL_RISK,
            hover_name=COL_COUNTRY,
            color_continuous_scale="Sunsetdark",
            size_max=42,
        )
        apply_chart_style(fig_scatter, height=350)
        fig_scatter.update_layout(xaxis_title="Ortalama CO₂", yaxis_title="Ortalama Risk")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)

    section_header("İklim Olay Analitiği", "Sıcaklık ve aşırı hava olayları için ek görünüm")
    row3_col1, row3_col2 = st.columns(2, gap="large")

    with row3_col1:
        chart_title("Sıcaklık Trend Grafiği", "Seçili ülkenin ortalama sıcaklık değişimi")
        fig_temp = px.line(detail_yearly, x=COL_YEAR, y=COL_TEMP, markers=True)
        fig_temp.update_traces(line=dict(width=3.2, color="#ef4444"), marker=dict(size=7))
        apply_chart_style(fig_temp, height=320)
        fig_temp.update_layout(xaxis_title="Yıl", yaxis_title="Sıcaklık")
        st.plotly_chart(fig_temp, use_container_width=True)

    with row3_col2:
        chart_title("Aşırı Hava Olayları Trend Grafiği", "Yıllara göre iklim olay sıklığındaki hareket")
        fig_events = px.bar(
            detail_yearly,
            x=COL_YEAR,
            y=COL_EVENTS,
            color=COL_EVENTS,
            color_continuous_scale="Bluered",
        )
        apply_chart_style(fig_events, height=320)
        fig_events.update_layout(
            xaxis_title="Yıl",
            yaxis_title="Olay Sayısı",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_events, use_container_width=True)

    st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)

    section_header("Uyarı Merkezi", "Yüksek karbon emisyonu sinyali taşıyan kayıtlar")
    chart_title("Karbon Emisyonu Uyarıları", "Filtrelenmiş veri içinde eşik değeri aşan kayıtların özet görünümü")

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