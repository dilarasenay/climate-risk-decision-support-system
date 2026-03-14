import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="İklim Karar Destek Sistemi",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# DATA LOAD
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clean" / "climate_data_scored.csv"

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🌍 Filtre Paneli")

country_list = ["Tümü"] + sorted(df["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Ülke Seç", country_list)

threshold = st.sidebar.slider(
    "Karbon Emisyonu Eşik Değeri",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05
)

# Dinamik uyarı üret
df["dynamic_alert"] = df["co2_emissions_tons_capita"].apply(
    lambda x: "Yüksek Karbon Emisyonu Uyarısı" if x > threshold else "Normal Durum"
)

# Filtreleme
if selected_country != "Tümü":
    filtered_df = df[df["country"] == selected_country].copy()
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("Seçilen filtreye ait veri bulunamadı.")
    st.stop()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef6ff, #f8fbff);
    border-right: 1px solid rgba(15,23,42,0.08);
}

/* Hero */
.hero-box {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #e0f2fe 0%, #f0fdf4 45%, #dbeafe 100%);
    border-radius: 28px;
    padding: 34px 26px;
    box-shadow: 0 10px 30px rgba(15,23,42,0.08);
    border: 1px solid rgba(15,23,42,0.06);
    margin-bottom: 24px;
}
.hero-sun {
    position: absolute;
    top: 16px;
    right: 28px;
    font-size: 44px;
    opacity: 0.95;
}
.hero-leaf {
    position: absolute;
    left: 24px;
    bottom: 18px;
    font-size: 34px;
    opacity: 0.75;
}
.hero-bird {
    position: absolute;
    right: 110px;
    top: 22px;
    font-size: 24px;
    opacity: 0.75;
}
.hero-wave {
    position: absolute;
    bottom: 10px;
    right: 26px;
    font-size: 28px;
    opacity: 0.7;
}
.hero-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
}
.hero-subtitle {
    text-align: center;
    font-size: 18px;
    color: #334155;
}

/* KPI kartları */
.kpi-card {
    background: white;
    border-radius: 22px;
    padding: 18px;
    text-align: center;
    border: 1px solid rgba(15,23,42,0.07);
    box-shadow: 0 8px 24px rgba(15,23,42,0.06);
}
.kpi-title {
    color: #475569;
    font-size: 14px;
    margin-bottom: 10px;
}
.kpi-value {
    color: #0f172a;
    font-size: 34px;
    font-weight: 800;
}

/* Alert boxes */
.alert-red {
    background: linear-gradient(90deg, #fee2e2, #fecaca);
    color: #991b1b;
    padding: 18px;
    border-radius: 18px;
    font-size: 20px;
    font-weight: 800;
    text-align: center;
    border: 1px solid #fca5a5;
    margin-bottom: 18px;
}
.alert-green {
    background: linear-gradient(90deg, #dcfce7, #bbf7d0);
    color: #166534;
    padding: 18px;
    border-radius: 18px;
    font-size: 20px;
    font-weight: 800;
    text-align: center;
    border: 1px solid #86efac;
    margin-bottom: 18px;
}

/* Charts */
[data-testid="stPlotlyChart"] {
    background: white;
    border-radius: 22px;
    padding: 10px;
    border: 1px solid rgba(15,23,42,0.07);
    box-shadow: 0 8px 24px rgba(15,23,42,0.06);
}

/* Table */
[data-testid="stDataFrame"] {
    background: white;
    border-radius: 18px;
    padding: 8px;
    border: 1px solid rgba(15,23,42,0.07);
}

/* Section titles */
.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 12px;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(15,23,42,0.12), transparent);
    margin: 22px 0;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-sun">☀️</div>
    <div class="hero-bird">🕊️</div>
    <div class="hero-leaf">🌿🌳</div>
    <div class="hero-wave">🌊</div>

    <div class="hero-title">🌍 İklim Karar Destek Sistemi</div>
    <div class="hero-subtitle">
        Doğa, iklim riski ve karbon emisyonu göstergeleriyle karar desteği
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ALERT AREA
# --------------------------------------------------
if selected_country != "Tümü":
    alert_rows = filtered_df[filtered_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı"]

    if not alert_rows.empty:
        max_co2 = round(alert_rows["co2_emissions_tons_capita"].max(), 3)
        st.markdown(
            f"""
            <div class="alert-red">
                🚨 {selected_country} için yüksek karbon emisyonu tespit edildi — 
                En yüksek CO₂ değeri: {max_co2}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="alert-green">
                ✅ {selected_country} için kritik karbon emisyonu uyarısı bulunmuyor
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# KPI HESAPLARI
# --------------------------------------------------
avg_temp = round(filtered_df["avg_temperature_°c"].mean() * 100, 1)
avg_co2 = round(filtered_df["co2_emissions_tons_capita"].mean() * 100, 1)
avg_sea = round(filtered_df["sea_level_rise_mm"].mean() * 100, 1)
avg_events = round(filtered_df["extreme_weather_events"].mean() * 100, 1)
avg_risk = round(filtered_df["climate_risk_score"].mean(), 1)
alert_count = int((filtered_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı").sum())

def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    kpi_card("🌡 Ortalama Sıcaklık", avg_temp)
with k2:
    kpi_card("🏭 Ortalama CO₂", avg_co2)
with k3:
    kpi_card("🌊 Deniz Seviyesi", avg_sea)
with k4:
    kpi_card("⛈ Aşırı Hava Olayı", avg_events)
with k5:
    kpi_card("📊 Risk Skoru", avg_risk)
with k6:
    kpi_card("🚨 Uyarı Sayısı", alert_count)

st.markdown("---")

# --------------------------------------------------
# AGGREGATED DATA
# --------------------------------------------------
yearly = (
    filtered_df.groupby("year", as_index=False)
    .agg({
        "avg_temperature_°c": "mean",
        "co2_emissions_tons_capita": "mean",
        "climate_risk_score": "mean"
    })
    .sort_values("year")
)

country_risk_map = (
    df.groupby("country", as_index=False)
    .agg({
        "climate_risk_score": "mean",
        "co2_emissions_tons_capita": "mean"
    })
)

country_risk_bar = (
    filtered_df.groupby("country", as_index=False)["climate_risk_score"]
    .mean()
    .sort_values("climate_risk_score", ascending=False)
)

if selected_country == "Tümü":
    country_risk_bar = country_risk_bar.head(10)

alerts = (
    filtered_df[filtered_df["dynamic_alert"] == "Yüksek Karbon Emisyonu Uyarısı"]
    .sort_values("co2_emissions_tons_capita", ascending=False)
)

plot_bg = "white"
paper_bg = "white"
font_color = "#0f172a"

# --------------------------------------------------
# TOP ROW
# --------------------------------------------------
left, right = st.columns([1.2, 1])

with left:
    fig_temp = px.line(
        yearly,
        x="year",
        y="avg_temperature_°c",
        markers=True,
        title="🌡 Yıllara Göre Ortalama Sıcaklık Eğilimi"
    )
    fig_temp.update_traces(line=dict(width=4, color="#f97316"))
    fig_temp.update_layout(
        height=420,
        xaxis_title="Yıl",
        yaxis_title="Ortalama Sıcaklık",
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font_color=font_color
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with right:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_risk,
            title={"text": "🔥 İklim Risk Skoru"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#0f172a"},
                "bar": {"color": "#ef4444"},
                "steps": [
                    {"range": [0, 25], "color": "#22c55e"},
                    {"range": [25, 50], "color": "#eab308"},
                    {"range": [50, 75], "color": "#f97316"},
                    {"range": [75, 100], "color": "#ef4444"},
                ]
            }
        )
    )
    fig_gauge.update_layout(
        height=420,
        paper_bgcolor=paper_bg,
        font_color=font_color
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# MAP + CO2 ROW
# --------------------------------------------------
left2, right2 = st.columns(2)

with left2:
    fig_map = px.choropleth(
        country_risk_map,
        locations="country",
        locationmode="country names",
        color="climate_risk_score",
        hover_name="country",
        color_continuous_scale="YlOrRd",
        title="🗺 Dünya Haritası Üzerinde İklim Risk Skoru"
    )
    fig_map.update_layout(
        height=450,
        geo=dict(
            bgcolor="white",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#94a3b8",
            projection_type="natural earth"
        ),
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font_color=font_color
    )
    st.plotly_chart(fig_map, use_container_width=True)

with right2:
    fig_co2 = px.line(
        yearly,
        x="year",
        y="co2_emissions_tons_capita",
        markers=True,
        title="🏭 Yıllara Göre Ortalama CO₂ Eğilimi"
    )
    fig_co2.update_traces(line=dict(width=4, color="#38bdf8"))
    fig_co2.update_layout(
        height=450,
        xaxis_title="Yıl",
        yaxis_title="CO₂ Emisyonu",
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font_color=font_color
    )
    st.plotly_chart(fig_co2, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# BOTTOM ROW
# --------------------------------------------------
left3, right3 = st.columns([1, 1.2])

with left3:
    if selected_country == "Tümü":
        fig_bar = px.bar(
            country_risk_bar,
            x="climate_risk_score",
            y="country",
            orientation="h",
            color="climate_risk_score",
            color_continuous_scale="Sunsetdark",
            title="🚩 En Riskli 10 Ülke"
        )
        fig_bar.update_yaxes(categoryorder="total ascending")
        fig_bar.update_layout(
            xaxis_title="Risk Skoru",
            yaxis_title="Ülke"
        )
    else:
        fig_bar = px.bar(
            country_risk_bar,
            x="country",
            y="climate_risk_score",
            color="climate_risk_score",
            color_continuous_scale="Sunsetdark",
            title=f"📍 {selected_country} Risk Skoru"
        )
        fig_bar.update_layout(
            xaxis_title="Ülke",
            yaxis_title="Risk Skoru"
        )

    fig_bar.update_layout(
        height=420,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font_color=font_color
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with right3:
    st.markdown('<div class="section-title">🚨 Karbon Emisyonu Uyarıları</div>', unsafe_allow_html=True)

    if len(alerts) > 0:
        st.warning(f"{len(alerts)} kayıt için yüksek karbon emisyonu uyarısı bulundu.")

        alert_table = alerts[
            ["country", "year", "co2_emissions_tons_capita", "climate_risk_score", "dynamic_alert"]
        ].head(12).rename(
            columns={
                "country": "Ülke",
                "year": "Yıl",
                "co2_emissions_tons_capita": "CO₂ Emisyonu",
                "climate_risk_score": "Risk Skoru",
                "dynamic_alert": "Uyarı"
            }
        )

        st.dataframe(alert_table, use_container_width=True, hide_index=True)
    else:
        st.success("Kritik karbon emisyonu uyarısı bulunmuyor.")