import pandas as pd
from pathlib import Path

# Dosya yolları
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "clean" / "climate_data_clean.csv"
OUTPUT_PATH = BASE_DIR / "data" / "clean" / "climate_data_scored.csv"

# Veriyi oku
df = pd.read_csv(INPUT_PATH)

# Gerekli kolonlar
required_cols = [
    "country",
    "avg_temperature_°c",
    "co2_emissions_tons_capita",
    "sea_level_rise_mm",
    "extreme_weather_events",
    "renewable_energy_",
    "forest_area_"
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Eksik kolonlar var: {missing_cols}")

# İklim risk skoru hesapla
# Pozitif etki edenler: sıcaklık, CO2, deniz seviyesi, aşırı hava olayları
# Riski azaltanlar: yenilenebilir enerji, orman alanı
df["climate_risk_score"] = (
    df["avg_temperature_°c"] * 0.25 +
    df["co2_emissions_tons_capita"] * 0.30 +
    df["sea_level_rise_mm"] * 0.15 +
    df["extreme_weather_events"] * 0.20 -
    df["renewable_energy_"] * 0.05 -
    df["forest_area_"] * 0.05
)

# Skoru 0-100 ölçeğine çek
min_score = df["climate_risk_score"].min()
max_score = df["climate_risk_score"].max()

if max_score != min_score:
    df["climate_risk_score"] = (
        (df["climate_risk_score"] - min_score) / (max_score - min_score) * 100
    )
else:
    df["climate_risk_score"] = 50

df["climate_risk_score"] = df["climate_risk_score"].round(2)

# Risk seviyesi
def classify_risk(score):
    if score >= 75:
        return "Kritik Risk"
    elif score >= 50:
        return "Yüksek Risk"
    elif score >= 25:
        return "Orta Risk"
    return "Düşük Risk"

df["risk_level"] = df["climate_risk_score"].apply(classify_risk)

# Uyarı mekanizması: CO2 per capita
THRESHOLD = 0.70  # veri normalize olduğu için 0-1 arası eşik
df["carbon_alert"] = df["co2_emissions_tons_capita"].apply(
    lambda x: "Yüksek Karbon Emisyonu Uyarısı" if x > THRESHOLD else "Normal Durum"
)

# Kaydet
df.to_csv(OUTPUT_PATH, index=False)

# Özet çıktı
print("Risk skoru hesaplandı ve dosya kaydedildi.")
print(f"Çıktı dosyası: {OUTPUT_PATH}")
print(df[["country", "climate_risk_score", "risk_level", "carbon_alert"]].head(10))