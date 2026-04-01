import os
import pandas as pd
from dotenv import load_dotenv
from db import get_engine

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "clean", "climate_data_scored.csv")
TABLE_NAME = os.getenv("DB_TABLE", "climate_risk_data")

def main():
    print("CSV okunuyor...")
    df = pd.read_csv(CSV_PATH)

    print("Satir sayisi:", len(df))
    print("Kolon sayisi:", len(df.columns))

    engine = get_engine()

    df.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists="replace",
        index=False
    )

    print("Veri basariyla MySQL'e aktarildi.")
    print("Tablo adi:", TABLE_NAME)

if __name__ == "__main__":
    main()
