# scripts/generate_district_mapping.py
import pandas as pd
from pathlib import Path

DATA = Path("data/processed/final_cleaned_district_dataset.csv")
OUT = Path("data/processed/district_codes.csv")

df = pd.read_csv(DATA)
codes = sorted(df["District"].unique())

out_df = pd.DataFrame({
    "ADM2_PCODE": codes,
    "District_Name": [""] * len(codes)   # blank names for you to fill
})

OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT, index=False)
print("Wrote", OUT, "with", len(codes), "codes. Fill District_Name column.")
