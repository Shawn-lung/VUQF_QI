from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"

# Keep the CRSP flag "NA" as text; blank returns are missing values.
df = pd.read_csv(DATA / "ehxu8muioash8w4y.csv", dtype=str, keep_default_na=False)
raw_rows = len(df)

keys = ["PERMNO", "DlyCalDt"]
df = df.drop_duplicates()
duplicate_rows = raw_rows - len(df)
if df.duplicated(keys).any():
    raise ValueError("Conflicting records for the same stock and date.")

df["PERMNO"] = pd.to_numeric(df["PERMNO"], errors="raise").astype(int)
df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"], errors="raise")
df["DlyPrevDt"] = pd.to_datetime(df["DlyPrevDt"], errors="raise")
df["DlyRet"] = pd.to_numeric(df["DlyRet"].replace("", np.nan), errors="raise")
df["DlyPrc"] = pd.to_numeric(df["DlyPrc"].replace("", np.nan), errors="raise")

missing_returns = df["DlyRet"].isna().sum()
df = df.dropna(subset=["DlyRet"])
if not (np.isfinite(df["DlyRet"]) & df["DlyRet"].ge(-1)).all():
    raise ValueError("Invalid total return; check the original data.")


multi_day_returns = df["DlyRetDurFlg"].eq("P1")
multi_day_rows = multi_day_returns.sum()
df = df.loc[~multi_day_returns].copy()

df = df.sort_values(keys).reset_index(drop=True)
df = df[["PERMNO", "DlyCalDt", "DlyRet", "DlyPrc", "Ticker"]]
df["month"] = df["DlyCalDt"].dt.to_period("M")
df.to_csv(DATA / "daily_returns.csv", index=False)

print(f"Raw rows: {raw_rows:,}")
print(f"Duplicate rows removed: {duplicate_rows:,}")
print(f"Missing return rows removed: {missing_returns:,}")
print(f"Multi-trading-day return rows removed: {multi_day_rows:,}")
print(f"Output rows: {len(df):,}; PERMNOs: {df['PERMNO'].nunique()}")
