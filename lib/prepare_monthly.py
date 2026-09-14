from pathlib import Path
import pandas as pd
from lib.get_data_permno import add_ranking_flag

DATA = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA / "QI_JT_2016-2025_raw_monthly.csv")
df = df[["PERMNO", "MthCalDt", "MthRet",'MthPrc','Ticker']]
raw_rows = len(df)

keys = ["PERMNO", "MthCalDt"]
assert df.groupby(keys)["MthRet"].nunique(dropna=False).le(1).all()
df = df.drop_duplicates(subset=keys)
duplicate_rows = raw_rows - len(df)

missing_returns = df["MthRet"].isna().sum()
missing_prices = df['MthPrc'].isna().sum()
df = df.dropna(subset=["MthRet", 'MthPrc'])

df = df.sort_values(keys).reset_index(drop=True)

df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
df["month"] = df["MthCalDt"].dt.to_period("M")

df = add_ranking_flag(df)
df.to_csv(DATA / "monthly_returns.csv", index=False, mode="w")

print(f"Raw rows: {raw_rows:,}")
print(f"Duplicate rows removed: {duplicate_rows:,}")
print(f"Missing return rows removed: {missing_returns:,}")
print(f"Missing price rows removed: {missing_prices:,}")
print(f"Ranking eligible rows: {df['eligible_6m'].sum():,}")
print(f"Ranking ineligible rows: {(~df['eligible_6m']).sum():,}")
print(f"Output rows: {len(df):,}; PERMNOs: {df['PERMNO'].nunique()}")
