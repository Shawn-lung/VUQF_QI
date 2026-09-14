# Data

- `data2025.csv`: Original daily data provided by the lecturer.
- `ehxu8muioash8w4y.csv`: Original WRDS CRSP daily data for 2016–2025 and the same 521 PERMNOs, including `DlyRet` (daily total return), prices, previous-price dates and return flags. Intended for individual-stock covariance estimation; raw duplicates and missing values are retained.
- `daily_returns.csv`: Cleaned daily data from `prepare_daily.py`, with exact duplicates, missing returns and the two `P1` returns spanning multiple trading days removed. Columns: `PERMNO, DlyCalDt, DlyRet, DlyPrc, Ticker, month`, ordered like the monthly data; `month` is YYYY-MM. `DlyRet` is a daily total return in decimal form; 0.01 means 1%. CRSP flags remain in the raw file; the six-month ranking flag belongs to the monthly data.
- `QI_JT_2016-2025_raw_monthly.csv`: Original CRSP monthly data downloaded from WRDS for 2016–2025, using the 521 PERMNOs in the lecturer's data.
- `monthly_returns.csv`: Cleaned monthly data with duplicate stock-month records and rows missing return or price removed. Includes the six-month ranking eligibility flag.
- `ivohlb3uqcydohyv.csv`: Original WRDS CRSP delisting extract: 12 exit events, consideration types, values and dates.
- `zx5pdxpj17thwruq.csv`: Original WRDS OTC daily pricing extract queried for SMCI and WGHTQ; prices require date and corporate-action checks before use as returns.

## Columns in monthly_returns.csv

| Column | Meaning |
| --- | --- |
| `PERMNO` | CRSP security identifier. |
| `MthCalDt` | Monthly observation date. |
| `MthRet` | Monthly total return, including dividends; 0.05 means 5%. |
| `MthPrc` | CRSP monthly price, not an adjusted-price series. |
| `Ticker` | Stock ticker. |
| `month` | Calendar month in YYYY-MM format. |
| `eligible_6m` | True if the current month and previous five months have consecutive, nonmissing returns; otherwise False.|
