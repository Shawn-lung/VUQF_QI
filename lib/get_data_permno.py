from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def add_ranking_flag(df):
    """Flag six consecutive nonmissing return months, including the current month."""
    df = df.copy()
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    df["month"] = df["MthCalDt"].dt.to_period("M")
    df = df.sort_values(["PERMNO", "month"]).reset_index(drop=True)
    if df.duplicated(["PERMNO", "month"]).any():
        raise ValueError("Remove duplicate stock-month records before flagging.")

    month_number = df["month"].astype("int64")
    fifth_previous = month_number.groupby(df["PERMNO"]).shift(5)
    six_consecutive_months = (month_number - fifth_previous).eq(5)

    six_valid_returns = df.groupby("PERMNO")["MthRet"].transform(
        lambda s: s.notna().rolling(6, min_periods=6).sum().eq(6)
    )
    df["eligible_6m"] = six_consecutive_months & six_valid_returns
    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA / "data2025.csv", usecols=["PERMNO"])
    permnos = sorted(df["PERMNO"].unique())
    print(" ".join(map(str, permnos)))