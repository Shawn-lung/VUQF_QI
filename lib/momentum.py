
import numpy as np
import pandas as pd
from lib.exit_handling import load_verified_exits, cohort_returns_with_cash, apply_otc_exits
from pathlib import Path

J = 6
K = 6
N_DECILES = 10


def load_monthly_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["MthCalDt"])
    df["month"] = df["MthCalDt"].values.astype("datetime64[M]")
    return df.sort_values(["PERMNO", "month"]).reset_index(drop=True)


def compute_formation_returns(df: pd.DataFrame, J: int = J) -> pd.DataFrame:
    """Trailing J-month compounded return, ending in (and including) each month."""
    df = df.copy()
    log_ret = np.log1p(df["MthRet"])
    trailing_log_ret = (
        log_ret.groupby(df["PERMNO"])
        .rolling(J, min_periods=J)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["formation_ret"] = np.expm1(trailing_log_ret)
    return df


def add_formation_status(df: pd.DataFrame, raw_path: str) -> pd.DataFrame:
    """Exclude recorded security endings from new formations, not earlier holdings."""
    raw = pd.read_csv(
        raw_path,
        usecols=["PERMNO", "SecurityEndDt", "MthRetFlg"],
        parse_dates=["SecurityEndDt"],
    )
    # SecurityEndDt alone can just be the dataset endpoint for an active stock.
    endings = raw.loc[
        raw["MthRetFlg"].eq("DE"), ["PERMNO", "SecurityEndDt"]
    ].drop_duplicates()
    if endings["SecurityEndDt"].isna().any() or endings["PERMNO"].duplicated().any():
        raise ValueError("Missing or conflicting security end dates; inspect raw event records.")
    end_dates = endings.set_index("PERMNO")["SecurityEndDt"]
    df = df.copy()
    df["recorded_end_date"] = df["PERMNO"].map(end_dates)
    formation_month = df["month"].dt.to_period("M")
    end_month = df["recorded_end_date"].dt.to_period("M")
    df["not_ended_at_formation"] = (
        df["recorded_end_date"].isna() | formation_month.lt(end_month)
    )
    return df


def assign_deciles(df: pd.DataFrame, n_deciles: int = N_DECILES) -> pd.DataFrame:
    
    df = df.copy()
    eligible = (
        df["eligible_6m"]
        & df["formation_ret"].notna()
        & df["not_ended_at_formation"]
    )
    df["decile"] = np.nan
    df.loc[eligible, "decile"] = (
        df.loc[eligible]
        .groupby("month")["formation_ret"]
        .transform(lambda x: pd.qcut(x, n_deciles, labels=False, duplicates="drop") + 1) 
    )
    return df

def build_overlapping_decile_returns(
    df: pd.DataFrame, K: int = K, exits=None, diagnostics_dir=None
) -> pd.DataFrame:
    vintages = df.loc[df["decile"].notna(), ["PERMNO", "month", "decile"]].rename(
        columns={"month": "vintage_month"}
    )
    returns = df[["PERMNO", "month", "MthRet"]]
    sample_end = returns["month"].max()

    vintage_month_returns = []
    for offset in range(1, K + 1):
        shifted = vintages.copy()
        shifted["month"] = shifted["vintage_month"] + pd.DateOffset(months=offset)
        # Future months beyond the dataset are outside this backtest.
        shifted = shifted.loc[shifted["month"] <= sample_end]
        merged = shifted.merge(
            returns,
            on=["PERMNO", "month"],
            how="left",
            validate="many_to_one",
        )
        vintage_month_returns.append(merged)
    panel = pd.concat(vintage_month_returns, ignore_index=True)

    panel["exit_month"] = pd.NaT
    if exits is not None:
        panel["exit_month"] = panel["PERMNO"].map(exits.set_index("PERMNO")["exit_month"])
        if panel["vintage_month"].eq(panel["exit_month"]).any():
            raise ValueError("A security entered a new cohort in its exit month.")
        # An OTC-exited PERMNO can later resume exchange trading. Its earlier
        # exit applies only to cohorts formed before that event.
        panel["exit_month"] = panel["exit_month"].where(panel["vintage_month"] < panel["exit_month"])
    panel["after_exit"] = panel["month"] > panel["exit_month"]
    otc_permnos = [] if exits is None or "delpaymenttype" not in exits else exits.loc[
        exits["delpaymenttype"].eq("OTC_FIRST_CLOSE"), "PERMNO"
    ]
    if (panel["after_exit"] & panel["MthRet"].notna() & ~panel["PERMNO"].isin(otc_permnos)).any():
        raise ValueError("Observed returns after a verified terminal event need review.")

    # Stop before mean() can silently drop a missing holding return.
    missing = panel.loc[panel["MthRet"].isna() & ~panel["after_exit"]]
    missing_months = (
        missing.groupby(["PERMNO", "month"]).size().reset_index(name="affected_vintages")
    )
    if diagnostics_dir is not None:
        from pathlib import Path
        diagnostic_path = Path(diagnostics_dir)
        diagnostic_path.mkdir(parents=True, exist_ok=True)
        missing_months.to_csv(diagnostic_path / "missing_holding_returns.csv", index=False)
        if exits is not None:
            exits.to_csv(diagnostic_path / "exit_events_audit.csv", index=False)
    if not missing.empty:
        print("Missing holding-period returns within the sample:")
        print(missing_months.to_string(index=False))
        raise ValueError(
            f"Missing returns for {len(missing_months)} stock-months "
            f"across {missing['PERMNO'].nunique()} stocks "
            f"({len(missing)} overlapping holding records). "
            "Investigate these returns before averaging; performance CSVs were not refreshed."
        )

    # equal-weighted return
    if exits is None:
        vintage_decile_month = (
            panel.groupby(["vintage_month", "decile", "month"])["MthRet"].mean().reset_index()
        )
    else:
        vintage_decile_month = cohort_returns_with_cash(panel)

    decile_month = (
        vintage_decile_month.groupby(["month", "decile"])["MthRet"].mean().unstack("decile")
    )
    decile_month.columns = [int(c) for c in decile_month.columns]
    return decile_month.sort_index(axis=1)


def summarise_table1(decile_month: pd.DataFrame) -> pd.DataFrame:
    avg = decile_month.mean()
    se = decile_month.std() / np.sqrt(decile_month.count())
    t_stat = avg / se

    spread = decile_month[N_DECILES] - decile_month[1]
    spread_avg = spread.mean()
    spread_t = spread_avg / (spread.std() / np.sqrt(spread.count()))

    table = pd.DataFrame(
        {
            "avg_monthly_return": avg,
            "annualised_return": (1 + avg) ** 12 - 1,
            "t_stat": t_stat,
            "n_months": decile_month.count(),
        }
    )
    table.loc["10-1 spread"] = [spread_avg, (1 + spread_avg) ** 12 - 1, spread_t, spread.count()]
    return table


def latest_formation_deciles(df: pd.DataFrame) -> pd.DataFrame:

    last_month = df["month"].max()
    latest = df.loc[df["month"] == last_month, ["PERMNO", "Ticker", "formation_ret", "decile", "eligible_6m", "not_ended_at_formation"]]
    return latest.rename(columns={"formation_ret": "formation_ret_latest", "decile": "decile_latest"})


def run_momentum(data_dir=None, output_dir=None):
    from pathlib import Path

    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent.parent / "data"
    raw_path = data_dir / "QI_JT_2016-2025_raw_monthly.csv"
    df = load_monthly_returns(data_dir / "monthly_returns.csv")
    df = add_formation_status(df, raw_path)
    exits = load_verified_exits(raw_path, data_dir / "ivohlb3uqcydohyv.csv")
    df, otc_exits = apply_otc_exits(df, raw_path, data_dir / "zx5pdxpj17thwruq.csv")
    exits = pd.concat([exits, otc_exits], ignore_index=True)
    df = compute_formation_returns(df, J=J)
    df = assign_deciles(df, n_deciles=N_DECILES)
    decile_month = build_overlapping_decile_returns(
        df, K=K, exits=exits, diagnostics_dir=output_dir
    )
    table = summarise_table1(decile_month)
    latest = latest_formation_deciles(df)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        monthly_output = decile_month.copy()
        monthly_output["10-1 spread"] = decile_month[N_DECILES] - decile_month[1]
        monthly_output.to_csv(output_dir / "decile_monthly_returns.csv", index_label="month")
        table.to_csv(output_dir / "table1_summary.csv", index_label="decile")
        latest.to_csv(output_dir / "latest_formation_deciles.csv", index=False)
    return decile_month, table, latest, df


def save_winners(df, output_path):
    winners = df.loc[
        df["decile"] == 10,
        ["month", "PERMNO", "Ticker", "formation_ret", "decile"]
    ].copy()

    winners = winners.rename(
        columns={"month": "formation_month"}
    )

    winners["entry_month"] = (
        winners["formation_month"] + pd.DateOffset(months=1)
    )
    winners = winners.set_index("entry_month")
    winners = winners.sort_index()

    winners_test = winners.loc["2021":"2025"]
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    winners_test.to_csv(
        output_path / "monthly_decile_10.csv",
        index=True
    )
    return winners_test


def save_expected_returns(monthly, output_dir):
    train = monthly.loc["2016":"2020"]
    expected_returns = train.mean()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_returns.rename("expected_monthly_return").to_csv(
        output_dir / "expected_returns_2016_2020.csv",
        index_label="decile",
    )
    return expected_returns


