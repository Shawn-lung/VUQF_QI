
import numpy as np
import pandas as pd


def apply_otc_exits(df, raw_path, otc_path):
    """Liquidate WW/SMCI at the first OTC close after Nasdaq suspension.

Only the working return panel is amended; original/cleaned input CSVs remain
unchanged. These two dates/identifiers were checked against issuer announcements
and the downloaded OTC extract. Later eligible post-relisting cohorts can enter.
"""
    raw = pd.read_csv(raw_path)
    otc = pd.read_csv(otc_path, parse_dates=["closingbestbiddate"])
    df = df.copy()
    records = []
    cases = [(89244, "WGHTQ", "98262P101", "2025-05-16"),
             (91907, "SMCI", "86800U104", "2018-08-23")]
    for permno, symbol, cusip, date in cases:
        quotes = otc.loc[otc["symbol"].eq(symbol)].sort_values("closingbestbiddate")
        if quotes.empty or quotes["closingbestbiddate"].duplicated().any():
            raise ValueError(f"Missing/duplicate OTC quotes for {symbol}.")
        first = quotes.iloc[0]
        if (first["closingbestbiddate"] != pd.Timestamp(date)
                or str(first["cusip"]) != cusip
                or not np.isfinite(first["lastprice"]) or first["lastprice"] <= 0
                or not first["sharevolume"] > 0):
            raise ValueError(f"First OTC close needs review for {symbol}.")
        month = pd.Timestamp(date).to_period("M").to_timestamp()
        raw_month = pd.to_datetime(raw["MthCalDt"]).dt.to_period("M")
        source = raw.loc[raw["PERMNO"].eq(permno) & raw_month.eq(month.to_period("M"))].drop_duplicates()
        selected = df["PERMNO"].eq(permno) & df["month"].eq(month)
        if len(source) != 1 or selected.sum() != 1:
            raise ValueError(f"Expected one partial CRSP month for {symbol}.")
        row = source.iloc[0]
        if (row["MthRetFlg"] != "IP" or row["MthDelFlg"] != "N"
                or not np.isfinite(row["MthPrc"]) or row["MthPrc"] <= 0
                or not np.isfinite(row["MthRet"]) or row["MthRet"] < -1):
            raise ValueError(f"CRSP partial-return bridge needs review for {symbol}.")
        # No distribution/share-unit change between the last Nasdaq close and
        # first OTC close in these reviewed cases. Later OTC months are not used.
        terminal_return = (1 + row["MthRet"]) * first["lastprice"] / row["MthPrc"] - 1
        df.loc[selected, "MthRet"] = terminal_return
        df.loc[selected, "not_ended_at_formation"] = False
        records.append({"PERMNO": permno, "exit_month": month, "deldlydt": pd.Timestamp(date),
                        "delpaymenttype": "OTC_FIRST_CLOSE", "exit_value": first["lastprice"],
                        "partial_crsp_price": row["MthPrc"], "original_MthRet": row["MthRet"],
                        "MthRet": terminal_return, "value_in_later_month": False,
                        "policy": "First OTC close liquidation; 0% cash until cohort expiry"})
    return df, pd.DataFrame(records)


def load_verified_exits(raw_path, event_path):
    """Match event evidence to terminal monthly rows; never add DelRet again."""
    raw = pd.read_csv(raw_path, keep_default_na=False)
    events = pd.read_csv(event_path, keep_default_na=False).rename(columns={"permno": "PERMNO"})
    if events["PERMNO"].duplicated().any():
        raise ValueError("Multiple exit events per PERMNO need individual review.")
    terminal = raw.loc[raw["MthRetFlg"].eq("DE"), [
        "PERMNO", "SecurityEndDt", "MthCalDt", "MthRet", "MthDelFlg"
    ]].drop_duplicates()
    if terminal["PERMNO"].duplicated().any():
        raise ValueError("Conflicting terminal monthly rows.")
    audit = events.merge(terminal, on="PERMNO", how="left", validate="one_to_one")
    for column in ["delistingdt", "deldlydt", "SecurityEndDt", "MthCalDt"]:
        audit[column] = pd.to_datetime(audit[column], errors="raise")
    for column in ["delret", "deldtprc", "delnextprc", "deldivamt", "MthRet"]:
        audit[column] = pd.to_numeric(audit[column], errors="coerce")
    audit["exit_value"] = np.where(
        audit["delpaymenttype"].eq("PRCF"), audit["delnextprc"], audit["deldivamt"]
    )
    known_type = audit["delpaymenttype"].isin(["CASH", "STK", "CST", "PRCF"])
    valid = (
        known_type
        & audit["delistingdt"].eq(audit["SecurityEndDt"])
        & audit["delistingdt"].dt.to_period("M").eq(audit["MthCalDt"].dt.to_period("M"))
        & audit["deldlydt"].ge(audit["delistingdt"])
        & audit["MthDelFlg"].isin(["A", "P"])
        & audit["delretmisstype"].eq("NA")
        & audit["MthRet"].ge(-1) & np.isfinite(audit["MthRet"])
        & audit["deldtprc"].gt(0) & audit["exit_value"].ge(0)
        & np.isclose(audit["exit_value"] / audit["deldtprc"] - 1,
                     audit["delret"], atol=2e-6, rtol=0)
    )
    if not valid.all():
        raise ValueError(f"Exit evidence needs review: {audit.loc[~valid, 'PERMNO'].tolist()}")
    audit["exit_month"] = audit["MthCalDt"].dt.to_period("M").dt.to_timestamp()
    audit["value_in_later_month"] = (
        audit["deldlydt"].dt.to_period("M") > audit["MthCalDt"].dt.to_period("M")
    )
    audit["policy"] = "CRSP monthly exit value to 0% cash/receivable; no reinvestment"
    return audit


def cohort_returns_with_cash(panel):
    """Self-financing monthly cohorts; missing active returns must be checked first."""
    records = []
    for (vintage, decile), cohort in panel.groupby(["vintage_month", "decile"]):
        risky, cash = 1.0, 0.0
        for month, holdings in cohort.groupby("month", sort=True):
            before = risky + cash
            if before <= 0:
                raise ValueError("A cohort lost all wealth; subsequent percentage returns are undefined.")
            active = holdings.loc[~holdings["after_exit"]]
            if active.empty:
                if not np.isclose(risky, 0):
                    raise ValueError("Risky capital disappeared without a terminal return.")
            else:
                returns = active["MthRet"].to_numpy(dtype=float)
                if not np.isfinite(returns).all() or (returns < -1).any():
                    raise ValueError("Invalid active holding return; cannot value cohort.")
                values = (risky / len(active)) * (1 + returns)
                exiting = active["exit_month"].eq(month).to_numpy()
                cash += values[exiting].sum()
                risky = values[~exiting].sum()
            after = risky + cash
            records.append((vintage, decile, month, after / before - 1))
    return pd.DataFrame(records, columns=["vintage_month", "decile", "month", "MthRet"])
