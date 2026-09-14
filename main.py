from pathlib import Path
import pandas as pd
from lib.momentum import run_momentum, save_winners, save_expected_returns
from lib.optimization import winner_portfolio
from lib.export_portfolio import export_to_excel
if __name__=="__main__":
    output_dir = Path(__file__).resolve().parent / "outputs"
    capital = 10000000
    rf_monthly = 1.01**(1/12)
    monthly, summary, latest, df = run_momentum()
    prepare_csv = False
    if prepare_csv:
        save_winners(df, output_dir)
        save_expected_returns(monthly, output_dir)

    winners_test = pd.read_csv(
        output_dir / "monthly_decile_10.csv",
        index_col="entry_month",
        parse_dates=["entry_month", "formation_month"]
    )

    expected_returns = pd.read_csv(
        output_dir / "expected_returns_2016_2020.csv",
        index_col="decile"
    )["expected_monthly_return"]

    portfolio = winner_portfolio(winners_test, expected_returns, prepare_csv)

    backtest = portfolio.backtest(capital, rf_monthly)
    values = portfolio.organize_output(capital, backtest, rf_monthly, prepare_csv=False)
    stock_columns = portfolio.weights["PERMNO"].unique()
    last_month = backtest.loc[
        backtest["month"] == "2025-12",
        stock_columns
    ]
    stock_values = last_month.sum()
    final_weights = stock_values/stock_values.sum()
    test = export_to_excel(final_weights, capital)
    #print(final_weights)