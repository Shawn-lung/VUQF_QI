import pandas as pd
import numpy as np
import scipy.optimize as opt
from pathlib import Path
from lib.exit_handling import load_verified_exits, apply_otc_exits

class winner_portfolio:
    def __init__(self, winner_test, expected_return, prepare_csv=False):
        output_path = (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "optimal_weights_cap10.csv"
        )
        if output_path.exists() and not prepare_csv:
            self.weights = pd.read_csv(
                output_path,
                parse_dates=["entry_month"]
            )
        else:
            self.daily = pd.read_csv(
                Path(__file__).resolve().parent.parent / "data" / "daily_returns.csv",
                parse_dates=["DlyCalDt"]
            )

            all_weights = []

            for month in winner_test.index.unique():
                weights = self.del_assets(winner_test.loc[month], month)

                weights = weights.rename_axis("PERMNO").reset_index()
                weights["entry_month"] = month
                all_weights.append(weights)

            self.weights = pd.concat(all_weights, ignore_index=True)
            self.weights = self.weights[
                ["entry_month", "PERMNO", "weight"]
            ]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.weights.to_csv(output_path, index=False)

        data_dir = Path(__file__).resolve().parent.parent / "data"
        self.monthly = pd.read_csv(
            data_dir / "monthly_returns.csv",
            parse_dates=["MthCalDt"]
        )
        self.monthly["month"] = (
            self.monthly["MthCalDt"].dt.to_period("M").dt.to_timestamp()
        )
        self.exits = load_verified_exits(
            data_dir / "QI_JT_2016-2025_raw_monthly.csv",
            data_dir / "ivohlb3uqcydohyv.csv"
        )
        self.monthly, otc_exits = apply_otc_exits(
            self.monthly,
            data_dir / "QI_JT_2016-2025_raw_monthly.csv",
            data_dir / "zx5pdxpj17thwruq.csv"
        )
        self.exits = pd.concat([self.exits, otc_exits], ignore_index=True)

        self.monthly["month"] = self.monthly["month"].dt.to_period("M")
        self.exits["exit_month"] = self.exits["exit_month"].dt.to_period("M")

    def cal_covariance(self, df):
        returns = df.pivot(index='DlyCalDt', columns='PERMNO', values='DlyRet')
        covariance = returns.cov()
        return self.optimal_weights(covariance)

    def del_assets(self, winners, month):
        start = month - pd.DateOffset(years=2)

        train = self.daily.loc[
            (self.daily["DlyCalDt"] >= start)
            & (self.daily["DlyCalDt"] < month)
        ]

        df = train.loc[train["PERMNO"].isin(winners['PERMNO'])]
        #counts = df.groupby("PERMNO")["DlyRet"].count()
        #counts = counts.reindex(winners["PERMNO"], fill_value=0)
        n_days = train["DlyCalDt"].nunique()
        df_after = df.groupby("PERMNO").filter(
            lambda stock: stock["DlyRet"].count() == n_days
        )

        return self.cal_covariance(df_after)

    def portfolio_variance(self, weights, cov):
        return weights @ cov @ weights
    
    def optimal_weights(self, covariance):
        cov = covariance.to_numpy()
        n_assets = len(covariance)
        initial_weights = np.ones(n_assets) / n_assets
        res = opt.minimize(
            fun=self.portfolio_variance,
            x0=initial_weights,
            args=(cov,),
            method="SLSQP",
            bounds=[(0,0.1)] * n_assets,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-12, "maxiter": 1000}
        )
        if not res.success:
            raise RuntimeError(res.message)
        return pd.Series(
            res.x,
            index=covariance.columns,
            name="weight"
        )

    def cal_return(self, weights, month, capital):
        start = month.to_period("M")
        end = start + 6
        returns = self.monthly.loc[
            (self.monthly["month"] >=start)
            & (self.monthly["month"] < end)
        ]
        returns = returns.pivot(
            index="month",
            columns="PERMNO",
            values="MthRet"
        )
        returns = returns.reindex(columns=weights.index)
        for _, event in self.exits.iterrows():
            permno = event["PERMNO"]
            exit_month = event["exit_month"]

            if permno in returns.columns and start <= exit_month:
                returns.loc[returns.index > exit_month, permno] = 0
        if returns.isna().any().any():
            raise ValueError(f"{month}: missing return")
        gross_returns = 1 + returns
        growth = gross_returns.cumprod()
        asset_values = growth.mul(weights, axis="columns") * capital
        portfolio_values = asset_values.sum(axis=1)
        previous_values = portfolio_values.shift(1)
        previous_values.iloc[0] = capital
        result = pd.DataFrame({
            "begin_value": previous_values,
            "end_value": portfolio_values
        })

        result["profit"] = result["end_value"] - result["begin_value"]
        result["portfolio_return"] = result["profit"] / result["begin_value"]
        result["entry_month"] = month.to_period("M")

        return result.reset_index()


    def backtest(self, initial_capital, rf):
        capital = [initial_capital / 6 * rf**i for i in range(6)] 
        results = []

        for i, (month, df) in enumerate(
            self.weights.groupby("entry_month", sort=True)
        ):
            slot = i % 6
            weights = df.set_index("PERMNO")["weight"]

            result = self.cal_return(weights, month, capital[slot])
            results.append(result)

            capital[slot] = result["end_value"].iloc[-1]

        return pd.concat(results, ignore_index=True)

    def organize_output(self, capital, backtest_result, rf_gross, prepare_csv:bool = False):
        values = backtest_result.groupby("month")[["profit", "begin_value"]].sum()
        for i, month in enumerate(values.index[:5]):
            cash_begin = (5 - i) * capital / 6 * rf_gross**i

            values.loc[month, "begin_value"] += cash_begin
            values.loc[month, "profit"] += cash_begin * (rf_gross - 1)
        values["strategy_return"] = (
            values["profit"] / values["begin_value"]
        )
        values["cumulative_return"] = (
            (1 + values["strategy_return"]).cumprod() - 1
        )
        values["wealth"] = capital * (1 + values["cumulative_return"])
        returns = values["strategy_return"]
        n_months = len(returns)

        annualized_return = (
            (1 + returns).prod() ** (12 / n_months) - 1
        )

        annualized_volatility = returns.std() * np.sqrt(12)

        excess_returns = returns - (rf_gross - 1)
        sharpe_ratio = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(12)
        )
        summary = pd.Series({
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio
        }, name="value")

        if prepare_csv:
            output_path = (
                Path(__file__).resolve().parent.parent
                / "outputs"
                / "values.csv"
            )
            values.to_csv(output_path, index=True)
            summary.to_csv(output_path.parent / "performance_summary.csv")

        return values     