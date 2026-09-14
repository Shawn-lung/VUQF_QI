"""Small hand-calculated checks for missing returns and self-financing cash."""
import unittest

import numpy as np
import pandas as pd

from lib.exit_handling import cohort_returns_with_cash
from lib.momentum import build_overlapping_decile_returns


class ExitHandlingTests(unittest.TestCase):
    def panel(self, returns_a, returns_b, exit_a=None, exit_b=None):
        rows = []
        for month, a, b in zip(pd.date_range("2020-02-01", periods=len(returns_a), freq="MS"), returns_a, returns_b):
            for permno, ret, end in [(1, a, exit_a), (2, b, exit_b)]:
                end = pd.Timestamp(end) if end else pd.NaT
                rows.append([pd.Timestamp("2020-01-01"), 1, month, permno, ret, end, month > end])
        return pd.DataFrame(rows, columns=["vintage_month", "decile", "month", "PERMNO", "MthRet", "exit_month", "after_exit"])

    def test_no_exit_keeps_monthly_equal_weights(self):
        out = cohort_returns_with_cash(self.panel([.2, .3], [0, 0]))
        np.testing.assert_allclose(out.MthRet, [.1, .15])

    def test_cash_proceeds_are_not_reset_to_equal_weight(self):
        out = cohort_returns_with_cash(self.panel([1, np.nan, np.nan], [0, 1, 0], "2020-02-01"))
        np.testing.assert_allclose(out.MthRet, [.5, 1/3, 0])

    def test_all_exit_then_cash_only(self):
        out = cohort_returns_with_cash(self.panel([.2, np.nan], [-.2, np.nan], "2020-02-01", "2020-02-01"))
        np.testing.assert_allclose(out.MthRet, [0, 0], atol=1e-12)

    def test_active_missing_is_rejected(self):
        with self.assertRaises(ValueError):
            cohort_returns_with_cash(self.panel([np.nan], [.1]))

    def test_unobserved_future_is_not_a_missing_return(self):
        df = pd.DataFrame({"PERMNO": [1, 1], "month": pd.to_datetime(["2020-01-01", "2020-02-01"]), "MthRet": [0, .1], "decile": [1, np.nan]})
        out = build_overlapping_decile_returns(df, K=6)
        np.testing.assert_allclose(out[1], [.1])

    def test_terminal_missing_is_not_excused_by_event(self):
        df = pd.DataFrame({"PERMNO": [1, 1], "month": pd.to_datetime(["2020-01-01", "2020-02-01"]), "MthRet": [0, np.nan], "decile": [1, np.nan]})
        exits = pd.DataFrame({"PERMNO": [1], "exit_month": [pd.Timestamp("2020-02-01")]})
        with self.assertRaises(ValueError):
            build_overlapping_decile_returns(df, K=1, exits=exits)

    def test_relisting_does_not_reinvest_old_cash_but_allows_new_cohort(self):
        df = pd.DataFrame({"PERMNO": [1]*5,
            "month": pd.date_range("2020-01-01", periods=5, freq="MS"),
            "MthRet": [0, .2, np.nan, .5, .1], "decile": [1, np.nan, np.nan, 1, np.nan]})
        exits = pd.DataFrame({"PERMNO": [1], "exit_month": [pd.Timestamp("2020-02-01")],
                               "delpaymenttype": ["OTC_FIRST_CLOSE"]})
        out = build_overlapping_decile_returns(df, K=3, exits=exits)
        np.testing.assert_allclose(out[1], [.2, 0, 0, .1], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
