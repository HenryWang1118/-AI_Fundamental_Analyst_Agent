import os
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from datetime import datetime

try:
    import yfinance as yf
except Exception:
    yf = None

from .config import Config


@dataclass
class MarketData:
    market_cap: Optional[float]
    stock_price: Optional[float]
    shares_outstanding: Optional[float]


def fetch_market_data(symbol: str) -> MarketData:
    """Fetch market data from Yahoo Finance."""
    if yf is None:
        return MarketData(None, None, None)
    t = yf.Ticker(symbol)
    info = getattr(t, "info", {}) or {}
    return MarketData(
        market_cap=info.get("marketCap"),
        stock_price=info.get("currentPrice"),
        shares_outstanding=info.get("sharesOutstanding"),
    )


def load_financial_statements(symbol: str) -> Dict[str, pd.DataFrame]:
    """Load financial statements from CSV files."""
    base = Config.DATA_PROCESSED_DIR
    s = symbol.lower()
    paths = {
        "income": os.path.join(base, f"{s}_income_statement.csv"),
        "balance": os.path.join(base, f"{s}_balance_sheet.csv"),
        "cashflow": os.path.join(base, f"{s}_cashflow_statement.csv"),
    }
    dfs: Dict[str, pd.DataFrame] = {}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing processed CSV: {p}")
        df = pd.read_csv(p)
        dfs[k] = df
    return dfs


def calculate_fcf(cashflow_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Free Cash Flow per year: FCF = Operating Cash Flow - Capital Expenditure."""
    df = cashflow_df.copy()
    ocf_col_candidates = ["operatingCashflow", "operating_cash_flow", "net_cash_provided_by_operating_activities"]
    capex_col_candidates = ["capitalExpenditures", "capital_expenditure", "capital_expenditures", "purchase_of_property_plant_and_equipment"]
    
    ocf_col = next((c for c in ocf_col_candidates if c in df.columns), None)
    capex_col = next((c for c in capex_col_candidates if c in df.columns), None)
    if ocf_col is None or capex_col is None:
        raise ValueError("Required cash flow columns not found for FCF calculation.")
    
    df["fcf"] = df[ocf_col] - df[capex_col]
    return df[["year", "fcf"]]


def _compute_cagr_from_fcf(fcf_series: pd.Series) -> float:
    """Compute CAGR from historical FCF series."""
    if fcf_series.empty:
        return 0.0
    df = fcf_series.reset_index(drop=True)
    pos_idx = [i for i, v in enumerate(df) if pd.notna(v) and v > 0]
    if len(pos_idx) < 2:
        return 0.0
    first_i, last_i = pos_idx[0], pos_idx[-1]
    first_val, last_val = float(df[first_i]), float(df[last_i])
    periods = last_i - first_i
    if periods <= 0 or first_val <= 0:
        return 0.0
    cagr = (last_val / first_val) ** (1.0 / periods) - 1.0
    return float(max(min(cagr, 0.5), -0.5))


def _project_fcf_with_cagr(last_fcf: float, cagr: float, years: int) -> List[float]:
    """Project FCF for given years using CAGR."""
    projections: List[float] = []
    base = max(last_fcf, 0.0)
    for t in range(1, years + 1):
        projections.append(base * ((1.0 + cagr) ** t))
    return projections


def _discount(values: List[float], discount_rate: float) -> List[float]:
    """Discount values to present value."""
    return [v / ((1.0 + discount_rate) ** t) for t, v in enumerate(values, start=1)]


def dcf_scenarios(
    historical_fcf: pd.Series,
    discount_rate: float,
    projection_years: int,
    terminal_growth_scenarios: Dict[str, float],
    cagr_override: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Build DCF projections and valuations."""
    if historical_fcf.empty:
        raise ValueError("Historical FCF is empty.")

    cagr = _compute_cagr_from_fcf(historical_fcf) if cagr_override is None else cagr_override
    last_fcf = float(historical_fcf.iloc[-1]) if pd.notna(historical_fcf.iloc[-1]) else 0.0
    proj_values = _project_fcf_with_cagr(last_fcf, cagr, projection_years)

    try:
        start_year = int(historical_fcf.index[-1])
        years = [start_year + i for i in range(1, projection_years + 1)]
    except Exception:
        years = list(range(1, projection_years + 1))
    
    projections_df = pd.DataFrame({"year": years, "projected_fcf": proj_values, "cagr": cagr})
    discounted_fcf = _discount(proj_values, discount_rate)
    pv_fcf = sum(discounted_fcf)

    valuations: Dict[str, Dict[str, float]] = {}
    last_proj = proj_values[-1]
    for name, g in terminal_growth_scenarios.items():
        if discount_rate <= g:
            valuations[name] = {"pv_fcf": pv_fcf, "pv_terminal": float("nan"), "enterprise_value": float("nan"), "g": g}
            continue
        terminal_value = last_proj * (1.0 + g) / (discount_rate - g)
        pv_terminal = terminal_value / ((1.0 + discount_rate) ** projection_years)
        enterprise_value = pv_fcf + pv_terminal
        valuations[name] = {
            "pv_fcf": pv_fcf,
            "pv_terminal": pv_terminal,
            "enterprise_value": enterprise_value,
            "g": g,
        }

    return projections_df, valuations


def multiples_valuation(income_df: pd.DataFrame, market: MarketData) -> Dict[str, Optional[float]]:
    """Compute P/E multiples valuation."""
    pe = None
    if market.stock_price and market.shares_outstanding and "net_income" in income_df.columns:
        latest = income_df.sort_values("year").iloc[-1]
        eps = latest["net_income"] / market.shares_outstanding if market.shares_outstanding else None
        pe = (market.stock_price / eps) if eps and eps != 0 else None
    return {"pe": pe, "market_cap": market.market_cap, "stock_price": market.stock_price}


def _safe_write_csv(path: str, df: pd.DataFrame, force_refresh: bool = True) -> str:
    """Write CSV with force refresh."""
    target = path
    if force_refresh and os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            base, ext = os.path.splitext(path)
            target = f"{base}_{ts}{ext}"
    df.to_csv(target, index=False)
    return target


def _extract_net_debt_and_shares(dfs: Dict[str, pd.DataFrame], market: MarketData) -> Tuple[Optional[float], Optional[float]]:
    """Extract net debt and shares outstanding for implied price calculation."""
    total_debt = None
    total_cash = None
    shares = None

    try:
        bal = dfs.get("balance")
        if bal is not None and not bal.empty:
            latest = bal.sort_values("year").iloc[-1]
            for c in ["totalDebt", "shortLongTermDebtTotal", "total_debt"]:
                if c in latest.index and pd.notna(latest[c]):
                    total_debt = float(latest[c])
                    break
            for c in ["totalCash", "cashAndCashEquivalentsAtCarryingValue", "total_cash"]:
                if c in latest.index and pd.notna(latest[c]):
                    total_cash = float(latest[c])
                    break
    except Exception:
        total_debt = None
        total_cash = None

    if market and market.shares_outstanding:
        shares = market.shares_outstanding
    else:
        if market and market.market_cap and market.stock_price:
            try:
                shares = market.market_cap / market.stock_price
            except Exception:
                shares = None

    net_debt = None
    if total_debt is not None:
        c = total_cash or 0.0
        net_debt = total_debt - c

    return net_debt, shares


def run_valuation(symbol: str = None, projection_years: int = 5, force_refresh: bool = True) -> Dict[str, Dict]:
    """Run DCF valuation with two scenarios: optimistic (WACC 7%) and pessimistic (WACC 9%)."""
    if symbol is None:
        symbol = Config.TARGET_COMPANY["symbol"]
    
    dfs = load_financial_statements(symbol)
    fcf_df = calculate_fcf(dfs["cashflow"]).sort_values("year")
    hist_series = fcf_df.set_index("year")["fcf"]
    
    # Two scenarios: optimistic and pessimistic
    scenarios = [
        {"name": "optimistic", "discount_rate": 0.079, "g": 0.035},
        {"name": "pessimistic", "discount_rate": 0.098, "g": 0.020},
    ]
    
    valuations = {}
    proj_df_list = []
    
    for sc in scenarios:
        proj_df_temp, vals_temp = dcf_scenarios(
            hist_series, 
            discount_rate=sc["discount_rate"], 
            projection_years=projection_years, 
            terminal_growth_scenarios={sc["name"]: sc["g"]}
        )
        proj_df_temp["scenario"] = sc["name"]
        proj_df_list.append(proj_df_temp)
        valuations[sc["name"]] = vals_temp.get(sc["name"], {})
    
    proj_df = pd.concat(proj_df_list, ignore_index=True) if proj_df_list else pd.DataFrame()

    # Fetch market data
    market = fetch_market_data(symbol)
    mult = multiples_valuation(dfs["income"], market)

    # Build result
    result = {
        "fcf": fcf_df.to_dict(orient="records"),
        "dcf_projections": proj_df.to_dict(orient="records"),
        "dcf_scenarios": valuations,
        "multiples": mult,
    }

    # Save projections CSV
    base_dir = Config.DATA_PROCESSED_DIR
    proj_path = os.path.join(base_dir, f"{symbol.lower()}_dcf_projections.csv")
    proj_path = _safe_write_csv(proj_path, proj_df, force_refresh=force_refresh)
    print(f"💾 Saved DCF projections → {proj_path}")

    # Build and save scenarios CSV with implied prices
    scen_rows = []
    for sc in scenarios:
        vals = valuations.get(sc["name"], {})
        scen_rows.append({
            "scenario": sc["name"],
            "discount_rate": sc["discount_rate"],
            "g": sc["g"],
            "pv_fcf": vals.get("pv_fcf"),
            "pv_terminal": vals.get("pv_terminal"),
            "enterprise_value": vals.get("enterprise_value"),
            "pe": mult.get("pe"),
            "market_cap": mult.get("market_cap"),
            "stock_price": mult.get("stock_price"),
            "implied_equity_value": None,
            "implied_price": None,
        })

    # Fill in implied prices
    try:
        net_debt, shares = _extract_net_debt_and_shares(dfs, market)
        for r in scen_rows:
            ev = r.get("enterprise_value")
            if ev is None or pd.isna(ev):
                continue
            implied_equity = ev if net_debt is None else (ev - net_debt)
            r["implied_equity_value"] = implied_equity
            if shares and shares != 0:
                r["implied_price"] = implied_equity / shares
            else:
                r["implied_price"] = None
    except Exception as e:
        print(f"⚠️ Could not calculate implied prices: {e}")

    # Save scenarios CSV
    scen_path = os.path.join(base_dir, f"{symbol.lower()}_dcf_scenarios.csv")
    scen_path = _safe_write_csv(scen_path, pd.DataFrame(scen_rows), force_refresh=force_refresh)
    print(f"💾 Saved DCF scenarios → {scen_path}")

    return result


if __name__ == "__main__":
    res = run_valuation()
    print("\n📈 DCF Valuation Result (preview):")
    print(res["dcf_scenarios"])
