"""
financial_ratios.py
--------------------------------------
Financial Ratios Calculation Module for Fundamental Analyst Agent
Company: Apple Inc. (AAPL)
Years: 2020–2024
--------------------------------------
Functions:
1. Load financial data from /data/processed/ CSV files
2. Calculate key financial ratios:
   - Profitability: Gross Margin, Net Margin, ROE, ROA
   - Liquidity: Current Ratio, Quick Ratio
   - Solvency: Debt-to-Equity, Debt-to-Assets
   - Efficiency: Asset Turnover
   - Cash Flow: Free Cash Flow, OCF to Net Income
3. Save calculated ratios to /data/processed/
--------------------------------------
Usage:
    python src/financial_ratios.py
--------------------------------------
"""

import os
import pandas as pd
from .config import Config
from .data_ingestion import run_ingestion_pipeline

PEER_SYMBOLS = ["MSFT", "AAPL", "GOOGL", "AMZN", "IBM", "ORCL", "CRM", "ADBE", "INTC", "CSCO", "SAP"]


# -------------------------------
# 1. LOAD FINANCIAL DATA
# -------------------------------
def load_financial_statements(symbol=None, verbose=True):
    """
    Load financial statements from CSV files
    
    Args:
        symbol: Stock symbol (default: from Config.TARGET_COMPANY)
        verbose: Whether to print loading status (default: True)
    
    Returns:
        tuple: (income_df, balance_df, cashflow_df)
    """
    if symbol is None:
        symbol = Config.TARGET_COMPANY["symbol"].lower()
    else:
        symbol = symbol.lower()
    
    # File paths
    income_path = os.path.join(Config.DATA_PROCESSED_DIR, f"{symbol}_income_statement.csv")
    balance_path = os.path.join(Config.DATA_PROCESSED_DIR, f"{symbol}_balance_sheet.csv")
    cashflow_path = os.path.join(Config.DATA_PROCESSED_DIR, f"{symbol}_cashflow_statement.csv")
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(income_path):
        missing_files.append("income_statement.csv")
    if not os.path.exists(balance_path):
        missing_files.append("balance_sheet.csv")
    if not os.path.exists(cashflow_path):
        missing_files.append("cashflow_statement.csv")
    
    if missing_files:
        if verbose:
            print(f"⚠️  Missing required files: {', '.join(missing_files)}")
            print(f"🔄 Automatically fetching data from Alpha Vantage...\n")
        
        # 自动运行数据获取
        success = run_ingestion_pipeline(symbol=symbol.upper(), verbose=verbose, force_refresh=False)
        
        if not success:
            raise FileNotFoundError(
                f"❌ Failed to fetch financial data.\n"
                f"   Please check your API key and network connection."
            )
        
        if verbose:
            print("")  # 空行分隔
    
    # Load data
    if verbose:
        print(f"📂 Loading financial statements for {symbol.upper()}...")
    
    income_df = pd.read_csv(income_path)
    balance_df = pd.read_csv(balance_path)
    cashflow_df = pd.read_csv(cashflow_path)
    
    if verbose:
        print(f"   ✅ Loaded {len(income_df)} years of data")
        print(f"      - Income Statement: {len(income_df.columns)} columns")
        print(f"      - Balance Sheet: {len(balance_df.columns)} columns")
        print(f"      - Cash Flow: {len(cashflow_df.columns)} columns")
    
    return income_df, balance_df, cashflow_df


# -------------------------------
# 2. CALCULATE FINANCIAL RATIOS
# -------------------------------
def calculate_profitability_ratios(income_df, balance_df):
    """
    Calculate profitability ratios
    
    Returns:
        pandas.DataFrame: Profitability ratios by year
    """
    ratios = pd.DataFrame()
    ratios['year'] = income_df['year']
    
    # Gross Margin = Gross Profit / Revenue
    ratios['gross_margin'] = (income_df['grossProfit'] / income_df['totalRevenue']) * 100
    
    # Net Margin = Net Income / Revenue
    ratios['net_margin'] = (income_df['netIncome'] / income_df['totalRevenue']) * 100
    
    # Operating Margin = Operating Income / Revenue
    ratios['operating_margin'] = (income_df['operatingIncome'] / income_df['totalRevenue']) * 100
    
    # ROE = Net Income / Total Shareholders Equity
    ratios['roe'] = (income_df['netIncome'] / balance_df['totalShareholderEquity']) * 100
    
    # ROA = Net Income / Total Assets
    ratios['roa'] = (income_df['netIncome'] / balance_df['totalAssets']) * 100
    
    return ratios


def calculate_liquidity_ratios(balance_df):
    """
    Calculate liquidity ratios
    
    Returns:
        pandas.DataFrame: Liquidity ratios by year
    """
    ratios = pd.DataFrame()
    ratios['year'] = balance_df['year']
    
    # Current Ratio = Current Assets / Current Liabilities
    ratios['current_ratio'] = balance_df['totalCurrentAssets'] / balance_df['totalCurrentLiabilities']
    
    # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
    ratios['quick_ratio'] = (
        (balance_df['totalCurrentAssets'] - balance_df['inventory']) / 
        balance_df['totalCurrentLiabilities']
    )
    
    # Cash Ratio = Cash / Current Liabilities
    ratios['cash_ratio'] = balance_df['cashAndCashEquivalentsAtCarryingValue'] / balance_df['totalCurrentLiabilities']
    
    return ratios


def calculate_solvency_ratios(balance_df):
    """
    Calculate solvency ratios
    
    Returns:
        pandas.DataFrame: Solvency ratios by year
    """
    ratios = pd.DataFrame()
    ratios['year'] = balance_df['year']
    
    # Debt-to-Equity = Total Liabilities / Total Equity
    ratios['debt_to_equity'] = balance_df['totalLiabilities'] / balance_df['totalShareholderEquity']
    
    # Debt-to-Assets = Total Liabilities / Total Assets
    ratios['debt_to_assets'] = (balance_df['totalLiabilities'] / balance_df['totalAssets']) * 100
    
    # Equity Ratio = Total Equity / Total Assets
    ratios['equity_ratio'] = (balance_df['totalShareholderEquity'] / balance_df['totalAssets']) * 100
    
    return ratios


def calculate_efficiency_ratios(income_df, balance_df):
    """
    Calculate efficiency ratios
    
    Returns:
        pandas.DataFrame: Efficiency ratios by year
    """
    ratios = pd.DataFrame()
    ratios['year'] = income_df['year']
    
    # Asset Turnover = Revenue / Total Assets
    ratios['asset_turnover'] = income_df['totalRevenue'] / balance_df['totalAssets']
    
    # Inventory Turnover = Cost of Revenue / Inventory
    ratios['inventory_turnover'] = income_df['costOfRevenue'] / balance_df['inventory']
    
    return ratios


def calculate_cashflow_ratios(income_df, cashflow_df):
    """
    Calculate cash flow ratios
    
    Returns:
        pandas.DataFrame: Cash flow ratios by year
    """
    ratios = pd.DataFrame()
    ratios['year'] = cashflow_df['year']
    
    # Free Cash Flow = Operating Cash Flow - Capital Expenditures
    ratios['free_cash_flow'] = (
        cashflow_df['operatingCashflow'] + cashflow_df['capitalExpenditures']
    )  # capitalExpenditures is negative in Alpha Vantage
    
    # OCF to Net Income = Operating Cash Flow / Net Income
    ratios['ocf_to_net_income'] = cashflow_df['operatingCashflow'] / income_df['netIncome']
    
    # Cash Flow Margin = Operating Cash Flow / Revenue
    ratios['cash_flow_margin'] = (cashflow_df['operatingCashflow'] / income_df['totalRevenue']) * 100
    
    return ratios


def add_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add YoY growth/decline rates (% change) for all numeric metrics (excluding year)."""
    df = df.sort_values("year").reset_index(drop=True)
    growth_df = df.copy()
    numeric_cols = [c for c in df.columns if c != "year"]
    for col in numeric_cols:
        growth_col = f"{col}_growth_pct"
        # Fix FutureWarning: specify fill_method=None
        growth_df[growth_col] = df[col].pct_change(fill_method=None) * 100
    return growth_df


# -------------------------------
# 3. COMBINE AND SAVE
# -------------------------------
def calculate_all_ratios(symbol=None, verbose=True):
    """
    Calculate all financial ratios and combine into single DataFrame
    
    Args:
        symbol: Stock symbol (default: from Config.TARGET_COMPANY)
        verbose: Whether to print progress (default: True)
    
    Returns:
        pandas.DataFrame: Combined financial ratios
    """
    # 加载数据
    income_df, balance_df, cashflow_df = load_financial_statements(symbol, verbose)
    
    if verbose:
        print("\n📊 Calculating financial ratios...")
    
    # 计算各类比率
    profitability = calculate_profitability_ratios(income_df, balance_df)
    liquidity = calculate_liquidity_ratios(balance_df)
    solvency = calculate_solvency_ratios(balance_df)
    efficiency = calculate_efficiency_ratios(income_df, balance_df)
    cashflow = calculate_cashflow_ratios(income_df, cashflow_df)
    
    # 合并所有比率
    all_ratios = profitability.merge(liquidity, on='year') \
                              .merge(solvency, on='year') \
                              .merge(efficiency, on='year') \
                              .merge(cashflow, on='year')

    # 增长率
    all_ratios = add_growth_rates(all_ratios)

    # 四舍五入
    numeric_cols = all_ratios.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    all_ratios[numeric_cols] = all_ratios[numeric_cols].round(2)
    
    if verbose:
        print(f"   ✅ Calculated {len(all_ratios.columns) - 1} financial ratios")
        print(f"      Years: {all_ratios['year'].min()} - {all_ratios['year'].max()}")
    
    return all_ratios


def save_ratios(ratios_df, symbol=None, verbose=True):
    """
    Save financial ratios to CSV
    
    Args:
        ratios_df: DataFrame with calculated ratios
        symbol: Stock symbol (default: from Config.TARGET_COMPANY)
        verbose: Whether to print save confirmation (default: True)
    """
    if symbol is None:
        symbol = Config.TARGET_COMPANY["symbol"].lower()
    else:
        symbol = symbol.lower()
    
    filename = f"{symbol}_financial_ratios.csv"
    filepath = os.path.join(Config.DATA_PROCESSED_DIR, filename)
    
    ratios_df.to_csv(filepath, index=False)
    
    if verbose:
        print(f"\n💾 Saved ratios → {filepath}")


# -------------------------------
# 4. MAIN PIPELINE
# -------------------------------
def run_ratio_calculation_pipeline(symbol=None, verbose=True):
    """
    Execute the complete ratio calculation pipeline
    
    Args:
        symbol: Stock symbol (default: from Config.TARGET_COMPANY)
        verbose: Whether to print detailed progress (default: True)
    
    Returns:
        pandas.DataFrame: Calculated financial ratios
    """
    if symbol is None:
        symbol = Config.TARGET_COMPANY["symbol"]
        company_name = Config.TARGET_COMPANY.get("name", symbol)
    else:
        company_name = symbol
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🧮 Calculating Financial Ratios for {company_name} ({symbol})")
        print(f"{'='*60}")
    
    try:
        # 计算比率
        ratios = calculate_all_ratios(symbol, verbose)
        
        # 保存结果
        save_ratios(ratios, symbol, verbose)
        
        if verbose:
            print(f"\n{'='*60}")
            print("✅ Ratio Calculation Complete!")
            print(f"{'='*60}")
            print(f"\n📈 Summary Statistics:")
            print(ratios.describe().round(2))
            print(f"\n💡 Next Step: Generate financial analysis report")
            print(f"{'='*60}\n")
        
        return ratios
        
    except FileNotFoundError as e:
        if verbose:
            print(f"\n{str(e)}")
        return None
    except Exception as e:
        if verbose:
            print(f"\n❌ Error during ratio calculation: {str(e)}")
        raise


# -------------------------------
# 5. EXECUTION
# -------------------------------
if __name__ == "__main__":
    ratios_df = run_ratio_calculation_pipeline()
    exit(0 if ratios_df is not None else 1)


def run_ratios_for_symbols(symbols=None, verbose=True, force_refresh=False):
    """Batch calculate ratios for a list of symbols (default: MSFT + peers)."""
    if symbols is None:
        symbols = PEER_SYMBOLS

    combined = []
    target_only = None

    for idx, sym in enumerate(symbols):
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 Ratios for {sym} ({idx+1}/{len(symbols)})")
            print(f"{'='*60}")
        try:
            ratios = calculate_all_ratios(sym, verbose=verbose)
            ratios["symbol"] = sym.upper()
            combined.append(ratios)
            if sym.upper() == Config.TARGET_COMPANY["symbol"].upper():
                target_only = ratios.copy()
            save_ratios(ratios, sym, verbose=verbose)
        except Exception as e:
            print(f"❌ Failed ratios for {sym}: {e}")

    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
        combined_path = os.path.join(Config.DATA_PROCESSED_DIR, "peers_financial_ratios.csv")
        combined_df.to_csv(combined_path, index=False)
        if verbose:
            print(f"\n💾 Saved combined ratios → {combined_path}")
    else:
        combined_df = None

    if target_only is None:
        target_only = next((df for df in combined if df.get("symbol") is not None and df["symbol"].iloc[0].upper() == Config.TARGET_COMPANY["symbol"].upper()), None)

    return {"target": target_only, "combined": combined_df}
