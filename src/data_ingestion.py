"""
data_ingestion.py
--------------------------------------
Hybrid Data Ingestion Module for Fundamental Analyst Agent
Company: Microsoft Corporation (MSFT)
Years: 2020–2024
--------------------------------------
Functions:
1. Fetch 5-year annual reports (income, balance sheet, cashflow) from Alpha Vantage
2. Clean and save structured CSV files to /data/processed/
--------------------------------------
Usage:
    python src/data_ingestion.py
--------------------------------------
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from .config import Config

# Default peer list for multi-company ingestion (all 11 companies including MSFT)
PEER_SYMBOLS = ["MSFT", "AAPL", "GOOGL", "AMZN", "IBM", "ORCL", "CRM", "ADBE", "INTC", "CSCO", "SAP"]


# -------------------------------
# 1. FETCH FROM ALPHA VANTAGE
# -------------------------------
def fetch_alpha_vantage_report(function_name, symbol=None, api_key=None, verbose=True, retry_delay=13):
    """
    Fetch structured financial statements from Alpha Vantage
    
    Args:
        function_name: One of ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']
        symbol: Stock symbol (default: from Config.TARGET_COMPANY)
        api_key: Alpha Vantage API key (default: from Config.ALPHA_VANTAGE_KEY)
        verbose: Whether to print progress messages (default: True)
        retry_delay: Delay in seconds between API calls (default: 15s for free tier)
    
    Returns:
        pandas.DataFrame: Filtered annual reports for specified years
    """
    if symbol is None:
        symbol = Config.TARGET_COMPANY["symbol"]
    if api_key is None:
        api_key = Config.ALPHA_VANTAGE_KEY
    
    url = f"{Config.ALPHA_VANTAGE_BASE_URL}?function={function_name}&symbol={symbol}&apikey={api_key}"
    
    if verbose:
        print(f"📡 Fetching {function_name} for {symbol}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        response = requests.get(url, timeout=30)
        data = response.json()

        # Check for rate limits
        if "Information" in data and "API call frequency" in str(data.get("Information")):
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                if verbose:
                    print(f"   ⏳ Rate limit detected. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                raise ValueError(
                    f"❌ Alpha Vantage API rate limit exceeded.\n"
                    f"   Free tier: 25 requests/day, 5 requests/minute.\n"
                    f"   Please wait a few minutes and try again.\n"
                    f"   Tip: Use force_refresh=False to skip already downloaded data."
                )

        # Check if data exists
        if "annualReports" not in data:
            error_msg = data.get("Note") or data.get("Error Message") or data
            raise ValueError(f"❌ No data found for {function_name}. Response: {error_msg}")

        # Convert to DataFrame
        df = pd.DataFrame(data["annualReports"])
        df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
        df["year"] = df["fiscalDateEnding"].dt.year
        
        # Filter specific years
        years_to_keep = list(range(Config.START_YEAR, Config.END_YEAR + 1))
        df = df[df["year"].isin(years_to_keep)].sort_values("year", ascending=True)
        
        if verbose:
            print(f"   ✅ Retrieved {len(df)} annual reports ({df['year'].min()} - {df['year'].max()})")
        
        return df
    
    raise ValueError(f"Failed to fetch {function_name} after {max_retries} attempts")


def clean_financial_data(df):
    """
    Clean and standardize financial data
    - Convert numeric columns to float
    - Handle 'None' values
    - Remove unnecessary columns
    """
    # Convert numeric columns
    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        if col not in ['fiscalDateEnding', 'reportedCurrency']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def save_report(df, report_name, output_dir=None, verbose=True, symbol=None):
    """
    Save dataframe as CSV to processed data directory
    
    Args:
        df: DataFrame to save
        report_name: Name of the report (e.g., 'income_statement')
        output_dir: Output directory (default: Config.DATA_PROCESSED_DIR)
        verbose: Whether to print save confirmation (default: True)
    """
    if output_dir is None:
        output_dir = Config.DATA_PROCESSED_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    target_symbol = symbol or Config.TARGET_COMPANY["symbol"]
    filename = f"{target_symbol.lower()}_{report_name}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    if verbose:
        print(f"   💾 Saved → {filepath}")


# -------------------------------
# 2. MAIN PIPELINE
# -------------------------------
def run_ingestion_pipeline(symbol=None, verbose=True, force_refresh=False):
    """
    Execute the complete data ingestion pipeline
    
    Args:
        symbol: Stock symbol (default: from Config.TARGET_COMPANY)
        verbose: Whether to print detailed progress (default: True)
        force_refresh: Whether to force re-fetch data even if files exist (default: False)
    
    Steps:
    1. Validate configuration
    2. Check if data already exists (skip if force_refresh=False)
    3. Fetch income statement, balance sheet, cash flow
    4. Clean and save data
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate configuration
    try:
        if verbose:
            Config.validate_keys(verbose=True)
            Config.create_directories(verbose=True)
        else:
            # 静默模式：仍然验证和创建，但不打印
            if not Config.ALPHA_VANTAGE_KEY:
                raise ValueError("Missing ALPHA_VANTAGE_API_KEY")
            os.makedirs(Config.DATA_PROCESSED_DIR, exist_ok=True)
    except ValueError as e:
        if verbose:
            print(str(e))
        return False
    
    if symbol is None:
        symbol = Config.TARGET_COMPANY["symbol"]
        company_name = Config.TARGET_COMPANY.get("name", symbol)
    else:
        company_name = symbol
    
    # Check if files already exist
    symbol_lower = symbol.lower()
    required_files = [
        f"{symbol_lower}_income_statement.csv",
        f"{symbol_lower}_balance_sheet.csv",
        f"{symbol_lower}_cashflow_statement.csv"
    ]
    files_exist = all(
        os.path.exists(os.path.join(Config.DATA_PROCESSED_DIR, f)) 
        for f in required_files
    )
    
    if files_exist and not force_refresh:
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ Data already exists for {company_name} ({symbol})")
            print(f"{'='*60}")
            print(f"\n📁 Using existing files in: {Config.DATA_PROCESSED_DIR}")
            print(f"   - {symbol_lower}_income_statement.csv")
            print(f"   - {symbol_lower}_balance_sheet.csv")
            print(f"   - {symbol_lower}_cashflow_statement.csv")
            print(f"\n💡 Tip: Use force_refresh=True to re-fetch data")
            print(f"{'='*60}\n")
        return True
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 Starting Data Ingestion for {company_name} ({symbol})")
        print(f"   Years: {Config.START_YEAR}-{Config.END_YEAR}")
        print(f"{'='*60}\n")
        print("📊 Fetching Financial Statements...")
    
    try:
        # Fetch 3 major financial statements (add delay to avoid rate limits)
        income = fetch_alpha_vantage_report("INCOME_STATEMENT", symbol=symbol, verbose=verbose)
        income_clean = clean_financial_data(income)
        save_report(income_clean, "income_statement", verbose=verbose, symbol=symbol)
        
        if verbose:
            print("   ⏳ Waiting 13s to avoid rate limit...")
        time.sleep(13)  # Alpha Vantage Free Tier limit: 5 requests/min (12s minimum safe interval)
        
        balance = fetch_alpha_vantage_report("BALANCE_SHEET", symbol=symbol, verbose=verbose)
        balance_clean = clean_financial_data(balance)
        save_report(balance_clean, "balance_sheet", verbose=verbose, symbol=symbol)
        
        if verbose:
            print("   ⏳ Waiting 13s to avoid rate limit...")
        time.sleep(13)
        
        cashflow = fetch_alpha_vantage_report("CASH_FLOW", symbol=symbol, verbose=verbose)
        cashflow_clean = clean_financial_data(cashflow)
        save_report(cashflow_clean, "cashflow_statement", verbose=verbose, symbol=symbol)
        
        if verbose:
            print(f"\n{'='*60}")
            print("✅ Data Ingestion Complete!")
            print(f"{'='*60}")
            print(f"\n📁 Files saved to: {Config.DATA_PROCESSED_DIR}")
            print(f"   - {symbol_lower}_income_statement.csv")
            print(f"   - {symbol_lower}_balance_sheet.csv")
            print(f"   - {symbol_lower}_cashflow_statement.csv")
            print(f"\n💡 Next Step: Run financial ratio analysis")
            print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Error during data ingestion: {str(e)}")
        raise


def run_ingestion_for_peers(symbols=None, verbose=True, force_refresh=False, pause_between_symbols=13):
    """Fetch 5-year financials for multiple symbols (peers)."""
    if symbols is None:
        symbols = PEER_SYMBOLS

    results = {}
    for idx, sym in enumerate(symbols):
        print(f"\n{'='*60}")
        print(f"🚀 Starting peer ingestion: {sym} ({idx+1}/{len(symbols)})")
        print(f"{'='*60}")
        try:
            ok = run_ingestion_pipeline(symbol=sym, verbose=verbose, force_refresh=force_refresh)
            results[sym] = ok
        except Exception as e:
            results[sym] = False
            print(f"❌ {sym} ingestion failed: {e}")
        # Respect Alpha Vantage free tier rate limits between different symbols
        if idx < len(symbols) - 1:
            if verbose:
                print(f"⏳ Cooling down {pause_between_symbols}s before next symbol to avoid rate limits...")
            time.sleep(pause_between_symbols)

    return results


# -------------------------------
# 3. EXECUTION
# -------------------------------
if __name__ == "__main__":
    success = run_ingestion_pipeline()
    # To fetch peers in one shot (MSFT, GOOGL, AMZN, META), uncomment:
    # peer_results = run_ingestion_for_peers()
    exit(0 if success else 1)
