import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import time
import io
from contextlib import redirect_stdout

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data_ingestion import run_ingestion_pipeline, run_ingestion_for_peers
from src.financial_ratios import run_ratio_calculation_pipeline, run_ratios_for_symbols
from src.valuation_models import run_valuation
from src.peer_analysis import run_peer_analysis, generate_peer_comparison_report, save_peer_report, save_markdown_report
from src.ai_report_generator import AIReportGenerator

st.set_page_config(page_title="AI Fundamental Analyst", page_icon="📈", layout="wide")

st.title("📈 AI Fundamental Analyst Agent")
st.markdown("An automated AI agent that fetches financial data, performs valuation, and generates professional investment memos.")

# Sidebar configuration
st.sidebar.header("Configuration")
target_symbol = st.sidebar.text_input("Target Company Symbol", value="MSFT").upper()
memo_type = st.sidebar.selectbox("Memo Type", ["detailed", "combined_zh", "executive_summary"])
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.3)

st.sidebar.header("Pipeline Steps")
do_fetch_target = st.sidebar.checkbox("1. Fetch Target Financials", value=False)
do_fetch_peers = st.sidebar.checkbox("2. Fetch Peer Financials", value=False)
do_analyze = st.sidebar.checkbox("3. Run Analysis & Valuation", value=True)
do_llm = st.sidebar.checkbox("4. Generate AI Memo", value=True)
force_refresh = st.sidebar.checkbox("Force Refresh Data", value=False)

run_button = st.sidebar.button("🚀 Run Agent", type="primary")

# Update Config dynamically if symbol changes
if target_symbol != Config.TARGET_COMPANY["symbol"]:
    Config.TARGET_COMPANY["symbol"] = target_symbol
    Config.TARGET_COMPANY["name"] = target_symbol + " Corporation"

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📝 AI Investment Memo", "📊 Financial Data", "💰 Valuation", "🏢 Peer Comparison"])

def run_pipeline():
    log_container = st.empty()
    log_text = ""
    
    class StreamlitRedirect:
        def write(self, text):
            nonlocal log_text
            log_text += text
            log_container.code(log_text, language="text")
        def flush(self):
            pass

    with redirect_stdout(StreamlitRedirect()):
        print(f"Starting pipeline for {target_symbol}...")
        
        if do_fetch_target:
            print("\n[1/4] Fetching target financials...")
            run_ingestion_pipeline(symbol=target_symbol, verbose=True, force_refresh=force_refresh)
            
        if do_fetch_peers:
            print("\n[2/4] Fetching peer financials...")
            run_ingestion_for_peers(verbose=True, force_refresh=force_refresh)
            
        if do_analyze:
            print("\n[3/4] Running analysis and valuation...")
            run_ratio_calculation_pipeline(symbol=target_symbol, verbose=True)
            if do_fetch_peers:
                run_ratios_for_symbols(verbose=True)
            run_valuation(symbol=target_symbol, force_refresh=force_refresh)
            run_peer_analysis(symbol=target_symbol, force_refresh=force_refresh)
            peer_report = generate_peer_comparison_report(verbose=True)
            save_peer_report(peer_report, output_dir=Config.DATA_PROCESSED_DIR)
            save_markdown_report(peer_report, output_dir=Config.DATA_PROCESSED_DIR)
            
        if do_llm:
            print("\n[4/4] Generating AI Investment Memo...")
            generator = AIReportGenerator(model=Config.QWEN_MODEL, temperature=temperature)
            generator.symbol = target_symbol
            result = generator.generate_ai_memo(memo_type=memo_type)
            if result.get("success"):
                print(f"\n✅ AI memo generated successfully: {result['report_path']}")
                st.session_state['latest_report'] = result['report_path']
            else:
                print(f"\n❌ Generation failed: {result.get('error')}")
                
        print("\nPipeline completed!")

if run_button:
    with st.spinner("Agent is working..."):
        run_pipeline()

# Display results in tabs
with tab1:
    st.header("AI Investment Memo")
    report_path = st.session_state.get('latest_report')
    
    if not report_path:
        reports_dir = Path(Config.REPORTS_DIR)
        if reports_dir.exists():
            reports = list(reports_dir.glob(f"{target_symbol}_ai_*.md"))
            if reports:
                reports.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                report_path = str(reports[0])

    if report_path and os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        st.markdown(content)
    else:
        st.info("No report generated yet. Click 'Run Agent' to generate one.")

with tab2:
    st.header("Processed Financial Data")
    col1, col2 = st.columns(2)
    
    inc_path = Path(Config.DATA_PROCESSED_DIR) / f"{target_symbol.lower()}_income_statement.csv"
    bal_path = Path(Config.DATA_PROCESSED_DIR) / f"{target_symbol.lower()}_balance_sheet.csv"
    
    with col1:
        st.subheader("Income Statement")
        if inc_path.exists():
            st.dataframe(pd.read_csv(inc_path))
        else:
            st.warning("Data not found.")
            
    with col2:
        st.subheader("Balance Sheet")
        if bal_path.exists():
            st.dataframe(pd.read_csv(bal_path))
        else:
            st.warning("Data not found.")

with tab3:
    st.header("Valuation Models (DCF)")
    dcf_path = Path(Config.DATA_PROCESSED_DIR) / f"{target_symbol.lower()}_dcf_scenarios.csv"
    if dcf_path.exists():
        st.dataframe(pd.read_csv(dcf_path))
    else:
        st.info("Run Analysis to generate valuation data.")

with tab4:
    st.header("Peer Comparison")
    peer_path = Path(Config.DATA_PROCESSED_DIR) / "peers_financial_ratios.csv"
    if peer_path.exists():
        st.dataframe(pd.read_csv(peer_path))
    else:
        st.info("Run Analysis with peers to generate comparison data.")
