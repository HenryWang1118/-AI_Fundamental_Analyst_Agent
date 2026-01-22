"""Demo entrypoint for end-to-end pipeline: fetch data -> analyze -> generate AI memo.

Usage examples (run from repo root):
    # Full pipeline: fetch target+peers, run analyses, then LLM memo
    python demo/run_agent.py --end-to-end --memo-type combined_zh

    # Only generate memo using existing processed data
    python demo/run_agent.py --memo-type combined_zh

What this script does (configurable by flags):
    - optionally fetch raw financials from Alpha Vantage (target and/or peers)
    - calculate financial ratios (target and peers), valuation, and peer comparison artifacts
    - generate the requested memo type (default: combined_zh) and save under reports/comprehensive

Requirements:
    - ALPHA_VANTAGE_API_KEY (for fetching financial statements)
    - DASHSCOPE_API_KEY + DASHSCOPE_BASE_URL (for LLM generation)
    - Network access for yfinance peer metrics (optional but recommended)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make project root importable when running from repo root or demo/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_report_generator import AIReportGenerator  # noqa: E402
from src.config import Config  # noqa: E402
from src.data_ingestion import run_ingestion_pipeline, run_ingestion_for_peers  # noqa: E402
from src.financial_ratios import (  # noqa: E402
    run_ratio_calculation_pipeline,
    run_ratios_for_symbols,
)
from src.valuation_models import run_valuation  # noqa: E402
from src.peer_analysis import (  # noqa: E402
    run_peer_analysis,
    generate_peer_comparison_report,
    save_peer_report,
    save_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI investment memo (comprehensive demo)")
    parser.add_argument(
        "--memo-type",
        default="detailed",
        choices=["detailed"],
        help="Memo template to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the LLM (default: 0.3).",
    )
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="Run full pipeline: fetch target+peers -> analyses -> LLM memo.",
    )
    parser.add_argument(
        "--fetch-target",
        action="store_true",
        help="Fetch target company's financials before analysis (Alpha Vantage).",
    )
    parser.add_argument(
        "--fetch-peers",
        action="store_true",
        help="Fetch peer financials before analysis (Alpha Vantage).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-fetch even if processed CSVs exist.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis steps (ratios, valuation, peer comparison) before LLM.",
    )
    parser.add_argument(
        "--skip-peer-report",
        action="store_true",
        help="Skip writing peer comparison JSON/Markdown reports (still runs core analytics).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation (useful to just refresh data/analytics).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default behavior: If no flags set, run end-to-end (fetch target+peers, analyze, LLM)
    no_flags = not any([args.end_to_end, args.fetch_target, args.fetch_peers, args.analyze, args.no_llm])

    do_fetch_target = args.end_to_end or args.fetch_target or no_flags
    do_fetch_peers = args.end_to_end or args.fetch_peers or no_flags
    do_analyze = args.end_to_end or args.analyze or no_flags
    do_llm = (not args.no_llm)  # LLM is enabled by default unless explicitly disabled
    skip_peer_report = args.skip_peer_report

    print("\n================ AI Fundamental Analyst Demo ===============")
    print(f"Workspace root: {ROOT}")
    print(f"Target company: {Config.TARGET_COMPANY['symbol']}")
    print(f"Memo type     : {args.memo_type}")
    print(f"Fetch target  : {do_fetch_target}")
    print(f"Fetch peers   : {do_fetch_peers}")
    print(f"Run analyses  : {do_analyze}")
    print(f"Generate LLM  : {do_llm}")
    print("===========================================================\n")

    # 1) Optional data fetching
    if do_fetch_target:
        print("[1/3] Fetching target financials...")
        ok = run_ingestion_pipeline(symbol=Config.TARGET_COMPANY["symbol"], verbose=True, force_refresh=args.force_refresh)
        if not ok:
            print("❌ Target ingestion failed; aborting.")
            sys.exit(1)
    if do_fetch_peers:
        print("[1b/3] Fetching peer financials...")
        run_ingestion_for_peers(verbose=True, force_refresh=args.force_refresh)

    # 2) Optional analytics
    if do_analyze:
        print("[2/3] Running ratio calculation (target)...")
        run_ratio_calculation_pipeline(symbol=Config.TARGET_COMPANY["symbol"], verbose=True)

        if do_fetch_peers:
            print("[2b/3] Running ratio calculation for peers and combined file...")
            run_ratios_for_symbols(verbose=True)

        print("[2c/3] Running valuation (DCF scenarios)...")
        run_valuation(symbol=Config.TARGET_COMPANY["symbol"], force_refresh=args.force_refresh)

        print("[2d/3] Running peer analysis and comparison report...")
        run_peer_analysis(symbol=Config.TARGET_COMPANY["symbol"], force_refresh=args.force_refresh)
        peer_report = generate_peer_comparison_report(verbose=True)
        if not skip_peer_report:
            # Save peer reports to processed data directory as requested
            save_peer_report(peer_report, output_dir=Config.DATA_PROCESSED_DIR)
            save_markdown_report(peer_report, output_dir=Config.DATA_PROCESSED_DIR)

    # 3) LLM generation
    if do_llm:
        generator = AIReportGenerator(model=Config.QWEN_MODEL, temperature=args.temperature)
        result = generator.generate_ai_memo(memo_type=args.memo_type)
        if not result.get("success"):
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            print("Hint: ensure DashScope API key/base URL are set in Config/.env, and required data is present.")
            sys.exit(1)

        print("✅ AI memo generated successfully")
        print(f"Saved to   : {result['report_path']}")
        print(f"Length     : {result['report_length']} chars")
        print(f"Model used : {result['model']}")
        print(f"Memo type  : {result['memo_type']}")

        print("\nYou can find the memo under reports/comprehensive.\n")
    else:
        print("Skipped LLM generation (--no-llm). Data and analytics steps completed.")


if __name__ == "__main__":
    main()
