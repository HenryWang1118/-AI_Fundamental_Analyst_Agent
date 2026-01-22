"""
AI Investment Report Generator
Generating professional, personalized investment memos using LLM
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import openai
from openai import OpenAI

from .config import Config
import pandas as pd
REPORTS_DIR: Path = Path(Config.REPORTS_DIR)
TARGET_COMPANY = Config.TARGET_COMPANY
DATA_PROCESSED_DIR: Path = Path(Config.DATA_PROCESSED_DIR)

# Optional Visualization Library (Skip chart generation if unavailable)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class AIReportGenerator:
    """AI Report Generator"""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.3):
        self.symbol = TARGET_COMPANY["symbol"]
        self.model = model
        self.temperature = temperature
        self.client = self._initialize_client()
        self.analysis_data = {}
        self.provider = "none"
        self._current_memo_type: Optional[str] = None
        
    def _initialize_client(self) -> Optional[OpenAI]:
        """Initialize LLM client: prioritize DashScope compatible endpoint, then OpenAI, finally mock"""
        # Use DashScope compatible OpenAI protocol only
        if Config.DASHSCOPE_API_KEY:
            try:
                client = OpenAI(api_key=Config.DASHSCOPE_API_KEY, base_url=Config.DASHSCOPE_BASE_URL)
                # Set model to Qwen (if not explicitly specified)
                if not self.model or self.model == "gpt-4":
                    self.model = Config.QWEN_MODEL
                client.models.list()
                self.provider = "dashscope"
                logger.info(f"Connected to DashScope (OpenAI-compatible). Model: {self.model}")
                return client
            except Exception as e:
                logger.error(f"Failed to initialize DashScope client: {e}")
                return None

        # Return None if DashScope API Key is not configured (no OpenAI or mock fallback)
        logger.error("DashScope API key not configured. LLM provider unavailable.")
        self.provider = "none"
        return None
    
    def load_analysis_data(self) -> bool:
        """Load analysis data"""
        try:
            data_dir = REPORTS_DIR
            
            # Load results from various analysis modules
            data_files = {
                "ratios": data_dir / f"{self.symbol}_ratios_analysis.json",
                "valuation": data_dir / f"{self.symbol}_dcf_results.json",
                "peer_analysis": data_dir / f"{self.symbol}_peer_analysis.json",
                "comparable": data_dir / f"{self.symbol}_comparable_analysis.md"
            }
            
            # Prioritize reports directory; if missing, skip silently and fallback to processed
            for key, path in data_files.items():
                if path.exists():
                    try:
                        if path.suffix == '.json':
                            with open(path, 'r') as f:
                                self.analysis_data[key] = json.load(f)
                        elif path.suffix == '.md':
                            with open(path, 'r', encoding='utf-8') as f:
                                self.analysis_data[key] = f.read()
                        logger.info(f"Loaded {key} data")
                    except Exception as e:
                        logger.error(f"Error loading {key}: {e}")
                        continue
            
            return len(self.analysis_data) > 0
            
        except Exception as e:
            logger.error(f"Failed to load analysis data: {e}")
            return False

    # ----------------------- Handle Fallback for Processed Directory -----------------------
    def _fallback_load_ratios_from_processed(self) -> None:
        ratios_csv = DATA_PROCESSED_DIR / f"{self.symbol.lower()}_financial_ratios.csv"
        if ratios_csv.exists():
            try:
                df = pd.read_csv(ratios_csv).sort_values("year")
                latest = df.iloc[-1]

                def pct(v):
                    try:
                        return float(v) / 100.0
                    except Exception:
                        return 0.0

                def f(v):
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

                # Simple FCF CAGR estimation
                fcf_cagr = 0.0
                try:
                    series = df["free_cash_flow"].dropna().astype(float)
                    if len(series) >= 2 and series.iloc[0] > 0:
                        periods = len(series) - 1
                        fcf_cagr = (series.iloc[-1] / series.iloc[0]) ** (1.0 / periods) - 1.0
                except Exception:
                    fcf_cagr = 0.0

                self.analysis_data.setdefault("ratios", {})
                self.analysis_data["ratios"]["key_ratios"] = {
                    "gross_margin": pct(latest.get("gross_margin")),
                    "operating_margin": pct(latest.get("operating_margin")),
                    "net_margin": pct(latest.get("net_margin")),
                    "roe": pct(latest.get("roe")),
                    "roa": pct(latest.get("roa")),
                    "current_ratio": f(latest.get("current_ratio")),
                    "quick_ratio": f(latest.get("quick_ratio")),
                    "debt_to_equity": f(latest.get("debt_to_equity")),
                }
                self.analysis_data["ratios"]["growth_metrics"] = {
                    "fcf_cagr": fcf_cagr,
                }
                logger.info(f"Loaded key ratios from processed CSV: {ratios_csv}")
            except Exception as e:
                logger.error(f"Failed to parse ratios CSV: {e}")

    def _fallback_load_valuation_from_scenarios(self) -> None:
        dcf_csv = DATA_PROCESSED_DIR / f"{self.symbol.lower()}_dcf_scenarios.csv"
        if dcf_csv.exists() and "valuation" not in self.analysis_data:
            try:
                df = pd.read_csv(dcf_csv)
                priced = df.dropna(subset=["implied_price"]) if "implied_price" in df.columns else df
                if not priced.empty:
                    intrinsic = float(priced["implied_price"].mean())
                    current_price = float(priced["stock_price"].iloc[0]) if "stock_price" in priced.columns else 0.0
                    margin = ((intrinsic - current_price) / current_price) if current_price else 0.0
                    base_row = priced.iloc[0]
                    self.analysis_data["valuation"] = {
                        "intrinsic_value_per_share": intrinsic,
                        "current_price": current_price,
                        "margin_of_safety": margin,
                        "discount_rate": float(base_row.get("discount_rate", 0.0)),
                        "terminal_growth": float(base_row.get("g", 0.0)),
                        "recommendation": "BUY" if current_price and margin > 0.1 else "HOLD",
                    }
                    logger.info(f"Built valuation data from DCF scenarios CSV: {dcf_csv}")
            except Exception as e:
                logger.error(f"Failed to parse DCF scenarios CSV: {e}")

    def _fallback_load_peer_from_processed(self) -> None:
        try:
            candidates = sorted(DATA_PROCESSED_DIR.glob("peer_comparison_report_*.json"))
            if candidates and "peer_analysis" not in self.analysis_data:
                latest = candidates[-1]
                with open(latest, "r", encoding="utf-8") as f:
                    self.analysis_data["peer_analysis"] = json.load(f)
                logger.info(f"Loaded latest peer comparison report: {latest}")
        except Exception as e:
            logger.error(f"Failed to load peer report from processed: {e}")

    def _fallback_load_peer_markdown_from_processed(self) -> None:
        """Load latest peer Markdown summary from processed directory (if exists)."""
        try:
            candidates = sorted(DATA_PROCESSED_DIR.glob("peer_comparison_report_*.md"))
            if candidates:
                latest = candidates[-1]
                with open(latest, "r", encoding="utf-8") as f:
                    self.analysis_data["peer_markdown"] = f.read()
                logger.info(f"Loaded latest peer comparison markdown: {latest}")
        except Exception as e:
            logger.error(f"Failed to load peer markdown from processed: {e}")

    def _load_latest_traditional_memo_text(self) -> None:
        """Read latest traditional memo text from reports/investment_memos for context fusion."""
        try:
            inv_dir = REPORTS_DIR / "investment_memos"
            if inv_dir.exists():
                candidates = sorted(inv_dir.glob(f"{self.symbol}_traditional_memo_*.md"))
                if candidates:
                    latest = candidates[-1]
                    with open(latest, "r", encoding="utf-8") as f:
                        self.analysis_data["traditional_memo_text"] = f.read()
                    logger.info(f"Loaded latest traditional memo: {latest}")
        except Exception as e:
            logger.error(f"Failed to load traditional memo: {e}")

    def _fallback_load_peer_multiples_csv(self) -> None:
        """Load peer valuation multiples CSV from processed directory and build summary table."""
        try:
            path = DATA_PROCESSED_DIR / f"{self.symbol.lower()}_peer_multiples.csv"
            if path.exists():
                df = pd.read_csv(path)
                # Keep only common fields
                cols = [
                    "symbol", "price", "market_cap", "enterprise_value",
                    "pe", "pb", "ps", "peg", "dividend_yield"
                ]
                df = df[[c for c in cols if c in df.columns]]
                # Convert to record list for text rendering
                self.analysis_data["peer_multiples"] = df.to_dict(orient="records")
                logger.info(f"Loaded peer multiples CSV: {path}")
        except Exception as e:
            logger.error(f"Failed to load peer multiples CSV: {e}")

    def _build_dcf_sensitivity_from_scenarios(self) -> None:
        """Build sensitivity table from DCF scenarios CSV (mean intrinsic value by WACC and perpetual growth)."""
        try:
            dcf_csv = DATA_PROCESSED_DIR / f"{self.symbol.lower()}_dcf_scenarios.csv"
            if not dcf_csv.exists():
                return
            df = pd.read_csv(dcf_csv)
            if "implied_price" not in df.columns:
                return
            # Select typical points: Mean of unique combinations of WACC and g
            for col in ["discount_rate", "g"]:
                if col not in df.columns:
                    return
            sens = []
            try:
                group_wacc = df.groupby("discount_rate")["implied_price"].mean().reset_index()
                group_g = df.groupby("g")["implied_price"].mean().reset_index()
                # Take top 5 points for display
                sens_wacc = group_wacc.sort_values("discount_rate").head(5).to_dict(orient="records")
                sens_g = group_g.sort_values("g").head(5).to_dict(orient="records")
                self.analysis_data["dcf_sensitivity"] = {
                    "by_wacc": sens_wacc,
                    "by_growth": sens_g,
                }
                logger.info("Built DCF sensitivity tables from scenarios CSV")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed building DCF sensitivity: {e}")

    def load_analysis_data_with_fallbacks(self) -> bool:
        """Comprehensive loading: prioritize reports, fallback to processed to build context"""
        ok = self.load_analysis_data()
        if "ratios" not in self.analysis_data:
            self._fallback_load_ratios_from_processed()
        if "valuation" not in self.analysis_data:
            self._fallback_load_valuation_from_scenarios()
        if "peer_analysis" not in self.analysis_data:
            self._fallback_load_peer_from_processed()
        # Supplement peer Markdown and traditional memo text
        self._fallback_load_peer_markdown_from_processed()
        self._load_latest_traditional_memo_text()
        # Supplement peer multiples and DCF sensitivity
        self._fallback_load_peer_multiples_csv()
        self._build_dcf_sensitivity_from_scenarios()
        return len(self.analysis_data) > 0
    
    def prepare_context_data(self) -> str:
        """Prepare context data for LLM (English, combining traditional memo and peer analysis)"""
        context = f"""# Analysis Context - {self.symbol}

## Company Overview
{self.symbol} is a leading global technology company. Analysis Period: 2020-2024.

## Financial Analysis Results
"""
        
        # Add Financial Ratios
        if "ratios" in self.analysis_data and "key_ratios" in self.analysis_data["ratios"]:
            ratios = self.analysis_data["ratios"]["key_ratios"]
            fcf_cagr = self.analysis_data["ratios"].get("growth_metrics", {}).get("fcf_cagr")
            context += "\n### Key Financial Ratios (Latest)\n"
            context += "| Metric | Value |\n|---|---|\n"
            def fmt_pct(v):
                return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "N/A"
            def fmt_num(v):
                return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"
            context += f"| Gross Margin | {fmt_pct(ratios.get('gross_margin'))} |\n"
            context += f"| Operating Margin | {fmt_pct(ratios.get('operating_margin'))} |\n"
            context += f"| Net Margin | {fmt_pct(ratios.get('net_margin'))} |\n"
            context += f"| ROE | {fmt_pct(ratios.get('roe'))} |\n"
            context += f"| ROA | {fmt_pct(ratios.get('roa'))} |\n"
            context += f"| Current Ratio | {fmt_num(ratios.get('current_ratio'))} |\n"
            context += f"| Quick Ratio | {fmt_num(ratios.get('quick_ratio'))} |\n"
            context += f"| Debt/Equity | {fmt_num(ratios.get('debt_to_equity'))} |\n"
            if isinstance(fcf_cagr, (int, float)):
                context += f"| FCF CAGR | {fcf_cagr*100:.1f}% |\n"
        
        # Add Valuation Data
        if "valuation" in self.analysis_data:
            val = self.analysis_data["valuation"]
            context += "\n### Valuation Summary (DCF)\n"
            intrinsic = val.get("intrinsic_value_per_share")
            current_price = val.get("current_price")
            mos = val.get("margin_of_safety")
            dr = val.get("discount_rate")
            tg = val.get("terminal_growth")
            context += "| Item | Value |\n|---|---|\n"
            context += f"| Intrinsic Value Per Share | ${intrinsic:.2f} |\n" if isinstance(intrinsic, (int, float)) else ""
            context += f"| Current Price | ${current_price:.2f} |\n" if isinstance(current_price, (int, float)) else ""
            context += f"| Margin of Safety | {mos*100:.1f}% |\n" if isinstance(mos, (int, float)) else ""
            context += f"| Discount Rate (WACC) | {dr:.2f} |\n" if isinstance(dr, (int, float)) else ""
            context += f"| Terminal Growth | {tg:.2f} |\n" if isinstance(tg, (int, float)) else ""
            if val.get("recommendation"):
                context += f"| Recommendation | {val['recommendation']} |\n"

        # DCF Sensitivity
        sens = self.analysis_data.get("dcf_sensitivity")
        if sens:
            context += "\n### DCF Sensitivity (Intrinsic Value vs WACC/Growth)\n"
            by_wacc = sens.get("by_wacc", [])
            if by_wacc:
                context += "**By WACC**\n| WACC | Mean Intrinsic Value |\n|---|---|\n"
                for row in by_wacc:
                    context += f"| {row.get('discount_rate')} | ${row.get('implied_price'):.2f} |\n"
            by_g = sens.get("by_growth", [])
            if by_g:
                context += "\n**By Perpetual Growth**\n| g | Mean Intrinsic Value |\n|---|---|\n"
                for row in by_g:
                    context += f"| {row.get('g')} | ${row.get('implied_price'):.2f} |\n"

        # Traditional Memo Summary Fusion
        tmemo = self.analysis_data.get("traditional_memo_text")
        if tmemo:
            snippet = tmemo.strip().replace("\r", " ")
            snippet = snippet[:500]
            context += "\n### Traditional Memo Summary (Excerpt)\n"
            context += snippet + "\n"
        
        # Peer Comparison
        if "peer_analysis" in self.analysis_data and "comparative_analysis" in self.analysis_data["peer_analysis"]:
            comp = self.analysis_data["peer_analysis"]["comparative_analysis"]
            context += "\n### Peer Comparison Highlights (Significant Deviations)\n"
            rows = []
            for metric, data in comp.items():
                if isinstance(data, dict) and "deviation_from_average_pct" in data:
                    dev = data["deviation_from_average_pct"]
                    if abs(dev) > 5:
                        name = data.get('name', metric)
                        direction = "above" if dev > 0 else "below"
                        rows.append((name, f"{abs(dev):.1f}% {direction}"))
            if rows:
                context += "| Metric | vs Peers |\n|---|---|\n"
                for name, val in rows[:6]:
                    context += f"| {name} | {val} |\n"

        # Peer Valuation Multiples
        pm = self.analysis_data.get("peer_multiples")
        if pm:
            context += "\n### Peer Valuation Multiples (Current)\n"
            context += "| Company | P/E | P/B | P/S | Div Yield |\n|---|---|---|---|---|\n"
            for rec in pm[:8]:
                def fmt(v):
                    try:
                        return f"{float(v):.2f}"
                    except Exception:
                        return "N/A"
                div = rec.get("dividend_yield")
                div_s = f"{float(div)*100:.1f}%" if isinstance(div, (int, float)) else "N/A"
                context += f"| {rec.get('symbol','N/A')} | {fmt(rec.get('pe'))} | {fmt(rec.get('pb'))} | {fmt(rec.get('ps'))} | {div_s} |\n"

        # Management Quality
        pa = self.analysis_data.get("peer_analysis", {})
        mg = pa.get("management_quality", {}).get(self.symbol)
        if mg:
            context += "\n### Management Quality Assessment\n"
            context += f"- Score: {mg.get('score','N/A')} | Rating: {mg.get('rating','N/A')}\n"
            factors = mg.get('factors', [])
            if factors:
                context += "- Key Factors:\n"
                for fx in factors[:5]:
                    context += f"  - {fx.get('name','factor')}: {fx.get('importance','')}\n"

        # Catalysts
        cat_msft = pa.get("catalysts", {}).get(self.symbol, {}).get("catalysts", [])
        if cat_msft:
            context += "\n### Catalysts\n"
            for c in cat_msft[:5]:
                title = c.get('title','event')
                date = c.get('date','')
                context += f"- {date}: {title}\n"

        # Peer Markdown Excerpt
        pmd = self.analysis_data.get("peer_markdown")
        if pmd:
            context += "\n### Peer Analysis Markdown Excerpt\n"
            context += pmd.strip()[:300] + "\n"
        
        # Risks & Investment Thesis
        context += """
    ### Key Risk Factors (Analysis Based)
    1. Cloud Competition: Intense pressure from AWS and Google Cloud
    2. Regulatory & Compliance: Antitrust and privacy scrutiny on big tech
    3. Macro Sensitivity: Corporate IT spending cycles and FX impact
    4. AI Execution Risk: Adoption and monetization pace of Copilot/Azure AI
    5. Security/Compliance Events: Reputational impact of enterprise security breaches

    ### Investment Thesis
    - Strong Cloud & Software Ecosystem (Azure/Office/Dynamics) driving recurring revenue
    - AI Integration across stack enhancing customer value and lock-in
    - Robust Balance Sheet with growing Free Cash Flow
    - High margin software/services structure
    - Cross-platform distribution and network effects
    """
        
        return context
    
    def create_prompt(self, context: str, memo_type: str = "detailed") -> str:
        """Create LLM Prompt"""
        
        prompt_templates = {
            "detailed": f"""You are a fundamental research analyst. Write a comprehensive investment report for {self.symbol}.

Analysis Context: {context}

Structure:
1. Executive Summary with Investment Rating
2. Company Overview & Business Model
3. Detailed Financial Analysis (trends, ratios, comparisons)
4. Valuation Models (DCF, multiples, sensitivity)
5. Industry & Competitive Analysis
6. Management & Governance Assessment
7. Risk Analysis (quantitative and qualitative)
8. Investment Conclusion & Implementation

Length: 3-4 pages. Include specific data points and charts descriptions.
"""
        }
        
        return prompt_templates.get(memo_type, prompt_templates["detailed"])

    def _beautify_spacing(self, text: str) -> str:
        """Beautify paragraph spacing: unify blank lines, leave one line before headers, remove excess blank lines."""
        lines = text.splitlines()
        out = []
        prev_empty = True
        for i, l in enumerate(lines):
            is_header = l.startswith("#")
            if is_header and out and out[-1].strip() != "":
                out.append("")
            if l.strip() == "":
                if not prev_empty:
                    out.append("")
                prev_empty = True
            else:
                out.append(l.rstrip())
                prev_empty = False
        beautified = "\n".join(out).strip()
        return beautified + "\n"

    def _generate_visuals(self) -> list:
        """Generate charts and return list of relative paths. Skip if library unavailable or insufficient data."""
        visual_paths = []
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available. Skipping visuals.")
            return visual_paths

        try:
            figs_dir = REPORTS_DIR
            figs_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1) Financial Ratios Bar Chart
            ratios = self.analysis_data.get("ratios", {}).get("key_ratios", {})
            if ratios:
                labels = [
                    "Gross Margin", "Op Margin", "Net Margin", "ROE", "ROA"
                ]
                keys = [
                    "gross_margin", "operating_margin", "net_margin", "roe", "roa"
                ]
                values = [ratios.get(k) for k in keys]
                vals = [v * 100 if isinstance(v, (int, float)) else None for v in values]
                valid = [v for v in vals if v is not None]
                if valid:
                    plt.figure(figsize=(8, 4))
                    sns.barplot(x=labels, y=[v if v is not None else 0 for v in vals], color="#4C78A8")
                    plt.ylabel("Percentage (%)")
                    plt.title(f"{self.symbol} Key Profitability Ratios")
                    plt.tight_layout()
                    p = figs_dir / f"{self.symbol}_ratios_{ts}.png"
                    plt.savefig(p, dpi=120)
                    plt.close()
                    visual_paths.append(p.name)

            # 2) DCF Sensitivity Line Chart (WACC / Perpetual Growth)
            sens = self.analysis_data.get("dcf_sensitivity", {})
            if sens:
                by_wacc = sens.get("by_wacc", [])
                if by_wacc:
                    xs = [d.get("discount_rate") for d in by_wacc]
                    ys = [d.get("implied_price") for d in by_wacc]
                    if xs and ys and None not in ys:
                        plt.figure(figsize=(6, 4))
                        sns.lineplot(x=xs, y=ys, marker="o")
                        plt.xlabel("WACC")
                        plt.ylabel("Intrinsic Value (Mean)")
                        plt.title(f"{self.symbol} DCF Sensitivity: WACC")
                        plt.tight_layout()
                        p = figs_dir / f"{self.symbol}_dcf_wacc_{ts}.png"
                        plt.savefig(p, dpi=120)
                        plt.close()
                        visual_paths.append(p.name)

                by_g = sens.get("by_growth", [])
                if by_g:
                    xs = [d.get("g") for d in by_g]
                    ys = [d.get("implied_price") for d in by_g]
                    if xs and ys and None not in ys:
                        plt.figure(figsize=(6, 4))
                        sns.lineplot(x=xs, y=ys, marker="o", color="#E45756")
                        plt.xlabel("Perpetual Growth g")
                        plt.ylabel("Intrinsic Value (Mean)")
                        plt.title(f"{self.symbol} DCF Sensitivity: g")
                        plt.tight_layout()
                        p = figs_dir / f"{self.symbol}_dcf_g_{ts}.png"
                        plt.savefig(p, dpi=120)
                        plt.close()
                        visual_paths.append(p.name)

            # 3) Peer P/E Bar Chart
            pm = self.analysis_data.get("peer_multiples", [])
            if pm:
                companies = [d.get("symbol") for d in pm[:6]]
                pe_vals = [d.get("pe") for d in pm[:6]]
                valid = [v for v in pe_vals if isinstance(v, (int, float))]
                if companies and valid:
                    plt.figure(figsize=(8, 4))
                    sns.barplot(x=companies, y=[v if isinstance(v, (int, float)) else 0 for v in pe_vals], color="#72B7B2")
                    plt.ylabel("P/E")
                    plt.title(f"{self.symbol} P/E Comparison with Peers")
                    plt.xticks(rotation=20, ha='right')
                    plt.tight_layout()
                    p = figs_dir / f"{self.symbol}_peers_pe_{ts}.png"
                    plt.savefig(p, dpi=120)
                    plt.close()
                    visual_paths.append(p.name)

        except Exception as e:
            logger.error(f"Failed to generate visuals: {e}")

        return visual_paths
    
    def generate_with_llm(self, prompt: str) -> str:
        """Generate report using LLM"""
        if self.client is None:
            raise RuntimeError("LLM provider not configured (DashScope required).")
        
        try:
            logger.info(f"Calling {self.model} with prompt length: {len(prompt)}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst writing investment research reports."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=Config.MAX_TOKENS,
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.1
            )
            
            content = response.choices[0].message.content
            logger.info(f"Generated {len(content)} characters")
            
            return content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}.")
            raise

    def enforce_length_limit(self, text: str, max_chars: int = 1200) -> str:
        """Trim to specified character limit while trying to keep paragraphs intact."""
        if len(text) <= max_chars:
            return text
        # First trim to limit, then try to find nearest paragraph break
        cut = text[:max_chars]
        last_break = max(cut.rfind("\n## "), cut.rfind("\n### "), cut.rfind("\n- "))
        if last_break > max_chars * 0.7:
            return cut[:last_break].rstrip()
        return cut.rstrip()
    
    def _generate_mock_response(self) -> str:
        """Generate mock response (when API is unavailable, output English memo; supports detailed/standard)"""
        # Extract key numbers from loaded data
        val = self.analysis_data.get("valuation", {})
        intrinsic = val.get("intrinsic_value_per_share")
        current_price = val.get("current_price")
        margin = val.get("margin_of_safety")
        discount = val.get("discount_rate")
        tg = val.get("terminal_growth")
        reco = val.get("recommendation", "HOLD")

        ratios = self.analysis_data.get("ratios", {}).get("key_ratios", {})

        def fmt_pct(x):
            try:
                return f"{float(x)*100:.1f}%"
            except Exception:
                return "N/A"

        def fmt_num(x):
            try:
                return f"{float(x):.2f}"
            except Exception:
                return "N/A"

        # Target price approximated by DCF intrinsic value (if available)
        target_price = intrinsic if intrinsic is not None else None

        # Peer significant deviations (if available, max 3)
        peer_lines = []
        try:
            comp = self.analysis_data.get("peer_analysis", {}).get("comparative_analysis", {})
            for metric, data in comp.items():
                if isinstance(data, dict) and "deviation_from_average_pct" in data:
                    dev = data["deviation_from_average_pct"]
                    if abs(dev) > 5:
                        direction = "above" if dev > 0 else "below"
                        name = data.get("name", metric)
                        peer_lines.append(f"- {name}: {abs(dev):.1f}% {direction} peer average")
            peer_lines = peer_lines[:3]
        except Exception:
            peer_lines = []

        # combined version (~1200-1500 words, 7 sections, fused context)
        if self._current_memo_type == "combined" or self._current_memo_type == "combined_zh":
            peer_block_lines = []
            try:
                comp = self.analysis_data.get("peer_analysis", {}).get("comparative_analysis", {})
                for metric, data in comp.items():
                    if isinstance(data, dict) and "deviation_from_average_pct" in data:
                        dev = data["deviation_from_average_pct"]
                        if abs(dev) > 5:
                            direction = "above" if dev > 0 else "below"
                            name = data.get("name", metric)
                            peer_block_lines.append(f"- {name}: {abs(dev):.1f}% {direction} peer average")
                peer_block_lines = peer_block_lines[:4]
            except Exception:
                pass

            pm = self.analysis_data.get("peer_multiples", [])
            multiples_line = ""
            if pm:
                # Build short multiples summary (only MSFT row if exists)
                msft_rec = next((r for r in pm if str(r.get("symbol")).upper() == self.symbol), None)
                if msft_rec:
                    def fmt(v):
                        try:
                            return f"{float(v):.1f}"
                        except Exception:
                            return "N/A"
                    div = msft_rec.get("dividend_yield")
                    div_s = f"{float(div)*100:.1f}%" if isinstance(div, (int, float)) else "N/A"
                    multiples_line = f"P/E {fmt(msft_rec.get('pe'))} | P/B {fmt(msft_rec.get('pb'))} | P/S {fmt(msft_rec.get('ps'))} | Div Yield {div_s}"

            # Catalysts (max 3)
            pa = self.analysis_data.get("peer_analysis", {})
            cat_msft = pa.get("catalysts", {}).get(self.symbol, {}).get("catalysts", [])
            catalyst_lines = []
            for c in cat_msft[:3]:
                title = c.get('title','event')
                date = c.get('date','')
                catalyst_lines.append(f"- {date}: {title}")

            return f"""**{TARGET_COMPANY['name']} ({self.symbol}) Investment Memo (Combined)**
**Date:** {datetime.now().strftime("%B %d, %Y")}
**Rating:** {reco}

---
### 1. Executive Summary
Based on DCF analysis, {self.symbol} has an intrinsic value of approx. {fmt_num(intrinsic) if intrinsic is not None else 'N/A'}, lower than current price {fmt_num(current_price) if current_price is not None else 'N/A'}, implying a margin of safety of {fmt_pct(margin) if margin is not None else 'N/A'}. While fundamentals and cash flow quality remain excellent, valuation is full. Maintain **{reco}**.

---
### 2. Investment Thesis
- **Cloud & Ecosystem**: Azure, Office, and Dynamics drive high-quality recurring revenue and strong lock-in.
- **AI Integration**: Copilot and AI stack integration across products enhance ARPU and stickiness.
- **Financial Strength**: Robust balance sheet and Free Cash Flow generation support AI capex and shareholder returns.

---
### 3. Financial & Valuation Summary
| Metric | Value |
|---|---|
| Gross Margin | {fmt_pct(ratios.get('gross_margin'))} |
| Operating Margin | {fmt_pct(ratios.get('operating_margin'))} |
| ROE | {fmt_pct(ratios.get('roe'))} |
| FCF CAGR | {fmt_pct(self.analysis_data.get('ratios', {}).get('growth_metrics', {}).get('fcf_cagr'))} |
| Debt/Equity | {fmt_num(ratios.get('debt_to_equity'))} |

Valuation Assumptions: WACC {fmt_num(discount) if discount is not None else 'N/A'}, Terminal Growth {fmt_num(tg) if tg is not None else 'N/A'}; {('Multiples: ' + multiples_line) if multiples_line else ''}

---
### 4. Peer Comparison Highlights
{('\n'.join(peer_block_lines)) if peer_block_lines else '- Key metrics largely in line with major cloud/software peers; focus on Azure and AI monetization relative to competitors.'}

---
### 5. Catalysts & Risks
**Catalysts**:
{('\n'.join(catalyst_lines)) if catalyst_lines else '- AI adoption acceleration, Copilot penetration, Azure customer expansion.'}
**Risks**:
- Intense cloud competition; Regulatory/Antitrust scrutiny; Macro IT spending cycles; AI monetization delays.

---
### 6. Recommendation
Maintain **{reco}**; Target price reference: DCF Intrinsic Value. If intrinsic value is below current price, suggest accumulating on dips and verifying growth assumptions quarterly.

---
### 7. Implementation
- Monitor Azure YoY growth and Copilot commercialization metrics.
- Track margin trends and FCF conversion.
- Use peer multiple ranges for relative valuation sanity check.
- Re-evaluate upon significant M&A or regulatory events.
"""

        # detailed version (approx 2 pages, more comprehensive)
        if self._current_memo_type == "detailed" or self._current_memo_type == "detailed_zh":
            peer_block = ("\n".join(peer_lines)) if peer_lines else "- Generally inline with cloud/software peers; key focus is Azure/AI momentum."
            return f"""# {self.symbol} Investment Research Report (AI Generated - Local Mock)

    **Date:** {datetime.now().strftime("%B %d, %Y")}
    **Rating:** {reco}
    **Current Price:** {fmt_num(current_price) if current_price is not None else 'N/A'}
    **Intrinsic Value (DCF):** {fmt_num(intrinsic) if intrinsic is not None else 'N/A'}
    **Margin of Safety:** {fmt_pct(margin) if margin is not None else 'N/A'}

    ## 1. Executive Summary
    underpinned by a strong cloud and enterprise software ecosystem, {self.symbol} demonstrates robust profitability and free cash flow generation. Based on project-generated DCF and ratio analysis, the current recommendation is **{reco}**, with a target price referenced to the intrinsic value. Caution is advised regarding model assumptions.

    ## 2. Company Overview & Business Model
    {self.symbol} leverages a highly synergistic ecosystem across Azure cloud services, Office productivity suite, and enterprise solutions. Cross-platform distribution, ecosystem lock-in, and AI integration across the stack drive customer value, switching costs, and long-term recurring revenue visibility.

    ## 3. Financial Analysis (Key Ratios & Trends)
    | Metric | Value |
    |---|---|
    | Gross Margin | {fmt_pct(ratios.get('gross_margin'))} |
    | Operating Margin | {fmt_pct(ratios.get('operating_margin'))} |
    | Net Margin | {fmt_pct(ratios.get('net_margin'))} |
    | ROE | {fmt_pct(ratios.get('roe'))} |
    | ROA | {fmt_pct(ratios.get('roa'))} |
    | Current Ratio | {fmt_num(ratios.get('current_ratio'))} |
    | Quick Ratio | {fmt_num(ratios.get('quick_ratio'))} |
    | Debt/Equity | {fmt_num(ratios.get('debt_to_equity'))} |

    The ratios above indicate solid profitability and asset efficiency. Free Cash Flow growth validates the cloud business expansion:
    **FCF CAGR**: {fmt_pct(self.analysis_data.get('ratios', {}).get('growth_metrics', {}).get('fcf_cagr'))}

    ## 4. Valuation Analysis (DCF & Assumptions)
    | Item | Value |
    |---|---|
    | Intrinsic Value Per Share | {fmt_num(intrinsic) if intrinsic is not None else 'N/A'} |
    | Current Price | {fmt_num(current_price) if current_price is not None else 'N/A'} |
    | Margin of Safety | {fmt_pct(margin) if margin is not None else 'N/A'} |
    | WACC | {fmt_num(discount) if discount is not None else 'N/A'} |
    | Terminal Growth | {fmt_num(tg) if tg is not None else 'N/A'} |

    If Intrinsic Value < Current Price, review growth assumptions (Cloud/AI penetration) and cyclical factors. Relative valuation (EV/FCF, P/E) vs peers should be used as a sanity check.

    ## 5. Industry & Competition (Peer Comparison)
    {peer_block}

    ## 6. Catalysts & Risks
    - **Catalysts**: AI adoption acceleration, Copilot penetration, Azure customer expansion/ARPU growth, Enterprise security demand.
    - **Risks**: Cloud price wars, Regulatory/Privacy scrutiny, Macro IT spending cycles, AI monetization delays, Security/Compliance incidents.

    ## 7. Conclusion & Recommendation
    Synthesizing fundamentals and valuation, our recommendation for {self.symbol} is **{reco}**. Target price based on DCF Intrinsic Value. Strategy: Accumulate on dips and monitor quarterly progression.
    """

        # Standard version (Shorter)
        return f"""# AI-Generated Investment Memo - {self.symbol}

**Date:** {datetime.now().strftime("%B %d, %Y")}
**Analyst:** AI Fundamental Analyst Agent

## Executive Summary
Based on project analysis data, {self.symbol} shows robust fundamentals and cash flow quality. Combining DCF and relative valuation, current recommendation: **{reco}**.

**Current Price:** {fmt_num(current_price) if current_price is not None else 'N/A'}
**DCF Intrinsic Value:** {fmt_num(intrinsic) if intrinsic is not None else 'N/A'}
**Margin of Safety:** {fmt_pct(margin) if margin is not None else 'N/A'}
**12m Target:** {fmt_num(target_price) if target_price is not None else 'N/A'}

## Investment Thesis
- Cloud & Enterprise Software Ecosystem (Azure/Office/Dynamics) provides high-quality recurring revenue.
- AI Integration (Copilot/Security/Dev Tools) enhances customer value and lock-in.
- Strong Balance Sheet and Profitability support long-term capex and returns.

## Financial Analysis (Key Ratios)
| Metric | Value |
|---|---|
| Gross Margin | {fmt_pct(ratios.get('gross_margin'))} |
| Operating Margin | {fmt_pct(ratios.get('operating_margin'))} |
| Net Margin | {fmt_pct(ratios.get('net_margin'))} |
| ROE | {fmt_pct(ratios.get('roe'))} |
| ROA | {fmt_pct(ratios.get('roa'))} |
| Current Ratio | {fmt_num(ratios.get('current_ratio'))} |
| Quick Ratio | {fmt_num(ratios.get('quick_ratio'))} |
| Debt/Equity | {fmt_num(ratios.get('debt_to_equity'))} |

## Valuation Assessment
- DCF Intrinsic Value (per share): {fmt_num(intrinsic) if intrinsic is not None else 'N/A'}; Current: {fmt_num(current_price) if current_price is not None else 'N/A'}.
- Margin of Safety: {fmt_pct(margin) if margin is not None else 'N/A'}.
- Key Assumptions: WACC approx {fmt_num(discount) if discount is not None else 'N/A'}; Terminal Growth approx {fmt_num(tg) if tg is not None else 'N/A'}.

## Competitive Positioning
{('\n'.join(peer_lines)) if peer_lines else '- Metrics generally inline with peers; monitor Azure/AI pace.'}

## Catalysts & Risks
- **Catalysts**: AI acceleration, Azure momentum, Pricing power, Security demand.
- **Risks**: Competition, Regulation, Macro cycles, AI monetization lag.

## Recommendation
**{reco}**; Target price based on DCF. Caution advised if data is limited.

*Note: This report is a local mock generation.*
"""
    
    def post_process_report(self, report: str, context: str, minimal: bool = False, visuals: Optional[list] = None) -> str:
        """Post-process report (add metadata, format), and append chart links at end. If minimal=True, return only body + charts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if minimal:
            body = self._beautify_spacing(report)
            if visuals:
                body += "\n## Visual Figures\n"
                for i, p in enumerate(visuals, 1):
                    body += f"![Figure {i}]({p})\n"
            return body
        
        processed_report = f"""# AI-GENERATED INVESTMENT MEMORANDUM

**Company:** {self.symbol}
**Report ID:** {self.symbol}_AI_{timestamp}
**Date:** {datetime.now().strftime("%B %d, %Y")}
**Model:** {self.model}
    **Provider:** {self.provider}
**Temperature:** {self.temperature}
**Analyst:** AI Fundamental Analyst Agent
**Confidentiality:** For Academic/Educational Use Only

{'='*80}

{report}

{'='*80}

## AI Report Metadata

### Generation Details
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Model:** {self.model}
- **Temperature:** {self.temperature}
- **Context Length:** {len(context)} characters
- **Report Length:** {len(report)} characters

### Data Sources (Used in Analysis)
1. Project Reports: reports/financial_analysis, reports/valuation, reports/peer_analysis, reports/comparable_analysis
2. Processed Data: data/processed/{self.symbol.lower()}_financial_ratios.csv, {self.symbol.lower()}_dcf_scenarios.csv, latest peer_comparison_report_*.json
3. Note: Generation based solely on local project artifacts; no external data calls

### Methodology Notes
- This report was generated by an AI system trained on financial analysis
- All conclusions are based on the provided analysis data
- The AI system does not have real-time market access
- Recommendations are for educational purposes only

### Important Disclosures
**EDUCATIONAL USE ONLY** - This report is generated as part of an MSc coursework project.
**NOT INVESTMENT ADVICE** - This is a simulation for academic purposes.
**LIMITATIONS** - AI models may have biases and limitations in financial analysis.
**PAST PERFORMANCE** - Not indicative of future results.

### Version Information
- Report Version: 1.0
- AI Agent Version: 2.1
"""
        
        # Append charts at the end, not counting towards body word count (body length controlled during generation)
        if visuals:
            processed_report += "\n## Visual Figures\n"
            for i, p in enumerate(visuals, 1):
                processed_report += f"![Figure {i}]({p})\n"
        
        return processed_report
    
    def save_ai_report(self, report: str, raw_prompt: str, raw_response: str) -> Path:
        """Save AI report and raw data to reports directory"""
        output_dir = REPORTS_DIR
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save final report
        report_filename = f"{self.symbol}_ai_memo_{timestamp}.md"
        report_path = output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save raw data (for traceability)
        raw_data = {
            "timestamp": timestamp,
            "symbol": self.symbol,
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "prompt": raw_prompt,
            "raw_response": raw_response,
            "final_report": report,
            "analysis_data_summary": {
                "valuation": self.analysis_data.get("valuation", {}).get("intrinsic_value_per_share", "N/A"),
                "current_price": self.analysis_data.get("valuation", {}).get("current_price", "N/A"),
                "recommendation": self.analysis_data.get("valuation", {}).get("recommendation", "N/A")
            }
        }
        
        raw_filename = f"{self.symbol}_ai_raw_{timestamp}.json"
        # Save raw data to processed directory to keep reports directory clean
        if not DATA_PROCESSED_DIR.exists():
            DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = DATA_PROCESSED_DIR / raw_filename
        
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"AI Report saved: {report_path}")
        logger.info(f"Raw data saved: {raw_path}")
        
        return report_path
    
    def generate_ai_memo(self, memo_type: str = "standard") -> Dict[str, Any]:
        """Generate AI Memo"""
        print(f"\nStarting AI Investment Memo Generation ({memo_type})...")
        self._current_memo_type = memo_type
        
        # 1. Load Data
        print("1. Loading analysis data...")
        if not self.load_analysis_data_with_fallbacks():
            print("❌ Failed to load analysis data")
            return {"error": "No analysis data available"}
        
        # 2. Prepare Context
        print("2. Preparing analysis context...")
        context = self.prepare_context_data()
        
        # 3. Create Prompt
        print("3. Creating LLM prompt...")
        prompt = self.create_prompt(context, memo_type)
        
        # 4. Generate Report
        print("4. Calling LLM to generate report...")
        try:
            raw_response = self.generate_with_llm(prompt)
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            return {"error": str(e), "success": False}
        
        # 5. Post-process
        print("5. Post-processing report...")
        final_body = raw_response
        visuals_paths = []
        if memo_type == "combined" or memo_type == "combined_zh":
            # For combined version, keep sections intact, beautify spacing; no charts generated here for now
            final_body = self._beautify_spacing(final_body)
            visuals_paths = []
        else:
            final_body = self._beautify_spacing(final_body)
        
        minimal = (memo_type == "combined" or memo_type == "combined_zh")
        final_report = self.post_process_report(final_body, context, minimal=minimal, visuals=visuals_paths)
        
        # 6. Save Report
        print("6. Saving report...")
        report_path = self.save_ai_report(final_report, prompt, raw_response)
        
        # Return Result
        return {
            "success": True,
            "report_path": str(report_path),
            "report_length": len(final_report),
            "symbol": self.symbol,
            "model": self.model,
            "memo_type": memo_type
        }


def compare_memos(traditional_path: Path, ai_path: Path) -> str:
    """Compare Traditional and AI Reports"""
    comparison = f"""# REPORT COMPARISON: Traditional vs AI-Generated

## Comparison Summary
- **Traditional Memo:** Rule-based, structured, consistent
- **AI Memo:** LLM-generated, flexible, creative
- **Purpose:** Demonstrate different approaches to automated research

## Key Differences

### 1. Generation Method
| Aspect | Traditional | AI |
|--------|------------|----|
| **Approach** | Template + Rules | LLM + Prompt Engineering |
| **Flexibility** | Low | High |
| **Consistency** | High | Medium |
| **Creativity** | Low | Medium-High |

### 2. Content Characteristics
| Aspect | Traditional | AI |
|--------|------------|----|
| **Structure** | Fixed sections | Flexible organization |
| **Language** | Standardized | Natural, varied |
| **Insights** | Data-driven | Data + contextual |
| **Personalization** | Low | High |

### 3. Educational Value
| Aspect | Traditional | AI |
|--------|------------|----|
| **Transparency** | High (rules visible) | Medium (prompt visible) |
| **Explainability** | High | Medium |
| **Learning Outcome** | Rule-based automation | LLM integration |

## Recommendations for Asset Management

### Use Cases for Traditional Approach:
1. **Standardized reporting** across large portfolios
2. **Regulatory compliance** requiring consistent format
3. **High-volume screening** with defined criteria
4. **Risk management** systems needing transparent logic

### Use Cases for AI Approach:
1. **Deep dive analysis** of complex companies
2. **Custom client reporting** with different preferences
3. **Idea generation** and hypothesis testing
4. **Sentiment analysis** and qualitative insights

## Conclusion
Both approaches have merits in asset management:
- **Traditional:** Better for scale, consistency, transparency
- **AI:** Better for depth, flexibility, qualitative analysis

**Ideal solution:** Hybrid approach using rules for data processing and LLMs for narrative generation.

*Comparison generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    return comparison


def main():
    """Main Function: Generate AI Report"""
    symbol = TARGET_COMPANY["symbol"]
    
    print(f"\n{'='*70}")
    print(f"AI Investment Memo Generation - {symbol}")
    print(f"{'='*70}\n")
    
    # Initialize Generator
    generator = AIReportGenerator(
        model="gpt-4",  # Fallback to mock if unauthorized
        temperature=0.3
    )
    
    # Generate Report
    result = generator.generate_ai_memo(memo_type="combined")
    
    if result.get("success"):
        print(f"\n{'='*70}")
        print(f"✅ AI Investment Memo Generated!")
        print(f"{'='*70}")
        print(f"Report saved to: {result['report_path']}")
        print(f"Report length: {result['report_length']} characters")
        print(f"Model used: {result['model']}")
        print(f"Provider: {generator.provider}")
        print(f"Memo type: {result['memo_type']}")
        
        # Read and display report head
        try:
            with open(result['report_path'], 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                print(f"\nReport Preview (First 20 lines):")
                print("-" * 60)
                for line in lines[:20]:
                    if line.strip():
                        print(line)
                print("-" * 60)
        except Exception as e:
            print(f"Could not read report: {e}")
    
    else:
        print(f"❌ Failed to generate AI memo: {result.get('error', 'Unknown error')}")
    
    print(f"\n📁 AI memos are saved in: {REPORTS_DIR / 'comprehensive'}")


if __name__ == "__main__":
    main()
