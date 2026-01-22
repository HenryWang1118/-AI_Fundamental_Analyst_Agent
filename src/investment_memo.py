"""
report_templates.py
--------------------------------------
Traditional Investment Memo Template System
Generates standardized reports based on predefined templates and rules
Company: Microsoft Corporation (MSFT)
--------------------------------------
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TraditionalMemoGenerator:
    """Traditional Investment Memo Generator"""
    
    def __init__(self, symbol: str = "MSFT"):
        self.symbol = symbol
        self.template = self._load_template()
        
    def _load_template(self) -> Dict:
        """Load report template"""
        return {
            "header": {
                "title": "Investment Memorandum",
                "company": "Microsoft Corporation",
                "symbol": "MSFT",
                "date": "",
                "analyst": "AI Fundamental Analyst Agent",
                "report_id": ""
            },
            "sections": {
                "executive_summary": {
                    "title": "Executive Summary",
                    "content": "",
                    "required": True
                },
                "investment_thesis": {
                    "title": "Investment Thesis",
                    "content": "",
                    "required": True
                },
                "business_overview": {
                    "title": "Business Overview",
                    "content": "",
                    "required": True
                },
                "financial_analysis": {
                    "title": "Financial Analysis",
                    "subsections": [
                        "profitability",
                        "growth",
                        "leverage",
                        "liquidity",
                        "efficiency"
                    ],
                    "required": True
                },
                "valuation": {
                    "title": "Valuation",
                    "subsections": [
                        "dcf_valuation",
                        "relative_valuation",
                        "sensitivity_analysis"
                    ],
                    "required": True
                },
                "peer_comparison": {
                    "title": "Peer Comparison",
                    "content": "",
                    "required": True
                },
                "catalysts": {
                    "title": "Key Catalysts",
                    "content": "",
                    "required": True
                },
                "risks": {
                    "title": "Risk Factors",
                    "subsections": [
                        "business_risks",
                        "financial_risks",
                        "market_risks",
                        "regulatory_risks"
                    ],
                    "required": True
                },
                "recommendation": {
                    "title": "Investment Recommendation",
                    "subsections": [
                        "rating",
                        "price_target",
                        "time_horizon",
                        "position_size"
                    ],
                    "required": True
                },
                "appendix": {
                    "title": "Appendix",
                    "content": "",
                    "required": False
                }
            },
            "footer": {
                "disclaimer": "This report is for educational purposes only. Not investment advice.",
                "confidentiality": "Confidential - For internal use only",
                "version": "1.0"
            }
        }
    
    def load_analysis_data(self, data_dir: Path) -> Dict:
        """Load all analysis data"""
        analysis_data = {}

        project_root = Path(__file__).parent.parent
        dcf_path = project_root / "data" / "processed" / f"{self.symbol.lower()}_dcf_scenarios.csv"
        ratios_csv_path = project_root / "data" / "processed" / f"{self.symbol.lower()}_financial_ratios.csv"
        
        # Define data file paths
        data_files = {
            "ratios": data_dir / "financial_analysis" / f"{self.symbol}_ratios_analysis.json",
            "valuation": data_dir / "valuation" / f"{self.symbol}_dcf_results.json",
            "peer_analysis": data_dir / "peer_analysis" / f"{self.symbol}_peer_analysis.json",
            "comparative": data_dir / "comparable_analysis" / f"{self.symbol}_comparable_analysis.md"
        }
        
        for key, path in data_files.items():
            if path.exists():
                try:
                    if path.suffix == '.json':
                        with open(path, 'r') as f:
                            analysis_data[key] = json.load(f)
                    elif path.suffix == '.md':
                        with open(path, 'r', encoding='utf-8') as f:
                            analysis_data[key] = f.read()
                except Exception as e:
                    logger.error(f"Failed to load {key} data: {e}")

        # Construct valuation data from DCF scenario CSV if valuation JSON is missing
        if "valuation" not in analysis_data and dcf_path.exists():
            try:
                df = pd.read_csv(dcf_path)
                priced = df.dropna(subset=["implied_price"])
                if not priced.empty:
                    intrinsic = float(priced["implied_price"].mean())
                    current_price = float(priced["stock_price"].iloc[0]) if "stock_price" in priced.columns else 0.0
                    margin = ((intrinsic - current_price) / current_price) if current_price else 0.0
                    base_row = priced.iloc[0]
                    analysis_data["valuation"] = {
                        "intrinsic_value_per_share": intrinsic,
                        "current_price": current_price,
                        "margin_of_safety": margin,
                        "discount_rate": float(base_row.get("discount_rate", 0.0)),
                        "terminal_growth": float(base_row.get("g", 0.0)),
                        "recommendation": "HOLD" if current_price == 0 else ("BUY" if margin > 0.1 else "HOLD"),
                    }
            except Exception as e:
                logger.error(f"Failed to construct valuation data from DCF scenario CSV: {e}")

        # Extract latest metrics from financial ratios CSV if ratios JSON is missing
        if "ratios" not in analysis_data and ratios_csv_path.exists():
            try:
                df = pd.read_csv(ratios_csv_path)
                df = df.sort_values("year")
                latest = df.iloc[-1]

                def pct(val):
                    try:
                        return float(val) / 100.0
                    except Exception:
                        return 0.0

                def safe_float(val):
                    try:
                        return float(val)
                    except Exception:
                        return 0.0

                # Simple FCF annualized growth estimation from first and last values
                fcf_cagr = 0.0
                try:
                    series = df["free_cash_flow"].dropna().astype(float)
                    if len(series) >= 2 and series.iloc[0] > 0:
                        periods = len(series) - 1
                        fcf_cagr = (series.iloc[-1] / series.iloc[0]) ** (1.0 / periods) - 1.0
                except Exception:
                    fcf_cagr = 0.0

                analysis_data["ratios"] = {
                    "key_ratios": {
                        "gross_margin": pct(latest.get("gross_margin")),
                        "operating_margin": pct(latest.get("operating_margin")),
                        "net_margin": pct(latest.get("net_margin")),
                        "roe": pct(latest.get("roe")),
                        "roa": pct(latest.get("roa")),
                        "current_ratio": safe_float(latest.get("current_ratio")),
                        "quick_ratio": safe_float(latest.get("quick_ratio")),
                        "debt_to_equity": safe_float(latest.get("debt_to_equity")),
                    },
                    "growth_metrics": {
                        "fcf_cagr": fcf_cagr,
                    },
                }
            except Exception as e:
                logger.error(f"Failed to construct ratios data from financial ratios CSV: {e}")
        
        return analysis_data
    
    def generate_executive_summary(self, analysis_data: Dict) -> str:
        """Generate Executive Summary"""
        # Get key info from valuation data
        valuation = analysis_data.get("valuation", {})
        peer = analysis_data.get("peer_analysis", {})
        
        intrinsic_value = valuation.get("intrinsic_value_per_share", 0)
        current_price = valuation.get("current_price", 0)
        recommendation = valuation.get("recommendation", "HOLD")
        
        summary = f"""## Executive Summary

    **Investment Recommendation:** {recommendation}

    **Key Metrics:**
    - **Current Price:** ${current_price:.2f}
    - **DCF Intrinsic Value:** ${intrinsic_value:.2f}
    - **Margin of Safety:** {((intrinsic_value - current_price) / current_price * 100):.1f}%

    **Investment Thesis:**
    Microsoft holds a solid industry position in cloud computing and artificial intelligence, with specific sticky business models and strong profitability, combined with a robust balance sheet and sustainable free cash flow, supporting long-term value creation.

    **Key Drivers:**
    1. **Azure Cloud Market Share & High-Margin Recurring Revenue**
    2. **Deep AI Integration & Comprehensive Product Empowerment**
    3. **Enterprise Software Moat & High Customer Lock-in**
    4. **Strong Free Cash Flow & Shareholder Return Capability**
    5. **Subscription-based Recurring Revenue Enhances Predictability**

    **Target Price (12mo):** ${intrinsic_value:.2f}
    **Holding Period:** 12-24 Months
    **Risk Rating:** Medium
    """
        
        return summary
    
    def generate_financial_analysis(self, analysis_data: Dict) -> str:
        """Generate Financial Analysis Section"""
        ratios = analysis_data.get("ratios", {})
        
        content = """## Financial Analysis

    ### Profitability

    Microsoft's high-margin software and cloud service model supports industry-leading profitability levels.
    """

        if "key_ratios" in ratios:
            key_ratios = ratios["key_ratios"]
            content += f"""
    - **Gross Margin:** {key_ratios.get('gross_margin', 0):.1%}
    - **Operating Margin:** {key_ratios.get('operating_margin', 0):.1%}
    - **Net Margin:** {key_ratios.get('net_margin', 0):.1%}
    - **ROE:** {key_ratios.get('roe', 0):.1%}
    - **ROA:** {key_ratios.get('roa', 0):.1%}
    """

        content += """
    ### Growth Analysis

    Cloud penetration, AI acceleration, and digital transformation drive Microsoft's continuous growth.
    """
        if "growth_metrics" in ratios:
            growth = ratios["growth_metrics"]
            content += f"""
    - **Revenue CAGR:** {growth.get('revenue_cagr', 0):.1%}
    - **EPS CAGR:** {growth.get('eps_cagr', 0):.1%}
    - **FCF CAGR:** {growth.get('fcf_cagr', 0):.1%}
    """

        content += """
    ### Financial Health

    Microsoft maintains a high credit rating and robust liquidity.
    """
        if "leverage_ratios" in ratios:
            leverage = ratios["leverage_ratios"]
            content += f"""
    - **Debt to Equity:** {leverage.get('debt_to_equity', 0):.2f}
    - **Interest Coverage:** {leverage.get('interest_coverage', 0):.1f}x
    - **Current Ratio:** {leverage.get('current_ratio', 0):.2f}
    - **FCF Yield:** Strong
    """

        return content
    def generate_valuation_section(self, analysis_data: Dict) -> str:
        """Generate Valuation Section"""
        valuation = analysis_data.get("valuation", {})
        peer = analysis_data.get("peer_analysis", {})
        
        content = """## Valuation Analysis

    ### DCF Valuation

    DCF reflects Microsoft's robust cash flow and moderate growth assumptions.
    """
        
        if "error" not in valuation:
            content += f"""
    - **Intrinsic Value Per Share:** ${valuation.get('intrinsic_value_per_share', 0):.2f}
    - **Current Price:** ${valuation.get('current_price', 0):.2f}
    - **Margin of Safety:** {valuation.get('margin_of_safety', 0)*100:.1f}%
    - **Key DCF Assumptions:**
      - Revenue Growth: {valuation.get('growth_rate', 0)*100:.1f}% (Gradually converging)
      - Discount Rate (WACC): {valuation.get('discount_rate', 0)*100:.1f}%
      - Terminal Growth: {valuation.get('terminal_growth', 0)*100:.1f}%
"""
        
        content += """
    ### Relative Valuation

    Current pricing shows a premium relative to the market, consistent with its growth and quality positioning.
    """
        if "comparative_analysis" in peer:
            comp = peer["comparative_analysis"]
            pe_data = comp.get("pe_ratio", {})
            if pe_data:
                content += f"""
    - **P/E Ratio:** {pe_data.get('target_value', 0):.1f} 
      (Peer Average: {pe_data.get('peer_average', 0):.1f})
      Cloud and high-margin businesses support a certain premium
"""
            
            pb_data = comp.get("pb_ratio", {})
            if pb_data:
                content += f"""
    - **P/B Ratio:** {pb_data.get('target_value', 0):.2f} 
      (Peer Average: {pb_data.get('peer_average', 0):.2f})
      Reflects software company valuation premium
"""
        
        content += """
    ### Valuation Conclusion
    - Relative to Growth: Reasonable range for 10%+ growth
    - Relative to History: Close to historical mean
    - Absolute View: Current risk-reward is balanced
    """
        
        return content
    
    def generate_peer_comparison(self, analysis_data: Dict) -> str:
        """Generate Peer Comparison"""
        peer = analysis_data.get("peer_analysis", {})
        
        content = """## Peer Comparison

### Competitive Positioning

Microsoft covers cloud, productivity software, gaming, and professional services, competing across diverse sectors with: AWS, Google Cloud, Salesforce, Adobe, etc.
"""
        
        if "competitive_positioning" in peer:
            cp = peer["competitive_positioning"]
            content += f"""
- **Market Cap Rank:** #{cp.get('market_leadership', {}).get('market_cap_rank', 'N/A')}
- **Profitability:** Top tier among peers
- **Cloud Market Share:** Global #2, approx 23%+ share
"""
        
        content += """
### Key Competitive Advantages

1. **Integrated Ecosystem**: Windows + Office 365 + Azure creates strong cross-selling and high lock-in
2. **Deep Enterprise Relationships**: Long-term IT partnerships, trusted vendor for critical workloads
3. **Tech & AI Leadership**: High R&D investment, leading AI/ML capabilities and partnerships
4. **Operating Leverage**: Software and cloud scale effects drive margin expansion
5. **Global Distribution**: 190+ countries, diversified revenue reduces geographic risk

### Financial Comparison
Microsoft typically demonstrates: Top-tier profitability and cash flow generation; Excellent capital efficiency; Above-average growth; Conservatively leveraged.
"""
        
        return content
    
    def generate_risk_assessment(self) -> str:
        """Generate Risk Assessment"""
        return """## Risk Assessment

### Business Risks
1. **Cloud Competition**: AWS leads, GCP price competition; Azure mitigates via integrated solutions and enterprise ties.
2. **Regulatory Scrutiny**: Antitrust/privacy compliance on OS/bundling; Continuous investment in compliance and local data centers.
3. **AI Adoption Uncertainty**: Copilot/AI enterprise adoption pace and pricing validation take time; Integration reduces churn.
4. **Macro Sensitivity**: Economic downturns squeeze IT budgets; Core software is resilient but growth slowing risks exist.

### Financial Risks
1. **FX Volatility**: >50% overseas revenue, strong USD is a headwind.
2. **Interest Rate Hikes**: Increases WACC, compresses valuation; Higher debt refinancing costs.
3. **M&A Integration**: Execution risks and goodwill impairment in large acquisitions.

### Mitigating Factors
- Cash rich, Investment Grade rating;
- Diversified revenue (Software, Cloud, Gaming, Ads);
- Ecosystem lock-in and high switching costs;
- Stable management execution.

**Overall Risk Rating: Medium**
"""
    
    def generate_recommendation(self, analysis_data: Dict) -> str:
        """Generate Investment Recommendation"""
        valuation = analysis_data.get("valuation", {})
        
        intrinsic_value = valuation.get("intrinsic_value_per_share", 0)
        current_price = valuation.get("current_price", 0)
        
        if intrinsic_value > 0 and current_price > 0:
            premium = (intrinsic_value - current_price) / current_price * 100
            
            if premium > 20:
                rating = "STRONG BUY"
                rationale = "Significantly undervalued, strong fundamentals and growth momentum"
                target_range = f"${intrinsic_value * 0.9:.2f} - ${intrinsic_value:.2f}"
            elif premium > 10:
                rating = "BUY"
                rationale = "Attrative valuation, positive Cloud & AI outlook"
                target_range = f"${intrinsic_value * 0.95:.2f} - ${intrinsic_value:.2f}"
            elif premium > -10:
                rating = "HOLD"
                rationale = "Valuation near fair value range, suitable for long-term hold"
                target_range = f"${current_price * 0.9:.2f} - ${intrinsic_value:.2f}"
            else:
                rating = "SELL"
                rationale = "Premium relative to intrinsic value"
                target_range = f"${intrinsic_value:.2f} - ${current_price * 0.9:.2f}"
        else:
            rating = "HOLD"
            rationale = "Insufficient data, maintain neutral"
            target_range = "N/A"
        
        return f"""## Investment Recommendation

**Rating:** {rating}

**Rationale:**
{rationale}

Cloud & AI growth, strong FCF, and diversified business moats support long-term value.

**Target Price (12mo):** ${intrinsic_value:.2f}
**Target Range:** {target_range}
**Current Price:** ${current_price:.2f}
**Upside:** {premium:.1f}%

**Holding Period:** 12-24 months (Core holding 3-5 years)
**Position Sizing:** Core 4-6%, Aggressive 8%, Concentration Cap 10%

**Key Monitorables:**
- Azure Growth (Target 25%+/yr)
- Gross & Operating Margin Trends
- FCF Growth (Target 10%+/yr)
- Copilot/AI Monetization & Penetration

**Risk Triggers:**
- Azure growth below 20% for 2 consecutive quarters
- Significant market share loss / Major regulatory setback
- Sustained margin contraction

**Potential Catalysts:**
- AI/Copilot better-than-expected monetization
- Government/Enterprise cloud spending acceleration
- Cloud scale effects driving margin expansion
"""
    
    def generate_full_memo(self, analysis_data: Dict) -> str:
        """Generate Full Memo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"{self.symbol}_TRAD_{timestamp}"
        
        # Generates all sections
        sections = [
            self.generate_executive_summary(analysis_data),
            self.generate_financial_analysis(analysis_data),
            self.generate_valuation_section(analysis_data),
            self.generate_peer_comparison(analysis_data),
            self.generate_risk_assessment(),
            self.generate_recommendation(analysis_data)
        ]
        
        # Build full report
        memo = f"""# Traditional Investment Memorandum

    **Company:** Microsoft Corporation
    **Symbol:** {self.symbol}
    **Report ID:** {report_id}
    **Date:** {datetime.now().strftime("%Y-%m-%d")}
    **Analyst:** AI Fundamental Analyst Agent
    **Confidentiality:** Internal Use Only

    {'='*80}

    """
        
        memo += "\n\n".join(sections)
        
        memo += f"""

{'='*80}

## Appendix

### Data Sources
1. Financial Statements: Local processed CSV (income/balance/cashflow)
2. Market Data: Yahoo Finance (Replaceable with local price files)
3. Peer Data: Public financial reports/industry documents
4. Economic Data: Public macro data

### Valuation Methods
**DCF:** Forecast free cash flow, discounted by WACC, terminal value via perpetual growth.
**Comparable Companies:** Peer multiples, adjusted for growth/profitability/risk differences.
**Relative Valuation:** P/E, P/B, EV/Sales vs historical and peers.

### Key Assumptions
- Revenue Growth (2024-2028): ~10-12%
- Perpetual Growth: ~2.5%
- WACC: ~7.5%
- FCF Margin: ~30-32%

### Important Disclosures
- For educational purposes only, not investment advice.
- Markets and fundamentals may change rapidly; self-assessment and continuous updates required.

### Report Metadata
- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Template Version: 1.0
- Target Company: Microsoft Corporation (MSFT)
- Currency: USD
"""
        
        return memo
    
    def save_memo(self, memo_content: str, output_dir: Path) -> Path:
        """Save Memo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.symbol}_traditional_memo_{timestamp}.md"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(memo_content)
        
        logger.info(f"Traditional memo saved: {filepath}")
        return filepath


def main():
    """Main Function: Generate Traditional Investment Memo"""
    import sys
    from pathlib import Path
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "reports"
    
    symbol = "MSFT"
    
    print(f"\n{'='*70}")
    print(f"Generating Traditional Investment Memo - {symbol}")
    print(f"{'='*70}\n")
    
    # Initialize generator
    generator = TraditionalMemoGenerator(symbol)
    
    # Load analysis data
    print("1. Loading analysis data...")
    analysis_data = generator.load_analysis_data(reports_dir)
    
    if not analysis_data:
        print("❌ No analysis data found. Please run analysis modules first.")
        return
    
    print(f"   ✓ Loaded {len(analysis_data)} analysis datasets")
    
    # Generate memo
    print("2. Generating traditional memo...")
    memo_content = generator.generate_full_memo(analysis_data)
    
    # Save memo
    print("3. Saving memo...")
    output_dir = reports_dir / "investment_memos"
    output_dir.mkdir(exist_ok=True)
    
    memo_path = generator.save_memo(memo_content, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ Traditional Investment Memo Generated!")
    print(f"{'='*70}")
    print(f"Saved to: {memo_path}")
    
    # Display summary
    lines = memo_content.split('\n')
    print("\nMemo Summary (First 20 lines):")
    print("-" * 70)
    for line in lines[:20]:
        if line.strip():
            print(line)
    print("-" * 70)
    print(f"\nTotal memo length: {len(memo_content)} characters, {len(lines)} lines")


if __name__ == "__main__":
    main()
