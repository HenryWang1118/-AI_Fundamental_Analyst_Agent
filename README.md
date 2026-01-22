#  AI Fundamental Analyst Agent

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
##  English Version

### Overview
A comprehensive AI-driven fundamental analysis agent that integrates financial statement analysis, valuation models, peer comparison, and automated investment memo generation. It fetches data, performs quantitative analysis, and uses LLMs to write professional-grade investment reports.

###  Key Features

1.  ** Financial Analysis**
    *   Ingests Income Statement, Balance Sheet, Cash Flow.
    *   Calculates 16+ key financial ratios (Profitability, Liquidity, Solvency, Efficiency).
    *   5-year historical trend analysis.

2.  ** Valuation Modeling**
    *   **DCF Model**: Discounted Cash Flow analysis with scenarios.
    *   **Relative Valuation**: Peer multiple comparison (P/E, etc.).

3.  ** Peer Analysis**
    *   Compare against key competitors.
    *   Automatic metric comparison and ranking.

4.  ** AI Report Generation**
    *   Generates professional detailed investment memos.
    *   **Default**: "Detailed" English report (~1500-2000 words).
    *   Powered by LLMs (e.g., Qwen/GPT) via DashScope/OpenAI compatible API.

###  Project Structure

```text
AI_Fundamental_Analyst_Agent/
 data/
    processed/       # Processed CSVs (financials, ratios, etc.)
 reports/             # All generated reports (Markdown, JSON, Figures)
 src/                 # Core source code
    ai_report_generator.py
    data_ingestion.py
    valuation_models.py
    ...
 demo/
    run_agent.py     # Main entry point script
 PROMPTS.md           # Prompt templates documentation
 requirements.txt     # Python dependencies
```

###  Quick Start

**Requirements**: Python 3.10 or higher is recommended.

#### 1. Installation
```bash
# Clone repository
git clone <repository-url>
cd AI_Fundamental_Analyst_Agent

# Create & activate virtual environment (recommended)
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configuration
Create a .env file in the root directory (based on src/config.py):
```ini
ALPHA_VANTAGE_API_KEY=your_key_here
DASHSCOPE_API_KEY=your_key_here
# Optional
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3-max
```

#### 3. Usage
Run the demo script to fetch data, analyze, and generate a report:

```bash
# Run the full pipeline (Fetch Data -> Analyze -> Generate Report)
python demo/run_agent.py
```

The output report will be saved directly to the reports/ folder.