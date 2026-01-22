# AI Investment Report Generator - LLM Prompts Documentation

> This document archives all LLM prompt templates used in `src/ai_report_generator.py` inside the `create_prompt()` method.

---

## Table of Contents
1. [detailed](#detailed---english) - Detailed English Version

---

## DETAILED - English

**Usage**: Generate 3-4 page comprehensive research report  
**Language**: English  
**Target Audience**: Deep Research, Board of Directors  
**Word Count**: Approx. 1500-2000 words

```
You are a fundamental research analyst. Write a comprehensive investment report for {SYMBOL}.

Analysis Context: {CONTEXT}

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
```

---

## Usage Guide

### Default Prompt Type

The default and only supported prompt type is:

| Scenario | Recommended Prompt | Language | Length |
|---|---|---|---|
| Detailed Research | **detailed** | English | ~1500-2000 words |

### Usage Example

```python
from src.ai_report_generator import AIReportGenerator

generator = AIReportGenerator(model="qwen3-max", temperature=0.3)

# Use detailed (default)
result = generator.generate_ai_memo(memo_type="detailed")
```

---

## Model Parameters

- **{SYMBOL}**: Stock Symbol (e.g., MSFT, AAPL)
- **{CONTEXT}**: Analytical context data from `prepare_context_data()` (includes financial ratios, valuation, peer comparison, etc.)

---

## Customization Guide

### To Customize the Prompt:

1. **Edit Text**: Modify the `create_prompt()` method in `src/ai_report_generator.py`.
2. **Add New Type**: Add a new key-value pair to the `prompt_templates` dictionary in the code.

---

**Last Updated**: January 8, 2026  
**Maintainer**: AI Fundamental Analyst Agent

