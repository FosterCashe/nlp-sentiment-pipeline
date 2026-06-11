# NLP Sentiment Analysis Pipeline

A Python/SQL pipeline that classifies sentiment and extracts themes from product reviews at scale, delivering actionable insights through a self-service dashboard.

## Overview

Built to replace manual review analysis across multiple DTC e-commerce accounts. The pipeline ingests thousands of product reviews, runs sentiment classification and theme extraction, and surfaces results in a dashboard used directly by product and marketing teams.

## Files

| File | Description |
|------|-------------|
| `review_analyzer.py` | Local version — ingests CSV exports from Yotpo, Okendo, etc. |
| `review_analyzer_pipeline.py` | Production version — pulls from Snowflake, writes results back to warehouse |
| `requirements.txt` | Python dependencies |
| `.env.example` | Snowflake credential template (copy to `.env`, never commit) |

## Tech Stack

- **Python** — pipeline orchestration, NLP modeling
- **SQL** — data transformation and aggregation
- **Snowflake / BigQuery** — cloud data warehouse
- **scikit-learn** — TF-IDF keyword extraction, LDA topic modeling
- **TextBlob** — sentiment classification
- **Looker** — self-service dashboard layer

## How It Works

1. Ingests raw product review data (CSV or direct from warehouse)
2. Classifies sentiment per review (positive / neutral / negative)
3. Extracts top keywords via TF-IDF scoring
4. Assigns dominant topic per review via LDA topic modeling
5. Writes structured output back to warehouse for downstream BI consumption

## Usage

**Local (CSV):**
