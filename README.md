# NLP Sentiment Analysis Pipeline

A Python/SQL pipeline that classifies sentiment and extracts themes from product reviews at scale, delivering actionable insights through a self-service dashboard. Supports both local CSV analysis and production warehouse deployments on Snowflake and BigQuery.

## Overview

Built to replace manual review analysis across multiple DTC e-commerce accounts. The pipeline ingests thousands of product reviews, runs sentiment classification and theme extraction, and surfaces results in a dashboard used directly by product and marketing teams.

## Files

| File | Description |
|------|-------------|
| `review_analyzer.py` | Local version — ingests CSV exports from Yotpo, Okendo, etc. |
| `review_analyzer_pipeline.py` | Production version — pulls from Snowflake or BigQuery, writes results back to warehouse |
| `customer_reviews_sample.csv` | Sample dataset for testing |
| `requirements.txt` | Python dependencies |
| `.env.example` | Credential template for Snowflake and BigQuery (copy to `.env`, never commit) |

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

**Try it with sample data:**

    python review_analyzer.py --input customer_reviews_sample.csv --review_col "reviews/content"

**Local (CSV):**

    python review_analyzer.py --input customer_reviews.csv --review_col "reviews/content"

**Warehouse — Snowflake:**

    cp .env.example .env
    python review_analyzer_pipeline.py \
      --warehouse snowflake \
      --source_table RAW.REVIEWS.YOTPO \
      --dest_table ANALYTICS.NLP.SENTIMENT_RESULTS

**Warehouse — BigQuery:**

    cp .env.example .env
    python review_analyzer_pipeline.py \
      --warehouse bigquery \
      --source_table myproject.raw_reviews.yotpo \
      --dest_table myproject.analytics.sentiment_results

## Taking It Further: AI-Powered Advertiser Insights

One of the most underrated truths in DTC marketing: **customers are usually better copywriters than the brand.**

Advertisers spend weeks workshopping headlines and angles, when the raw material already exists in their reviews. Phrases like *"my coworkers keep asking what I'm doing differently"* or *"I've tried every eye cream on the market and this is the best"* are ready-made ad copy — they just need to be found.

This pipeline includes an optional `--generate_insights` flag that pipes the classified review data into Claude to surface:

- **Emotional themes** — the underlying drivers behind why people love or reject a product, beyond surface-level keywords
- **Standout customer phrases** — verbatim language from real buyers that works as social proof or headline copy
- **Ad copy angles** — fully written headline and body copy based on what customers actually say
- **Objection mapping** — the top complaints from negative reviews that smart ads should proactively address

The output turns a sentiment dataset into a creative brief. Run it with:

    python review_analyzer.py --input customer_reviews_sample.csv --review_col "reviews/content" --generate_insights

Requires an `ANTHROPIC_API_KEY` in your `.env` file.

## Impact

- Processed thousands of reviews across multiple DTC client accounts
- Insights directly informed product development and feature prioritization
- Replaced hours of manual tagging with an automated, repeatable pipeline
