# NLP Sentiment Analysis Pipeline

A Python/SQL pipeline that classifies sentiment and extracts themes from product reviews at scale, delivering actionable insights through a self-service dashboard.

## Overview

Built to replace manual review analysis across multiple DTC e-commerce accounts. The pipeline ingests thousands of product reviews, runs sentiment classification and theme extraction, and surfaces results in a dashboard used directly by product and marketing teams.

## Tech Stack

- **Python** — pipeline orchestration, NLP modeling
- **SQL** — data transformation and aggregation
- **Snowflake / BigQuery** — cloud data warehouse
- **Looker** — self-service dashboard layer

## What It Does

1. Ingests raw product review data at scale
2. Classifies sentiment (positive / neutral / negative) per review
3. Extracts recurring themes and topics across review sets
4. Loads structured output into a warehouse table
5. Powers a dashboard for product, marketing, and CX teams

## Impact

- Processed thousands of reviews across multiple client accounts
- Insights directly informed product development and feature prioritization
- Replaced hours of manual tagging with automated, repeatable pipeline
