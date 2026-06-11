"""
review_analyzer_pipeline.py
----------------------------
Production warehouse version of the NLP sentiment analysis pipeline.
Supports both Snowflake and BigQuery via the --warehouse flag.
Pulls product reviews directly from your warehouse, runs sentiment classification,
keyword extraction, and topic modeling, then writes results back to a
destination table for downstream BI/dashboard consumption.

Setup:
    1. Copy .env.example to .env and fill in your credentials
    2. pip install -r requirements.txt
    3. See usage examples below

Usage:

    Snowflake:
        python review_analyzer_pipeline.py \
            --warehouse snowflake \
            --source_table RAW.REVIEWS.YOTPO \
            --dest_table ANALYTICS.NLP.SENTIMENT_RESULTS

    BigQuery:
        python review_analyzer_pipeline.py \
            --warehouse bigquery \
            --source_table myproject.raw_reviews.yotpo \
            --dest_table myproject.analytics.sentiment_results
"""

import argparse
import os
import re
import logging
from datetime import datetime
from abc import ABC, abstractmethod

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Warehouse Connectors ──────────────────────────────────────────────────────

class WarehouseConnector(ABC):
    """Abstract base class for warehouse connectors."""

    @abstractmethod
    def fetch(self, source_table: str, review_col: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def write(self, df: pd.DataFrame, dest_table: str) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class SnowflakeConnector(WarehouseConnector):
    """Snowflake connector using environment variable credentials."""

    def __init__(self):
        import snowflake.connector
        self.conn = snowflake.connector.connect(
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
        )
        log.info("Connected to Snowflake.")

    def fetch(self, source_table: str, review_col: str) -> pd.DataFrame:
        query = f"""
            SELECT *
            FROM {source_table}
            WHERE {review_col} IS NOT NULL
              AND TRIM({review_col}) != ''
        """
        log.info(f"Fetching reviews from {source_table}...")
        cursor = self.conn.cursor()
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
        log.info(f"Fetched {len(df):,} reviews.")
        return df

    def write(self, df: pd.DataFrame, dest_table: str) -> None:
        from snowflake.connector.pandas_tools import write_pandas
        log.info(f"Writing {len(df):,} rows to {dest_table}...")
        success, nchunks, nrows, _ = write_pandas(
            self.conn,
            df,
            table_name=dest_table.split(".")[-1],
            database=dest_table.split(".")[0] if dest_table.count(".") >= 2 else None,
            schema=dest_table.split(".")[1] if dest_table.count(".") >= 2 else None,
            auto_create_table=True,
            overwrite=True,
        )
        if success:
            log.info(f"Successfully wrote {nrows:,} rows in {nchunks} chunk(s).")
        else:
            raise RuntimeError("Snowflake write failed.")

    def close(self) -> None:
        self.conn.close()
        log.info("Snowflake connection closed.")


class BigQueryConnector(WarehouseConnector):
    """BigQuery connector using a service account JSON key."""

    def __init__(self):
        from google.cloud import bigquery
        from google.oauth2 import service_account

        key_path = os.environ.get("BIGQUERY_KEY_PATH")
        project = os.environ["BIGQUERY_PROJECT"]

        if key_path:
            credentials = service_account.Credentials.from_service_account_file(key_path)
            self.client = bigquery.Client(project=project, credentials=credentials)
        else:
            # Falls back to application default credentials (e.g. gcloud auth)
            self.client = bigquery.Client(project=project)

        self.project = project
        log.info(f"Connected to BigQuery project: {project}")

    def fetch(self, source_table: str, review_col: str) -> pd.DataFrame:
        query = f"""
            SELECT *
            FROM `{source_table}`
            WHERE {review_col} IS NOT NULL
              AND TRIM(CAST({review_col} AS STRING)) != ''
        """
        log.info(f"Fetching reviews from {source_table}...")
        df = self.client.query(query).to_dataframe()
        log.info(f"Fetched {len(df):,} reviews.")
        return df

    def write(self, df: pd.DataFrame, dest_table: str) -> None:
        from google.cloud import bigquery
        log.info(f"Writing {len(df):,} rows to {dest_table}...")
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = self.client.load_table_from_dataframe(df, dest_table, job_config=job_config)
        job.result()
        log.info(f"Successfully wrote {len(df):,} rows to {dest_table}.")

    def close(self) -> None:
        self.client.close()
        log.info("BigQuery connection closed.")


def get_connector(warehouse: str) -> WarehouseConnector:
    """Factory function — returns the correct connector based on --warehouse flag."""
    if warehouse == "snowflake":
        return SnowflakeConnector()
    elif warehouse == "bigquery":
        return BigQueryConnector()
    else:
        raise ValueError(f"Unsupported warehouse: {warehouse}. Choose 'snowflake' or 'bigquery'.")


# ── Text Preprocessing ────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Lowercase, strip special characters and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


# ── Sentiment Analysis ────────────────────────────────────────────────────────

def analyze_sentiment(reviews: pd.Series) -> pd.Series:
    """Classify each review as positive, neutral, or negative via TextBlob."""
    def classify(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    return reviews.apply(classify)


# ── Keyword Extraction ────────────────────────────────────────────────────────

def extract_keywords(reviews: pd.Series, n: int = 10) -> list[tuple]:
    """Return top n keywords by TF-IDF score."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(reviews)
    keywords = vectorizer.get_feature_names_out()
    word_scores = tfidf_matrix.sum(axis=0).A1
    ranked = sorted(zip(keywords, word_scores), key=lambda x: x[1], reverse=True)
    return ranked[:n]


# ── Topic Modeling ────────────────────────────────────────────────────────────

def topic_modeling(reviews: pd.Series, n_topics: int = 5, n_words: int = 10) -> list[list]:
    """Run LDA topic modeling and return top words per topic."""
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()
    return [
        [feature_names[i] for i in topic.argsort()[-n_words:]]
        for topic in lda.components_
    ]


def assign_topic(review: str, lda, vectorizer) -> int:
    """Assign the dominant LDA topic index to a single review."""
    vec = vectorizer.transform([review])
    return int(lda.transform(vec).argmax(axis=1)[0])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NLP Sentiment Pipeline (Snowflake + BigQuery)")
    parser.add_argument("--warehouse", default="snowflake", choices=["snowflake", "bigquery"],
                        help="Target warehouse (default: snowflake)")
    parser.add_argument("--source_table", required=True, help="Fully qualified source table")
    parser.add_argument("--review_col", default="REVIEW_BODY", help="Column containing review text")
    parser.add_argument("--dest_table", required=True, help="Fully qualified destination table for results")
    parser.add_argument("--n_topics", type=int, default=5, help="Number of LDA topics")
    args = parser.parse_args()

    connector = get_connector(args.warehouse)

    try:
        # Fetch
        df = connector.fetch(args.source_table, args.review_col)

        # Preprocess
        df['CLEANED_REVIEW'] = df[args.review_col].fillna("").apply(preprocess_text)
        df = df[df['CLEANED_REVIEW'].str.strip() != ""].reset_index(drop=True)
        log.info(f"{len(df):,} reviews after cleaning.")

        # Sentiment
        df['SENTIMENT'] = analyze_sentiment(df['CLEANED_REVIEW'])
        log.info(f"Sentiment distribution:\n{df['SENTIMENT'].value_counts().to_string()}")

        # Keywords (logged, not written to table)
        log.info("Top Keywords (TF-IDF):")
        for word, score in extract_keywords(df['CLEANED_REVIEW']):
            log.info(f"  {word}: {score:.3f}")

        # Topic Modeling
        count_vec = CountVectorizer(stop_words='english')
        doc_term_matrix = count_vec.fit_transform(df['CLEANED_REVIEW'])
        lda = LatentDirichletAllocation(n_components=args.n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        df['DOMINANT_TOPIC'] = df['CLEANED_REVIEW'].apply(lambda r: assign_topic(r, lda, count_vec))

        feature_names = count_vec.get_feature_names_out()
        topic_labels = {
            i: ", ".join([feature_names[j] for j in topic.argsort()[-5:]])
            for i, topic in enumerate(lda.components_)
        }
        df['TOPIC_KEYWORDS'] = df['DOMINANT_TOPIC'].map(topic_labels)

        log.info("LDA Topics:")
        for i, label in topic_labels.items():
            log.info(f"  Topic {i + 1}: {label}")

        # Pipeline metadata
        df['PIPELINE_RUN_AT'] = datetime.utcnow().isoformat()
        df['SOURCE_TABLE'] = args.source_table

        # Write results
        output_cols = [args.review_col, 'CLEANED_REVIEW', 'SENTIMENT', 'DOMINANT_TOPIC',
                       'TOPIC_KEYWORDS', 'PIPELINE_RUN_AT', 'SOURCE_TABLE']
        connector.write(df[output_cols], args.dest_table)

    finally:
        connector.close()


if __name__ == "__main__":
    main()
