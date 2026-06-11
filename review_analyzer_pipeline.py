"""
review_analyzer_pipeline.py
----------------------------
Production warehouse version of the NLP sentiment analysis pipeline.
Pulls product reviews directly from Snowflake, runs sentiment classification,
keyword extraction, and topic modeling, then writes results back to a
destination table for downstream BI/dashboard consumption.

Setup:
    1. Copy .env.example to .env and fill in your Snowflake credentials
    2. pip install -r requirements.txt
    3. python review_analyzer_pipeline.py --source_table RAW.REVIEWS.YOTPO --dest_table ANALYTICS.NLP.SENTIMENT_RESULTS

Usage:
    python review_analyzer_pipeline.py \
        --source_table RAW.REVIEWS.YOTPO \
        --review_col REVIEW_BODY \
        --dest_table ANALYTICS.NLP.SENTIMENT_RESULTS \
        --n_topics 5
"""

import argparse
import os
import re
import logging
from datetime import datetime

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Snowflake Connection ──────────────────────────────────────────────────────

def get_snowflake_connection():
    """Create and return a Snowflake connection using environment variables."""
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def fetch_reviews(conn, source_table: str, review_col: str) -> pd.DataFrame:
    """Pull all reviews from the specified source table."""
    query = f"""
        SELECT *
        FROM {source_table}
        WHERE {review_col} IS NOT NULL
          AND TRIM({review_col}) != ''
    """
    log.info(f"Fetching reviews from {source_table}...")
    cursor = conn.cursor()
    cursor.execute(query)
    df = cursor.fetch_pandas_all()
    log.info(f"Fetched {len(df):,} reviews.")
    return df


def write_results(conn, df: pd.DataFrame, dest_table: str) -> None:
    """Write sentiment results back to Snowflake destination table."""
    from snowflake.connector.pandas_tools import write_pandas

    log.info(f"Writing {len(df):,} rows to {dest_table}...")
    success, nchunks, nrows, _ = write_pandas(
        conn,
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
    parser = argparse.ArgumentParser(description="NLP Sentiment Pipeline (Snowflake)")
    parser.add_argument("--source_table", required=True, help="Fully qualified source table (DB.SCHEMA.TABLE)")
    parser.add_argument("--review_col", default="REVIEW_BODY", help="Column containing review text")
    parser.add_argument("--dest_table", required=True, help="Fully qualified destination table for results")
    parser.add_argument("--n_topics", type=int, default=5, help="Number of LDA topics")
    args = parser.parse_args()

    conn = get_snowflake_connection()

    try:
        # Fetch
        df = fetch_reviews(conn, args.source_table, args.review_col)

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

        # Topic Modeling — assign dominant topic per review
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

        # Add pipeline metadata
        df['PIPELINE_RUN_AT'] = datetime.utcnow().isoformat()
        df['SOURCE_TABLE'] = args.source_table

        # Write results
        output_cols = [args.review_col, 'CLEANED_REVIEW', 'SENTIMENT', 'DOMINANT_TOPIC', 'TOPIC_KEYWORDS', 'PIPELINE_RUN_AT', 'SOURCE_TABLE']
        write_results(conn, df[output_cols], args.dest_table)

    finally:
        conn.close()
        log.info("Snowflake connection closed.")


if __name__ == "__main__":
    main()
