"""
review_analyzer.py
------------------
Local CSV version of the NLP sentiment analysis pipeline.
Ingests product reviews from a CSV export (Klaviyo, Yotpo, etc.),
runs sentiment classification, keyword extraction, and topic modeling,
and outputs results to a summary CSV.

Usage:
    python review_analyzer.py --input customer_reviews.csv --review_col "reviews/content"
"""

import argparse
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt


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
    """
    Classify each review as positive, neutral, or negative
    using TextBlob polarity scoring.
    """
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
    """Return the top n keywords by TF-IDF score across all reviews."""
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


# ── Visualization ─────────────────────────────────────────────────────────────

def create_wordcloud(reviews: pd.Series, title: str = "Review Word Cloud") -> None:
    """Generate and display a word cloud from review text."""
    text = " ".join(reviews)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_sentiment_distribution(sentiment_series: pd.Series) -> None:
    """Bar chart of sentiment label counts."""
    counts = sentiment_series.value_counts()
    counts.plot(kind='bar', color=['#4CAF50', '#9E9E9E', '#F44336'], edgecolor='black')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Review Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NLP Sentiment Analysis Pipeline (Local CSV)")
    parser.add_argument("--input", default="customer_reviews.csv", help="Path to input CSV file")
    parser.add_argument("--review_col", default="reviews/content", help="Column name containing review text")
    parser.add_argument("--output", default="sentiment_results.csv", help="Path to output CSV file")
    parser.add_argument("--n_topics", type=int, default=5, help="Number of LDA topics")
    args = parser.parse_args()

    # Load
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    if args.review_col not in df.columns:
        raise ValueError(f"Column '{args.review_col}' not found. Available columns: {list(df.columns)}")

    # Preprocess
    df['cleaned_review'] = df[args.review_col].fillna("").apply(preprocess_text)
    df = df[df['cleaned_review'].str.strip() != ""].reset_index(drop=True)
    print(f"Processing {len(df):,} reviews...")

    # Sentiment
    df['sentiment'] = analyze_sentiment(df['cleaned_review'])
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts().to_string())

    # Keywords
    print("\nTop Keywords (TF-IDF):")
    for word, score in extract_keywords(df['cleaned_review']):
        print(f"  {word}: {score:.3f}")

    # Topics
    print("\nLDA Topics:")
    topics = topic_modeling(df['cleaned_review'], n_topics=args.n_topics)
    for i, topic in enumerate(topics):
        print(f"  Topic {i + 1}: {', '.join(topic)}")

    # Visualize
    create_wordcloud(df['cleaned_review'])
    plot_sentiment_distribution(df['sentiment'])

    # Export
    df[[args.review_col, 'cleaned_review', 'sentiment']].to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
