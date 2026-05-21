import re
import html
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SENTIMENT140_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]


def load_sentiment140(file_path):
    """
    Load Sentiment140 dataset.

    The original Sentiment140 file does not contain column names,
    so column names are assigned manually.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {file_path}")

    df = pd.read_csv(
        file_path,
        encoding="latin-1",
        names=SENTIMENT140_COLUMNS,
        engine="python",
        quotechar='"',
        on_bad_lines="skip"
    )

    return df


def convert_labels(df):
    """
    Convert original Sentiment140 labels:
    0 -> 0 negative
    4 -> 1 positive
    """
    df = df.copy()

    df["label"] = df["target"].map({
        0: 0,
        4: 1
    })

    df["sentiment"] = df["label"].map({
        0: "negative",
        1: "positive"
    })

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    return df


def clean_tweet(text):
    """
    Clean tweet text for sentiment classification.
    """
    text = str(text)

    # Convert HTML entities, for example &amp; -> &
    text = html.unescape(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Replace usernames with common token
    text = re.sub(r"@\w+", " user ", text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)

    # Keep only English letters, apostrophes, and spaces
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_text_cleaning(df):
    """
    Apply text cleaning to the tweet text column.
    """
    df = df.copy()

    df["clean_text"] = df["text"].apply(clean_tweet)

    # Remove empty text rows after cleaning
    df = df[df["clean_text"].str.len() > 0].copy()

    return df


def create_balanced_sample(df, sample_per_class=100000, random_state=42):
    """
    Create a balanced sample with the same number of negative and positive tweets.
    """
    negative_data = df[df["label"] == 0]
    positive_data = df[df["label"] == 1]

    negative_sample = negative_data.sample(
        n=min(sample_per_class, len(negative_data)),
        random_state=random_state
    )

    positive_sample = positive_data.sample(
        n=min(sample_per_class, len(positive_data)),
        random_state=random_state
    )

    df_sample = pd.concat([negative_sample, positive_sample], axis=0)
    df_sample = df_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_sample


def split_data(df, test_size=0.2, val_size=0.5, random_state=42):
    """
    Split data into train, validation, and test sets.

    First split:
    80% train, 20% temporary

    Second split:
    10% validation, 10% test
    """
    X = df["clean_text"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir="processed"):
    """
    Save processed train, validation, and test files for Week 3 deep learning models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    train_df = pd.DataFrame({
        "clean_text": X_train.values,
        "label": y_train.values
    })

    val_df = pd.DataFrame({
        "clean_text": X_val.values,
        "label": y_val.values
    })

    test_df = pd.DataFrame({
        "clean_text": X_test.values,
        "label": y_test.values
    })

    train_df.to_csv(output_dir / "week02_train.csv", index=False)
    val_df.to_csv(output_dir / "week02_val.csv", index=False)
    test_df.to_csv(output_dir / "week02_test.csv", index=False)

    print("Processed files saved:")
    print(output_dir / "week02_train.csv")
    print(output_dir / "week02_val.csv")
    print(output_dir / "week02_test.csv")
