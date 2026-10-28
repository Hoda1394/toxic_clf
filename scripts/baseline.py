"""TF-IDF + logistic regression baseline for toxic comment classification.

Uses the same train/val/test split as scripts/train.py, so its numbers are
directly comparable to the transformer's. The decision threshold is tuned on
the validation set and then applied unchanged to the test set.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_civil_comments_data


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Return the probability threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    # The last precision/recall pair has no matching threshold
    return float(thresholds[f1[:-1].argmax()])


def report(name: str, labels: np.ndarray, probs: np.ndarray, threshold: float):
    """Print threshold-free and thresholded metrics."""
    preds = (probs >= threshold).astype(int)
    print(
        f"{name}: "
        f"roc_auc={roc_auc_score(labels, probs):.4f}  "
        f"pr_auc={average_precision_score(labels, probs):.4f}  "
        f"precision={precision_score(labels, preds):.4f}  "
        f"recall={recall_score(labels, preds):.4f}  "
        f"f1={f1_score(labels, preds):.4f}  "
        f"(threshold={threshold:.2f})"
    )


def main(args):
    """Train and evaluate the baseline."""

    print(f"Loading data from {args.data_path}")
    train_df, val_df, test_df = load_civil_comments_data(
        args.data_path,
        text_column=args.text_column,
        target_column=args.target_column,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        dedup=args.dedup,
        official_test=args.official_test,
    )

    if args.max_train_samples is not None and args.max_train_samples < len(train_df):
        train_df = train_df.sample(args.max_train_samples, random_state=args.seed)
    print(f"Train size: {len(train_df)}, val size: {len(val_df)}, test size: {len(test_df)}")

    # Word n-grams capture phrases; character n-grams survive obfuscation
    # such as "sh*tty" and misspellings
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=args.max_features,
        min_df=3,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=args.max_features,
        min_df=5,
        sublinear_tf=True,
    )

    print("Fitting TF-IDF features...")
    train_texts = train_df[args.text_column]
    X_train = hstack([
        word_vectorizer.fit_transform(train_texts),
        char_vectorizer.fit_transform(train_texts),
    ]).tocsr()

    def transform(texts):
        return hstack([
            word_vectorizer.transform(texts),
            char_vectorizer.transform(texts),
        ]).tocsr()

    X_val = transform(val_df[args.text_column])
    X_test = transform(test_df[args.text_column])

    print("Training logistic regression...")
    model = LogisticRegression(C=args.C, max_iter=1000)
    # After preprocessing, the label column is binary (0 or 1) returned by load_civil_comments_data
    model.fit(X_train, train_df["label"])   

    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    val_labels = val_df["label"].values
    test_labels = test_df["label"].values

    threshold = best_f1_threshold(val_labels, val_probs)

    print("\nResults at the default threshold:")
    report("val ", val_labels, val_probs, 0.5)
    report("test", test_labels, test_probs, 0.5)

    print("\nResults at the threshold tuned on validation:")
    report("val ", val_labels, val_probs, threshold)
    report("test", test_labels, test_probs, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + logistic regression baseline"
    )

    # Data arguments (defaults match src/config.py so the split is identical)
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--target_column", type=str, default="target")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Collapse duplicate texts before splitting",
    )
    parser.add_argument(
        "--official_test",
        action="store_true",
        help="Test on the official hold-out set (test_public/private_expanded.csv); the CSV is split into train/val only",
    )

    # Baseline arguments
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=400_000,
        help="Subsample the training set to limit memory and time (0 uses the full training set)",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=100_000,
        help="Maximum features for each of the word and character vectorizers",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=4.0,
        help="Inverse regularization strength for logistic regression",
    )

    args = parser.parse_args()
    if args.max_train_samples == 0:
        args.max_train_samples = None
    main(args)
