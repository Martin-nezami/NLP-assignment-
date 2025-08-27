# Train_Model.py
import time
import random
import joblib
import pandas as pd
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import wikipediaapi


def get_wikipedia_content(title: str, lang: str = "en", user_agent: Optional[str] = None) -> Optional[str]:
    """
    Fetch the full text of a Wikipedia page. Returns None if the page doesn't exist.
    """
    if user_agent is None:
        user_agent = "NLP-Assignment/1.0 (contact: you@example.com)"
    wiki = wikipediaapi.Wikipedia(
        language=lang,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=user_agent,
    )
    page = wiki.page(title)
    if not page.exists():
        return None
    # Use .text which includes sections; .summary would be short.
    return page.text


def build_dataset(topics: Dict[str, List[str]], sleep_sec: float = 0.8) -> pd.DataFrame:
    """
    Given a mapping of label -> list of Wikipedia page titles, download content and
    return a DataFrame with columns ['text', 'label'].
    """
    rows = []
    for label, titles in topics.items():
        for t in titles:
            txt = get_wikipedia_content(t)
            # be nice to Wikipedia
            time.sleep(sleep_sec)
            if txt and len(txt.strip()) > 200:
                rows.append({"text": txt, "label": label})
            else:
                print(f"[warn] Skipping '{t}' (missing or too short)")
    if not rows:
        raise RuntimeError("No data fetched from Wikipedia. Check your internet connection or titles.")
    df = pd.DataFrame(rows)
    # Shuffle rows for good measure
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def main():
    # --- 1) Define a tiny, balanced toy dataset via Wikipedia titles ---
    topics = {
        "technology": [
            "Artificial intelligence",
            "Machine learning",
            "Computer hardware",
            "Software engineering",
            "Cloud computing",
        ],
        "sports": [
            "Association football",
            "Basketball",
            "Tennis",
            "Cricket",
            "Rugby union",
        ],
        "history": [
            "Ancient Rome",
            "French Revolution",
            "Industrial Revolution",
            "World War II",
            "Renaissance",
        ],
    }

    print("[1/5] Downloading Wikipedia pages…")
    dataset = build_dataset(topics)
    print(f"Fetched {len(dataset)} documents "
          f"({dataset['label'].value_counts().to_dict()})")

    # --- 2) Split ---
    print("[2/5] Splitting train/test…")
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"],
    )

    # --- 3) Build pipeline (vectorizer + classifier) ---
    print("[3/5] Building pipeline…")
    pipe = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,       # keep all terms (dataset is tiny)
            max_df=0.95,    # ignore ultra-common tokens
        ),
        LogisticRegression(
            max_iter=1000,
            n_jobs=None,    # keep default; set to None for portability
            solver="lbfgs",
            multi_class="auto",
        ),
    )

    # --- 4) Train & evaluate ---
    print("[4/5] Training…")
    pipe.fit(X_train, y_train)

    print("[5/5] Evaluating…")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    # Quick sanity predictions
    samples = [
        "The CPU and GPU are central to computer hardware, while cloud platforms host scalable services.",
        "The French monarchy faced widespread unrest that culminated in a revolution.",
        "The team scored two goals in a thrilling football match at the stadium.",
    ]
    for s in samples:
        print(f"[demo] {pipe.predict([s])[0]}  |  {s[:80]}…")

    # --- 5) Save model pipeline ---
    out_path = "text_clf.joblib"
    joblib.dump(pipe, out_path)
    print(f"\nSaved trained pipeline to: {out_path}")


if __name__ == "__main__":
    main()