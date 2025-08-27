# predict.py
import argparse
import sys
import joblib
from pathlib import Path


def load_pipeline(path: str = "text_clf.joblib"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing '{path}'. Train the model first by running: python Train_Model.py"
        )
    return joblib.load(p)


def main():
    parser = argparse.ArgumentParser(description="Predict a label for input text.")
    parser.add_argument(
        "-t", "--text",
        type=str,
        help="Text to classify. If omitted, reads from STDIN.",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="text_clf.joblib",
        help="Path to saved pipeline (.joblib).",
    )
    args = parser.parse_args()

    if args.text is None:
        print("Enter/paste text, then Ctrl-D (Unix) or Ctrl-Z + Enter (Windows):")
        text = sys.stdin.read().strip()
    else:
        text = args.text.strip()

    if not text:
        print("No text provided.")
        sys.exit(1)

    pipe = load_pipeline(args.model)
    pred = pipe.predict([text])[0]
    proba = None
    if hasattr(pipe[-1], "predict_proba"):
        # Get probability of the predicted class for a friendly confidence signal
        probs = pipe[-1].predict_proba(pipe[:-1].transform([text]))[0]
        # map index back to class name
        classes = list(pipe[-1].classes_)
        proba = dict(zip(classes, probs)).get(pred, None)

    if proba is not None:
        print(f"Prediction: {pred}  (confidence ~ {proba:.2f})")
    else:
        print(f"Prediction: {pred}")


if __name__ == "__main__":
    main()
