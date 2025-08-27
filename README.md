# Wikipedia Text Classifier

## Overview
This project implements a **text classification System** using **Wikipedia articles** as the dataset. through by Natural Language Processing (NLP) and Machine Learning (ML) techniques.
This project demonstrates how to build, train, and evaluate a text classifier with **scikit-learn**, and how to use the trained model to make predictions on new text.
The system leverages **TF-IDF vectorization** for feature extraction and **Logistic Regression** for classification, making it lightweight, interpretable, and effective for demonstration purposes.  

---

## Project Structure  

1. **Data Collection**  
   - Uses the `wikipedia-api` library to fetch full-text articles from Wikipedia.  
   - Articles are grouped into predefined categories (technology, sports, history).  
   - Builds a labeled dataset (`text`, `label`) for training.  

2. **Preprocessing & Feature Extraction**  
   - Applies **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical vectors.  
   - Supports unigrams and bigrams for richer context.  

3. **Model Training**  
   - Implements a **Logistic Regression classifier** wrapped in a scikit-learn pipeline.  
   - Splits dataset into training and testing sets for evaluation.  
   - Trains and saves the pipeline as `text_clf.joblib`.  

4. **Evaluation**  
   - Computes classification accuracy, precision, recall, and F1-score.  
   - Displays predictions on sample test sentences for quick validation.  

5. **Prediction**  
   - A command-line tool (`predict.py`) is provided to classify arbitrary input text.  
   - Reports predicted label and, if available, the confidence score.  

---

## Results and Example Predictions

After training, you can expect outputs like:

```
Accuracy: 0.87

Prediction: technology  |  The CPU and GPU are central to computer hardware, while cloud platforms...
Prediction: history     |  The French monarchy faced widespread unrest that culminated in a revolution...
Prediction: sports      |  The team scored two goals in a thrilling football match at the stadium...
```

---

## Conclusion

This project demonstrates how to build a simple yet effective **NLP text classification system** with Python and scikit-learn.
By leveraging **Wikipedia articles** as a dataset and combining **TF-IDF features** with **Logistic Regression**, the system achieves reliable classification across multiple domains.

It serves as a clear, educational example of applying **NLP + ML** to real-world text classification tasks.

```

---

⚡ This is **Markdown-ready** — just paste it into your `README.md` file in VS Code and it will render cleanly on GitHub.  

---

## Features

* Automatically fetches Wikipedia pages for selected topics (`technology`, `sports`, `history`).
* Builds a **TF-IDF + Logistic Regression** pipeline.
* Trains and evaluates the model on a balanced dataset.
* Saves the trained model as `text_clf.joblib`.
* Provides a CLI tool (`predict.py`) to classify new input text.

---


## Project Structure
```
├── Train_Model.py      # Script to fetch data, train the model, and save it
├── predict.py          # CLI tool to load the model and classify text
├── requirements.txt    # Dependencies
├── text_clf.joblib     # Saved trained model (generated after training)
```

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/WikiTextClassifier.git
cd WikiTextClassifier
pip install -r requirements.txt
```

---

## Usage



### 1. Train the Model
Run the training script to fetch data from Wikipedia, train the model, evaluate it, and save the trained pipeline:

```bash
python Train_Model.py
```

This will generate a file named `text_clf.joblib`.

---



### 2. Predict with the Model
You can classify text using the `predict.py` script.

**Option A: Provide text via argument**

```bash
python predict.py -t "Basketball is a popular sport with millions of fans worldwide."
```

**Option B: Input text manually (STDIN)**

```bash
python predict.py
```

Then paste or type your text and press **Ctrl+D** (Linux/macOS) or **Ctrl+Z + Enter** (Windows).

**Output Example:**

```
Prediction: sports  (confidence ~ 0.88)
```

---

## Requirements

See [requirements.txt](requirements.txt).
Main dependencies include:

* `scikit-learn`
* `pandas`
* `spacy`
* `wikipedia-api`
* `joblib`

---

## Notes

* The training script downloads Wikipedia articles, so ensure you have a stable internet connection.
* Wikipedia API requests are throttled with a short sleep (`0.8s`) to avoid overloading servers.
* The dataset is small and meant for demonstration purposes—accuracy may vary.

---

## License

This project is open-source under the **MIT License**.