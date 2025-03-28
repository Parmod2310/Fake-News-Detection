# Fake News Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Recommendations](#recommendations)
- [References](#references)
- [License](#license)
- [Contact](#contact)
- [Contributing](#contributing)

## Project Overview
The **Fake News Detection** project aims to classify news articles as "Fake" or "Real" using advanced machine learning (ML) and deep learning (DL) techniques. Leveraging Natural Language Processing (NLP) and feature engineering, this project builds a robust system to detect misinformation in news articles. The datasets `fake.csv` and `true.csv` serve as the foundation for training, evaluating, and refining classification models.

---

## Dataset
The dataset comprises 44,898 news articles, each labeled as either "Fake" (0) or "Real" (1). Metadata includes article titles, text, subjects, and publication dates.

### Dataset Statistics
- **Total Samples**: 44,898
- **Columns**: `title`, `text`, `subject`, `date`, `label`
- **Label Distribution**:
  - Fake News: 52.3% (23,481 articles)
  - Real News: 47.7% (21,417 articles)
- **Unique Values**:
  - Titles: 38,729
  - Text: 38,646
  - Subjects: 8
  - Dates: 2,397

### Key Insights from Exploratory Data Analysis (EDA)
- **Class Distribution**: Near-balanced between fake and real articles, reducing bias in model training.
- **Subjects**: Common topics include `politicsNews`, `worldNews`, and `businessNews`. Fake news often focuses on sensational subjects.
- **Text Characteristics**: Average word count is 300 words, with real news tending to be longer and more detailed.
- **Word Clouds**: 
  - Fake News: Frequent words like "trump," "clinton," "scandal."
  - Real News: Terms like "government," "policy," "economy."

---

## Methodology

### Data Preprocessing
1. Removed special characters and punctuation.
2. Converted text to lowercase.
3. Tokenized text using NLTK.
4. Removed English stopwords.
5. Applied TF-IDF vectorization (max vocabulary: 5,000 words) to extract numerical features.

### Feature Engineering
- **TF-IDF Vectorization**: Converted text into numerical features, highlighting key terms like "election," "scandal," and "climate."

### Models
#### Machine Learning
Three traditional ML models were trained and evaluated:
1. **Logistic Regression**
2. **Random Forest**
3. **Support Vector Machine (SVM)**

#### Deep Learning
- **LSTM Model**: 
  - Architecture: Embedding layer (128 dimensions), two LSTM layers (128 and 64 units), dropout layers, and a dense layer with sigmoid activation.
  - Tokenizer: Limited to 5,000 words, sequences padded to 200 tokens.

---

## Results

### Machine Learning Models
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 98.90%   | 0.99      | 0.99   | 0.99     |
| Random Forest       | **99.72%** | 1.00    | 1.00   | 1.00     |
| SVM                 | 99.48%   | 0.99      | 0.99   | 0.99     |

- **Random Forest Confusion Matrix**: 
  - True Positives: 4,330
  - True Negatives: 4,650
  - False Positives/Negatives: 0

### Deep Learning Model (LSTM)
- **Training Accuracy**: 99.73%
- **Validation Accuracy**: 99.70%
- **Classification Report**:
  - Precision, Recall, F1-Score: 1.00 for both classes.

### Key Observations
- **Random Forest** and **LSTM** achieved near-perfect performance, making them strong candidates for deployment.
- Real news uses formal, structured language; fake news often employs emotional, exaggerated tones.

---

## Installation

### Prerequisites
- Python 3.8+
- Required libraries (install via `pip`):
  ```bash
  pip install pandas numpy scikit-learn tensorflow nltk transformers matplotlib seaborn wordcloud joblib
  ```

## Setup
### 1. Clone the repository
  
  ```bash
  git clone https://github.com/Parmod2310/Fake-News-Detection.git
  cd fake-news-detection
  ```

  ### 2. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```
  ### 3. Download NLTK resources

  ```bash
  import nltk
  nltk.download('stopwords')
  nltk.download('punkt')
  ```

  ### Dataset
  - Place fake.csv and true.csv in the Dataset/ directory.
  ---
## Usage
### Training Models
### Run the preprocessing and model training script:

```bash
python train.py
```
- This will preprocess the data, train ML and DL models, and save the best model (fake_news_model.pkl).

## Predicting New Articles
### Load the trained model and predict:

```bash
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load('fake_news_model.pkl')
tfidf = TfidfVectorizer(max_features=5000)  # Ensure same settings as training
new_text = ["Breaking news! NASA discovers water on Mars."]
new_text_clean = [preprocess_text(text) for text in new_text]
new_text_vectorized = tfidf.transform(new_text_clean)
prediction = model.predict(new_text_vectorized)
print("Prediction (0: Fake, 1: True):", prediction[0])

```

---

## Recommendations
### Deployment
- Use Random Forest for lightweight, interpretable applications.
- Deploy LSTM or BERT for high-accuracy, context-aware scenarios.
### Future Work
- Explore ensemble methods combining ML and DL models.
- Extend to multilingual fake news detection.
### Scalability
- Implement real-time scraping and classification pipelines.

---

## References
- [Fake News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

## License
This project is licensed under the MIT License - see the  file for details.

---

## Contact

For questions or contributions, please reach out via p921035@gmail.com.

---

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---
