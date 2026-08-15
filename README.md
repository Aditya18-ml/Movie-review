# 🎬 IMDB Sentiment Analysis API

An end-to-end **Machine Learning and Natural Language Processing (NLP)** project that classifies IMDB movie reviews as **positive or negative**.

The project covers the complete ML pipeline, including data exploration, text preprocessing, TF-IDF feature extraction, model training, evaluation, and deployment through a **FastAPI REST API**.

## 🚀 Features

* **Exploratory Data Analysis (EDA)**

  * Dataset inspection
  * Shape and data-type checking
  * Duplicate detection and removal
  * Missing-value analysis

* **NLP Preprocessing**

  * HTML tag removal
  * Regex-based text cleaning
  * Lowercasing
  * Stopword removal
  * WordNet lemmatization using NLTK

* **Feature Extraction**

  * TF-IDF vectorization
  * Unigrams and bigrams

* **Machine Learning**

  * Logistic Regression classifier
  * Model evaluation
  * Serialized model and vectorizer artifacts using Joblib

* **REST API**

  * FastAPI-based prediction service
  * Uvicorn server
  * Interactive Swagger API documentation

## 🧠 Machine Learning Pipeline

```text
IMDB Review
     ↓
Data Cleaning
     ↓
HTML & Regex Cleaning
     ↓
Lowercasing
     ↓
Stopword Removal
     ↓
Lemmatization
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Positive / Negative Sentiment
```

## 📂 Project Structure

```text
Movie-review/
│
├── IMDB Dataset.csv       # IMDB movie review dataset
├── eda.py                 # Data inspection and sanity checks
├── preprocessing.py       # Text cleaning and NLP preprocessing
├── train_test.py          # Model training and evaluation
├── main.py                # Training pipeline entry point
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Aditya18-ml/Movie-review.git
cd Movie-review
```

### 2. Create and Activate a Virtual Environment

#### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ How to Run

### 1. Train the Model

Run the main training pipeline:

```bash
python main.py
```

This performs preprocessing, TF-IDF feature extraction, Logistic Regression training, evaluation, and generates the required `.joblib` model artifacts.

### 2. Start the FastAPI Server

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 3. Open Swagger UI

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

You can use the interactive Swagger interface to test the `/predict` endpoint.

## 🧪 API Usage

### Endpoint

```text
POST /predict
```

### Request

```json
{
  "review": "The movie had fantastic visual effects and compelling performances!"
}
```

### Response

```json
{
  "review": "The movie had fantastic visual effects and compelling performances!",
  "sentiment": "positive"
}
```

## 🧰 Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| Pandas       | Data manipulation              |
| NumPy        | Numerical operations           |
| NLTK         | NLP preprocessing              |
| Scikit-learn | TF-IDF and Logistic Regression |
| Joblib       | Model serialization            |
| FastAPI      | REST API                       |
| Uvicorn      | API server                     |
| Git & GitHub | Version control                |

## 📊 Model

The project uses:

**TF-IDF Vectorizer**

* Unigrams + bigrams
* Converts processed text into numerical feature vectors

**Logistic Regression**

* Binary classification
* Predicts either `positive` or `negative` sentiment

## 🔮 Future Improvements

* Add additional ML models such as SVM, Random Forest, and XGBoost
* Compare different NLP preprocessing techniques
* Add model confidence/probability scores
* Build a frontend interface for real-time predictions
* Containerize the application using Docker
* Deploy the API using a cloud platform
* Add automated testing and CI/CD

## 👨‍💻 Author

**Aditya Kumar**

BTech — Computer Science & Engineering
Data Science & Machine Learning

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
