# 📰 Fake News Detection System

## 📌 Overview

This project is a **Fake News Detection System** that uses Natural Language Processing (NLP) and Machine Learning techniques to classify news articles as **Real** or **Fake**.

The system processes textual data, applies preprocessing techniques, and uses a trained model to make predictions.

---

## 🚀 Features

* 🧹 Data preprocessing (cleaning, tokenization, normalization)
* 📊 Dataset integration (LIAR + WELFake datasets)
* 🤖 Model training using NLP techniques
* 📈 Evaluation on test data
* ⚡ Modular backend structure
* 🌐 Ready for frontend integration

---

## 🗂️ Project Structure

```
fake-news-detector/
│
├── backend/
│   ├── models/
│   │   ├── train.py
│   │   └── test_model.py
│   │
│   └── utils/
│       ├── preprocess.py
│       ├── combine_datasets.py
│       ├── download_data.py
│       └── download_welfake.py
│
├── frontend/              # (UI - optional / future work)
├── data/                  # (ignored in GitHub)
├── venv/                  # (ignored in GitHub)
│
├── config.py
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rudraksha-30/fake-news-bias-detection.git
cd fake-news-bias-detection
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

This project uses:

* **LIAR Dataset**
* **WELFake Dataset**

⚠️ Note: Dataset files are not included in the repository due to size.

👉 You can download datasets from:

* LIAR: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
* WELFake: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

---

## 🤖 Model

The model is trained using NLP techniques (tokenization, embeddings, etc.) and saved locally.

⚠️ Model files are not uploaded due to large size.

👉 Download trained model from:
*(Add your Google Drive link here)*

---

## 🧪 How to Run

### 1. Train the model

```bash
python backend/models/train.py
```

### 2. Test the model

```bash
python backend/models/test_model.py
```

---

## 🧠 How It Works

1. Data is collected from multiple datasets
2. Preprocessing is applied:

   * Lowercasing
   * Removing punctuation
   * Tokenization
3. Data is combined and split
4. Model is trained on processed data
5. Predictions are made on new input

---

## 📈 Future Improvements

* 🌐 Full frontend UI integration
* ☁️ Deploy model as API
* 📊 Improve accuracy with advanced models (BERT, etc.)
* 📱 Mobile/web application

---

## 👨‍💻 Author

**Rudraksha Chouhan**
B.Tech CSE Student

---

## ⭐ Acknowledgements

* LIAR Dataset
* WELFake Dataset
* Open-source NLP libraries

---

## 📌 Note

This project is built for **learning and academic purposes** and demonstrates the application of AI/ML in fake news detection.
