# Fake News & Bias Detection

An AI-powered web application and browser extension for detecting potentially fake news and analyzing the credibility/bias of news content.

The project uses a machine-learning model to classify news text and provides a simple interface for submitting and analyzing news articles.

## ✨ Features

* 🤖 AI-powered fake news detection
* 📰 News text analysis
* 🔍 Bias/credibility analysis
* 🌐 Browser extension for analyzing news while browsing
* 🖥️ Web-based frontend
* ⚡ Python backend API
* 🧠 Pre-trained NLP model
* 📊 Dataset preprocessing and combination utilities
* 📦 Large model weights managed using Git LFS

## 🏗️ Project Structure

```text
fake-news-bias-detection/
│
├── backend/
│   ├── api/
│   │   └── app.py
│   │
│   ├── models/
│   │   ├── fake_news_model/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.txt
│   │   │
│   │   ├── train.py
│   │   └── test_model.py
│   │
│   └── utils/
│       ├── combine_datasets.py
│       ├── download_data.py
│       ├── download_welfake.py
│       └── preprocess.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── extension/
│   ├── background.js
│   ├── content.js
│   ├── manifest.json
│   ├── popup.css
│   └── popup.html
│
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
│       └── index.html
│
├── notebooks/
│
├── config.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

> **Note:** `venv/`, datasets, and `.env` are intentionally excluded from GitHub. The model weights are stored using Git LFS.

## 🧰 Technologies Used

### Backend

* Python
* Flask
* REST API
* Natural Language Processing
* Hugging Face Transformers

### Frontend

* HTML5
* CSS3
* JavaScript

### Browser Extension

* JavaScript
* Chrome Extension APIs
* Manifest V3

### Machine Learning

* NLP-based text classification
* Pre-trained transformer model
* WELFake dataset
* LIAR dataset

### Version Control

* Git
* GitHub
* Git LFS

## 📋 Requirements

Before starting, make sure you have:

* Python 3.9+
* Git
* Git LFS
* A modern web browser such as Google Chrome

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rudraksha-30/fake-news-bias-detection.git
cd fake-news-bias-detection
```

### 2. Install Git LFS

If Git LFS is not already installed:

```bash
git lfs install
```

Pull the model:

```bash
git lfs pull
```

Verify that the model is available:

```bash
git lfs ls-files
```

You should see:

```text
backend/models/fake_news_model/model.safetensors
```

### 3. Create a virtual environment

#### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
# Add required environment variables here
```

Do **not** commit your `.env` file.

For contributors, create a `.env.example` containing placeholder values instead of real credentials.

## 📊 Dataset Setup

The datasets are intentionally not stored in the Git repository because they are relatively large.

The project contains utility scripts for downloading and processing the datasets.

Depending on the implementation, run the appropriate scripts:

```bash
python backend/utils/download_data.py
```

```bash
python backend/utils/download_welfake.py
```

Then preprocess the datasets:

```bash
python backend/utils/preprocess.py
```

If required, combine the datasets:

```bash
python backend/utils/combine_datasets.py
```

The resulting files should be placed inside:

```text
data/
├── raw/
└── processed/
```

These directories are excluded from Git.

## ▶️ Running the Backend

From the project root:

```bash
python backend/api/app.py
```

The backend API will start locally.

The exact host and port depend on the configuration in `backend/api/app.py`.

## 🌐 Running the Frontend

The frontend is located in:

```text
frontend/
```

If the Flask application serves the frontend templates directly, simply start the backend and open the provided local address in your browser.

## 🧩 Installing the Browser Extension

1. Open Google Chrome.
2. Navigate to:

```text
chrome://extensions/
```

3. Enable **Developer mode**.
4. Click **Load unpacked**.
5. Select the project's:

```text
extension/
```

directory.
6. The extension will now appear in your installed extensions.

## 🧠 Model

The trained model is stored at:

```text
backend/models/fake_news_model/
```

The large:

```text
model.safetensors
```

file is managed using **Git Large File Storage (Git LFS)**.

This prevents the large model from being stored directly in the normal Git history.

To download the model after cloning:

```bash
git lfs pull
```

## 📁 Files Excluded from Git

The following files/directories are intentionally excluded:

```text
venv/
.env
data/raw/
data/processed/
```

Large machine-learning artifacts such as model weight files are also handled separately through Git LFS.

This keeps the Git repository lightweight while preserving the complete project setup.

## 🔄 Development Workflow

After making changes:

```bash
git add .
git commit -m "Describe your changes"
git push
```

Git LFS automatically handles the tracked model files.

## 🧪 Testing

Model testing utilities are available in:

```text
backend/models/test_model.py
```

Run:

```bash
python backend/models/test_model.py
```

## 🛠️ Troubleshooting

### Model is missing

Run:

```bash
git lfs install
git lfs pull
```

Then verify:

```bash
git lfs ls-files
```

### Python dependencies are missing

Make sure the virtual environment is activated:

```powershell
.\venv\Scripts\Activate.ps1
```

Then:

```bash
pip install -r requirements.txt
```

### Dataset files are missing

Run the dataset download and preprocessing scripts located in:

```text
backend/utils/
```

## 🔒 Security

Never commit:

* API keys
* Passwords
* Access tokens
* Database credentials
* `.env` files
* Other private credentials

The `.env` file is intentionally ignored by Git.

If a secret is accidentally pushed to GitHub, revoke/rotate it immediately.

## 📌 Future Improvements

Potential future improvements include:

* Real-time news source verification
* Improved bias classification
* Multi-language news analysis
* Article URL analysis
* More detailed credibility scoring
* Additional datasets
* Model performance dashboard
* Improved browser extension UI
* Deployment of the backend API
* Automated model updates

## 👨‍💻 Author

**Rudraksha Chouhan**

Computer Science Engineering Student

## ⭐ Acknowledgements

This project uses publicly available datasets and open-source machine-learning technologies for research and educational purposes.

---

⭐ If you find this project useful, consider giving the repository a star!
