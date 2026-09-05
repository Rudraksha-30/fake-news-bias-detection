import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths
    DATA_RAW_PATH = "data/raw/"
    DATA_PROCESSED_PATH = "data/processed/"
    MODEL_PATH = "backend/models/"

    # Model
    BASE_MODEL = "distilbert-base-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # API Keys
    FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")

    # Labels
    FAKE_NEWS_LABELS = {0: "REAL", 1: "FAKE"}
    BIAS_LABELS = ["political", "gender", "racial", "religious", "age", "occupational"]
```

**`.gitignore`**:
```
venv/
__pycache__/
*.pyc
.env
backend/models/*.pt
data/raw/
*.egg-info/
.DS_Store