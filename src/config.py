# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class Config:
    # API Keys
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

    # DashScope Compatible OpenAI Protocol Configuration
    DASHSCOPE_BASE_URL = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    QWEN_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3-max")

    # Path Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # DATA_RAW_DIR removed as per user request
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    # FIGURES_DIR removed


    # LLM Configuration
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

    # Alpha Vantage Configuration
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

    # Target Company Configuration
    TARGET_COMPANY = {
        "symbol": "MSFT",
        "name": "Microsoft Corporation",
        "cik": "0000789019",
    }

    # Financial Data Year Range
    START_YEAR = 2020
    END_YEAR = 2024

    @classmethod
    def validate_keys(cls, verbose=True):
        """Validate if required API Keys exist"""
        missing_keys = []
        if not cls.DASHSCOPE_API_KEY:
            missing_keys.append("DASHSCOPE_API_KEY")
        if not cls.ALPHA_VANTAGE_KEY:
            missing_keys.append("ALPHA_VANTAGE_API_KEY")

        if missing_keys:
            raise ValueError(f"❌ Missing required API Keys: {', '.join(missing_keys)}. Please check .env file.")
        if verbose:
            print("✅ Configuration loaded successfully")

    @classmethod
    def create_directories(cls, verbose=True):
        """Create necessary directories"""
        dirs = [cls.DATA_PROCESSED_DIR, cls.REPORTS_DIR]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
        if verbose:
            print("✅ Directory structure created")

