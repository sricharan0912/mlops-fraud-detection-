from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "models/lgbm.pkl"
    database_url: str = "postgresql://fraud_user:changeme@localhost:5432/fraud"

    model_config = {"env_file": ".env", "case_sensitive": False}


settings = Settings()
