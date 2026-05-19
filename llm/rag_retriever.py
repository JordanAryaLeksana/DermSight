import pandas as pd
from pathlib import Path
from difflib import get_close_matches
from typing import Any, Dict

from utils.logger import get_logger


logger = get_logger(__name__)


class SkinDiseaseRetriever:
    def __init__(self, disease_list_path: str):
        self.disease_list = Path(disease_list_path)

        logger.info("Initializing SkinDiseaseRetriever")
        logger.info("Loading disease list from: %s", self.disease_list)

        if not self.disease_list.exists():
            logger.error("Disease list file not found at: %s", self.disease_list)
            raise FileNotFoundError(
                f"Disease list file not found at {self.disease_list}"
            )

        try:
            self.df = pd.read_csv(self.disease_list)
            logger.info(
                "Disease list loaded successfully | rows=%s | columns=%s",
                len(self.df),
                list(self.df.columns),
            )
        except Exception:
            logger.exception("Failed to read disease list CSV: %s", self.disease_list)
            raise

        required_columns = {"model_label", "disease", "llm_context"}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            logger.error(
                "Missing required columns in disease list CSV | missing=%s",
                missing_columns,
            )
            raise ValueError(
                f"Missing required columns in disease list CSV: {missing_columns}"
            )

        self.df["model_label"] = self.df["model_label"].astype(str).str.strip()
        self.df["disease"] = self.df["disease"].astype(str).str.strip()
        self.df["llm_context"] = self.df["llm_context"].astype(str).str.strip()

        self.labels = self.df["model_label"].tolist()

        logger.info(
            "SkinDiseaseRetriever initialized successfully | total_labels=%s",
            len(self.labels),
        )

    def retrieve(self, predicted_label: str) -> Dict[str, Any]:
        logger.info("Retrieving disease context | input_label=%s", predicted_label)

        if not predicted_label or not isinstance(predicted_label, str):
            logger.warning(
                "Invalid predicted_label received | value=%s | type=%s",
                predicted_label,
                type(predicted_label),
            )
            raise ValueError("predicted_label must be a non-empty string")

        predicted_label = predicted_label.strip()

        true_label = self.df[self.df["model_label"] == predicted_label]

        if not true_label.empty:
            row = true_label.iloc[0]

            logger.info(
                "Exact disease label match found | input_label=%s | disease=%s",
                predicted_label,
                row["disease"],
            )

            return {
                "matched": True,
                "match_type": "exact",
                "input_label": predicted_label,
                "matched_label": row["model_label"],
                "disease": row["disease"],
                "context": row["llm_context"],
            }

        logger.info(
            "No exact match found. Trying fuzzy match | input_label=%s",
            predicted_label,
        )

        matches = get_close_matches(
            predicted_label,
            self.labels,
            n=1,
            cutoff=0.65,
        )

        if matches:
            matched_label = matches[0]
            row = self.df[self.df["model_label"] == matched_label].iloc[0]

            logger.info(
                "Fuzzy disease label match found | input_label=%s | matched_label=%s | disease=%s",
                predicted_label,
                matched_label,
                row["disease"],
            )

            return {
                "matched": True,
                "match_type": "fuzzy",
                "input_label": predicted_label,
                "matched_label": row["model_label"],
                "disease": row["disease"],
                "context": row["llm_context"],
            }

        logger.warning(
            "No disease label match found | input_label=%s",
            predicted_label,
        )

        return {
            "matched": False,
            "match_type": None,
            "input_label": predicted_label,
            "matched_label": None,
            "disease": None,
            "context": None,
        }