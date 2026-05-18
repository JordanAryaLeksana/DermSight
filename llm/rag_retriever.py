import pandas as pd
from pathlib import Path
from difflib import get_close_matches


class SkinDiseaseRetriever:
    def __init__(self, disease_list_path: str):
        self.disease_list = Path(disease_list_path)

        if not self.disease_list.exists():
            raise FileNotFoundError(
                f"Disease list file not found at {self.disease_list}"
            )

        self.df = pd.read_csv(self.disease_list)

        required_columns = {"model_label", "disease", "llm_context"}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in disease list CSV: {missing_columns}"
            )

        self.df["model_label"] = self.df["model_label"].astype(str).str.strip()
        self.df["disease"] = self.df["disease"].astype(str).str.strip()
        self.df["llm_context"] = self.df["llm_context"].astype(str).str.strip()

        self.labels = self.df["model_label"].tolist()
        
    def retrieve(self, predicted_label: str):
        predicted_label = predicted_label.strip()
        
        true_label = self.df[self.df["model_label"] == predicted_label]
        if not true_label.empty:
            row = true_label.iloc[0]
            return {
                "matched": True,
                "match_type": "exact",
                "input_label": predicted_label,
                "matched_label": row["model_label"],
                "disease": row["disease"],
                "context": row["llm_context"],
            }
        matches = get_close_matches(predicted_label, self.labels, n=1, cutoff=0.65)
        if matches:
            matched_label = matches[0]
            row = self.df[self.df["model_label"] == matched_label].iloc[0]
            return {
                "matched": True,
                "match_type": "fuzzy",
                "input_label": predicted_label,
                "matched_label": row["model_label"],
                "disease": row["disease"],
                "context": row["llm_context"],
            }
        return {
            "matched": False,
            "match_type": None,
            "input_label": predicted_label,
            "matched_label": None,
            "disease": None,
            "context": None,
        }