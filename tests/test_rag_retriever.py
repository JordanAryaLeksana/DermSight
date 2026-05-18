import pandas as pd
import pytest

from llm.rag_retriever import SkinDiseaseRetriever


@pytest.fixture
def sample_disease_csv(tmp_path):


    data = {
        "model_label": ["Acne", "FU-ringworm", "BA-impetigo"],
        "disease": ["Acne", "Fungal ringworm", "Bacterial impetigo"],
        "llm_context": [
            "Context about acne",
            "Context about fungal ringworm",
            "Context about bacterial impetigo",
        ],
    }

    df = pd.DataFrame(data)

    csv_path = tmp_path / "sample_disease_list.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def test_retrieve_exact_match(sample_disease_csv):
    retriever = SkinDiseaseRetriever(sample_disease_csv)

    result = retriever.retrieve("FU-ringworm")

    assert result["matched"] is True
    assert result["match_type"] == "exact"
    assert result["input_label"] == "FU-ringworm"
    assert result["matched_label"] == "FU-ringworm"
    assert result["disease"] == "Fungal ringworm"
    assert result["context"] == "Context about fungal ringworm"


def test_retrieve_fuzzy_match(sample_disease_csv):
    retriever = SkinDiseaseRetriever(sample_disease_csv)

    result = retriever.retrieve("FU-ringwrom")

    assert result["matched"] is True
    assert result["match_type"] == "fuzzy"
    assert result["matched_label"] == "FU-ringworm"
    assert result["disease"] == "Fungal ringworm"


def test_retrieve_no_match(sample_disease_csv):
    retriever = SkinDiseaseRetriever(sample_disease_csv)

    result = retriever.retrieve("UnknownDisease")

    assert result["matched"] is False
    assert result["match_type"] is None
    assert result["matched_label"] is None
    assert result["disease"] is None
    assert result["context"] is None


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        SkinDiseaseRetriever("file_yang_tidak_ada.csv")


def test_missing_required_columns_raises_error(tmp_path):
    data = {
        "label": ["Acne"],
        "description": ["Some context"],
    }

    df = pd.DataFrame(data)

    csv_path = tmp_path / "invalid.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        SkinDiseaseRetriever(csv_path)


def test_retrieve_strips_input_whitespace(sample_disease_csv):
    retriever = SkinDiseaseRetriever(sample_disease_csv)

    result = retriever.retrieve("  Acne  ")

    assert result["matched"] is True
    assert result["match_type"] == "exact"
    assert result["matched_label"] == "Acne"