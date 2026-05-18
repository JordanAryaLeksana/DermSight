import pytest 

from llm.prompts import build_skin_disease_prompt, SYSTEM_PROMPT

@pytest.fixture

def mock_retrieval_result():
    return {
        "matched": True,
        "match_type": "exact",
        "input_label": "Acne",
        "matched_label": "Acne",
        "disease": "Acne",
        "context": "Acne is a skin condition that may cause pimples, blackheads, and oily skin.",
    }


def test_system_prompt_exists():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0
    assert "medical diagnosis" in SYSTEM_PROMPT.lower()
    
def test_build_skin_disease_prompt_valid_input(mock_retrieval_result):
    results = build_skin_disease_prompt(
        predicted_label="Acne",
        confidence=0.85,
        retrieval_result=mock_retrieval_result,
    )
    
    assert isinstance(results, list)
    assert len(results) == 2

    
def test_prompt_has_system_and_user_roles(mock_retrieval_result):
    messages = build_skin_disease_prompt(
        predicted_label="Acne",
        confidence=0.85,
        retrieval_result=mock_retrieval_result,
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_prompt_contains_prediction_data(mock_retrieval_result):
    messages = build_skin_disease_prompt(
        predicted_label="Acne",
        confidence=0.85,
        retrieval_result=mock_retrieval_result,
    )

    user_prompt = messages[1]["content"]

    assert "Acne" in user_prompt
    assert "85.0%" in user_prompt
    assert "exact" in user_prompt
    assert "pimples" in user_prompt


def test_prompt_contains_required_sections(mock_retrieval_result):
    messages = build_skin_disease_prompt(
        predicted_label="Acne",
        confidence=0.85,
        retrieval_result=mock_retrieval_result,
    )

    user_prompt = messages[1]["content"]

    assert "Possible condition:" in user_prompt
    assert "AI confidence:" in user_prompt
    assert "Simple explanation:" in user_prompt
    assert "Common signs:" in user_prompt
    assert "Suggested next steps:" in user_prompt
    assert "Medical disclaimer:" in user_prompt


def test_prompt_handles_unmatched_result():
    retrieval_result = {
        "matched": False,
        "match_type": None,
        "input_label": "UnknownLabel",
        "matched_label": None,
        "disease": None,
        "context": None,
    }

    messages = build_skin_disease_prompt(
        predicted_label="UnknownLabel",
        confidence=0.40,
        retrieval_result=retrieval_result,
    )

    user_prompt = messages[1]["content"]

    assert "Unknown skin condition" in user_prompt
    assert "No reliable disease context" in user_prompt
    assert "40.0%" in user_prompt


def test_empty_predicted_label_raises_error(mock_retrieval_result):
    with pytest.raises(ValueError):
        build_skin_disease_prompt(
            predicted_label="",
            confidence=0.85,
            retrieval_result=mock_retrieval_result,
        )


# def test_invalid_confidence_type_raises_error(mock_retrieval_result):
#     with pytest.raises(TypeError):
#         build_skin_disease_prompt(
#             predicted_label="Acne",
#             confidence="high",
#             retrieval_result=mock_retrieval_result,
#         )


def test_confidence_below_zero_raises_error(mock_retrieval_result):
    with pytest.raises(ValueError):
        build_skin_disease_prompt(
            predicted_label="Acne",
            confidence=-0.1,
            retrieval_result=mock_retrieval_result,
        )


def test_confidence_above_one_raises_error(mock_retrieval_result):
    with pytest.raises(ValueError):
        build_skin_disease_prompt(
            predicted_label="Acne",
            confidence=1.2,
            retrieval_result=mock_retrieval_result,
        )


def test_invalid_retrieval_result_type_raises_error():
    with pytest.raises(TypeError):
        build_skin_disease_prompt(
            predicted_label="Acne",
            confidence=0.85,
            retrieval_result=None,
        )