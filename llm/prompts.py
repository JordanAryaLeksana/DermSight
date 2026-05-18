from typing import Dict, List, Any


SYSTEM_PROMPT = """
You are a medical education assistant specializing in skin disease information.

Your task is to explain AI-based skin disease classification results in a safe,
clear, and patient-friendly way.

Rules:
- Do not provide a definitive medical diagnosis.
- Always explain that the result comes from an AI image classification model.
- Use only the disease context provided by the knowledge base.
- If the context is missing or unavailable, clearly say that the information is limited.
- Do not recommend prescription medication.
- Do not provide emergency diagnosis.
- Recommend consulting a qualified doctor or dermatologist.
- Use simple language that non-medical users can understand.
- Be calm, respectful, and reassuring.
""".strip()


def build_skin_disease_prompt(
    predicted_label: str,
    confidence: float,  
    retrieval_result: Dict[str, Any],
) -> List[Dict[str, str]]:
    if not isinstance(predicted_label, str) or not predicted_label.strip():
        raise ValueError("Predicted label must be a non-empty string.")
    
    if not isinstance(confidence, (float, int)):
        raise ValueError("Confidence must be a number (float or int).")
    
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")

    if not isinstance(retrieval_result, dict):
        raise TypeError("retrieval_result must be a dictionary")
    
    matched = retrieval_result.get("matched", False)
    match_type = retrieval_result.get("match_type")
    matched_label = retrieval_result.get("matched_label")
    disease = retrieval_result.get("disease")
    context = retrieval_result.get("context")
    
    if matched:
        disease_name = str(disease).strip() if disease else "Unknown skin condition"
        disease_context = str(context).strip() if context else "No disease context was provided."
        matched_label_text = str(matched_label).strip() if matched_label else "Unknown"
        match_type_text = str(match_type).strip() if match_type else "unknown"
    else:
        disease_name = "Unknown skin condition"
        disease_context = "No reliable disease context was found for this prediction."
        matched_label_text = "None"
        match_type_text = "none"
        
    confidence_percent = round(float(confidence * 100), 2)
    user_prompt = f"""
An AI image classification model produced the following result:

Predicted model label:
{predicted_label.strip()}

Matched knowledge base label:
{matched_label_text}

Disease name:
{disease_name}

Model confidence:
{confidence_percent}%

Match type:
{match_type_text}

Knowledge base context:
{disease_context}

Generate a user-facing explanation in the following format:

Possible condition:
Explain the possible skin condition detected by the AI model.

AI confidence:
Explain the confidence score in simple terms. Do not overstate certainty.

Simple explanation:
Explain the condition using simple non-technical language.

Common signs:
List common visible signs or symptoms based only on the provided context.

Suggested next steps:
Give safe and practical next steps, such as monitoring symptoms and consulting
a doctor or dermatologist.

Medical disclaimer:
State clearly that this is not a medical diagnosis and that the user should
consult a qualified healthcare professional for confirmation.
""".strip()

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]