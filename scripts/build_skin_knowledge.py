import time
import re
import html
import csv
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from difflib import SequenceMatcher


BASE_DIR = Path("/home/jordan/Backup HDD/AI_Machine Learning/DermSight")

DATASET_DIR = BASE_DIR / "/test"     
LLM_DATA_DIR = BASE_DIR / "llm/data"

DISEASE_LIST_PATH = LLM_DATA_DIR / "skin_disease_list.csv"
EXPANDED_PATH = LLM_DATA_DIR / "skin_knowledge_expanded.csv"
RAG_PATH = LLM_DATA_DIR / "skin_knowledge_rag.csv"

SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"

LABEL_KNOWLEDGE_MAP = {
    "Vascular_Tumors": {
        "disease": "Vascular skin tumors",
        "queries": ["hemangioma", "cherry angioma", "pyogenic granuloma", "vascular skin lesion"]
    },
    "Vasculitis": {
        "disease": "Cutaneous vasculitis",
        "queries": ["cutaneous vasculitis", "vasculitis skin"]
    },
    "SkinCancer": {
        "disease": "Skin cancer",
        "queries": ["skin cancer", "melanoma", "basal cell carcinoma", "squamous cell carcinoma"]
    },
    "Bullous": {
        "disease": "Bullous skin diseases",
        "queries": ["bullous pemphigoid", "pemphigus", "blistering skin disease"]
    },
    "Infestations_Bites": {
        "disease": "Infestations and bites",
        "queries": ["insect bites", "scabies", "lice", "skin infestation"]
    },
    "Lupus": {
        "disease": "Cutaneous lupus",
        "queries": ["cutaneous lupus", "lupus skin rash"]
    },
    "Lichen": {
        "disease": "Lichen skin diseases",
        "queries": ["lichen planus", "lichen sclerosus"]
    },
    "Acne": {
        "disease": "Acne",
        "queries": ["acne", "acne vulgaris"]
    },
    "Rosacea": {
        "disease": "Rosacea",
        "queries": ["rosacea"]
    },
    "Vitiligo": {
        "disease": "Vitiligo",
        "queries": ["vitiligo"]
    },
    "DrugEruption": {
        "disease": "Drug eruption",
        "queries": ["drug eruption", "drug rash", "adverse drug skin reaction"]
    },
    "Sun_Sunlight_Damage": {
        "disease": "Sun damage",
        "queries": ["sunburn", "sun damage skin", "photodamage", "actinic damage"]
    },
    "Eczema": {
        "disease": "Eczema / Atopic dermatitis",
        "queries": ["eczema", "atopic dermatitis"]
    },
    "Actinic_Keratosis": {
        "disease": "Actinic keratosis",
        "queries": ["actinic keratosis"]
    },
    "Psoriasis": {
        "disease": "Psoriasis",
        "queries": ["psoriasis"]
    },
    "Benign_tumors": {
        "disease": "Benign skin tumors",
        "queries": ["benign skin tumor", "dermatofibroma", "lipoma", "skin cyst"]
    },
    "Seborrh_Keratoses": {
        "disease": "Seborrheic keratosis",
        "queries": ["seborrheic keratosis"]
    },
    "Moles": {
        "disease": "Moles / Nevi",
        "queries": ["moles", "nevus", "skin mole"]
    },
    "Unknown_Normal": {
        "disease": "Normal skin / unknown lesion",
        "queries": ["normal skin", "benign skin lesion", "skin self examination"]
    },
    "Warts": {
        "disease": "Warts",
        "queries": ["warts", "human papillomavirus skin warts"]
    },
    "Tinea": {
        "disease": "Tinea / Ringworm",
        "queries": ["ringworm", "tinea", "athlete's foot"]
    },
    "Candidiasis": {
        "disease": "Cutaneous candidiasis",
        "queries": ["cutaneous candidiasis", "candida skin infection"]
    },
}

DEFAULT_MANUAL = {
    "symptoms": "Information varies by subtype. Common signs may include rash, color change, bumps, scaling, itching, pain, swelling, blisters, crusting, or abnormal skin growth.",
    "causes": "Possible causes vary by condition and may include infection, inflammation, immune reaction, allergy, irritation, sun exposure, genetics, medications, or benign/malignant skin growth.",
    "treatment": "Treatment depends on the exact diagnosis. General care includes gentle skin care, avoiding triggers, keeping the area clean, and seeking medical evaluation when symptoms are severe or persistent.",
    "medications": "Medicines depend on the diagnosis and may include topical creams, antifungal agents, antibiotics, anti-inflammatory medicines, antivirals, antiparasitic medicines, or specialist-prescribed therapy.",
    "prevention": "Use sun protection, avoid sharing towels or personal items, maintain hygiene, avoid known triggers, protect skin from injury, and seek early care for suspicious or worsening lesions.",
    "doctor_advice": "See a doctor if the lesion changes rapidly, bleeds, is painful, spreads, shows infection signs, affects the eye/genitals, or does not improve."
}

MANUAL_KNOWLEDGE = {
    "Acne": {
        "symptoms": "Pimples, blackheads, whiteheads, oily skin, inflamed bumps, and possible scarring.",
        "causes": "Clogged hair follicles, excess oil, bacteria, inflammation, and hormonal changes.",
        "treatment": "Clean skin gently, avoid squeezing pimples, and use acne treatment when needed.",
        "medications": "Benzoyl peroxide, salicylic acid, topical retinoids, topical antibiotics, oral antibiotics, or isotretinoin under doctor supervision.",
        "prevention": "Use gentle cleanser, avoid comedogenic products, wash after sweating, and avoid picking acne.",
        "doctor_advice": "See a doctor if acne is severe, painful, leaves scars, or does not improve."
    },
    "Eczema": {
        "symptoms": "Dry skin, itching, redness, rash, swelling, cracked skin, and repeated flare-ups.",
        "causes": "Skin barrier problems, irritants, allergens, stress, environmental triggers, and genetic factors.",
        "treatment": "Moisturize regularly, avoid triggers, use gentle skin care, and treat inflammation during flare-ups.",
        "medications": "Moisturizers, topical corticosteroids, topical calcineurin inhibitors, antihistamines, or other prescription medicines.",
        "prevention": "Avoid harsh soaps, fragrances, irritating fabrics, allergens, and keep skin moisturized.",
        "doctor_advice": "See a doctor if eczema is severe, infected, painful, or disrupts sleep."
    },
    "Psoriasis": {
        "symptoms": "Red or darker raised plaques, silvery scale, itching, burning, cracked skin, or nail changes.",
        "causes": "Immune system inflammation, genetic tendency, infections, stress, skin injury, or medication triggers.",
        "treatment": "Moisturizers, trigger control, topical therapy, light therapy, or systemic treatment for severe disease.",
        "medications": "Topical corticosteroids, vitamin D analogs, retinoids, phototherapy, methotrexate, biologics, or other specialist medicines.",
        "prevention": "Avoid triggers, manage stress, avoid skin injury, moisturize, and follow treatment consistently.",
        "doctor_advice": "See a doctor if plaques are widespread, painful, infected, or joint pain occurs."
    },
    "Rosacea": {
        "symptoms": "Facial redness, flushing, visible blood vessels, acne-like bumps, burning, stinging, or eye irritation.",
        "causes": "Chronic inflammatory skin condition triggered by heat, sunlight, alcohol, spicy food, stress, or certain products.",
        "treatment": "Avoid triggers, use gentle skin care, sun protection, and prescription treatment if needed.",
        "medications": "Metronidazole, azelaic acid, ivermectin, oral doxycycline, or laser therapy for visible vessels.",
        "prevention": "Use sunscreen, avoid personal triggers, and avoid harsh skin products.",
        "doctor_advice": "See a doctor if symptoms worsen, eyes are irritated, or redness becomes persistent."
    },
    "Vitiligo": {
        "symptoms": "White or lighter patches of skin, often on face, hands, arms, feet, or areas around body openings.",
        "causes": "Loss of pigment-producing cells, often related to autoimmune factors and genetic risk.",
        "treatment": "Sun protection, camouflage, topical therapy, phototherapy, or specialist treatment.",
        "medications": "Topical corticosteroids, calcineurin inhibitors, phototherapy, or other dermatologist-directed therapy.",
        "prevention": "Protect depigmented skin from sunburn and avoid skin trauma when possible.",
        "doctor_advice": "See a doctor if new white patches appear or spread."
    },
    "Warts": {
        "symptoms": "Rough bumps on skin, sometimes painful, flat, filiform, or clustered depending on type.",
        "causes": "Human papillomavirus infection spread by contact or skin breaks.",
        "treatment": "Many warts resolve slowly; treatment may include topical agents, freezing, or removal.",
        "medications": "Salicylic acid, cryotherapy, cantharidin, or other clinician-directed treatments.",
        "prevention": "Avoid picking, cover warts, wear footwear in public showers, and avoid sharing personal items.",
        "doctor_advice": "See a doctor if warts are painful, spreading, bleeding, on face/genitals, or if immunity is weak."
    },
    "Tinea / Ringworm": {
        "symptoms": "Itchy, red, scaly, ring-shaped rash or peeling skin, depending on body location.",
        "causes": "Dermatophyte fungal infection spread by people, animals, clothing, towels, or damp surfaces.",
        "treatment": "Keep area dry and use antifungal medicine.",
        "medications": "Clotrimazole, terbinafine, miconazole, ketoconazole, or oral antifungals for scalp/widespread infection.",
        "prevention": "Avoid sharing towels/clothes, keep skin dry, treat infected pets, and wash hands.",
        "doctor_advice": "See a doctor if rash is on scalp, widespread, recurrent, or not improving."
    },
    "Cutaneous candidiasis": {
        "symptoms": "Red itchy rash, moist patches, satellite bumps, cracking, or soreness in skin folds.",
        "causes": "Candida yeast overgrowth in warm, moist areas, often worsened by sweating, diabetes, antibiotics, or weak immunity.",
        "treatment": "Keep skin dry, reduce friction, and use antifungal treatment.",
        "medications": "Clotrimazole, miconazole, nystatin, or oral fluconazole for selected cases under medical advice.",
        "prevention": "Keep folds dry, change sweaty clothes, manage diabetes, and avoid prolonged moisture.",
        "doctor_advice": "See a doctor if rash spreads, recurs often, is painful, or occurs with diabetes/weak immunity."
    },
    "Skin cancer": {
        "symptoms": "New or changing mole, non-healing sore, bleeding lesion, irregular border, color change, or growing bump.",
        "causes": "UV exposure, tanning beds, fair skin, age, genetic risk, immune suppression, or previous skin cancer.",
        "treatment": "Needs medical diagnosis. Treatment may include excision, topical therapy, cryotherapy, radiation, or oncology care depending on type.",
        "medications": "Medication depends on cancer type; may include topical fluorouracil/imiquimod for selected lesions or systemic therapy for advanced disease.",
        "prevention": "Use sunscreen, protective clothing, avoid tanning beds, and perform skin self-checks.",
        "doctor_advice": "See a doctor urgently for ABCDE mole changes, bleeding, rapid growth, or non-healing sores."
    },
    "Actinic keratosis": {
        "symptoms": "Rough, scaly, dry patches on sun-exposed skin, sometimes pink, red, brown, or tender.",
        "causes": "Long-term ultraviolet exposure causing precancerous skin changes.",
        "treatment": "Dermatology treatment may include freezing, topical medicines, curettage, or photodynamic therapy.",
        "medications": "Fluorouracil, imiquimod, diclofenac gel, tirbanibulin, or other doctor-prescribed field therapy.",
        "prevention": "Sun protection, sunscreen, hats, protective clothing, and avoiding tanning beds.",
        "doctor_advice": "See a doctor because actinic keratosis can resemble or progress to skin cancer."
    }
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(label: str) -> str:
    return str(label).strip().replace(" ", "_")


def humanize_label(label: str) -> str:
    return normalize_label(label).replace("_", " ")


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def parse_document(document):
    result = {
        "matched_title": "",
        "description": "",
        "source_url": "",
    }

    for content in document.findall("content"):
        name = content.attrib.get("name")
        value = clean_text("".join(content.itertext()))

        if name == "title":
            result["matched_title"] = value
        elif name == "FullSummary":
            result["description"] = value
        elif name == "url":
            result["source_url"] = value

    return result


def search_medlineplus(query: str, retmax: int = 5):
    params = {
        "db": "healthTopics",
        "term": query,
        "retmax": retmax,
    }

    response = requests.get(
        SEARCH_URL,
        params=params,
        timeout=20,
        headers={"User-Agent": "DermSight-Capstone/1.0"},
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    documents = root.findall(".//document")

    candidates = []

    for document in documents:
        item = parse_document(document)
        title = item["matched_title"]
        desc = item["description"]

        score = similarity(query, title) * 3

        if query.lower() in title.lower():
            score += 3

        if query.lower() in desc.lower():
            score += 1

        if item["description"]:
            candidates.append((score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in candidates]


def scrape_label_knowledge(model_label: str, queries: list[str]):
    all_items = []
    seen_urls = set()
    seen_titles = set()

    for query in queries:
        print(f"  - query: {query}")

        try:
            items = search_medlineplus(query, retmax=5)
        except Exception as e:
            print(f"    error: {e}")
            items = []

        for item in items:
            key_url = item.get("source_url", "")
            key_title = item.get("matched_title", "").lower()

            if key_url and key_url in seen_urls:
                continue

            if key_title and key_title in seen_titles:
                continue

            item["query"] = query
            all_items.append(item)

            if key_url:
                seen_urls.add(key_url)

            if key_title:
                seen_titles.add(key_title)

        time.sleep(0.7)

    return all_items


def combine_scraped_items(items: list[dict]) -> dict:
    if not items:
        return {
            "matched_title": "",
            "description": "",
            "source_url": "",
            "all_topics": "",
            "all_sources": "",
        }

    matched_titles = []
    descriptions = []
    source_urls = []

    for item in items:
        title = clean_text(item.get("matched_title", ""))
        desc = clean_text(item.get("description", ""))
        url = clean_text(item.get("source_url", ""))
        query = clean_text(item.get("query", ""))

        if title:
            matched_titles.append(title)

        if desc:
            descriptions.append(f"Topic: {title or query}. {desc}")

        if url:
            source_urls.append(url)

    return {
        "matched_title": " | ".join(matched_titles),
        "description": "\n\n".join(descriptions),
        "source_url": " | ".join(source_urls),
        "all_topics": " | ".join(matched_titles),
        "all_sources": " | ".join(source_urls),
    }


def get_manual_knowledge(disease: str, model_label: str) -> dict:
    return (
        MANUAL_KNOWLEDGE.get(disease)
        or MANUAL_KNOWLEDGE.get(model_label)
        or DEFAULT_MANUAL
    )


def build_llm_context(row: dict) -> str:
    return f"""
Model label:
{row["model_label"]}

Disease / category:
{row["disease"]}

Medical topics:
{row["matched_title"]}

Description:
{row["description"]}

Symptoms:
{row["symptoms"]}

Causes:
{row["causes"]}

Treatment / Management:
{row["treatment"]}

Possible medicines:
{row["medications"]}

Prevention:
{row["prevention"]}

When to see a doctor:
{row["doctor_advice"]}

Safety note:
This information is educational. It should not replace diagnosis or treatment from a healthcare professional.
""".strip()


def build_disease_list_from_folders(dataset_dir: Path, output_path: Path):
    rows = []

    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue

        model_label = normalize_label(folder.name)

        config = LABEL_KNOWLEDGE_MAP.get(model_label, {
            "disease": humanize_label(model_label),
            "queries": [humanize_label(model_label)]
        })

        rows.append({
            "model_label": model_label,
            "disease": config["disease"],
            "queries": " | ".join(config["queries"]),
            "image_count": len(list(folder.glob("*")))
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    return df


def load_or_create_disease_list():
    if DATASET_DIR.exists():
        print(f"Building disease list from folder: {DATASET_DIR}")
        return build_disease_list_from_folders(DATASET_DIR, DISEASE_LIST_PATH)

    print(f"Dataset folder not found. Reading existing CSV: {DISEASE_LIST_PATH}")
    return pd.read_csv(DISEASE_LIST_PATH)


def main():
    LLM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    disease_df = load_or_create_disease_list()
    rows = []

    for _, item in disease_df.iterrows():
        model_label = normalize_label(item["model_label"])

        config = LABEL_KNOWLEDGE_MAP.get(model_label, {
            "disease": item.get("disease", humanize_label(model_label)),
            "queries": [item.get("disease", humanize_label(model_label))]
        })

        disease = config["disease"]
        queries = config["queries"]

        print(f"\nSearching knowledge for: {model_label} -> {disease}")

        scraped_items = scrape_label_knowledge(model_label, queries)
        scraped = combine_scraped_items(scraped_items)
        extra = get_manual_knowledge(disease, model_label)

        final_row = {
            "model_label": model_label,
            "disease": disease,
            "queries": " | ".join(queries),
            "matched_title": scraped["matched_title"],
            "description": scraped["description"],
            "symptoms": extra["symptoms"],
            "causes": extra["causes"],
            "treatment": extra["treatment"],
            "medications": extra["medications"],
            "prevention": extra["prevention"],
            "doctor_advice": extra["doctor_advice"],
            "source_url": scraped["source_url"],
            "all_topics": scraped["all_topics"],
            "all_sources": scraped["all_sources"],
            "image_count": item.get("image_count", ""),
        }

        final_row["llm_context"] = build_llm_context(final_row)
        rows.append(final_row)

    output_df = pd.DataFrame(rows)

    for col in output_df.columns:
        output_df[col] = (
            output_df[col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace("nan", "")
        )

    output_df.to_csv(
        EXPANDED_PATH,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL
    )

    rag_df = output_df[[
        "model_label",
        "disease",
        "queries",
        "matched_title",
        "llm_context",
        "source_url"
    ]]

    rag_df.to_csv(
        RAG_PATH,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL
    )

    print(f"\nSaved disease list to: {DISEASE_LIST_PATH}")
    print(f"Saved expanded CSV to: {EXPANDED_PATH}")
    print(f"Saved RAG CSV to: {RAG_PATH}")

    print(output_df[[
        "model_label",
        "disease",
        "queries",
        "matched_title",
        "source_url"
    ]])


if __name__ == "__main__":
    main()